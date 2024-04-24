from pathlib import Path
import argparse
import os
import random
import neptune
from torch import nn, optim
from torch.utils.data import Dataset
# from utilscapsnet import AverageMeter
import torch
import numpy as np
import resnet
# from capsule_network import resnet20 
# ref https://discuss.pytorch.org/t/difference-between-torch-manual-seed-and-torch-cuda-manual-seed/13848/7
seed_value = 42
# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

# 4. Set `pytorch` pseudo-random generator at a fixed value
import torch
torch.manual_seed(seed_value)
from data_loader import get_test_loader, get_train_valid_loader, VIEWPOINT_EXPS

class AverageMeter(object):
    """
    Computes and stores the average and
    current value.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

 

def get_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate a pretrained model on CIFAR-10"
    )

    # Data
    parser.add_argument("--data-dir", type=Path, help="path to dataset")
    
    # Checkpoint
    parser.add_argument("--pretrained", type=Path, help="path to pretrained model")
    parser.add_argument(
        "--exp-dir",
        default="./checkpoint/lincls/",
        type=Path,
        metavar="DIR",
        help="path to checkpoint directory",
    )

    # Model
    parser.add_argument("--arch", type=str, default="resnet18")

    # Optim
    parser.add_argument(
        "--epochs",
        default=100,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--batch-size", default=256, type=int, metavar="N", help="mini-batch size"
    )
    parser.add_argument(
        "--lr-backbone",
        default=0.0,
        type=float,
        metavar="LR",
        help="backbone base learning rate",
    )
     
    # Running
    parser.add_argument(
        "--workers",
        default=8,
        type=int,
        metavar="N",
        help="number of data loader workers",
    )

    return parser


def main():
    parser = get_arguments()
    args = parser.parse_args()
    args.ngpus_per_node = torch.cuda.device_count()
    # single-gpu training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    main_worker(device, args)

def exclude_bias_and_norm(p):
    return p.ndim == 1

def main_worker(gpu, args):
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    
    torch.backends.cudnn.benchmark = True

    # run = neptune.init_run(project = 'enter your username and project name', dependencies="infer",
    # api_token="enter your api token")

    backbone, _ = resnet.__dict__[args.arch](
            zero_init_residual=True
        )
    state_dict = torch.load("model.pth", map_location="cpu")
    model_state_dict = state_dict['model']
    print(model_state_dict.keys())
    backbone_state_dict = {k[len("backbone."):]: v for k, v in model_state_dict.items() if k.startswith("backbone.")}

    backbone.load_state_dict(backbone_state_dict, strict=True)
     
    head = nn.Linear(512, 5)
    head.weight.data.normal_(mean=0.0, std=0.01)
    head.bias.data.zero_()
    model = nn.Sequential(backbone, head)
    # print(model)
    model.cuda(gpu)
    
    backbone.requires_grad_(False)
    head.requires_grad_(True)

    # check if frozen or not
    model.to(gpu)
    kwargs = {'num_workers': 0, 'pin_memory': False}

    # instantiate data loaders
    data_loader = get_train_valid_loader(
            "./data","smallnorb", args.batch_size,
            seed_value, "azimuth", "0.1",
            True, **kwargs
        ) 
    train_loader = data_loader[0]
    valid_loader = data_loader[1]
    num_train = len(train_loader.dataset)
    num_valid = len(valid_loader.dataset)  

    # familiar = False means that the dataset is not familiar to the model, so we test on NOVEL.
    test_loader = get_test_loader(
     "./data", "smallnorb", args.batch_size, "azimuth", False,
    **kwargs
    )  
    num_test = len(test_loader.dataset)
    print("Length of testing dataset", len(test_loader.dataset))
     
    test_loader_familiar = get_test_loader(
     "./data", "smallnorb", args.batch_size, "azimuth", True,
    **kwargs
    )  
    
    print("Length of testing familiar dataset", len(test_loader_familiar.dataset))
    
    print("num_train", num_train)
    
    print("num_valid", num_valid)
     
    print("Data loaders created successfully")
    
    print("Length of training dataset", len(train_loader.dataset))
    print("Length of validation dataset", len(valid_loader.dataset))
 
     
    print("\n[*] Train on {} samples, validate on {} samples".format(
        num_train, num_valid)
    )
    
    criterion = nn.CrossEntropyLoss().to(gpu)
    optimizer = optim.SGD(head.parameters(), lr=0.3, momentum=0.9, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    best_valid_acc = 0
    for epoch in range(0, args.epochs):
        # get current lr
        for i, param_group in enumerate(optimizer.param_groups):
            lr = float(param_group['lr'])
            break
        
        # train for 1 epoch
        # train_loss, train_acc, model, optimizer, scheduler = train_one_epoch(epoch)
        model.eval()
        backbone.eval()
        head.train()

        losses = AverageMeter()
        accs = AverageMeter()
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(gpu), y.to(gpu)
 
            out = model(x)
            
            loss = criterion(out, y)

            # compute accuracy
            pred = torch.max(out, 1)[1]
            correct = (pred == y).float()
            acc = 100 * (correct.sum() / len(y))

            # store
            losses.update(loss.data.item(), x.size()[0])
            accs.update(acc.data.item(), x.size()[0])

            # compute gradients and update SGD
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss, train_acc = losses.avg, accs.avg
        # run["after_pretraining/training/epoch/loss"].log(train_loss)
        # run["after_pretraining/training/epoch/acc"].log(train_acc)
        # evaluate on validation set
        with torch.no_grad():
            
            model.eval()

            losses = AverageMeter()
            accs = AverageMeter()

            for i, (x, y) in enumerate(valid_loader):
                x, y = x.to(gpu), y.to(gpu)

                out = model(x)
                
                loss = criterion(out, y)

                # compute accuracy
                pred = torch.max(out, 1)[1]
                correct = (pred == y).float()
                acc = 100 * (correct.sum() / len(y))

                # store
                losses.update(loss.data.item(), x.size()[0])
                accs.update(acc.data.item(), x.size()[0])
        
        valid_loss, valid_acc = losses.avg, accs.avg

        # run["after_pretraining/validation/epoch/loss"].log(valid_loss)
        # run["after_pretraining/validation/epoch/acc"].log(valid_acc)  
        
        # decay lr
        scheduler.step()
    save_checkpoint(
        {   'epoch': epoch + 1,
            'model_state': model.state_dict(),
            'optim_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'valid_acc': valid_acc
        } 
        )
    # Testing
    correct = 0
    model.eval()

    for i, (x, y) in enumerate(test_loader):
        x, y = x.to(gpu), y.to(gpu)

        out = model(x)

        # compute accuracy
        pred = torch.max(out, 1)[1]
        correct += pred.eq(y.data.view_as(pred)).cpu().sum()

    perc = (100. * correct.data.item()) / (num_test)
    error = 100 - perc
    print(
        '[*] Test Acc: {}/{} ({:.2f}% - {:.2f}%)'.format(
            correct, num_test, perc, error)
    )
    
    # run["after_pretraining/testing/epoch/loss"].log(error)
    # run["after_pretraining/testing/epoch/acc"].log(perc)
    
    correct = 0
    model.eval()
    num_test_familiar = len(test_loader_familiar.dataset)
    for i, (x, y) in enumerate(test_loader_familiar):
        x, y = x.to(gpu), y.to(gpu)

        out = model(x)

        # compute accuracy
        pred = torch.max(out, 1)[1]
        correct += pred.eq(y.data.view_as(pred)).cpu().sum()

    perc = (100. * correct.data.item()) / (num_test_familiar)
    error = 100 - perc
    print(
        '[*] Test Acc: {}/{} ({:.2f}% - {:.2f}%)'.format(
            correct, num_test_familiar, perc, error)
    )
    
    # run["after_pretraining/testing/FAMILIAR/epoch/loss"].log(error)
    # run["after_pretraining/testing/FAMILIAR/epoch/acc"].log(perc)
    
    # run.stop()

def save_checkpoint(state):
    epoch_num = state['epoch']
    filename = "vicreg_linear_evaluation_after_pretrained" + '_ckpt_epoch_' + str(epoch_num) +'.pth.tar'
    ckpt_path = os.path.join("checkpoints", filename)
    torch.save(state, ckpt_path)
    
class ApplyTransform(Dataset):
    # reference:https://stackoverflow.com/questions/56582246/correct-data-loading-splitting-and-augmentation-in-pytorch
    """
    Apply transformations to a Dataset

    Arguments:
        dataset (Dataset): A Dataset that returns (sample, target)
        transform (callable, optional): A function/transform to be applied on the sample
        target_transform (callable, optional): A function/transform to be applied on the target

    """
    def __init__(self, dataset, transform=None, target_transform=None):
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform
        if transform is None and target_transform is None:
            print("Transforms have failed")

    def __getitem__(self, idx):
        sample, target = self.dataset[idx]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self.dataset)
     
 
if __name__ == "__main__":
    main()
