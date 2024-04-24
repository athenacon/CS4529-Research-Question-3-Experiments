from pathlib import Path
import argparse
import os
import random
import neptune
from torch import nn, optim
from torch.utils.data import Dataset
from utilscapsnet import AverageMeter
import torch
import resnet
from capsule_network import resnet20 
import numpy as np

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
    #  
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
        "--batch-size", default=2, type=int, metavar="N", help="mini-batch size"
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
        default=0,
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

    # load the pre-trained model
    model = VICReg(args).cuda(gpu)
    # print(model) 
    import os
    print("Current working directory:", os.getcwd())

    checkpoint = torch.load("check_results_checkpoints/epoch_100.pth.tar", map_location='cuda')  # or 'cpu'
    model.load_state_dict(checkpoint['model_state'], strict=True)
    # print(model)
    print("Model loaded")  

    for param in model.parameters():
        param.requires_grad = False
 
    for param in model.projection_head.fc.parameters():
        param.requires_grad = True
    
    # check if frozen or not
    model.to(gpu) 
   
    kwargs = {}
    if torch.cuda.is_available(): 
        kwargs = {'num_workers': 0, 'pin_memory': False}
        torch.cuda.set_device("cuda:0")
        print("GPU is available")
           

    # familiar = False means that the dataset is not familiar to the model, so we test on NOVEL.
    test_loader = get_test_loader(
     "./data", "smallnorb", args.batch_size, "elevation", False,
    **kwargs
    )  
    num_test = len(test_loader.dataset)
    print("Length of testing dataset", len(test_loader.dataset))
     
    test_loader_familiar = get_test_loader(
     "./data", "smallnorb", args.batch_size, "elevation", True,
    **kwargs
    )  
    
    print("Length of testing familiar dataset", len(test_loader_familiar.dataset))
    
     
    print("Data loaders created successfully")
    
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
    
class VICReg(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_features = 2048
        self.backbone, _ = resnet.__dict__[args.arch](
            zero_init_residual=True 
        )
        self.projection_head = resnet20(16, {'size': 32, 'channels': 3, 'classes': 10}, 32, 16, 1, mode="EM").to("cuda")
    def forward(self, x ):
        x = self.projection_head(self.backbone(x))
        return x
 


if __name__ == "__main__":
    main()
