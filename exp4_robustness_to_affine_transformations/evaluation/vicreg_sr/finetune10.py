from pathlib import Path
import argparse
import os
import random
import neptune
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import  Dataset
from utilscapsnet import AverageMeter
import torch

import resnet
from capsule_network import resnet20 
import torch
import torchvision
import resnet
import numpy as np
from torch.utils.data import DataLoader
import scipy.io as sio
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


class affNIST(Dataset):
    # reference: https://github.com/fabio-deep/Variational-Capsule-Routing
    ''' In:
            data_path (string): path to the dataset split folder, i.e. train/valid/test
            transform (callable, optional): transform to be applied on a sample.
        Out:
            image, label: sample data and respective label'''

    def __init__(self, data_path, shuffle=False, transform=None):

        self.data_path = data_path
        self.shuffle = shuffle
        self.transform = transform
        self.split = self.data_path.split('/')[-1]

        if self.split == 'train':
            for i, file in enumerate(os.listdir(data_path)):
                # load dataset .mat file batch
                self.dataset = sio.loadmat(os.path.join(data_path, file))
                # concatenate the 32 .mat files to make full dataset
                if i == 0:
                    self.data = np.array(self.dataset['affNISTdata']['image'][0][0])
                    self.labels = np.array(self.dataset['affNISTdata']['label_int'][0][0])
                else:
                    self.data = np.concatenate((self.data,
                        np.array(self.dataset['affNISTdata']['image'][0][0])), axis=1)
                    self.labels = np.concatenate((self.labels,
                        np.array(self.dataset['affNISTdata']['label_int'][0][0])), axis=1)

            # (N, 1, 40, 40) <- (1, 40, 40, N) <- (40*40, N)
            self.data = np.moveaxis(self.data.reshape(1,40,40,-1), -1, 0)
            # (N,)
            # labels are 2D, squeeze to 1D
            self.labels = self.labels.squeeze()
        else:
            print("yes")
            # load valid/test dataset .mat file
            self.dataset = sio.loadmat(os.path.join(self.data_path, self.split+'.mat'))
            # (40*40, N)
            self.data = np.array(self.dataset['affNISTdata']['image'][0][0])
            # (N, 1, 40, 40) <- (1, 40, 40, N) <- (40*40, N)
            self.data = np.moveaxis(self.data.reshape(1,40,40,-1), -1, 0)
            # (N,)
            self.labels = np.array(self.dataset['affNISTdata']['label_int'][0][0]).squeeze()

        self.data = self.data.squeeze()

        if self.shuffle: # shuffle the dataset
            idx = np.random.permutation(self.data.shape[0])
            self.data = self.data[idx]
            self.labels = self.labels[idx]

    def __len__(self):
        # Returns the total number of samples in the dataset,
        # which is simply the first dimension of the data array 
        return self.data.shape[0]

    def __getitem__(self, idx):
        #  Fetches the idx-th sample from the dataset.
        # It applies any specified transformations to the image data 
        # before returning it along with its corresponding label.

        image = self.data[idx]

        if self.transform is not None:
            image = self.transform(image) 
        return image, self.labels[idx] # (X, Y)

def get_train_valid_loader(data_dir, batch_size, num_workers=4, pin_memory=False):
    from torchvision.transforms import InterpolationMode
    
    trans_train = transforms.Compose([
            # transforms.ToPILImage(), 
            transforms.Pad(6),
                transforms.RandomAffine(
                    degrees=0, # No rotation
                    translate=(0.2, 0.2), # Translate up to 20% of the image size per the attention paper
                    scale=None, # Keep original scale
                    shear=None, # No shear
                    interpolation=InterpolationMode.NEAREST, # Nearest neighbor interpolation
                    fill=0 # Fill with black color for areas outside the image
                ),
            transforms.ToTensor(),
            transforms.Normalize((0.13066047,), (0.30810780,))
            ])
    
    trans_valid = transforms.Compose([
                # transforms.ToPILImage(), 
                transforms.Pad(6),
                transforms.ToTensor(),
                transforms.Normalize((0.13066047,), (0.30810780,))
                ])
    
    data_dir = data_dir +  '/MNIST'
    dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=None)
    # targets = np.array(dataset.targets)
    n_samples_per_class = len(dataset) * 0.1 // 10
    n_samples_per_class = int(n_samples_per_class)
    # Organize indices by class
    class_indices = {i: [] for i in range(10)}  #  5 classes
    for idx, (_, target) in enumerate(dataset):
        class_indices[int(target)].append(idx)
    train_idx = []
    valid_idx = [] 
        
    for _, indices in class_indices.items():
        np.random.seed(seed_value)
        np.random.shuffle(indices)
        class_train_indices = indices[:n_samples_per_class]
        
        class_valid_indices = indices[n_samples_per_class:n_samples_per_class * 2]

        train_idx.extend(class_train_indices)
        valid_idx.extend(class_valid_indices)
 
    from torch.utils.data import Subset 
    train_set = Subset(dataset, indices=train_idx)
    valid_set = Subset(dataset, indices=valid_idx)

    # Applying transforms dynamically
    train_set = ApplyTransform(train_set, transform=trans_train)
    valid_set = ApplyTransform(valid_set, transform=trans_valid)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, valid_loader



def save_checkpoint(state):
    epoch_num = state['epoch']
    filename = "vicreg_linear_evaluation_after_pretrained" + '_ckpt_epoch_' + str(epoch_num) +'.pth.tar'
    ckpt_path = os.path.join("checkpoints_linear_evaluation", filename)
    torch.save(state, ckpt_path)

def save_checkpoint_final(state):
    epoch_num = state['epoch']
    filename = "final_mnist_acc92" + '_ckpt_epoch_' + str(epoch_num) +'.pth.tar'
    ckpt_path = os.path.join("checkpoints_linear_evaluation", filename)
    torch.save(state, ckpt_path)


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
    
    # load the pre-trained model
    model = VICReg(args).cuda(gpu)
    
    checkpoint = torch.load("pre_trained_model_epoch_1000.pth", map_location='cuda')  # or 'cpu'
    
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    print(model)
    print("Model loaded")  

    for param in model.parameters():
        param.requires_grad = True
    model.to(gpu)
    
    normalize = transforms.Normalize((0.13066047,), (0.30810780,))
    
    # train on random pad mnist
    
    kwargs = {'num_workers': 0, 'pin_memory': False}
    args.seed = 42
    # instantiate data loaders
    data_loader = get_train_valid_loader(
          'data/', args.batch_size,
         args.workers,  False
    )  
    train_loader = data_loader[0]
    # evaluate on just padded mnist (no random)
    valid_loader = data_loader[1]
    
    # test on padded mnist after each epoch until accuracy is >92.2 and <92.3
    transform_MNIST = transforms.Compose(
    [   transforms.Pad(6),
        transforms.ToTensor(),
        normalize   ]
    )
    test_MNIST_dataset = datasets.MNIST(
        'data/',
        train=False,  # Load the test set
        download=True,
        transform=transform_MNIST,
    ) 
    
    # loader for test_MNIST_dataset
    kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    ) 
    test_MNIST_loader = torch.utils.data.DataLoader(test_MNIST_dataset, shuffle=False, **kwargs)
    
    # then finally test on affnist.
    # affNIST dataset 
    transform_affNIST = transforms.Compose(
    [   
        transforms.ToTensor(),
        normalize
    ])

    # we only need testing dataset of affNIST
    test_dataset_path = 'affNist_transformed/test'
    test_dataset = affNIST(data_path=test_dataset_path, transform=transform_affNIST)
    print("Testing dataset loaded successfully")  
    
    # affNIST
    kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )  
    test_AffNIST_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, **kwargs)
    
    print("Data loaders created successfully")
    print("Length of training on random pad mnist: ", len(train_loader.dataset))
    print("Length of validation on just simply pad mnist: ", len(valid_loader.dataset))
    print("Length of testing on padded mnist: ", len(test_MNIST_loader.dataset))
    print("Length of testing on affnist: ", len(test_AffNIST_loader.dataset))
      
    num_test_mnist = len(test_MNIST_loader.dataset)
    num_test_affnist = len(test_AffNIST_loader.dataset)
 
    criterion = nn.NLLLoss().to(gpu) 
    finetune_1_percent_n_epochs = 100

    caps_net_fc_params = list(model.projection_head.fc.parameters())
    other_params = [p for p in model.parameters() if not any(p is pp for pp in caps_net_fc_params)]

    # sanity check
    print(f"Total model parameters: {len(list(model.parameters()))}")
    print(f"CapsNet FC parameters: {len(caps_net_fc_params)}")
    print(f"Other parameters: {len(other_params)}")

    param_groups = [
        {'params': other_params, 'lr': 3},
        {'params': caps_net_fc_params, 'lr': 1}
    ]
    
    optimizer = optim.SGD(param_groups, 0, momentum=0.9, weight_decay=0)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, finetune_1_percent_n_epochs)

    num_train = len(train_loader.dataset)
    num_valid = len(valid_loader.dataset)

    print("\n[*] Train on {} samples, validate on {} samples".format(
        num_train, num_valid)
    )
    end_lets_test_affnist = False

    for epoch in range(finetune_1_percent_n_epochs):
        # get current lr
        for i, param_group in enumerate(optimizer.param_groups):
            lr = float(param_group['lr'])
            break

        model.train()

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
        correct = 0
        model.eval()
        for i, (x, y) in enumerate(test_MNIST_loader):
            x, y = x.to(gpu), y.to(gpu)

            out = model(x)

            # compute accuracy
            pred = torch.max(out, 1)[1]
            correct += pred.eq(y.data.view_as(pred)).cpu().sum()

        perc = (100. * correct.data.item()) / (num_test_mnist)
        error = 100 - perc
        
        if perc >= 94.00 and perc < 94.10:
            end_lets_test_affnist = True
            save_checkpoint(
            {   'epoch': epoch + 1,
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'valid_acc': valid_acc
            } 
            )
        # run["after_pretraining/testing/justpaddedMNIST/epoch/loss"].log(error)
        # run["after_pretraining/testing/justpaddedMNIST/epoch/acc"].log(perc)

        if end_lets_test_affnist:
            correct = 0
            model.eval()

            for i, (x, y) in enumerate(test_AffNIST_loader):
                x, y = x.to(gpu), y.to(gpu)

                out = model(x)

                # compute accuracy
                pred = torch.max(out, 1)[1]
                correct += pred.eq(y.data.view_as(pred)).cpu().sum()

            perc = (100. * correct.data.item()) / (num_test_affnist)
            error = 100 - perc
            print(
                '[*] Test Acc: {}/{} ({:.2f}% - {:.2f}%)'.format(
                    correct, num_test_affnist, perc, error)
            )
            
            # run["after_pretraining/testing/AffNIST/epoch/loss"].log(error)
            # run["after_pretraining/testing/AffNIST/epoch/acc"].log(perc)
            
            # run.stop() 
            
    
    
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
        self.projection_head = resnet20(16, {'size': 32, 'channels': 3, 'classes': 10}, 32, 16, 1, mode="SR").to("cuda")


    def forward(self, x ):
        x = self.projection_head(self.backbone(x))
        return x

 

if __name__ == "__main__":
    main()
