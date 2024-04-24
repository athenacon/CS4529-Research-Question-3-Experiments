import os  
 

import numpy as np
import torch
import argparse
import neptune
# SimCLR
from simclr import SimCLR
from simclr.modules import NT_Xent, get_resnet
from simclr.modules.transformations import TransformsSimCLR
from simclr.modules.resnet_hacks import modify_resnet_model

from model import load_optimizer
from utils import yaml_config_hook
from norb import smallNORBViewPoint
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

# Capsule Network
from capsule_network import resnet20

def train(args, train_loader, model, criterion, optimizer, epoch):
    loss_epoch = 0
    for step, ((x_i, x_j), _) in enumerate(train_loader):
        optimizer.zero_grad()
        x_i = x_i.cuda(non_blocking=True)
        x_j = x_j.cuda(non_blocking=True)
        
        # positive pair, with encoding
        z_i, z_j = model(x_i, x_j)
        
        loss = criterion(z_i, z_j)
        loss.backward()

        optimizer.step()
        
        loss_epoch += loss.item()
        
    if (epoch + 1) == 1000:
        model_filename = f"pre_trained_model_epoch_{epoch + 1}.pth"
        state = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(state, os.path.join("checkpoints", model_filename))
    return loss_epoch

VIEWPOINT_EXPS = ['azimuth', 'elevation']
 
def get_train_valid_loader(data_dir,
                           dataset,
                           batch_size,
                           random_seed,
                           exp='elevation',
                           valid_size=0.1,
                           shuffle=True,
                           num_workers=4,
                           pin_memory=False):

    data_dir = data_dir + '/' + dataset
     
    train_set = smallNORBViewPoint(data_dir, exp=exp, train=True, download=True,
            transform=TransformsSimCLR())
     
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True
    )
 
    return train_loader 
def main(gpu, args):
    # run = neptune.init_run(project = 'enter your username and project name', dependencies="infer",
    # api_token="enter your api token")
    
    torch.backends.cudnn.benchmark = True 
    kwargs = {}
    if torch.cuda.is_available(): 
        kwargs = {'num_workers': 0, 'pin_memory': False}
        torch.cuda.set_device(gpu)
        print("GPU is available")
        
    # also config these
    train_loader = get_train_valid_loader(
        "./data", "smallNorb", args.batch_size,
        42, "elevation", 0.1,
        "True", **kwargs
    )
      
    num_train = len(train_loader.dataset)
    print("num_train", num_train)
    # num_train 16200
 
    # initialize ResNet
    encoder = get_resnet(args.resnet, pretrained=False)
    modified_resnet = modify_resnet_model(encoder)
    capsule_network = resnet20(16, {'size': 32, 'channels': 1, 'classes': 10}, 32, 16, 1, mode="EM").to(args.device)
    
    # initialize model
    model = SimCLR(modified_resnet, capsule_network)
    model = model.to(args.device)
    # print(model)
    
    # optimizer / loss
    optimizer, scheduler = load_optimizer(args, model)
    criterion = NT_Xent(args.batch_size, args.temperature, args.world_size)

    model = model.to(args.device)

    print("Start pre-training")
    args.global_step = 0
    args.current_epoch = 0
    for epoch in range(args.start_epoch, args.epochs):
        # print("epoch", epoch)
        lr = optimizer.param_groups[0]["lr"]
        loss_epoch = train(args, train_loader, model, criterion, optimizer, epoch)

        scheduler.step()

        # save lr and loss neptune
        # run["pre-training/epoch/loss"].log(loss_epoch/len(train_loader))
        # run["pre-training/epoch/learning_rate"].log(lr)
        

    # end training
    # run.stop()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="SimCLR")
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.num_gpus = 1
    args.world_size = 1

    main(0, args)
