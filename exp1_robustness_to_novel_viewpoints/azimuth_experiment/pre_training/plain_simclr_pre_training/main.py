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
from model import load_optimizer, save_model
from utils import yaml_config_hook
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

 
from norb import smallNORBViewPoint
 

def train(args, train_loader, model, criterion, optimizer):
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
    
    return loss_epoch


VIEWPOINT_EXPS = ['azimuth', 'elevation']


def get_train_valid_loader(data_dir,
                           dataset,
                           batch_size,
                           random_seed,
                           exp='azimuth',
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

    
    kwargs = {}
    torch.backends.cudnn.benchmark = True 
    if torch.cuda.is_available(): 
        kwargs = {'num_workers': 0, 'pin_memory': False}
        torch.cuda.set_device(gpu)
        print("GPU is available")
        
    # also config these
    train_loader = get_train_valid_loader(
        "./data", "smallNorb", args.batch_size,
        42, "azimuth", 0.1,
        "True", **kwargs
    )
  
    num_train = len(train_loader.dataset)
    print("num_train", num_train)
     

    # initialize ResNet
    encoder_previous = get_resnet(args.resnet, pretrained=False)
    encoder = modify_resnet_model(encoder_previous, cifar_stem=True, v1=True)
    n_features = encoder.fc.in_features  # get dimensions of fc layer

    # initialize model
    model = SimCLR(encoder, args.projection_dim, n_features)
    model = model.to(args.device)

    # optimizer / loss
    optimizer, scheduler = load_optimizer(args, model)
    criterion = NT_Xent(args.batch_size, args.temperature, 1)

    model = model.to(args.device)

    args.global_step = 0
    args.current_epoch = 0
    for epoch in range(args.start_epoch, args.epochs):
         
        lr = optimizer.param_groups[0]["lr"]
        loss_epoch = train(args, train_loader, model, criterion, optimizer)

        scheduler.step()

        save_model(args, model, optimizer)
        # save lr
        run["pre-training/epoch/lr"].log(lr)
        run["pre-training/epoch/loss"].log(loss_epoch/len(train_loader))
 
        if (epoch + 1) == 1000:
            model_filename = f"pre_trained_model_epoch_{epoch + 1}.pth"
            state = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, os.path.join("checkpoints", model_filename))

    run.stop()


if __name__ == "__main__":
    # run = neptune.init_run(project = 'enter your username and project name', dependencies="infer",
    # api_token="enter your api token")

    parser = argparse.ArgumentParser(description="SimCLR")
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 
    main(0, args)
