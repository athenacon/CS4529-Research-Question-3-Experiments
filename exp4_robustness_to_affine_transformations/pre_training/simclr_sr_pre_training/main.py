import os  
import numpy as np
import torch
import torchvision
import argparse
import neptune
# SimCLR
from simclr import SimCLR
from simclr.modules import NT_Xent, get_resnet
from simclr.modules.transformations import TransformsSimCLR
from simclr.modules.resnet_hacks import modify_resnet_model

from model import load_optimizer
from utils import yaml_config_hook

# Capsule Network
from capsule_network import resnet20
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
    # Save model after 100 epochs
    
    if (epoch + 1) % 100 == 0:
        if not os.path.exists("checkpoints"):
            os.makedirs("checkpoints")
        model_filename = f"pre_trained_model_epoch_{epoch + 1}.pth"
        state = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(state, os.path.join("checkpoints", model_filename))
    return loss_epoch


def main(gpu, args):

    # run = neptune.init_run(project = 'enter your username and project name', dependencies="infer",
    # api_token="enter your api token")
    
    torch.cuda.set_device(gpu)

    train_dataset = torchvision.datasets.MNIST(
        './data',
        train=True,
        download=True,
        transform=TransformsSimCLR(),
    )  

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
    )
    
    print("num_train_loader", len(train_loader))
    
    # initialize ResNet
    encoder = get_resnet(args.resnet, pretrained=False)
    modified_resnet = modify_resnet_model(encoder)
    # one channel for MNIST
    capsule_network = resnet20(16, {'size': 40, 'channels': 1, 'classes': 10}, 32, 16, 1, mode="SR").to(args.device)
    
    # initialize model
    model = SimCLR(modified_resnet, capsule_network)
    model = model.to(args.device)
    # print(model)
    # optimizer / loss
    optimizer, scheduler = load_optimizer(args, model)
    criterion = NT_Xent(args.batch_size, args.temperature, args.world_size)

    model = model.to(args.device)

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
