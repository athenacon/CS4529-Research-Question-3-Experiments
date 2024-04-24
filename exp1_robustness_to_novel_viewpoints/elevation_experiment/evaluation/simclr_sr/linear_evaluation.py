import os

import argparse
import torch
import numpy as np
import neptune
from simclr import SimCLR
from simclr.modules import get_resnet

from utils import yaml_config_hook
from torch import nn, optim

from simclr.modules.resnet_hacks import modify_resnet_model
# Capsule Network
from capsule_network import resnet20
from utilscapsnet import AverageMeter

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


def save_checkpoint(state):
    epoch_num = state['epoch']
    filename = "simclr_linear_evaluation_after_pretrained" + '_ckpt_epoch_' + str(epoch_num) +'.pth.tar'
    ckpt_path = os.path.join("checkpoints", filename)
    torch.save(state, ckpt_path)
        
 
from data_loader import get_test_loader, get_train_valid_loader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimCLR")
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
        
    
    # run = neptune.init_run(project = 'enter your username and project name', dependencies="infer",
    # api_token="enter your api token")

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    torch.backends.cudnn.benchmark = True 

    kwargs = {}
    if torch.cuda.is_available(): 
        kwargs = {'num_workers': 0, 'pin_memory': False}
        torch.cuda.set_device("cuda:0")
        print("GPU is available")
           
            
    # instantiate data loaders
    data_loader = get_train_valid_loader(
            "./data","smallnorb", args.logistic_batch_size,
            seed_value, "elevation", "0.1",
            True, **kwargs
        ) 
    train_loader = data_loader[0]
    valid_loader = data_loader[1]
    num_train = len(train_loader.dataset)
    num_valid = len(valid_loader.dataset)  

    # familiar = False means that the dataset is not familiar to the model, so we test on NOVEL.
    test_loader = get_test_loader(
     "./data", "smallnorb", args.logistic_batch_size, "elevation", False,
    **kwargs
    )  
    num_test = len(test_loader.dataset)
    print("Length of testing dataset", len(test_loader.dataset))
     
    test_loader_familiar = get_test_loader(
     "./data", "smallnorb", args.logistic_batch_size, "elevation", True,
    **kwargs
    )  
    
    print("Length of testing familiar dataset", len(test_loader_familiar.dataset))

    print("num_train", num_train)
    
    print("num_valid", num_valid)
     
    print("Data loaders created successfully")
    
    print("Length of training dataset", len(train_loader.dataset))
    print("Length of validation dataset", len(valid_loader.dataset))
 
    encoder = get_resnet(args.resnet, pretrained=False)
    modified_resnet = modify_resnet_model(encoder)
    capsule_network = resnet20(16, {'size': 40, 'channels': 1, 'classes': 10}, 32, 16, 1, mode="SR").to(args.device)
    
    # initialize model
    model = SimCLR(modified_resnet, capsule_network)
     
    device = args.device
    checkpoint = torch.load("pre_trained_model_epoch_1000.pth", map_location=device)  
     
    state_dict = checkpoint['model_state_dict']
    model.load_state_dict(state_dict, strict=True)
    model = model.to(args.device)
    # print(model) 
    print("Model loaded from pre-trained model successfully")
    for param in model.parameters():
        param.requires_grad = False
 
    for param in model.caps_net.fc.parameters():
        param.requires_grad = True
     
    # check again if frozen
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
       
    model = model.to(args.device)
    
    start_epoch = 0
    criterion = nn.NLLLoss().to(args.device) 
    
    optimizer = optim.SGD(model.caps_net.fc.parameters(), lr=0.3, momentum=0.9, weight_decay=1e-6)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.logistic_epochs)

    best_valid_acc = 0
    for epoch in range(args.logistic_epochs):
        losses = AverageMeter()
        accs = AverageMeter()
        model.eval()
        model.caps_net.fc.train()
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(args.device), y.to(args.device) 
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
                x, y = x.to(args.device), y.to(args.device)

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
        {'epoch': epoch + 1,
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
        x, y = x.to(args.device), y.to(args.device)

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
    
    test_loader_familiar_len = len(test_loader_familiar.dataset) 
    
    # FAMILIAR:
    correct = 0
    model.eval()

    for i, (x, y) in enumerate(test_loader_familiar):
        x, y = x.to(args.device), y.to(args.device)

        out = model(x)

        # compute accuracy
        pred = torch.max(out, 1)[1]
        correct += pred.eq(y.data.view_as(pred)).cpu().sum()

    perc = (100. * correct.data.item()) / (test_loader_familiar_len)
    error = 100 - perc
    print(
        '[*] Test Acc: {}/{} ({:.2f}% - {:.2f}%)'.format(
            correct, test_loader_familiar_len, perc, error)
    )
    
    # run["after_pretraining/testing/FAMILIAR/epoch/loss"].log(error)
    # run["after_pretraining/testing/FAMILIAR/epoch/acc"].log(perc)

    # run.stop()     