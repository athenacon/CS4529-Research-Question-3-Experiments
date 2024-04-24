import os
import argparse
import torch
import numpy as np
import neptune

from simclr.modules import get_resnet
from utils import yaml_config_hook

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

from data_loader import get_test_loader, get_train_valid_loader, VIEWPOINT_EXPS

 
    
def save_checkpoint(state):
    epoch_num = state['epoch']
    filename = "simclr_linear_evaluation_after_pretrained" + '_ckpt_epoch_' + str(epoch_num) +'.pth.tar'
    ckpt_path = os.path.join("checkpoints_linear_evaluation", filename)
    torch.save(state, ckpt_path)

 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimCLR")
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    kwargs = {}
    if torch.cuda.is_available(): 
        kwargs = {'num_workers': 0, 'pin_memory': False}
        torch.cuda.set_device("cuda:0")
        print("GPU is available")
   
    # familiar = False means that the dataset is not familiar to the model, so we test on NOVEL.
    test_loader = get_test_loader(
     "./data", "smallnorb", args.logistic_batch_size, "azimuth", False,
    **kwargs
    )  
    num_test = len(test_loader.dataset)
    print("Length of testing dataset", len(test_loader.dataset))
     
    test_loader_familiar = get_test_loader(
     "./data", "smallnorb", args.logistic_batch_size, "azimuth", True,
    **kwargs
    )  
    
    print("Length of testing familiar dataset", len(test_loader_familiar.dataset))
    
     
    print("Data loaders created successfully")

    from simclr.modules.resnet_hacks import modify_resnet_model

    encoder_previous = get_resnet(args.resnet, pretrained=False)
    encoder = modify_resnet_model(encoder_previous, cifar_stem=True, v1=True)
    encoder.fc = torch.nn.Identity()
    
    import torch.nn as nn
    head = nn.Linear(512, 5) 
    head.weight.data.normal_(mean=0.0, std=0.01)
    head.bias.data.zero_()
    model = nn.Sequential(encoder, head)
    chpt = torch.load("check_results_checkpoints/epoch_100.pth.tar", map_location="cuda")
    model_state_dict = chpt['model_state']
    model.load_state_dict(model_state_dict, strict=True)
    # print(model) 
    model.cuda(args.device)

    
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