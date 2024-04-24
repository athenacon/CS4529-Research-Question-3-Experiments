import os
import torch 
import torchvision.transforms as transforms
import numpy as np   
from torchvision import transforms
import json

# Capsule Network 
from norb import smallNORB
from torch.utils.data import Subset

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
import torch
import torch.nn as nn

import os 
from models import *
from data_loader import DATASET_CONFIGS

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import torchvision
from torchvision.models.resnet import Bottleneck, ResNet


class Capsule(nn.Module):
    """
    We opt for simplicity and adopt the commonly used ResNet (He et al., 2016) to obtain hi = f(x ̃i) = ResNet(x ̃i) where hi ∈ Rd is the output after the average pooling layer.
    """

    def __init__(self, encoder, caps_net):
        super(Capsule, self).__init__()
        self.encoder = encoder
        self.caps_net = caps_net

        # Replace the fc layer with an Identity function
        self.encoder.fc = nn.Identity()
        self.encoder.avgpool = nn.Identity()


    def forward(self, x_i):      
        x_i = self.encoder.conv1(x_i) 
        x_i = self.encoder.bn1(x_i) 
        x_i = self.encoder.relu(x_i) 
        x_i = self.encoder.layer1(x_i)  
        x_i = self.encoder.layer2(x_i) 
        x_i = self.encoder.layer3(x_i) 
        h_i = self.encoder.layer4(x_i) 
        z_i = self.caps_net(h_i) 
        return z_i
    
def get_resnet(name, pretrained=False):
    resnets = {
        "resnet18": torchvision.models.resnet18(pretrained=pretrained),
        "resnet50": torchvision.models.resnet50(pretrained=pretrained),
    }
    if name not in resnets.keys():
        raise KeyError(f"{name} is not a valid ResNet version")
    return resnets[name]

def modify_resnet_model(model, *, cifar_stem=True, v1=True):
    """Modifies some layers of a given torchvision resnet model to
    match the one used for the CIFAR-10 experiments in the SimCLR paper.

    Parameters
    ----------
    model : ResNet
        Instance of a torchvision ResNet model.
    cifar_stem : bool
        If True, adapt the network stem to handle the smaller CIFAR images, following
        the SimCLR paper. Specifically, use a smaller 3x3 kernel and 1x1 strides in the
        first convolution and remove the max pooling layer.
    v1 : bool
        If True, modify some convolution layers to follow the resnet specification of the
        original paper (v1). torchvision's resnet is v1.5 so to revert to v1 we switch the
        strides between the first 1x1 and following 3x3 convolution on the first bottleneck
        block of each of the 2nd, 3rd and 4th layers.

    Returns
    -------
    Modified ResNet model.
    """
    assert isinstance(model, ResNet), "model must be a ResNet instance"
    if cifar_stem:
        conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        nn.init.kaiming_normal_(conv1.weight, mode="fan_out", nonlinearity="relu")
        model.conv1 = conv1
        model.maxpool = nn.Identity()
        model.fc = nn.Identity()
    if v1:
        for l in range(2, 5):
            layer = getattr(model, "layer{}".format(l))
            block = list(layer.children())[0]
            if isinstance(block, Bottleneck):
                assert block.conv1.kernel_size == (1, 1) and block.conv1.stride == (
                    1,
                    1,
                )
                assert block.conv2.kernel_size == (3, 3) and block.conv2.stride == (
                    2,
                    2,
                )
                assert block.conv2.dilation == (
                    1,
                    1,
                ), "Currently, only models with dilation=1 are supported"
                block.conv1.stride = (2, 2)
                block.conv2.stride = (1, 1)
    return model

 

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
    from torchvision.transforms import InterpolationMode

    trans = [  
                transforms.Resize(48,  interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(32),
                transforms.ToTensor(),
            ]
        
    dataset = smallNORB(data_dir, train=True, download=True,
                    transform = transforms.Compose(trans))

    num_train = len(dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
 
    valid_idx = indices[:split]
 
    valid_set = Subset(dataset, valid_idx)
 
    valid_loader = torch.utils.data.DataLoader(
        valid_set, batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    ) 
    
    return valid_loader
 

if __name__ == "__main__": 
     
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    kwargs = {}
    if torch.cuda.is_available():
        kwargs = {'num_workers': 0, 'pin_memory': False}
        torch.cuda.set_device("cuda:0")
        print("GPU is available")
           
    valid_loader = get_train_valid_loader(
        "./data", "smallNorb", 1,
        2018, "full", 0.1,
        "True", **kwargs
    )
     
    num_valid = len(valid_loader.dataset)   
    print("Length of validation dataset", len(valid_loader.dataset))
     
    # initialize ResNet
    
    encoder = get_resnet("resnet18", pretrained=False).to(device)
    modified_resnet = modify_resnet_model(encoder).to(device)
    
    
    capsule_network = resnet20(16, DATASET_CONFIGS["smallnorb"], 32, 16, 1, mode="EM").to(device)
    print("DATASET CONFIT", DATASET_CONFIGS["smallnorb"])
    model = Capsule(modified_resnet, capsule_network)
   
    # load checkpoint
    ckpt_dir = "./ckpt"
    print("[*] Loading model from {}".format(ckpt_dir))

    filename = "resnet_em_routing" + '_model_best.pth.tar'
    ckpt_path = os.path.join(ckpt_dir, filename)
    ckpt = torch.load(ckpt_path)

    # load variables from checkpoint
    start_epoch = ckpt['epoch']
    best_valid_acc = ckpt['best_valid_acc']
    model.load_state_dict(ckpt['model_state'])

    print(
            "[*] Loaded {} checkpoint @ epoch {} "
            "with best valid acc of {:.3f}".format(
                filename, ckpt['epoch'], ckpt['best_valid_acc'])
        )
     
    model.eval()
       
    model = model.to(device) 
    # breakpoint()
    representation_query_image = {}
    representations_except_query_image = {}
    model.eval() 
    with torch.no_grad():
        for i, (x, y) in enumerate(valid_loader):
            
            x, y = x.to(device), y.to(device)

            out = model(x) 
            if i == 0:
                # Convert the tensor to a list for JSON compatibility
                representation_query_list = out.cpu().numpy().tolist()
                representation_query_image[f'image_{i}'] = representation_query_list
            else:
                # Convert the tensor to a list for JSON compatibility
                representation_list = out.cpu().numpy().tolist()
                # We use the batch index as a simple identifier for the image.
                representations_except_query_image[f'image_{i}'] = representation_list
            

    # Save the representations to a JSON file
    with open('pose_rest_images.json', 'w') as f:
        json.dump(representations_except_query_image, f)
    with open('pose_query_image.json', 'w') as f:
        json.dump(representation_query_image, f)
    print("The number of keys in the dictionary:", len(representations_except_query_image))
    print("The number of keys in the first dictionary:", len(representation_query_image))