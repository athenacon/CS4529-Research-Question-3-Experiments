import torch 
import torchvision.transforms as transforms
import numpy as np 
from simclr import SimCLR
from simclr.modules import get_resnet 
from torchvision import transforms
import json

from simclr.modules.resnet_hacks import modify_resnet_model
# Capsule Network
from capsule_network import resnet20 
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
                transforms.Resize(48, interpolation=InterpolationMode.BICUBIC),
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
 
    # We set batch_size to 1 
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
    encoder = get_resnet("resnet18", pretrained=False)
    modified_resnet = modify_resnet_model(encoder)
    capsule_network = resnet20(16, {'size': 32, 'channels': 1, 'classes': 5}, 32, 16, 1, mode="EM").to(device)
    
    # initialize model
    model = SimCLR(modified_resnet, capsule_network)
      
    checkpoint = torch.load("simclr_linear_evaluation_after_pretrained_ckpt_epoch_100.pth.tar", map_location=device)  
    model.load_state_dict(checkpoint['model_state'], strict=True)
      
    model = model.to(device)
    model.eval()
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
        
    # Save the representations to a JSON file, so we can load them after and find the nearest neighbors
    with open('pose_rest_images.json', 'w') as f:
        json.dump(representations_except_query_image, f)
    with open('pose_query_image.json', 'w') as f:
        json.dump(representation_query_image, f)
    # Do sanity checks
    print("The number of keys in the dictionary:", len(representations_except_query_image))
    print("The number of keys in the first dictionary:", len(representation_query_image))