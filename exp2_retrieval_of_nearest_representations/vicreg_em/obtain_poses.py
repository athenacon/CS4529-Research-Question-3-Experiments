import argparse
import json
from torch import nn
from torchvision import transforms
import torch
import resnet
from capsule_network import resnet20 
from norb import smallNORB
from torch.utils.data import Subset
import numpy as np


def get_arguments():
    
    parser = argparse.ArgumentParser(
        description="Evaluate a pretrained model on CIFAR-10"
    ) 
    
    # Model
    parser.add_argument("--arch", type=str, default="resnet18")
 
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
    # single-gpu training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    main_worker(device, args)

def exclude_bias_and_norm(p):
    return p.ndim == 1

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

     
    valid_loader = torch.utils.data.DataLoader(
        valid_set, batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    ) 
    
    return valid_loader
 
def main_worker(gpu, args):
     
    # load the pre-trained model
    model = VICReg(args).cuda(gpu)
    
    checkpoint = torch.load("vicreg_linear_evaluation_after_pretrained_ckpt_epoch_100.pth.tar", map_location='cuda')  # or 'cpu'
    model.load_state_dict(checkpoint['model_state'], strict=True)
    print("Model loaded")  
    model.to(gpu)
    
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
    print("Data loaders created successfully")
   
    representation_query_image = {}
    representations_except_query_image = {}
    model.eval() 
    with torch.no_grad():
        for i, (x, y) in enumerate(valid_loader):
            
            x, y = x.to(gpu), y.to(gpu)

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


class VICReg(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.backbone, _ = resnet.__dict__[args.arch](
            zero_init_residual=True 
        ) 
        self.projection_head = resnet20(16, {'size': 32, 'channels': 1, 'classes': 10}, 32, 16, 1, mode="EM").to("cuda")

    def forward(self, x ):
        x = self.projection_head(self.backbone(x))
        return x

if __name__ == "__main__":
    main()
