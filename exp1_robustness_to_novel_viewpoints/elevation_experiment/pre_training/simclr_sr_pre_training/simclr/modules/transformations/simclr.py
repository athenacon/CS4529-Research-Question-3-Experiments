import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode        
class TransformsSimCLR:
    def __init__(self):
        self.transform = transforms.Compose(
            [         
                    transforms.Resize(48, interpolation=InterpolationMode.BICUBIC),
                    transforms.RandomCrop(32), 
                    transforms.ColorJitter(brightness=32./255, contrast=0.3),
                    transforms.ToTensor(), 
                 
            ]
        )
        self.transform_prime = transforms.Compose(
            [   
                transforms.Resize(48, interpolation=InterpolationMode.BICUBIC),
                transforms.RandomCrop(32), 
                transforms.ColorJitter(brightness=32./255, contrast=0.3),
                transforms.ToTensor(), 
            ]
        )

    def __call__(self, sample):
        x1 = self.transform(sample)
        x2 = self.transform_prime(sample)
        return x1, x2
