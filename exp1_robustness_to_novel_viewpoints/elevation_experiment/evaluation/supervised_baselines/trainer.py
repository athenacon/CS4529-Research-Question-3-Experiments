import torch
import torch.nn as nn
import torch.optim as optim

import os 
import shutil 
import neptune 
from utils import AverageMeter, save_config 

from models import *
from loss import *
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

class Trainer(object):
    """
    Trainer encapsulates all the logic necessary for
    training.

    All hyperparameters are provided by the user in the
    config file.
"""
    def __init__(self, config, data_loader):
        """
        Construct a new Trainer instance.

        Args
        ----
        - config: object containing command line arguments.
        - data_loader: data iterator
        """
        self.config = config

        # data params
        if config.is_train:
            self.train_loader = data_loader[0]
            self.valid_loader = data_loader[1]
            self.num_train = len(self.train_loader.dataset)
            self.num_valid = len(self.valid_loader.dataset)
            print("num_train", self.num_train)
            print("here")
            print("num_train", self.num_train)
            print("num_valid", self.num_valid)
            # breakpoint()
        else:
            self.test_loader = data_loader
            self.num_test = len(self.test_loader.dataset)
            print("num_test", self.num_test)
    


        # Display the images grid
        
        self.epochs = config.epochs
        self.start_epoch = 0
        self.momentum = config.momentum
        self.weight_decay = config.weight_decay
        self.lr = config.init_lr

        # misc params
        self.best = config.best
        self.ckpt_dir = config.ckpt_dir
        self.logs_dir = config.logs_dir
        self.best_valid_acc = 0.
        self.counter = 0
        self.train_patience = config.train_patience
        # self.use_tensorboard = config.use_tensorboard
        self.resume = config.resume
        self.print_freq = config.print_freq

        self.attack_type = config.attack_type
        self.attack_eps = config.attack_eps
        self.targeted = config.targeted

        self.name = config.name

        if config.name.endswith('dynamic_routing'):
            self.mode = 'DR'
        elif config.name.endswith('em_routing'):
            self.mode = 'EM'
        elif config.name.endswith('self_routing'):
            self.mode = 'SR'
        elif config.name.endswith('max'):
            self.mode = 'MAX'
        elif config.name.endswith('avg'):
            self.mode = 'AVG'
        elif config.name.endswith('fc'):
            self.mode = 'FC'
        else:
            raise NotImplementedError("Unknown model postfix")
        encoder = get_resnet("resnet18", pretrained=False).to(device)
        self.modified_resnet = modify_resnet_model(encoder).to(device)
        
        
        # initialize
        if config.name.startswith('resnet'):
            self.capsule_network = resnet20(config.planes, DATASET_CONFIGS[config.dataset], config.num_caps, config.caps_size, config.depth, mode=self.mode).to(device)
            print("DATASET CONFIT", DATASET_CONFIGS[config.dataset])
            self.model = Capsule(self.modified_resnet, self.capsule_network)
            # print(self.model) 
            print('[*] Number of capsulenetwor parameters: {:,}'.format(
            sum([p.data.nelement() for p in self.capsule_network.parameters()])))
            print('[*] Number of modified resnet50 parameters: {:,}'.format(
            sum([p.data.nelement() for p in self.modified_resnet.parameters()]))) 
        else:
            raise NotImplementedError("Unknown model prefix")

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)

        # self.loss = nn.CrossEntropyLoss().to(device)
        if self.mode in ['SR']:
            print("using NLL loss")
            self.loss = nn.NLLLoss().to(device)
        elif self.mode in ['EM']:
            print("using EM loss")
            self.loss = EmRoutingLoss(self.epochs).to(device)
        elif self.mode in ['AVG']:
            print("using cross entropy loss")
            self.loss = nn.CrossEntropyLoss().to(device)
            
        self.params = self.model.parameters()
        self.optimizer = optim.SGD(self.params, lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[150, 250], gamma=0.1)

        

        print('[*] Number of model parameters: {:,}'.format(
            sum([p.data.nelement() for p in self.model.parameters()])))
        print('[*] Number of trainable model parameters: {:,}'.format(
            sum(p.numel() for p in self.model.parameters() if p.requires_grad))) 
    def train(self):
        """
        Train the model on the training set.

        A checkpoint of the model is saved after each epoch
        and if the validation accuracy is improved upon,
        a separate ckpt is created for use on the test set.
        """
        # run = neptune.init_run(project = 'enter your username and project name', dependencies="infer",
        # api_token="enter your api token")
    
        # load the most recent checkpoint
        if self.resume:
            print("[*] Resuming from the most recent checkpoint")
            self.load_checkpoint(best=False)

        print("\n[*] Train on {} samples, validate on {} samples".format(
            self.num_train, self.num_valid)
        )

        for epoch in range(self.start_epoch, self.epochs):
            # get current lr
            for i, param_group in enumerate(self.optimizer.param_groups):
                lr = float(param_group['lr'])
                break
 
            # train for 1 epoch
            train_loss, train_acc = self.train_one_epoch(epoch)

            # evaluate on validation set
            with torch.no_grad():
                valid_loss, valid_acc = self.validate(epoch) 
            is_best = valid_acc > self.best_valid_acc 
            # run["training/epoch/loss"].log(train_loss)
            # run["training/epoch/acc"].log(train_acc)
            # run["validation/epoch/loss"].log(valid_loss)
            # run["validation/epoch/acc"].log(valid_acc)
             
            # check for improvement
            if not is_best:
                self.counter += 1
            '''
            if self.counter > self.train_patience:
                print("[!] No improvement in a while, stopping training.")
                return
            '''
            # decay lr
            self.scheduler.step()

            self.best_valid_acc = max(valid_acc, self.best_valid_acc)
            self.save_checkpoint(
                {'epoch': epoch + 1,
                 'model_state': self.model.state_dict(),
                 'optim_state': self.optimizer.state_dict(),
                 'scheduler_state': self.scheduler.state_dict(),
                 'best_valid_acc': self.best_valid_acc
                 }, is_best
            )
 
        print(self.best_valid_acc)

    def train_one_epoch(self, epoch):
        """
        Train the model for 1 epoch of the training set.

        An epoch corresponds to one full pass through the entire
        training set in successive mini-batches.

        This is used by train() and should not be called manually.
        """
        self.model.train()

        losses = AverageMeter()
        accs = AverageMeter()
 
        for i, (x, y) in enumerate(self.train_loader):
            x, y = x.to(device), y.to(device)

            b = x.shape[0]
            out = self.model(x)
            if isinstance(self.loss, EmRoutingLoss):
                loss = self.loss(out, y, epoch=epoch)
            else:
                loss = self.loss(out, y)

            # compute accuracy
            pred = torch.max(out, 1)[1]
            correct = (pred == y).float()
            acc = 100 * (correct.sum() / len(y))

            # store
            losses.update(loss.data.item(), x.size()[0])
            accs.update(acc.data.item(), x.size()[0])

            # compute gradients and update SGD
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
  
        return losses.avg, accs.avg

    def validate(self, epoch):
        """
        Evaluate the model on the validation set.
        """
        self.model.eval()

        losses = AverageMeter()
        accs = AverageMeter()

        for i, (x, y) in enumerate(self.valid_loader):
            x, y = x.to(device), y.to(device)

            out = self.model(x)
            if isinstance(self.loss, EmRoutingLoss):
                loss = self.loss(out, y, epoch=epoch)
            else:
                loss = self.loss(out, y)

            # compute accuracy
            pred = torch.max(out, 1)[1]
            correct = (pred == y).float()
            acc = 100 * (correct.sum() / len(y))

            # store
            losses.update(loss.data.item(), x.size()[0])
            accs.update(acc.data.item(), x.size()[0])
 
        return losses.avg, accs.avg

    def test(self):
        """
        Test the model on the held-out test data.
        This function should only be called at the very
        end once the model has finished training.
        """
        correct = 0

        # load the best checkpoint
        self.load_checkpoint(best=self.best)
        self.model.eval()

        for i, (x, y) in enumerate(self.test_loader):
            x, y = x.to(device), y.to(device)

            out = self.model(x)

            # compute accuracy
            pred = torch.max(out, 1)[1]
            correct += pred.eq(y.data.view_as(pred)).cpu().sum()

        perc = (100. * correct.data.item()) / (self.num_test)
        error = 100 - perc
        print(
            '[*] Test Acc: {}/{} ({:.2f}% - {:.2f}%)'.format(
                correct, self.num_test, perc, error)
        )


    def save_checkpoint(self, state, is_best):
        """
        Save a copy of the model so that it can be loaded at a future
        date. This function is used when the model is being evaluated
        on the test data.

        If this model has reached the best validation accuracy thus
        far, a seperate file with the suffix `best` is created.
        """
        # print("[*] Saving model to {}".format(self.ckpt_dir))

        filename = self.name + '_ckpt.pth.tar'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        torch.save(state, ckpt_path)

        if is_best:
            filename = self.name + '_model_best.pth.tar'
            shutil.copyfile(
                ckpt_path, os.path.join(self.ckpt_dir, filename)
            )

    def load_checkpoint(self, best=False):
        """
        Load the best copy of a model. This is useful for 2 cases:

        - Resuming training with the most recent model checkpoint.
        - Loading the best validation model to evaluate on the test data.

        Params
        ------
        - best: if set to True, loads the best model. Use this if you want
          to evaluate your model on the test data. Else, set to False in
          which case the most recent version of the checkpoint is used.
        """
        print("[*] Loading model from {}".format(self.ckpt_dir))

        filename = self.name + '_ckpt.pth.tar'
        if best:
            filename = self.name + '_model_best.pth.tar'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        ckpt = torch.load(ckpt_path)

        # load variables from checkpoint
        self.start_epoch = ckpt['epoch']
        self.best_valid_acc = ckpt['best_valid_acc']
        self.model.load_state_dict(ckpt['model_state'])
        self.optimizer.load_state_dict(ckpt['optim_state'])
        self.scheduler.load_state_dict(ckpt['scheduler_state'])

        if best:
            print(
                "[*] Loaded {} checkpoint @ epoch {} "
                "with best valid acc of {:.3f}".format(
                    filename, ckpt['epoch'], ckpt['best_valid_acc'])
            )
        else:
            print(
                "[*] Loaded {} checkpoint @ epoch {}".format(
                    filename, ckpt['epoch'])
            )

