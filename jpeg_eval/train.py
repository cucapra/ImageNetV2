"""
Finetuning Torchvision Models
=============================

**Author:** `Nathan Inkawhich <https://github.com/inkawhich>`__

"""



from __future__ import print_function 
from __future__ import division
import sys
import PIL
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import argparse
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)


######################################################################
#
def parse_args(args):
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    
    parser.add_argument('--data_dir', '-d', type=str,\
        default='/data/zhijing/jenna_data/', \
        help='Directory of the input data. \
        String. Default: /data/zhijing/jenna_data/')
    parser.add_argument('--model_name', '-m', type=str,\
        default='squeezenet',\
        help = 'NN models to choose from [resnet, alexnet, \
        vgg, squeezenet, densenet, inception]. \
        String. Default: squeezenet')
    
    parser.add_argument('--num_classes', '-c', type=int,\
        default = 3,\
        help = 'Number of classes in the dataset. \
        Integer. Default: 3')
    
    parser.add_argument('--batch_size', '-b', type=int,\
        default = 8,\
        help = 'Batch size for training (can change depending\
        on how much memory you have. \
        Integer. Default: 8)')
    
    
    parser.add_argument('-ep', '--num_epochs', type=int,\
        default = 25,\
        help = 'Number of echos to train for. \
        Integer. Default:25')
    
    parser.add_argument('--val_name', type=str,\
        default='/data/zhijing/flickrImageNetV2/matched_frequency_train/val/',\
        help='Directory for validation dataset. \
        String. Default:"val" ') 
    parser.add_argument('--feature_extract',action = 'store_true')
    args,unparsed = parser.parse_known_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.dir = os.path.dirname(__file__)
    print(args)
    return args
#####################################################################~~~~~~~~~~~~~~~~~~~~
## training and validation.
# 

def train_model(args, model, dataloaders, criterion, optimizer, is_inception=False):
    since = time.time()

    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    phases = ['train','val'] 
    for epoch in range(args.num_epochs):
        print('Epoch {}/{}'.format(epoch, args.num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in phases:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            #infos = []
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(args.device)
                labels = labels.to(args.device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    optimizer.zero_grad()

                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    reg_loss = 0
                    for param in model.parameters():
                        if param.requires_grad:
                            reg_loss += torch.norm(param, 1)
                    factor = 0.0005
                    loss += factor * reg_loss
                    _, preds = torch.max(outputs, 1)
                    #print(preds, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


######################################################################
# Set Model Parameters’ .requires_grad attribute
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## 
def set_parameter_requires_grad(model, feature_extract):
    if feature_extract:
        for name, param in model.named_parameters():
            param.requires_grad = False


######################################################################
# Initialize and Reshape the Networks

def initialize_model(args, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if args.model_name == "resnet":
        """ Resnet50
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft,\
           args.feature_extract)       
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, args.num_classes)
        input_size = 224

    elif args.model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft,\
           args.feature_extract)       
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, args.num_classes)
        input_size = 224

    elif args.model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft,\
           args.feature_extract)    
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, args.num_classes)
        input_size = 224

    elif args.model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft,\
           args.feature_extract)       
        model_ft.classifier[1] = nn.Conv2d(512, args.num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = args.num_classes
        input_size = 224

    elif args.model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft,\
           args.feature_extract)       
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, args.num_classes) 
        input_size = 224

    elif args.model_name == "inception":
        """ Inception v3 
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft,\
           args.feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, args.num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,args.num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()
    #load_from = os.path.join(args.dir,"verification.final")
    #print(model_ft) 
    model_ft = model_ft.to(args.device)
    return model_ft, input_size


######################################################################
# Load Data
# ---------

def load_data(args, input_size):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            #transforms.Resize(256),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    print("Initializing Datasets and Dataloaders...")
    folders = {
        'train':args.data_dir,#os.path.join(args.data_dir, 'train'),
        'val':args.val_name#os.path.join(args.data_dir, args.val_name) 
    }
    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(folders[x], data_transforms[x]) for x in ['train', 'val']}
    # Create training and validation dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}
    return image_datasets, dataloaders_dict
# Detect if we have a GPU available

######################################################################
# Create the Optimizer
def create_optimizer(args, model_ft):

    params_to_learn = [] #model_ft.parameters()
    print("Params to learn:")
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_learn.append(param)
            print(name)
                
    #optimizer_ft = optim.Adam(params_to_learn, lr=0.01)
    optimizer_ft = optim.SGD( model_ft.parameters(), lr = 0.001, momentum=0.9)
    return optimizer_ft

   

def run(args):


    args = parse_args(args)

    
    model_ft, input_size = initialize_model(args,use_pretrained=True)
    image_datasets, dataloaders_dict = load_data(args, input_size) 
    
    optimizer_ft = create_optimizer(args, model_ft)
    criterion = nn.CrossEntropyLoss()
    # Train and evaluate
    model_ft, hist = train_model( args=args, model=model_ft, dataloaders=dataloaders_dict, criterion=criterion, optimizer=optimizer_ft, is_inception=(args.model_name=="inception") )
    #torch.save(model_ft.state_dict(), os.path.join(args.dir,"verification.final"))
    #return hist[0].cpu().numpy()

if __name__=='__main__':
    sys.exit(run(sys.argv[1:]))
