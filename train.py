#written in Atom
#Nelson Walker started 01/16/2022
"""  Rubric for Train.py
Training a network	train.py successfully trains a new network on a dataset of images
Training validation log	The training loss, validation loss, and validation accuracy are printed out as a network trains
Model architecture	The training script allows users to choose from at least two different architectures available from torchvision.models
Model hyperparameters	The training script allows users to set hyperparameters for learning rate, number of hidden units, and training epochs
Training with GPU	The training script allows users to choose training the model on a GPU

python train.py --arch "vgg16" --hidden_units 512 --epochs 1 --gpu "True"
python train.py --arch "densenet121" --hidden_units 512 --epochs 4 --gpu "True"
python train.py --arch "densenet121" --hidden_units 512 --epochs 5 --gpu "True"
python train.py --arch "densenet121" --hidden_units 512 --epochs 2 --gpu "False"
python train.py --arch "densenet121" --hidden_units 512 --epochs 1 --gpu "True"

"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import time
import argparse

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torchvision.datasets import ImageFolder
from collections import OrderedDict

def define_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data_dir',
        dest="data_dir",
        action="store",
        help="image folder path",
        default="flowers")
    arg_parser.add_argument('--arch',
        dest="arch",
        action="store",
        help="pretrained network - such as densenet121",
        default="densenet121",
        choices="['densenet121', 'vgg16']")
    arg_parser.add_argument('--learning_rate',
        dest="learning_rate",
        action="store",
        help="learning rate",
        default=0.01)
    arg_parser.add_argument('--hidden_units',
        dest="hidden_units",
        action="store",
        help="num hidden units",
        default=512)
    arg_parser.add_argument('--epochs',
        dest="epochs",
        action="store",
        help="num epochs",
        default=5)
    arg_parser.add_argument('--gpu',
        dest="gpu",
        action="store",
        help="use gpu true/false",
        default=True)

    return arg_parser.parse_args()

def get_data(data_dir):

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    training_data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    testing_data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    validation_data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    img_datasets = {
            'training': datasets.ImageFolder(train_dir, transform=training_data_transforms),
            'validation': datasets.ImageFolder(valid_dir, transform=validation_data_transforms),
            'testing': datasets.ImageFolder(test_dir, transform=testing_data_transforms)
        }
    #change batch sizes from 64 to 32 for trainign & validation due to CUDA out of memory error
    img_dataloaders = {
            'training': torch.utils.data.DataLoader(img_datasets['training'], batch_size = 64, shuffle=True),
            'validation': torch.utils.data.DataLoader(img_datasets['validation'], batch_size = 32, shuffle=True),
            'testing': torch.utils.data.DataLoader(img_datasets['testing'], batch_size = 16, shuffle=False)
        }

    return img_datasets, img_dataloaders

def create_model(arch, hidden_units, class_to_idx, learning_rate):

    if arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        input_size = 1024
    else:
        model = models.vgg16(pretrained=True)
        input_size = 25088

    for parm in model.parameters():
        parm.requires_grad = False

    output_size = 102  #down from 128
    #this gets the input size from the model features
    #doesn't work for vgg16?
    #input_size = model.classifier.in_features

    #building the classifier to replace the pretrained model classifier
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(input_size, hidden_units)),
                              ('drop', nn.Dropout(p=0.1)),
                              ('relu', nn.ReLU()),
                              ('fc2', nn.Linear(hidden_units, output_size)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier
    model.class_to_idx = class_to_idx
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(),lr=learning_rate)

    return model, classifier, optimizer, criterion

def train_model(model, img_datasets, img_dataloaders, epochs, gpu, optimizer, criterion):

    training_data = img_dataloaders['training']
    validation_data = img_dataloaders['validation']

    if gpu == True:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print("CUDA device: {}".format(torch.cuda.get_device_name(0)))
    else:
        device = 'cpu'

    print("Using device: {}".format(device))
    print("Training dataloader size: {}".format(len(training_data)))
    print("Validation dataloader size: {}".format(len(validation_data)))
    print("---------------------------------------------------------")

    model.to(device)
    #Train model
    steps = 0
    print_every = 30
    start_time = time.time()
    for epoch in range(epochs):
        running_training_loss = 0
        for x, (inputs, labels) in enumerate(training_data):
            model.train()
            steps += 1
            step_start = time.time()
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_training_loss +=  loss.item()

            #Step out of training and run validation check
            if steps % print_every == 0:
                valid_loss, valid_accuracy = validation_check(validation_data, model, criterion, device)

                step_elapsed= time.time() - step_start
                print(f"Epoch {epoch + 1}/{epochs}.. Step {steps}.. "
                      f"Training loss: {running_training_loss/print_every:.3f}.. "
                      f"Valid loss: {valid_loss:.3f}.. "
                      f"Valid accuracy: {valid_accuracy:.3f}")

                running_training_loss = 0

                print(f"Elapsed: {(step_elapsed):.3f} seconds")

    total_elapsed= time.time() - start_time
    print("--------------------------------------------------------------")
    print(f"Device = {device}; Total elapsed: {(total_elapsed):.3f} seconds")
    print("----COMPLETE TRAINING----------------------------------------------------------")
    return model

def validation_check(dataloader, model, criterion, device):
    #in my jupyter notebook ipnyb for the final project,
    # I put this validation check inside the epoch loop
    # but I was having trouble trying to get my network to train properly
    # I studied several other projects in GitHub
    # and I found that using a validation function made the code cleaner
    # and easier to read.  I used their function as an idea for mine

    running_validation_loss = 0
    accuracy = 0

    #turn model into evaluation inference mode, turns off dropout
    model.eval()
    lendataloader = len(dataloader)
    for x, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)

        output = model.forward(inputs)
        running_validation_loss += criterion(output, labels).item()
        ps = torch.exp(output)
        # from my mentor Mr. Himanshu M
        doesequal = (labels.data == ps.max(dim = 1)[1])
        accuracy += doesequal.type_as(torch.FloatTensor()).mean()

    return (running_validation_loss/lendataloader),(accuracy/lendataloader)

def save_chkpoint(model, optimizer, criterion, classifier, args, filepath):

    checkpoint = {'model': model,
                  'arch': args.arch,
                  'learning_rate': args.learning_rate,
                  'hidden_units': args.hidden_units,
                  'classifier': classifier,
                  'epochs': args.epochs,
                  'criterion': criterion,
                  'optimizer': optimizer.state_dict(),
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx}

    torch.save(checkpoint, filepath)
#------------------------------------------------
# MAIN get started
#------------------------------------------------
def main():
    """rubric
    The training script allows users to choose from at least two different architectures available from torchvision.models
    The training loss, validation loss, and validation accuracy are printed out as a network trains
    The training script allows users to set hyperparameters for learning rate, number of hidden units, and training epochs
    The training script allows users to choose training the model on a GPU
    """
    args = define_args()
    data_dir = args.data_dir
    #chkpoint_dir = args.chkpoint_dir
    if args.gpu == "True":
        gpu = True
    else:
        gpu = False
    arch = args.arch
    learning_rate = float(args.learning_rate)
    epochs = int(args.epochs)
    hidden_units = int(args.hidden_units)
    print('--------------------------------------------------')
    print('Training model - arguments')
    print('--------------------------------------------------')
    print(f'data_dir: {data_dir}')
    #print(f'chkpoint_dir: {chkpoint_dir}')
    print(f'gpu: {gpu}')
    print(f'arch: {arch}')
    print(f'learning_rate: {learning_rate}')
    print(f'epochs: {epochs}')
    print(f'hidden_units: {hidden_units}')
    print('--------------------------------------------------')
    #load dataloaders
    img_datasets, img_dataloaders = get_data(data_dir)

    #create model
    training_data = img_datasets['training']
    class_to_idx = training_data.class_to_idx
    model, classifier, optimizer, criterion = create_model(arch, hidden_units, class_to_idx, learning_rate)

    #train model
    model = train_model(model, img_datasets, img_dataloaders, epochs, gpu, optimizer, criterion)

    #when training complete, save checkpoint
    model.to('cpu')

    chkpoint_save_path = 'checkpoint_cmdline3.pth'
    save_chkpoint(model, optimizer, criterion, classifier, args, chkpoint_save_path)
    print('--------------------------------------------------')
    print(f'Checkpoint saved to file: {chkpoint_save_path}')
    print('--------------------------------------------------')
    print("Training program complete")
    print('--------------------------------------------------')
    #--------- end of main ---------------

if __name__ == '__main__':
    main()
