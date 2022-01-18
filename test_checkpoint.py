#written in Atom
#Nelson Walker started 01/17/2022
#Working with my mentor Mr. Himanshu M to run my saved checkpoint using my test dataloader
'''
Looking at your code, I don't see any issue there. What I find different was that you have performed
validation on your test data before
saving the checkpoint in your training notebook. Could you run the following piece of code:
After you load your checkpoint as well, this will help ensure that your checkpoint is correct.
If you are getting simillar accuracy results as earlier your checkpoint is good.

What I believe you are confusing accuracy with is predicted probability on a certain
image for eg: 57%, 43% that you are getting on your inference image. But, that is just one example,
there might be other classes of images where your model is performing better and you get higher
probability during predictions.

Let me know how it goes either ways.

Himanshu M
Mentor    1/16/2022


#test commandlines
python test_checkpoint.py --checkpoint_file "checkpoint_cmdline.pth" --gpu "True"
python test_checkpoint.py --checkpoint_file "checkpoint_cmdline.pth" --gpu "True"
python test_checkpoint.py --checkpoint_file "checkpoint.pth" --gpu "True"
'''
import argparse
import numpy as np

import json
import torch
from torchvision import datasets, transforms, models
from PIL import Image

def define_args():
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('--checkpoint_file',
        dest="checkpoint_file",
        action="store",
        help="checkpoint file"
        #,default="checkpoint_cmdline.pth"
        )

    arg_parser.add_argument('--gpu',
        dest="gpu",
        action="store",
        help="use gpu to run process"
        #,default="True"
        )

    return arg_parser.parse_args()

def load_checkpoint(file_path):
    checkpoint = torch.load(file_path, map_location = 'cpu')
    model = checkpoint["model"]
    model.load_state_dict(checkpoint['state_dict'])
    #criterion = checkpoint["criterion"]
    return model #, criterion

def get_data():

    data_dir = 'flowers'
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
            'validation': torch.utils.data.DataLoader(img_datasets['validation'], batch_size = 64, shuffle=True),
            'testing': torch.utils.data.DataLoader(img_datasets['testing'], batch_size = 32, shuffle=False)
        }

    return img_datasets, img_dataloaders

#------------------------------------------------
# MAIN get started
#------------------------------------------------
def main():

    args = define_args()
    checkpoint_file = args.checkpoint_file
    gpu = args.gpu

    if gpu == "True":
        boolgpu = True
    else:
        boolgpu = False

    print('--------------------------------------------------')
    print('Run test dataset on saved checkpoint')
    print('--------------------------------------------------')
    print('Arguments')
    print(f'checkpoint_file: {checkpoint_file}')
    print(f'gpu: {gpu}')
    print('--------------------------------------------------')

    if boolgpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print("CUDA device: {}".format(torch.cuda.get_device_name(0)))
        print("Device: {}".format(device))
    else:
        device = 'cpu'
        print("Device: {}".format(device))

    #get dataset
    img_datasets, img_dataloaders = get_data()
    testing_dataloader = img_dataloaders['testing']

    #reconstruct model from saved checkpoint
    model = load_checkpoint(checkpoint_file)

    print("Using device: {}".format(device))
    print("Testing dataloader size: {}".format(len(testing_dataloader)))
    print("---------------------------------------------------------")

    # This code from mentor Himanshu M
    # TODO: Do validation on the test set
    # the next two lines should be run on the model loaded from checkpoint

    model.to(device)
    model.eval()
    print("Device: {}".format(device))
    test_loss = 0
    test_accuracy = 0
    with torch.no_grad():
        #RUN through testing dataset
        for images, labels in testing_dataloader:
            images, labels = images.to(device), labels.to(device)

            output = model.forward(images)
            #test_loss += criterion(output,labels).item()
            ps = torch.exp(output)
            doesequal = (labels.data == ps.max(dim = 1)[1])
            test_accuracy += doesequal.type(torch.FloatTensor).mean()

    print("Network accuracy on test image dataset: {:.4f}".format(test_accuracy/len(testing_dataloader)))
    print('-----Complete------------------------------------')

if __name__ == '__main__':
    main()
