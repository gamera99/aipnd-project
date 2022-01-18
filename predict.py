#written in Atom
#Nelson Walker 01/17/2022
#All of these functions were based on my prediction routines in my ipnyb jupyter notebook project

""" rubric for predict.py
Predicting classes	The predict.py script successfully reads in an image and a checkpoint then prints the most likely image class and it's associated probability
Top K classes	The predict.py script allows users to print out the top K classes along with associated probabilities
Displaying class names	The predict.py script allows users to load a JSON file that maps the class values to other category names
Predicting with GPU	The predict.py script allows users to use the GPU to calculate the predictions

"flowers/test/10/image_07090.jpg"
"flowers/test/20/image_04910.jpg"
"flowers/test/30/image_03488.jpg"
"flowers/test/40/image_04563.jpg"
"flowers/test/50/image_06320.jpg"

python predict.py --image "flowers/test/50/image_06320.jpg" --gpu=True
python predict.py --image "flowers/test/50/image_06320.jpg" --gpu=False
"""

import argparse
import numpy as np

import json
import torch
from torchvision import datasets, transforms, models
from PIL import Image

def define_args():
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('--image',
        dest="image",
        action="store",
        help="filepath for test image")
    arg_parser.add_argument('--topk',
        dest="topk",
        action="store",
        help="how many top probabilities to return",
        default=5)
    arg_parser.add_argument('--checkpoint_file',
        dest="checkpoint_file",
        action="store",
        help="checkpoint file",
        default="checkpoint_cmdline.pth")
    arg_parser.add_argument('--json_file',
        dest="json_file",
        action="store",
        help="category file",
        default="cat_to_name.json")
    arg_parser.add_argument('--gpu',
        dest="gpu",
        action="store",
        help="use gpu to predict",
        default="False")

    return arg_parser.parse_args()

# Nelson Walker 01/15/2022
def load_checkpoint(file_path):
    checkpoint = torch.load(file_path, map_location = 'cpu')
    model = checkpoint["model"]
    model.load_state_dict(checkpoint['state_dict'])
    return model

# Nelson Walker 01/16/2022
def process_image(str_image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    image1 = Image.open(str_image_path)

    new_size = 256
    image1 = image1.resize((new_size,new_size))

    new_dim = 224
    top_dim = (new_size - new_dim) / 2
    bottom_dim = (new_size + new_dim) / 2
    right_dim = (new_size + new_dim) / 2
    left_dim = (new_size - new_dim) / 2

    image1 = image1.crop((left_dim,top_dim,right_dim,bottom_dim))

    np_image = np.array(image1)/new_size

    #normalize image  mean and std dev
    means = np.array([0.485, 0.456, 0.406])
    std_dev = np.array([0.229, 0.224, 0.225])

    np_img = (np_image - means)/std_dev

    #The color channel needs to be first and retain the order of the other two dimensions.
    np_img = np_img.transpose((2,0,1))  #move 2 to first

    return np_img

#completely provided by Udacity from my ipynb project
#do not alter
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax

#Nelson Walker 01/16/2022  from my ipynb project
def predict(image_path, model, topk, gpu):

    if gpu == True:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print("CUDA device: {}".format(torch.cuda.get_device_name(0)))
    else:
        device = 'cpu'
    model.to(device)
    print("Using device: {}".format(device))
    # Go into evaluation inference, no dropout
    model.eval()

    # process the image with process_image function
    image = process_image(image_path)

    # tranfer image to tensor
    image = torch.from_numpy(np.array([image])).float()
    image = image.to(device)
    #run image through model
    with torch.no_grad():
        output = model.forward(image)

    pbs = torch.exp(output).data

    #print("torch.exp pbs")
    #print(torch.topk(pbs, topk)[0])
    #print(torch.topk(pbs, topk)[1])

    #top 5 probabilities
    pbs_topk = torch.topk(pbs, topk)[0].tolist()[0]
    #top 5 class index
    image_index_topk = torch.topk(pbs, topk)[1].tolist()[0]

    #print("image_index_topk")
    #print(image_index_topk)

    # all class numbers and names in json
    lenclass = len(model.class_to_idx.items())
    classnumbers = []
    for i in range(lenclass):
        classnumbers.append(list(model.class_to_idx.items())[i][0])

    #print(classnames)

    # transfer index to label for top 5 probabilities
    label_topk = []
    for i in range(5):
        label_topk.append(classnumbers[image_index_topk[i]])

    #print(pbs_topk,label_topk)

    return pbs_topk, label_topk
#------------------------------------------------
# MAIN get started
#------------------------------------------------
def main():
    args = define_args()
    image = args.image
    topk = int(args.topk)
    checkpoint_file = args.checkpoint_file
    json_file = args.json_file
    gpu = args.gpu
    if (gpu == "True"):
        gpu = True
    else:
        gpu = False
    print('--------------------------------------------------')
    print('Prediction - arguments')
    print('--------------------------------------------------')
    print(f'image: {image}')
    print(f'topk: {topk}')
    print(f'checkpoint_file: {checkpoint_file}')
    print(f'json_file: {json_file}')
    print(f'gpu: {gpu}')
    print('--------------------------------------------------')

    #reconstruct model from saved checkpoint
    saved_model = load_checkpoint(checkpoint_file)

    #Nelson Walker 01/16/2022
    # TODO: Display an image along with the top 5 classes
    #With help from Udacity def view_classify(img, ps,version)

    #print(saved_model)
    #Predict probabilities with my model
    #image = Image.open(test_image)
    #plt.imshow(image)

    probs, classes = predict(image, saved_model, topk, gpu)

    #print(probs)
    #print(classes)
    #print([cat_to_name[i] for i in classes])

    max_index = np.argmax(probs)
    #print(max_index)
    label_idx = classes[max_index]

    with open(json_file, 'r') as f:
        cat_to_name = json.load(f)

    p_labels = []
    for sclass in classes:
        p_labels.append(cat_to_name[sclass])

    print("Prediction Results - top {} results".format(topk))
    for i, index in enumerate(classes):
        print(" {:.2%}".format(probs[i]), "", cat_to_name[index])
      #--------- end of main ---------------

if __name__ == '__main__':
    main()
