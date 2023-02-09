import argparse
import json
import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision import transforms
import torch.nn.functional as F

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', action='store', type=str, default='flowers/test/58/image_02663.jpg')
    parser.add_argument('checkpoint', action='store', type=str, default='checkpoints.pth')
    #parser.add_argument('--checkpoint', metavar='checkpoint', type=str, default='vgg16_checkpoint.pth')
    #parser.add_argument('--checkpoint', action='store', default='vgg16_checkpoint.pth')
    parser.add_argument('--topk', action='store', dest="topk", type=int, default=5)
    parser.add_argument('--category_names', action='store', dest='category_names', type=str, default='cat_to_name.json')
    parser.add_argument('--gpu', action="store_true", default='gpu', dest="gpu")
    return parser.parse_args() 


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    _model = getattr(torchvision.models, checkpoint['arch'])(pretrained=True)
    _model.input_size = checkpoint['input_size']
    _model.output_size = checkpoint['output_size']
    _model.learning_rate = checkpoint['learning_rate']
    _model.hidden_units = checkpoint['hidden_units']
    _model.learning_rate = checkpoint['learning_rate']
    _model.classifier = checkpoint['classifier']
    _model.epochs = checkpoint['epoch']
    _model.load_state_dict(checkpoint['state_dict'])
    _model.class_to_idx = checkpoint['class_to_idx']
    _model.optimizer = checkpoint['optimizer']
    return _model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    args = parse_args()
    gpu = args.gpu
    if  gpu and torch.cuda.is_available() == True:
        device=torch.device('cuda')                    
    else:
        device=torch.device('cpu')
    # images have to be normalized
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    # preprocess step
    preprocess = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     normalize
                                    ])
    loaded_image = Image.open(image)
    img_tensor = preprocess(loaded_image)
    return img_tensor.to(device)

def predict(image_path, model, topk=5):
    
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    args = parse_args()
    gpu = args.gpu
    if  gpu and torch.cuda.is_available() == True:
        device=torch.device('cuda')                    
    else:
        device=torch.device('cpu')
    # TODO: Implement the code to predict the class from an image file
    model.eval()
    model.to(device)
    input_img = process_image(image_path)
    input_img = input_img.unsqueeze_(0)
    input_img = input_img.float()
    
    with torch.no_grad(): 
        output = model(input_img)
        ps = torch.exp(output)
        top_ps, top_classes = ps.topk(topk, dim=1)
        
        top_ps = top_ps.tolist()[0]
        top_classes = top_classes.tolist()[0]

        idx_to_class = {value:key for key, value in model.class_to_idx.items()}
        top_classes = [idx_to_class.get(x) for x in top_classes]
        
    return top_ps, top_classes

def load_names(category_names_file):
    with open(category_names_file) as f:
        category_names = json.load(f)
    return category_names


def main():
    args = parse_args()
    image_path = args.image_path
    checkpoint = args.checkpoint
    topk = args.topk
    category_names = args.category_names
    gpu = args.gpu
    
    model = load_checkpoint(checkpoint)

    top_p, top_classes = predict(image_path, model, topk)

    category_names = load_names(category_names)

    labels = [category_names[str(index)] for index in top_classes]

    print(f"Results for your File: {image_path}")
    print(labels)
    print(top_p)
    print()

    for i in range(len(labels)):
        print("{} - {} with a probability of {}".format((i+1), labels[i], top_p[i]))


if __name__ == "__main__":
    main()


