import numpy as np
import torchvision 
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models, transforms, datasets
import os
import json
import argparse
from torch import nn, optim
import time
import torch
import sys


def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data_dir', action='store')    #parser.add_argument('data_dir', metavar='data_dir', type = str)
    parser.add_argument('--save_dir', action = 'store', dest='save_dir',type = str, default='checkpoints.pth' )
    parser.add_argument('--arch', action='store', dest='arch', type=str, default='vgg16', choices=['vgg16', 'vgg19'])
    parser.add_argument('--learning_rate', action='store', dest='learning_rate', type =float, default=0.001)
    parser.add_argument('--hidden_units', action= 'store', dest='hidden_units', type= int, default = 4096)
    parser.add_argument('--epoch', dest='epoch', type = int, default='5')
    parser.add_argument('--gpu', action="store_true", default='gpu', dest="gpu")
    return parser.parse_args()

args = parse_args()
gpu = args.gpu

def load_model(arch, hidden_units, gpu):
  
    if  gpu and torch.cuda.is_available() == True:
        device=torch.device('cuda')                    
    else:
        device=torch.device('cpu')
    if arch == 'vgg16':
        model =models.vgg16(pretrained = True)
    elif arch == 'vgg19':
        model=models.vgg19(pretrained=True)
    else:
        print('Invalid Model is Choosen! Choose VGG16 or VGG19 Default is VGG16.')
        sys.exit()
  
    for param in model.parameters():
        param.required_grad = False
    num_in_features = 25088
    classifier=nn.Sequential(nn.Dropout(p=0.25),
                             nn.Linear(num_in_features,hidden_units),
                             nn.ReLU(),
                             nn.Dropout(p=0.2),
                             nn.Linear(hidden_units,2048),
                             nn.ReLU(),
                             nn.Dropout(p=0.2),
                             nn.Linear(2048, output_size),
                             nn.LogSoftmax(dim=1))
    model.classifier=classifier        
    model.to(device)
    return model, device, num_in_features
##train_model(arch, model,device, optimizer, criterion, trainloader, validloader,epoch)
def train_model(model,device, optimizer, criterion, trainloader, validloader,epoch): 
    
    train_loss=0
    count=20
    print('Training Start but you should have to wait b/c process is slow...')
    start = time.time()
    for e in range(epoch):   
        model.train()
        steps=0
        for images,labels in trainloader:
            steps +=1
            images, labels= images.to(device), labels.to(device)
            optimizer.zero_grad()
            output=model.forward(images)
            loss = criterion(output,labels)
            loss.backward()
            optimizer.step()
            train_loss +=loss.item()
            if steps % count == 0 or steps == len(trainloader):
                print("Epoch: {}/{} Batch % Complete: {:.2f}%".format(e+1, epoch, (steps)*100/len(trainloader)))
               
        valid_loss =0.0
        accuracy=0.0
        model.eval()
        with torch.no_grad():
            for images, labels in validloader:   
                images, labels=images.to(device), labels.to(device)
                output=model(images)
                loss = criterion(output,labels)
                valid_loss +=loss.item()
                ps=torch.exp(output)
                top_p, top_classes = ps.topk(1,dim=1)
                equal=top_classes==labels.view(*top_classes.shape)
                accuracy +=torch.mean(equal.type(torch.FloatTensor)).item()

        train_loss = train_loss/len(trainloader)
        valid_loss = valid_loss/len(validloader)
        accuracy = accuracy*100/len(validloader)
                
        print('Training Loss: {:.4f} \tValidation Loss:{:.4f} \t Accuracy:{:.2f}'.format(train_loss, valid_loss, accuracy))
    end = time.time()
    total_time = end - start
    print('Training time {:.0f}m : {:.0f}s'.format(total_time//60, total_time % 60))
 

def checkpoint_save(model, file_path, arch, train_dataset, input_size, output_size, hidden_units, optimizer, learning_rate, epoch):
    model.class_to_idx=train_dataset.class_to_idx
    checkpoint={
        'arch': arch,
        'input_size': input_size,
        'output_size': output_size,   
        'epoch': epoch,
        'learning_rate': learning_rate,
        'hidden_units': hidden_units,
        'classifier': model.classifier,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx,
        'optimizer': optimizer.state_dict()}
    torch.save(checkpoint, file_path)
def main():
    print('Data is Loading....')
    args=parse_args()
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    output_size = 102
    gpu = args.gpu
    file_path =args.save_dir
    Normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    train_transform = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          Normalize])
    valid_transform=transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        Normalize])
    print('transformation done...')
    train_dataset=datasets.ImageFolder(train_dir, transform=train_transform)
    valid_dataset=datasets.ImageFolder(valid_dir, transform=valid_transform)
    test_dataset=datasets.ImageFolder(test_dir, transform=valid_transform)
    
    print('dataset done...')
    trainloader=torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    validloader=torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=False)
    testloader=torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    with open('cat_to_name.json','r') as f:
        cat_to_name = json.load(f)
    
    print('now model_load() ftn calling...')
    model,device, num_in_features = load_model(args.arch, args.hidden_units, args.gpu)
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    print('now train_model() ftn calling...')
    train_model(model,device, optimizer, criterion, trainloader, validloader,args.epoch)

    print('now checkpoint_save() ftn calling...')
    checkpoint_save(model, file_path, args.arch, train_dataset, num_in_features, output_size, args.hidden_units, optimizer, args.learning_rate, args.epoch)        
 
if __name__ == '__main__':
    main()