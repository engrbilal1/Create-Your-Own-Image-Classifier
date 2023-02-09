# Create-Your-Own-Image-Classifier
```
Final Project of "AI Programming with Python" Udacity Nano Degree
```
### Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Hyperparameters](#hyperparameters)
4. [Training](#training)
5. [Predicting](#predicting)
6. [Conclusion](#conclusion)

## Overview <a name="overview"></a>
- AI with Python nanodegree final project, In this project we implemented an image classification application. This application will train a deep learning model on a dataset of images. It will then use the trained model to classify new images.

- In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice, you'd train this classifier, then export it for use in your application. We'll be using this dataset of 102 flower categories. Predict flower name from an image with predict.py along with the probability of that name. That is you'll pass in a single image /path/to/image and return the flower name and class probability

## Installation <a name="installation"></a>
This project requires Python 3.6.3 and the following Python libraries installed:
- PyTorch
- ArgParse
- Jason
- PIL
- NumPy
- Pandas
- matplotlib

## Hyperparameters <a name="hyperparameters"></a>
- Basically, anything in machine learning and deep learning that you decide their values or choose their configuration before training begins and whose values or configuration will remain the same when training ends is a hyperparameter. Here we will discuss two hyperparameters:
1. `Number of iterations (epochs)`:
    * By increasing the number of epochs the accuracy of the network on the training set gets better and better however be careful because if you pick a large number of epochs the network won't generalize well, that is to say it will have high accuracy on the training image and low accuracy on the test images. Eg: training for 12 epochs training accuracy: 87% Test accuracy: 83%. Training for 30 epochs training accuracy 94% test accuracy 60%.
2. `Learning rate (lr)`:
    * A big learning rate guarantees that the network will converge fast to a small error but it will constantly overshot
    * A small learning rate guarantees that the network will reach greater accuracies but the learning process will take longer

## Training <a name="training"></a>
- Train a new network on a data set with `train.py`.
  - Basic usage: python train.py data_directory
  - Prints out training loss, validation loss, and validation accuracy as the network trains
  - Options:
    - Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
    - Choose architecture: python train.py data_dir --arch "vgg16"
    - Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 4096 --epochs 15
    - Use GPU for training if available : python train.py data_dir --gpu

## Predicting <a name="predicting"></a>
- Predict flower name from an image with `predict.py`.
  - Basic usage: python predict.py /path/to/image checkpoint
  - Options:
    - Return top KK most likely classes: python predict.py input checkpoint --topk 5
    - Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
    - Use GPU for inference: python predict.py input checkpoint --gpu

## Conclusion <a name="conclusion"></a>
- As you know the process of training a model and then using it to predict classes of images. Also you saw, Transfer Learning is a powerful thing. We have successfully took a pre-trained Convolutional Neural Network, modified it, retrained it, and used it to predict species of 102 different flowers with over an 90% accuracy per testing results!
