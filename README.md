# Readme
# Tennis Image classification  
### Galvanize Capstone Project
  Code for building a transfer learning neural network using python's Keras library.  Weights for the transfer network can be obtained from [VGG16 Github](https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3).   


## Table of Contents
1. [Dependencies](##Dependencies)
2. [Motivation](##Motivation)
3. [Data](##Data)
4. [Model](##Model)
5. [Running Model](##Running Model)
6. [Performance](##Performance)

## Dependencies
  * Python 2.7
  * Keras
  * Numpy
  * Theano

## Motivation

The goal of the project is to build a neural network model to automate the classification of tennis video for the company Cizr.  This would allow the out of play portion of the video to be automatically cut and the active play portions can be packaged for players or coaches.

## Data
  The model takes images from directories that are named with the labels of the data.  All resizing and augmentations happen within the functions that generate batches for the neural network.  

## Model
This model employs supervised learning, specifically deep learning and transfer learning techiniques to deal with a complex dataset
### Deep Learning
Deep learning neural networks are powerful techniques for understanding and learning about complicated datasets.  Neural networks are made of layers that process data in different ways by aggregation or applying filters.  Deep neural networks have many layers that can be used understand minor details about a dataset that in aggregate allow a model to make accurate predictions about complicated data.  For example, datasets made of images are very complex and it is difficult to define features, so providing a network with many layers allows the network to be able to define features that it needs in order to learn things about a image set.  Unfortunately, deep networks take a very long time to train on existing hardware.  

![Neural Network](https://cloud.githubusercontent.com/assets/17914936/20403127/2862931e-acc5-11e6-853c-02cac20c4ce1.png?style=centerme)
Figure 1

### Transfer Learning

Transfer learning allows new models to reuse weights generated from previously trained neural networks(Figure 2).  This is useful because the first several layers of a network are learning general things about an image.  

![Transfer Network](https://cloud.githubusercontent.com/assets/17914936/20403126/286226fe-acc5-11e6-9855-693183fab83e.png?style=centerme)
Figure 2

## Running Model

###### The file structure for the repository:

```  
  ├── README.md  
  ├── .gitignore  
  ├── src   
  │     ├── generate_weights_for_transfer.py  
  |     ├── images_download.py  
  |     ├── new_image_prediction.py    
  │     ├── reload_transfer_tennis_cnn.py
  |     ├── text_file_formatter.py
  |     ├── transfer_tennis_cnn.py  
  ├── test  
  │     ├── inside_point  
  |     ├── outside_point  
  |     ├── serve  
  ├── train  
  │     ├── inside_point  
  |     ├── outside_point  
  |     ├── serve  
  └── weights
```

###### To format image names and download them into directories with the classification labels:
1. Run text_file_formater.py to get image names
2. Run images_downlad.py in the correctly labeled directory to add images to the directory

###### To train a new transfer network on new images:
1. Run generate_weights.py to build a fully trained classifier and save the weights
2. Run transfer_tennis_cnn.py to load your classifier and the vgg16 weights and fine tune the weights and save a final model

###### To make predictions on new images:
1. Edit new_image_prediction.py with the directory of your images and run new_image_prediction.py


## Performance

[include accuracy/epoch graphs, confusion matrix]
