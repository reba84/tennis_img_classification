# Readme
# Tennis Image classification

## Overview

The goal of the project is to build a neural network model to automate the classification of tennis video for the company [Cizr](http://www.cizr.com/).  This would allow the out of play portion of the video to be automatically cut and the active play portions can be packaged for players or coaches.

## Table of Contents
1. [Dependencies](https://github.com/reba84/tennis_img_classification/blob/master/README.md#dependencies)
2. [Motivation](https://github.com/reba84/tennis_img_classification/blob/master/README.md#motivation)
3. [Data](https://github.com/reba84/tennis_img_classification/blob/master/README.md#data)
4. [Model](https://github.com/reba84/tennis_img_classification/blob/master/README.md#model)
5. [Running Model](https://github.com/reba84/tennis_img_classification/blob/master/README.md#running-model)
6. [Performance](https://github.com/reba84/tennis_img_classification/blob/master/README.md#performance)

## Dependencies
  * [Python 2.7](https://www.python.org/download/releases/2.7/)
  * [Keras](https://keras.io/)
  * [Numpy](http://www.numpy.org/)
  * [Theano](http://deeplearning.net/software/theano/)

## Data
  The model takes images from directories that are named with the labels of the data.  All resizing and augmentations happen within the functions that generate batches for the neural network.  

## Model

This model employs supervised learning, specifically deep learning and transfer learning techiniques to deal with a complex dataset

### Deep Learning
Deep learning neural networks are powerful techniques for understanding and learning about complicated datasets.  Neural networks are made of layers that process data in different ways, for example by aggregating data or applying filters.  Deep neural networks have many layers that can be used understand minor details about a dataset that, in aggregate, allow a model to make accurate predictions about complicated data.  For example, datasets made of images are very complex and it is difficult to define features, so providing a network with many layers allows the network to be able to define features that it needs in order to learn useful details about a image set.  Unfortunately, deep networks take a very long time to train on existing hardware.  

![Neural Network](https://cloud.githubusercontent.com/assets/17914936/20403127/2862931e-acc5-11e6-853c-02cac20c4ce1.png?style=centerme)
Figure 1

### Transfer Learning

Transfer learning allows new models to reuse weights from layers generated from previously trained neural networks(Figure 2).  This is useful because the first several layers of a network are learning general things about an image. The biggest advantage to reusing weights from previously trained networks makes the training process much quicker on deep networks and higher model accuracy can be achieved.  

![Transfer Network](https://cloud.githubusercontent.com/assets/17914936/20403126/286226fe-acc5-11e6-9855-693183fab83e.png?style=centerme)
Figure 2

## Running the Model
 Weights for the transfer network can be obtained from [VGG16 Github](https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3).   
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
It was trained with 14,700 and validated against 6,300 images.  The model is best at predicting serves and overall has a 66% accuracy on the validation set.  If you consider serves as the one condition and the inside point and outside point as the other the model has 79% accuracy on serve images.

![Confusion_matrix](https://cloud.githubusercontent.com/assets/17914936/20437644/5eb3d288-ad7a-11e6-80df-d1e7203f1d31.png)

Accuracy and loss graphs showing model perfomance over 100 epochs

![Val_training_graph](https://cloud.githubusercontent.com/assets/17914936/20437409/78f83360-ad79-11e6-91d7-40f039ec7ebc.png)  
