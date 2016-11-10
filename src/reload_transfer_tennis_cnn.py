import os
import h5py
import numpy as np
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from datetime import datetime



def image_processing(train_dir, test_dir, img_width, img_height, batch_size):

    train_datagen = ImageDataGenerator(rescale=1./255,
                                        horizontal_flip = True,
                                        vertical_flip = False,
                                        rotation_range=20,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_processing = train_datagen.flow_from_directory(
            train_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            #color_mode = "grayscale",
            #class_mode='binary',
            shuffle=True)


    test_processing = test_datagen.flow_from_directory(
            test_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            #color_mode = "grayscale",
            #class_mode='binary',
            shuffle=True)

    n_train_samples = train_processing.nb_sample
    n_test_samples = test_processing.nb_sample

    return train_processing, test_processing, n_train_samples, n_test_samples

def load_net(weights_path, architecture_path):

        # load json and create model
    json_file = open(architecture_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(weights_path)
    print("Loaded model from disk")

    # compile the model with a SGD/momentum optimizer
    # and a very slow learning rate.
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    loaded_model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    return loaded_model

def fit_net(model, train_processing, test_processing, n_train_samples, n_test_samples, epoch):
    epoch_weights = ModelCheckpoint(filepath = 'transfer_weights/weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=0, save_best_only=True)
    fit = model.fit_generator(train_processing, samples_per_epoch = n_train_samples, nb_epoch = epoch, validation_data = test_processing, nb_val_samples = n_test_samples, callbacks = [epoch_weights])
    accuracy = 'acc: {}, loss: {}, val_acc: {}, val_loss: {}'.format(*fit.history.values())
    return fit


def save_model(fit, epoch, batch_size, n_train_samples, n_test_samples):
    ## --- Save Settings ---
    datetime_str = str(datetime.now()).split('.')[0]

    #Save Weights & Model
    weights_path = 'weights/'+ datetime_str + '.h5'
    architecture_path = 'weights/'+ datetime_str + '.json'
    model.save_weights(save_weights_path, overwrite=True)
    model_json = model.to_json()
    with open(architecture_path, "r+") as json_file:
        json_file.write(model_json)

    #Save Parameters and Accuracy
    parameters = '\nn_train_samples: {}, n_test_samples: {}, n_epoch: {}, batch_size: {}\n'.format(n_train_samples, n_test_samples, epoch, batch_size)
    accuracy = 'acc: {}, loss: {}, val_acc: {}, test_loss: {}'.format(*fit.history.values())
    text = '\n' + datetime_str + parameters + accuracy
    with open('log.txt', "r+") as myfile:
        myfile.write(text)

    print "Saved!"

if __name__ == '__main__':
    main()
    #Set Parameters for weights
    # weights_path = 'weights/3.h5'
    # architecture_path = 'weights/3.json'
