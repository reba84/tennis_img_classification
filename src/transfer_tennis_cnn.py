import os
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from datetime import datetime



def image_processing(train_dir, test_dir, img_width, img_height, batch_size):

    '''
    use once to vary images once cnn is working

    '''
    # keras.preprocessing.image.ImageDataGenerator(featurewise_center=False,
    #     samplewise_center=False,
    #     featurewise_std_normalization=False,
    #     samplewise_std_normalization=False,
    #     zca_whitening=False,
    #     rotation_range=0.,
    #     width_shift_range=0.,
    #     height_shift_range=0.,
    #     shear_range=0.,
    #     zoom_range=0.,
    #     channel_shift_range=0.,
    #     fill_mode='nearest',
    #     cval=0.,
    #     horizontal_flip=False,
    #     vertical_flip=False,
    #     rescale=None,
    #     dim_ordering=K.image_dim_ordering())

    train_datagen = ImageDataGenerator(rescale=1./255,
                                        horizontal_flip = True,
                                        vertical_flip = False,
                                        rotation_range= 10,
                                        width_shift_range=0,
                                        height_shift_range=0)

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

    classes = train_processing.nb_class
    n_train_samples = train_processing.nb_sample
    n_test_samples = test_processing.nb_sample

    return train_processing, test_processing, classes, n_train_samples, n_test_samples

def build_net(weights_path, top_model_weights_path, classes, img_width, img_height, nb_fitlers, pool_size, kernel_size):

    # build the VGG16 network
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, img_width, img_height)))

    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
    # model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # load the weights of the VGG16 networks
    # (trained on ImageNet, won the ILSVRC competition in 2014)
    # note: when there is a complete match between your model definition
    # and your weight savefile, you can simply call model.load_weights(filename)
    assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
    f = h5py.File(weights_path)
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            # we don't look at the last (fully-connected) layers in the savefile
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
    f.close()
    print('Model loaded.')

    # build a classifier model to put on top of the convolutional model
    top_model = Sequential()

    top_model.add(Convolution2D(512, 3, 3, input_shape=model.output_shape[1:]))
    top_model.add(Activation('relu'))
    top_model.add(MaxPooling2D((2,2)))
    top_model.add(Dropout(0.5))

    top_model.add(Convolution2D(512, 3, 3))
    top_model.add(Activation('relu'))
    top_model.add(Dropout(0.5))

    top_model.add(Flatten(name = 'my_1'))
    top_model.add(Dense(512, init = 'uniform', name = 'my_2'))
    top_model.add(Activation('relu',  name='my_3'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(3, init = 'uniform', name='my_4'))
    top_model.add(Activation('softmax', name='my_5'))

    # note that it is necessary to start with a fully-trained
    # classifier, including the top classifier,
    # in order to successfully do fine-tuning
    top_model.load_weights(top_model_weights_path)

    # add the model on top of the convolutional base
    model.add(top_model)

    # set the first 18 layers (up to the last conv block)
    # to non-trainable (weights will not be updated)
    for layer in model.layers[:18]:
        layer.trainable = False

    # compile the model with a SGD/momentum optimizer
    # and a very slow learning rate.
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    return model

def fit_net(model, train_processing, test_processing, n_train_samples, n_test_samples, epoch):
    epoch_weights = ModelCheckpoint(filepath = '../weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=0, save_best_only=True)
    fit = model.fit_generator(train_processing, samples_per_epoch = n_train_samples, nb_epoch = epoch, validation_data = test_processing, nb_val_samples = n_test_samples, callbacks = [epoch_weights])
    accuracy = 'acc: {}, loss: {}, val_acc: {}, val_loss: {}'.format(*fit.history.values())
    return fit


def save_model(fit, epoch, batch_size, classes, n_train_samples, n_test_samples):
    ## --- Save Settings ---
    datetime_str = str(datetime.now()).split('.')[0]
    # datetime = datetime_str.replace(" ", "_")

    #Save Weights & Model
    weights_path = '../weights/' + datetime_str + '.h5'
    architecture_path = '../weights/' + datetime_str + '.json'
    model.save_weights(weights_path, overwrite=True)
    model_json = model.to_json()
    with open(architecture_path, "w") as json_file:
        json_file.write(model_json)

    #Save Parameters and Accuracy
    parameters = '\nn_train_samples: {}, n_test_samples: {}, n_epoch: {}, batch_size: {}\n'.format(n_train_samples, n_test_samples, epoch, batch_size)
    accuracy = 'acc: {}, loss: {}, val_acc: {}, test_loss: {}'.format(*fit.history.values())
    text = '\n' + datetime_str + parameters + accuracy
    with open('log' + datetime_str + '.txt', "w") as myfile:
        myfile.write(text)

    print "Saved!"

if __name__ == '__main__':
    #Set Parameters fpr weights
    weights_path = '../../vgg16_weights.h5'
    top_model_weights_path = '21k_img_weights_matches_together.h5'
    #Set image perameters
    img_width, img_height = 150, 150
    train_dir = '../../train'
    test_dir = '../../test'
    epoch = 50
    batch_size = 32
    pool_size = (2, 2)
    kernel_size = (3, 3)
    nb_filters = 32

    #fit_image_generators, build CNN, train_network, save history
    train_processing, test_processing, classes, n_train_samples, n_test_samples = image_processing(train_dir, test_dir, img_width, img_height, batch_size)
    model = build_net(weights_path, top_model_weights_path, classes, img_width, img_height, nb_filters, pool_size, kernel_size)
    fit_model = fit_net(model, train_processing, test_processing, n_train_samples, n_test_samples, epoch)
    save_model(fit_model, epoch, batch_size, classes, n_train_samples, n_test_samples)
