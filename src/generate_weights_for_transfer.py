import os
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD
from keras.utils import np_utils


'''
This code is based on fchollet's github
https://gist.github.com/fchollet/f35fbc80e066a49d65f1688a7e99f069
'''

def load_VGG16_net(weights_path):
    '''
    This function loads all the weights for the transfered neural net from VGG16 weights
    '''
    datagen = ImageDataGenerator(rescale=1./255)

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

    '''

    extra layers which I decided to not use and train more convolutional layers in my
    model below

    '''

    #model.add(ZeroPadding2D((1, 1)))
    #model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    #model.add(ZeroPadding2D((1, 1)))
    #model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
    #model.add(ZeroPadding2D((1, 1)))
    #model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
    #model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    '''
    load the weights of the VGG16 networks
    (trained on ImageNet, won the ILSVRC competition in 2014)
    note: when there is a complete match between your model definition
    and your weight savefile, you can simply call model.load_weights(filename)
    '''
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

    return model

def generate_predictions(model, train_data_dir, validation_data_dir, img_width, img_height):

    '''
    generates predictions and saves them in numpy arrays

    '''
    datagen = ImageDataGenerator(rescale=1./255)

    train_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode=None,
        shuffle=False)

    print train_generator.class_indices
    train_labels = train_generator.classes

    nb_train_samples = train_generator.nb_sample
    bottleneck_features_train = model.predict_generator(train_generator, nb_train_samples)
    np.save(open('bottleneck_features_train.npy', 'w'), bottleneck_features_train)
    np.save(open('bottleneck_features_train_labels.npy', 'w'), train_labels)

    val_generator = datagen.flow_from_directory(
            validation_data_dir,
            target_size=(img_width, img_height),
            batch_size=32,
            class_mode=None,
            shuffle=False)
    print val_generator.class_indices
    validation_labels = val_generator.classes

    nb_validation_samples = val_generator.nb_sample
    bottleneck_features_validation = model.predict_generator(val_generator, nb_validation_samples)
    np.save(open('bottleneck_features_validation.npy', 'w'), bottleneck_features_validation)
    np.save(open('bottleneck_features_validation_labels.npy', 'w'), validation_labels)


def train_top_model(top_model_weights_path):
    '''
    loads in numpy arrays and trains model with layers defined below
    saves weights for the trained model
    '''
    train_data = np.load(open('bottleneck_features_train.npy'))
    train_labels = np.load(open('bottleneck_features_train_labels.npy'))
    validation_data = np.load(open('bottleneck_features_validation.npy'))
    validation_labels = np.load(open('bottleneck_features_validation_labels.npy'))

    train_labels = np_utils.to_categorical(train_labels)
    validation_labels = np_utils.to_categorical(validation_labels)

    model = Sequential()
    print train_data.shape[1:]
    print train_labels.shape

    model.add(Convolution2D(512, 3, 3, input_shape=train_data.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.5))

    model.add(Convolution2D(512, 3, 3))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Flatten(name = 'my_1'))
    model.add(Dense(512, init = 'uniform', name = 'my_2'))
    model.add(Activation('relu',  name='my_3'))
    model.add(Dropout(0.5))
    model.add(Dense(3, init = 'uniform', name='my_4'))
    model.add(Activation('softmax', name='my_5'))

    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              nb_epoch=nb_epoch, batch_size=32,
              validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path, overwrite = True)

if __name__ == '__main__':
    # path to the model weights files.
    weights_path = '../../vgg16_weights.h5'
    top_model_weights_path = '21k_img_weights_matchs_together.h5'
    # dimensions of our images.
    img_width, img_height = 150, 150
    #paths to data
    train_data_dir = '../../train'
    validation_data_dir = '../../test'

    nb_epoch = 48

    model = load_VGG16_net(weights_path)
    #generate_predictions(model, train_data_dir, validation_data_dir, img_width, img_height)
    train_top_model(top_model_weights_path)
