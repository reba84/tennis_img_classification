import os
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from datetime import datetime
import reload_transfer_tennis_cnn as reload_net
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score, recall_score


def predict_image(image_directory):
    datagen = ImageDataGenerator(rescale=1./255)
    val_generator = datagen.flow_from_directory(
            image_directory,
            target_size=(img_width, img_height),
            batch_size=32,
            shuffle=False)
    datetime_str = str(datetime.now()).split('.')[0]
    print datetime_str
    print val_generator.class_indices
    targets = val_generator.classes

    num_images = val_generator.nb_sample
    image_predictions = model.predict_generator(val_generator, num_images)
    pred_list = []
    for row in image_predictions:
        max_pred = np.argmax(row)
        pred_list.append(max_pred)

    return np.array(pred_list), targets
    #np.save(open('image_predictions' + datetime_str + '.npy', 'w'), image_predictions)
    #np.save(open('targets' + datetime_str + '.npy', 'w'), targets)
if __name__ == '__main__':
    image_directory = "test"
    img_width = 150
    img_height = 150
    weights_path = 'weights/2016-11-06 09:03:13.h5'
    architecture_path = 'weights/2016-11-06 09:03:13.json'

    #load net here
    model = reload_net.load_net(weights_path, architecture_path)
    image_predictions, targets = predict_image(image_directory)
    print image_predictions
    print confusion_matrix(targets, image_predictions)
    print accuracy_score(targets, image_predictions)
    print cohen_kappa_score(targets, image_predictions)
    #print recall_score(targets, image_predictions, average = 'samples')
