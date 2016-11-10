import numpy as np


def file_reader(text_file):
    '''
    Takes a text file and replaces new lines then splits on spaces.
    then appends the labels and image names as tuples to tup_list and
    returns the list
    '''
    f = open(text_file, 'r')
    text = f.read()

    text_clean = text.replace('\n', ' ')
    text_clean = text_clean.split()

    labels = text_clean[1::2]
    image = text_clean[0::2]

    tup_lst = []
    i = 0
    while i in range(len(image)):
        tup = (image[i],labels[i])
        tup_lst.append(tup)
        i += 1
    return tup_lst

def label_split(tup_lst):
    '''
    splits up the tuple list from file_reader based on labels and returns the list
    of the images with their labels.
    '''
    serve = []
    InsidePoint = []
    OutsidePoint = []
    other = []
    for tup in tup_lst:
        if tup[1] == 'Serve':
            serve.append(tup)
        if tup[1] == 'InsidePoint':
            InsidePoint.append(tup)
        if tup[1] == 'OutsidePoint':
            OutsidePoint.append(tup)
    return serve, InsidePoint, OutsidePoint

def random_sample(split_tup, size):
    '''
    If you need a random subset of the data you can use this to randomly
    sample the data.
    '''
    samples = np.random.choice(len(split_tup), size = size, replace = False)
    image_array = np.array(split_tup)
    to_download = image_array[samples]
    return to_download
