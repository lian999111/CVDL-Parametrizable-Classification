# Deep Learning for Computer Vision
# Parsing some of the used datasets for the exercises
#
# Sebastian Houben, sebastian.houben@ini.rub.de

import numpy as np
from tensorflow.keras.datasets import mnist
import os
import pickle
import matplotlib.pyplot as plt

# TODO: set path to the location to which CIFAR was extracted (where the file data_batch_1 is located)
cifar_path = r'PATH_TO_CIFAR'


def cull_unused_classes(x, y, used_labels):
    """
    Removes data x and label y entries for classes not in used_labels. 
    """
    idxs = [label in used_labels for label in y]
    x = x[idxs]
    y = y[idxs]

    return x, y


def make_successive_labels(y):
    """
    Replaces the integers in y such that only successive integers appear. 

    Example: [2 4 4 2 6 2 9 9 4 2] -> [0 1 1 0 2 0 3 3 1 0]
    """
    for (new_label, unique_label) in enumerate(np.unique(y)):
        y[y == unique_label] = new_label

    return y


def get_dataset(dataset='mnist',
                used_labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                training_size=60000,
                test_size=10000):
    """ 
    Reads and converts data from the MNIST dataset. 

    :param  dataset         'mnist' or 'cifar10'
    :param  used_labels     list of digit classes to include into the returned subset
    :param  training_size   number of images from training size
    :param  test_size       number of images from test_size

    :return x_train, y_train, x_test, y_test, class_names    
      x_train, x_test: training and test images (uint8 [0- 255], shape: training_size x 28 x 28, test_size x 28 x 28),
      y_train, y_test: corresponding labels (int32 [0 - len(used_labels)), shape: training_size / test_size)
      class_names: array with names of classes (size: len(used_labels))
    """

    num_classes = len(used_labels)
    max_num_classes = 10

    np.random.seed(4711)

    if dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()  # x_train.shape = 60,000 x 28 x 28

        class_names = map(str, used_labels)
    elif dataset == 'cifar10':

        if not os.path.isfile(cifar_path + r'\data_batch_1'):
            raise FileNotFoundError('The path %s does not seem to contain the CIFAR-10 dataset. ' % (cifar_path))

        np_x_train_file = cifar_path + r'\numpy_dump_x_train.npy'
        np_y_train_file = cifar_path + r'\numpy_dump_y_train.npy'
        np_x_test_file = cifar_path + r'\numpy_dump_x_test.npy'
        np_y_test_file = cifar_path + r'\numpy_dump_y_test.npy'

        if not os.path.isfile(np_x_train_file):  # first time loading

            file_idxs = [1, 2, 3, 4, 5]  # which cifar batches 1..5 should be loaded

            # fills the labels and data variables
            # labels contains the NR class labels 0..9 of each CIFAR image
            # data is an array NR x 3072 where each line contains the pixel information of a CIFAR image, the memory layout is RGB, column, row 
            for file_idx in file_idxs:
                with open((cifar_path + r'\data_batch_%d') % file_idx, 'rb') as cifar_file:
                    dict = pickle.load(cifar_file, encoding='bytes')
                    if not 'y_train' in locals():
                        y_train = np.array(dict[b'labels'], np.int32)
                        x_train = dict[b'data']
                    else:
                        y_train = np.concatenate((y_train, np.array(dict[b'labels'], np.int32)))
                        x_train = np.concatenate((x_train, dict[b'data']))

            x_train = np.transpose(np.reshape(x_train, [x_train.shape[0], 3, 32, 32]), [0, 2, 3, 1])

            with open(cifar_path + r'\test_batch', 'rb') as cifar_file:
                dict = pickle.load(cifar_file, encoding='bytes')
                y_test = np.array(dict[b'labels'], np.int32)
                x_test = dict[b'data']

            x_test = np.transpose(np.reshape(x_test, [x_test.shape[0], 3, 32, 32]), [0, 2, 3, 1])

            np.save(np_x_train_file, x_train)
            np.save(np_y_train_file, y_train)
            np.save(np_x_test_file, x_test)
            np.save(np_y_test_file, y_test)

        else:
            x_train = np.load(np_x_train_file)
            y_train = np.load(np_y_train_file)
            x_test = np.load(np_x_test_file)
            y_test = np.load(np_y_test_file)

        # names of the 10 CIFAR classes                
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        class_names = [class_names[idx] for idx in used_labels]


    else:
        raise ValueError('The variable dataset must either be set to "mnist" or "cifar10"')

    if num_classes != 10:
        x_train, y_train = cull_unused_classes(x_train, y_train, used_labels)
        x_test, y_test = cull_unused_classes(x_test, y_test, used_labels)

    # make the class indices consecutive (only if not all 10 digits are chosen in used_labels)

    y_train = make_successive_labels(y_train)
    y_test = make_successive_labels(y_test)

    original_training_size = x_train.shape[0]
    original_test_size = x_test.shape[0]

    # take a small random subset of images (size is given in training_size and test_size)
    training_idxs = np.arange(original_training_size)
    np.random.shuffle(training_idxs)
    training_idxs = training_idxs[0:training_size]
    x_train = x_train[training_idxs]
    y_train = y_train[training_idxs]
    y_train = y_train.astype(np.int32)

    test_idxs = np.arange(original_test_size)
    np.random.shuffle(test_idxs)
    test_idxs = test_idxs[0:test_size]
    x_test = x_test[test_idxs]
    y_test = y_test[test_idxs]
    y_test = y_test.astype(np.int32)

    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)

    return x_train, y_train, x_test, y_test, class_names


if __name__ == '__main__':

    x_train, y_train, x_test, y_test, class_names = get_dataset(dataset='cifar10', training_size=600, test_size=100)

    print(x_train.shape)

    # Show one example of each class 
    plt.figure()
    for class_id in range(len(class_names)):
        plt.subplot(2, 5, class_id + 1)
        plt.imshow(x_train[y_train == class_id][0].astype(
            np.uint8))  # plotting behaviour of imshow differs between float32 and uint8
        plt.title(class_names[class_id])
    plt.show()
