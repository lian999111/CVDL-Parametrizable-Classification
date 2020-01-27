# %%
# TODO: This makes sure the weights are initialized the same, but results after training still not reproducible
seed_value = 1
import os
os.environ['PYTHONHASHSEED']=str(seed_value)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'  # new flag present in tf 2.0+
import random
random.seed(seed_value)
import numpy as np 
np.random.seed(seed_value)
import tensorflow as tf 
tf.random.set_seed(seed_value)   

import DLCVDatasets
import models
import train_center_loss
import utils
import matplotlib.pyplot as plt

# %% Load and preprocess data
dataset_name = 'mnist'    # mnist or cifar10
train_size = 60000
test_size = 10000
used_labels = list(range(0, 9))    # the labels to be loaded
num_classes = len(used_labels)
x_train, y_train, x_test, y_test, class_names = DLCVDatasets.get_dataset(dataset_name,
                                                                         used_labels=used_labels,
                                                                         training_size=train_size,
                                                                         test_size=test_size)
# Normalization
x_train, x_test = x_train / 255.0, x_test / 255.0

# Reshape to add the channel dimension
x_train = np.reshape(x_train, x_train.shape+(1,))
x_test = np.reshape(x_test, x_test.shape+(1,))
input_shape = x_train.shape[1:]

# %% Load the model
model = tf.keras.models.load_model('log/centerloss/model_v3.h5')

# %% Get the model
encoding_dim = 2
normalized_encodings = False
model = models.get_model_v4(input_shape, encoding_dim, normalized_encodings)
model.summary()

# %% Train the model with center loss
use_last_bias = False
num_epochs = 20
batch_size = 128
learning_rate = 0.001
alpha = 0.5
ratio = 1
train_center_loss.train_model_with_centerloss(model, x_train, y_train,
                                              x_test, y_test, num_classes, encoding_dim, use_last_bias,
                                              num_epochs, batch_size,
                                              learning_rate, alpha, ratio)
model.save('log/centerloss/model_lenet++_no_9.h5')

# %% Evaluate the model
# Load the complete dataset, including 0 - 9
used_labels = list(range(0, 10))    # the labels to be loaded
x_train, y_train, x_test, y_test, class_names = DLCVDatasets.get_dataset(dataset_name,
                                                                         used_labels=used_labels,
                                                                         training_size=train_size,
                                                                         test_size=test_size)
# Reshape to add the channel dimension
x_train = np.reshape(x_train, x_train.shape+(1,))
x_test = np.reshape(x_test, x_test.shape+(1,))

# Sort test samples in ascending order (from 0 to 9)
sorted_idc = np.argsort(y_test, kind='stable')
x_test = x_test[sorted_idc]
y_test = y_test[sorted_idc]

# Normalized sorted test data for prediction
x_test_normalized = x_test / 255.0

# Store the indices for each digit in dict
digits_idc = {}
for idx in range(10):
    digits_idc[str(idx)] = np.argwhere(y_test == idx)

# Compute encodings and pairwise euclidean distances
encodings = model.predict(x_test_normalized)
normalized_encodings = utils.l2_normalize(encodings)
pairwise_dists = utils.cal_pairwise_dists(normalized_encodings)

# %%
img2_0_idx = digits_idc['2'][0]
img2_1_idx = digits_idc['2'][20]
img5_0_idx = digits_idc['5'][0]
img5_1_idx = digits_idc['5'][20]
img6_0_idx = digits_idc['6'][0]
img6_1_idx = digits_idc['6'][20]
img9_0_idx = digits_idc['9'][0]
img9_1_idx = digits_idc['9'][20]

encoding_2_0 = normalized_encodings[img2_0_idx]
encoding_2_1 = normalized_encodings[img2_1_idx]

encoding_5_0 = normalized_encodings[img5_0_idx]
encoding_5_1 = normalized_encodings[img5_1_idx]

encoding_6_0 = normalized_encodings[img6_0_idx]
encoding_6_1 = normalized_encodings[img6_1_idx]

encoding_9_0 = normalized_encodings[img9_0_idx]
encoding_9_1 = normalized_encodings[img9_1_idx]

# Visualization
plt.figure(1)
pixels = np.array(x_test[img2_0_idx], dtype='uint8')
pixels = pixels.reshape((28, 28))
plt.subplot(421)
plt.imshow(pixels, cmap='gray')

pixels = np.array(x_test[img2_1_idx], dtype='uint8')
pixels = pixels.reshape((28, 28))
plt.subplot(422)
plt.imshow(pixels, cmap='gray')

pixels = np.array(x_test[img5_0_idx], dtype='uint8')
pixels = pixels.reshape((28, 28))
plt.subplot(423)
plt.imshow(pixels, cmap='gray')

pixels = np.array(x_test[img5_1_idx], dtype='uint8')
pixels = pixels.reshape((28, 28))
plt.subplot(424)
plt.imshow(pixels, cmap='gray')

pixels = np.array(x_test[img6_0_idx], dtype='uint8')
pixels = pixels.reshape((28, 28))
plt.subplot(425)
plt.imshow(pixels, cmap='gray')

pixels = np.array(x_test[img6_1_idx], dtype='uint8')
pixels = pixels.reshape((28, 28))
plt.subplot(426)
plt.imshow(pixels, cmap='gray')

pixels = np.array(x_test[img9_0_idx], dtype='uint8')
pixels = pixels.reshape((28, 28))
plt.subplot(427)
plt.imshow(pixels, cmap='gray')

pixels = np.array(x_test[img9_1_idx], dtype='uint8')
pixels = pixels.reshape((28, 28))
plt.subplot(428)
plt.imshow(pixels, cmap='gray')
plt.show()

print('2 & 2: {}'.format(tf.norm(encoding_2_0 - encoding_2_1).numpy()))
print('2 & 5: {}'.format(tf.norm(encoding_2_0 - encoding_5_0).numpy()))
print('2 & 6: {}'.format(tf.norm(encoding_2_0 - encoding_6_0).numpy()))

print('5 & 5: {}'.format(tf.norm(encoding_5_0 - encoding_5_1).numpy()))
print('6 & 6: {}'.format(tf.norm(encoding_6_0 - encoding_6_1).numpy()))
print('5 & 6: {}'.format(tf.norm(encoding_5_0 - encoding_6_0).numpy()))

print('9 & 9: {}'.format(tf.norm(encoding_9_0 - encoding_9_1).numpy()))
print('5 & 9: {}'.format(tf.norm(encoding_5_0 - encoding_9_0).numpy()))
print('6 & 9: {}'.format(tf.norm(encoding_6_0 - encoding_9_0).numpy()))

# %% Intraclass test
test_num = 2
anchor_idx = 0
x = x_test[y_test == test_num]
encoding_anchor = tf.math.l2_normalize(model(x[[anchor_idx],]))
for idx in range(0, 100):
    encoding = tf.math.l2_normalize(model(x[[idx],]))
    print('Intraclass: No.{}, id{} & id{}: {}'.format(test_num, anchor_idx, idx, tf.norm(encoding - encoding_anchor).numpy()))

# %% Scatter plot 2D encodings
f = plt.figure(figsize=(8, 8))
c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff', 
    '#ff00ff', '#990000', '#999900', '#009900', '#009999']
for i in range(10):
    plt.scatter(encodings[y_test==i, 0], encodings[y_test==i, 1], alpha=0.5, s=2)
plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
plt.grid()
plt.show()

# %% Save results for embedding projector
import os
import csv

used_labels = list(range(0, 10))    # the labels to be loaded
x_train, y_train, x_test, y_test, class_names = DLCVDatasets.get_dataset(dataset_name,
                                                                         used_labels=used_labels,
                                                                         training_size=train_size,
                                                                         test_size=test_size)

# Normalize data
x_train, x_test = x_train / 255.0, x_test / 255.0

# Reshape to add the channel dimension
x_train = np.reshape(x_train, x_train.shape+(1,))
x_test = np.reshape(x_test, x_test.shape+(1,))

# Compute encodings and pairwise euclidean distances
encodings = model.predict(x_test)
normalized_encodings = utils.l2_normalize(encodings)

dirname = 'log/centerloss'
if not os.path.exists(dirname):
    os.makedirs(dirname)
with open(dirname+'/metadata.tsv', 'w') as metadata_file:
    metadata_file.write('Index\tLabel\n')
    for idx, row in enumerate(y_test):
        metadata_file.write('%d\t%d\n' % (idx, row))

with open(dirname+'/feature_vecs.tsv', 'w') as fw:
    csv_writer = csv.writer(fw, delimiter='\t')
    for vec in normalized_encodings:
        csv_writer.writerow(vec)

# %% Performance evaluation

utils.threshold_evaluation(pairwise_dists, y_test, 0.1, 1.2, 12, i_want_to_plot = True)

treshold = 0.75
(recall, FAR, precision) = utils.performance_test(pairwise_dists, y_test, treshold)
accuracy_table = utils.get_accuracy_table(pairwise_dists, y_test, treshold )
