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
from utils import cal_pairwise_dists, l2_normalize
import matplotlib.pyplot as plt
import train_triplet_loss
import os
import csv

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
#mean = np.mean(x_train)
#x_train -= mean
#x_test -= mean
x_train, x_test = x_train / 255.0, x_test / 255.0

# Reshape to add the channel dimension
x_train = np.reshape(x_train, x_train.shape+(1,))
x_test = np.reshape(x_test, x_test.shape+(1,))
input_shape = x_train.shape[1:]

# %% Get the model
encoding_dim = 64
normalized_encodings = True
model = models.get_model_v1(input_shape, encoding_dim, normalized_encodings)
model.summary()

# %% Train the model with triplet lossnum_epochs = 20
num_epochs=5
batch_size = 128
learning_rate = 0.0001
margin=0.5
triplet_loss_strategy="batch_hard"
train_triplet_loss.train_model_with_tripletloss(model, x_train, y_train,
                                              x_test, y_test, num_classes, encoding_dim,
                                              num_epochs, batch_size, learning_rate,
                                              margin,  triplet_loss_strategy)
#model.save('model_trained_TL_lenet_1.h5')

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
normalized_encodings = l2_normalize(encodings)
encoding_new = encodings.flatten(order='C')
# plot3D(encoding_new,y_test)
pairwise_dists = cal_pairwise_dists(normalized_encodings)

#  Save results for embedding projector

dirname = 'log/tripleLoss64'
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

# %% Performance test
threshold = 0.7

### Overall performance (temporal, can be calculated from table) ###
# Check if labels[i] == labels[j]
# Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
really_equal = np.equal(np.expand_dims(y_test, 0), np.expand_dims(y_test, 1))
predicted_equal = pairwise_dists < threshold
overall_accuracy = (np.sum(predicted_equal==really_equal)-len(y_test)) / (predicted_equal.size-len(y_test))

### Accuracy per class
accuracy_per_class = np.zeros((10, 10))
start_row = 0
start_column_next = 0

for row_class in range(10):
    row_span = len(np.argwhere(y_test == row_class))
    start_column = start_column_next
    start_column_next = start_column_next + row_span

    for column_class in range(row_class,10):
        column_span = len(np.argwhere(y_test == column_class))

        if(row_class == column_class):
            expected_value = 1
        else:
            expected_value = 0
        
        predicted_equal = pairwise_dists[start_row:start_row+row_span-1 , start_column:start_column+column_span-1] < threshold
        
        if(row_class == column_class):
            accuracy = (np.sum(predicted_equal==expected_value)-row_span) / (predicted_equal.size-row_span)
        else:
            accuracy = (np.sum(predicted_equal==expected_value)) / predicted_equal.size
        
        accuracy_per_class[row_class,column_class] = accuracy
        start_column = start_column + column_span
    
    start_row = start_row + row_span 


# %%
