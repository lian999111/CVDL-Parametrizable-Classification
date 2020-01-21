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
import lenet_model
import utils_center_loss
from utils import cal_pairwise_dists, l2_normalize
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
mean = np.mean(x_train)
x_train -= mean
x_test -= mean
x_train, x_test = x_train / 255.0, x_test / 255.0

# Reshape to add the channel dimension
x_train = np.reshape(x_train, x_train.shape+(1,))
x_test = np.reshape(x_test, x_test.shape+(1,))
input_shape = x_train.shape[1:]

# %% Get the model
encoding_dim = 16
lenet = lenet_model.get_lenet_model(input_shape=input_shape, encoding_dim=encoding_dim)
lenet.summary()

# %% Train the model with center loss
utils_center_loss.train_model_with_centerloss(lenet, x_train, y_train,
                                              x_test, y_test, num_classes=10, len_encoding=encoding_dim,
                                              num_epochs=2, batch_size=64,
                                              learning_rate=0.005, alpha=0.1, ratio=0.2)

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

# Store the indices for each digit in dict
digits_idc = {}
for idx in range(10):
    digits_idc[str(idx)] = np.argwhere(y_test == idx)

# Compute encodings and pairwise euclidean distances
encodings = lenet.predict(x_test)
normalized_encodings = l2_normalize(encodings)
pairwise_dists = cal_pairwise_dists(normalized_encodings)

# %%
img2_0_idx = digits_idc['2'][0]
img2_1_idx = digits_idc['2'][20]
img5_0_idx = digits_idc['5'][0]
img5_1_idx = digits_idc['5'][20]
img6_0_idx = digits_idc['6'][0]
img6_1_idx = digits_idc['6'][20]
img9_0_idx = digits_idc['9'][0]
img9_1_idx = digits_idc['9'][20]

encoding_2_0 = tf.math.l2_normalize(lenet(x_test[img2_0_idx]))
encoding_2_1 = tf.math.l2_normalize(lenet(x_test[img2_1_idx]))

x_5 = x_test[y_test == 5]
encoding_5_0 = tf.math.l2_normalize(lenet(x_test[img5_0_idx]))
encoding_5_1 = tf.math.l2_normalize(lenet(x_test[img5_1_idx]))

x_6 = x_test[y_test == 6]
encoding_6_0 = tf.math.l2_normalize(lenet(x_test[img6_0_idx]))
encoding_6_1 = tf.math.l2_normalize(lenet(x_test[img6_1_idx]))


x_9 = x_test[y_test == 9]
encoding_9_0 = tf.math.l2_normalize(lenet(x_test[img9_0_idx]))
encoding_9_1 = tf.math.l2_normalize(lenet(x_test[img9_1_idx]))

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
test_num = 0
anchor_idx = 0
x = x_test[y_test == test_num]
encoding_anchor = tf.math.l2_normalize(lenet(x[[anchor_idx],]))
for idx in range(0, 100):
    encoding = tf.math.l2_normalize(lenet(x[[idx],]))
    print('Intraclass: No.{}, id{} & id{}: {}'.format(test_num, anchor_idx, idx, tf.norm(encoding - encoding_anchor).numpy()))

# %% Save results for embedding projector
# import io

# out_v = io.open('vecs5.tsv', 'w', encoding='utf-8')
# out_m = io.open('meta5.tsv', 'w', encoding='utf-8')

# for idx, img in enumerate(x_test[:100]):
#     out_m.write(str(y_test[idx]) + "\t")
#     embedding = tf.math.l2_normalize(lenet(x_test[[idx],]))
#     out_v.write('\t'.join([str(x) for x in embedding.numpy().tostring()]) + "\n")

# out_v.close()
# out_m.close()

# %%
