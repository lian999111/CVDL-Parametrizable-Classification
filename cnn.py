# %%
from tensorflow.keras.optimizers import Adam
import numpy as np
import tensorflow as tf
import tensorflow.keras as k
import DLCVDatasets

# %% Prepare CIFAR-10 dataset
dataset_name = 'mnist'    # mnist or cifar10
train_size = 60000
test_size = 10000
used_labels = list(range(0, 10))    # the labels to be loaded
x_train, y_train, x_test, y_test, class_names = DLCVDatasets.get_dataset(dataset_name,
                                                                         used_labels=used_labels,
                                                                         training_size=train_size,
                                                                         test_size=test_size)
# Normalization
if dataset_name == 'mnist':
    x_train = np.reshape(x_train, x_train.shape+(1,))
    x_test = np.reshape(x_test, x_test.shape+(1,))
x_train, x_test = x_train / 255.0, x_test / 255.0
input_shape = x_train.shape[1:4]

# %% Define cnn model
cnn_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=input_shape),
    tf.keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
    tf.keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu', name='encoding'),
    tf.keras.layers.Dense(10)
])

cnn_model.summary()

# %% Callback for early stopping


class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') > 0.99):
            print("\nReached 99% accuracy so cancelling training!")
            self.model.stop_training = True


# %% Compile cnn_model
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
cnn_model.compile(optimizer=Adam(lr=0.01),
                  loss=loss,
                  metrics=['accuracy'])


# %% Train model
cnn_model.fit(x_train, y_train, validation_data=(x_test, y_test),
              epochs=15, batch_size=64, verbose=1, callbacks=[MyCallback()])
