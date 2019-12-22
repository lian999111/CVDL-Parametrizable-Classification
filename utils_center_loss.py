import tensorflow as tf 
import numpy as np 

# The signature is not compatible with keras fit(), thus can only be used with custimized training
class CenterLoss:
    def __init__(self, alpha, num_classes, len_encoding):
        self.alpha = alpha
        self.num_classes = num_classes
        self.len_encoding = len_encoding
        self.centers = tf.Variable(np.zeros((num_classes, len_encoding), dtype=np.float32))
    
    @tf.function
    def __call__(self, labels, encodings):
        labels = tf.reshape(labels, (-1,))   # to 1-D

        # Get the centers of each sample in this batch
        centers_batch = tf.gather(self.centers, labels)

        # Compute loss
        normalized_encodings = tf.math.l2_normalize(encodings)
        delta = tf.subtract(centers_batch, normalized_encodings)    # difference between encodings and centers
        loss = tf.nn.l2_loss(delta)

        # Update centers
        unique_labels, unique_idc, unique_counts = tf.unique_with_counts(labels)
        appear_times = tf.gather(unique_counts, unique_idc)
        appear_times = tf.reshape(appear_times, (-1, 1))
        delta = delta / tf.cast((1 + appear_times), tf.float32)
        delta = tf.scalar_mul(self.alpha, delta)
        labels = tf.expand_dims(labels, -1) # to match dim of self.centers
        self.centers.assign(tf.tensor_scatter_nd_sub(self.centers, labels, delta))

        return loss, tf.identity(self.centers)

@tf.function
def train_one_step_centerloss(model, additional_layer,
                x_batch_train, y_batch_train, 
                scce_fn, center_loss_fn, ratio, optimizer, metric):
    with tf.GradientTape(persistent=False) as tape:
        encodings = model(x_batch_train)
        logits = additional_layer(encodings)
        softmax_loss = scce_fn(y_batch_train, logits)
        center_loss, centers = center_loss_fn(y_batch_train, encodings)
        total_loss = softmax_loss + tf.math.scalar_mul(ratio, center_loss)
        metric.update_state(y_batch_train, tf.argmax(logits, 1))

    grads = tape.gradient(total_loss, [additional_layer.trainable_variables, model.trainable_variables])
    optimizer.apply_gradients(zip(grads[0], additional_layer.trainable_variables))
    optimizer.apply_gradients(zip(grads[1], model.trainable_variables))

    return softmax_loss, center_loss, total_loss, centers

def train_model_with_centerloss(model, train_data, train_labels,
                test_data, test_labels, num_classes, len_encoding,
                num_epochs=20, batch_size=128,
                learning_rate=0.001, ratio=0.1):

    # Generate tf.data.Dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

    # Additional layer for softmax loss (cross-entropy loss)
    additional_layer = tf.keras.layers.Dense(num_classes)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    metric = tf.metrics.Accuracy()

    # Get loss function objects
    scce_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    center_loss_fn = CenterLoss(0.2, num_classes, len_encoding)

    # Placeholder for total loss over an epoch
    overall_total_loss = tf.Variable(0, dtype=tf.float32)
    
    # Train network
    for epoch in range(num_epochs):
        # Iterate over minibatches
        metric.reset_states()
        overall_total_loss.assign(0.0)
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            softmax_loss, center_loss, total_loss, centers = train_one_step_centerloss(model, additional_layer, 
                                                                                       x_batch_train, y_batch_train, 
                                                                                       scce_fn, center_loss_fn, ratio, 
                                                                                       optimizer, metric)
        
            overall_total_loss.assign_add(softmax_loss + center_loss)
        train_accuracy = metric.result()
        print('Epoch: {}: Train Loss: {}, Train Accuracy: {}'.format(epoch, overall_total_loss.numpy(), train_accuracy))

        # Model evaluation on test set for this epoch
        metric.reset_states()
        test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_labels))
        test_dataset = test_dataset.shuffle(buffer_size=1024).batch(batch_size)
        for x_batch_test, y_batch_test in test_dataset:
            encodings_test = model(x_batch_test)
            logits = additional_layer(encodings_test)
            metric.update_state(y_batch_test, tf.argmax(logits, 1))
        test_accuracy = metric.result()
        print('Epoch: {}: Test Accuracy: {}'.format(epoch, test_accuracy))