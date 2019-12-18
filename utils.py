# %% 
import tensorflow as tf 
import numpy as np 

def triplet_loss(y_true, y_pred, alpha = 0.2):
    """
    Implementation of the triplet loss as defined by formula (3)
    
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)
    
    Returns:
    loss -- real number, value of the loss
    """
    
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    
    # Step 1: Compute the (encoding) distance between the anchor and the positive, you will need to sum over axis=-1
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis = -1)
    # Step 2: Compute the (encoding) distance between the anchor and the negative, you will need to sum over axis=-1
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis = -1)
    # Step 3: subtract the two previous distances and add alpha.
    basic_loss = pos_dist - neg_dist + alpha
    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0))
    
    return loss

# The signature is not compatible with keras fit(), thus can only be used with custimized training
class CenterLoss:
    def __init__(self, alpha, num_classes, len_encoding):
        self.alpha = alpha
        self.num_classes = num_classes
        self.len_encoding = len_encoding
        self.centers = tf.zeros(shape=(num_classes, len_encoding))
    
    # @tf.function
    def __call__(self, labels, encodings):
        labels = tf.reshape(labels, (-1,))   # to 1-D

        # Get the centers of each sample in this batch
        centers_batch = tf.gather(self.centers, labels)

        # Compute loss
        normalized_encodings = tf.math.l2_normalize(encodings)
        loss = tf.nn.l2_loss(encodings - centers_batch)

        # The difference between the encodings and their corresponding centers
        delta = tf.subtract(centers_batch, encodings)

        # Update centers
        unique_labels, unique_idc, unique_counts = tf.unique_with_counts(labels)
        appear_times = tf.gather(unique_counts, unique_idc)
        appear_times = tf.reshape(appear_times, (-1, 1))
        delta = delta / tf.cast((1 + appear_times), tf.float32)
        delta = tf.scalar_mul(self.alpha, delta)

        labels = tf.expand_dims(labels, -1)
        self.centers = tf.tensor_scatter_nd_sub(self.centers, labels, delta)

        return loss, tf.identity(self.centers)

total_loss = None
overall_total_loss = None
additional_layer = None
# %% 
# @tf.function
def train_model_with_centerloss(model, train_data, train_labels,
                test_data, test_labels, num_classes, len_encoding,
                num_epochs= 20, batch_size = 128,
                learning_rate=0.001, ratio = 0.1):

    # Generate tf.data.Dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    metric = tf.metrics.Accuracy()

    scce_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    center_loss_fn = CenterLoss(0.01, num_classes, len_encoding)

    # global total_loss
    # if total_loss is None:
    #     total_loss = tf.Variable(0, dtype=tf.float32)
    # global overall_total_loss
    # if overall_total_loss is None:
    #     overall_total_loss = tf.Variable(0, dtype=tf.float32)

    # global additional_layer
    # if additional_layer is None:
    #     additional_layer = tf.keras.layers.Dense(num_classes)

    total_loss = tf.Variable(0, dtype=tf.float32)
    overall_total_loss = tf.Variable(0, dtype=tf.float32)
    additional_layer = tf.keras.layers.Dense(num_classes)

    # Train network
    for epoch in range(num_epochs):
        # Iterate over minibatches
        metric.reset_states()
        overall_total_loss.assign(0.0)
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape(persistent=False) as tape:
                encodings = model(x_batch_train)
                logits = additional_layer(encodings)
                softmax_loss = scce_fn(y_batch_train, logits)
                center_loss, centers = center_loss_fn(y_batch_train, encodings)
                total_loss = softmax_loss + tf.math.scalar_mul(ratio, center_loss)
                overall_total_loss.assign_add(total_loss)
                metric.update_state(y_batch_train, tf.argmax(logits, 1))

            grads = tape.gradient(total_loss, [additional_layer.trainable_variables, model.trainable_variables])
            optimizer.apply_gradients(zip(grads[0], additional_layer.trainable_variables))
            optimizer.apply_gradients(zip(grads[1], model.trainable_variables))
        
        train_accuracy = metric.result()
        tf.print('Epoch: {}: Train Loss: {}, Train Accuracy: {}'.format(epoch, total_loss, train_accuracy))

        metric.reset_states()
        test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_labels))
        test_dataset = test_dataset.shuffle(buffer_size=1024).batch(batch_size)
        for x_batch_test, y_batch_test in test_dataset:
            encodings_test = model(x_batch_test)
            logits = additional_layer(encodings_test)
            metric.update_state(y_batch_test, tf.argmax(logits, 1))
        test_accuracy = metric.result()
        tf.print('Epoch: {}: Test Accuracy: {}'.format(epoch, test_accuracy))