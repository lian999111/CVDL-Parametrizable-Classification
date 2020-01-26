"""Define functions to train a model using triplet loss."""

import tensorflow as tf
import triplet_loss

@tf.function
def train_one_step_tripletloss(model, x_batch_train, y_batch_train, 
                triplet_loss_strategy, margin, optimizer):
    with tf.GradientTape(persistent=False) as tape:
        embeddings = model(x_batch_train)
        if(triplet_loss_strategy=="batch_hard"):
            triplet_loss_value = triplet_loss.batch_hard_triplet_loss(y_batch_train, embeddings, margin)
        elif(triplet_loss_strategy=="batch_all"):
            triplet_loss_value = triplet_loss.batch_all_triplet_loss(y_batch_train, embeddings, margin)

    grads = tape.gradient(triplet_loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return triplet_loss_value

def train_model_with_tripletloss(model, train_data, train_labels,
                test_data, test_labels, num_classes, len_encoding,
                num_epochs=20, batch_size=128, learning_rate=0.001,
                margin=0.5,  triplet_loss_strategy="batch_hard"):

    # Generate tf.data.Dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
    train_dataset = train_dataset.batch(batch_size)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Placeholder for total loss over an epoch
    overall_triplet_loss = tf.Variable(0, dtype=tf.float32)
    
    # Train network
    for epoch in range(num_epochs):
        # Iterate over minibatches
        overall_triplet_loss.assign(0.0)
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            triplet_loss_value = train_one_step_tripletloss(model, x_batch_train, y_batch_train, 
                                                          triplet_loss_strategy, margin, optimizer)
        
            overall_triplet_loss.assign_add(triplet_loss_value)
            #if(epoch==0):
            #    print('Step: {} Loss: {}'.format(step, triplet_loss_value))
        print('Epoch: {}: Average Train Loss: {} Last Batch Loss {}'.format(epoch, overall_triplet_loss.numpy()/batch_size, triplet_loss_value))

        # Model evaluation on test set for this epoch
        test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_labels))
        test_dataset = test_dataset.batch(batch_size)
        overall_triplet_loss.assign(0.0)
        for x_batch_test, y_batch_test in test_dataset:
            embeddings = model(x_batch_test)
            if(triplet_loss_strategy=="batch_hard"):
                triplet_loss_value = triplet_loss.batch_hard_triplet_loss(y_batch_test, embeddings, margin)
            elif (triplet_loss_strategy=="batch_all"):
                triplet_loss_value = triplet_loss.batch_all_triplet_loss(y_batch_test, embeddings, margin)
            overall_triplet_loss.assign_add(triplet_loss_value)
        print('Epoch: {}:  Average Test Loss: {} Last Batch Loss {}'.format(epoch, overall_triplet_loss.numpy()/batch_size, triplet_loss_value))