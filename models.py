import tensorflow as tf 

# The base LeNet-alike model. To use center loss, an additional layer has to be used.
def get_model_v1(input_shape, encoding_dim=64, normalize_encoding=True):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPool2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(encoding_dim, activation='linear')
    ])
    if normalize_encoding:
        model.add(tf.keras.layers.Lambda(lambda x: tf.keras.backend.l2_normalize(x, axis=1), name='norm_encoding'))
    
    return model

def get_model_v2(input_shape, encoding_dim=64, normalize_encoding=True):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPool2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(encoding_dim, activation='linear'),
    ])

    if normalize_encoding:
        model.add(tf.keras.layers.Lambda(lambda x: tf.keras.backend.l2_normalize(x, axis=1), name='norm_encoding'))
    return model

    # The base LeNet-alike model. To use center loss, an additional layer has to be used.
def get_model_v3(input_shape, encoding_dim=64, normalize_encoding=True):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (5, 5), padding='same', input_shape=input_shape),
        tf.keras.layers.PReLU(),
        tf.keras.layers.Conv2D(32, (5, 5), padding='same', input_shape=input_shape),
        tf.keras.layers.PReLU(),
        tf.keras.layers.MaxPool2D((2, 2)),
        tf.keras.layers.Conv2D(64, (5, 5), padding='same'),
        tf.keras.layers.PReLU(),
        tf.keras.layers.Conv2D(64, (5, 5), padding='same'),
        tf.keras.layers.PReLU(),
        tf.keras.layers.MaxPool2D((2, 2)),
        tf.keras.layers.Conv2D(128, (5, 5), padding='same'),
        tf.keras.layers.PReLU(),
        tf.keras.layers.Conv2D(128, (5, 5), padding='same'),
        tf.keras.layers.PReLU(),
        tf.keras.layers.MaxPool2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(encoding_dim, activation='linear')
    ])
    if normalize_encoding:
        model.add(tf.keras.layers.Lambda(lambda x: tf.keras.backend.l2_normalize(x, axis=1), name='norm_encoding'))
    
    return model
