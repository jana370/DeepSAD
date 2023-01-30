import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import Preprocessing
from sklearn import metrics


def convolutional_module(input, filters, kernel_size, strides=1, padding="same"): # strides and padding could currently be removed
    x = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=False)(input)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.MaxPooling2D((2, 2))(x)
    return x


def deconvolutional_module(input, filters, kernel_size, strides=1, padding="same", final_layer=False): # strides and padding could currently be removed
    x = layers.LeakyReLU(alpha=0.1)(input)
    x = layers.UpSampling2D((2, 2))(x)
    if not final_layer:
        x = layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                   use_bias=False)(x)
        x = layers.BatchNormalization()(x)
    else:
        x = layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding, use_bias=False,
                                   activation="sigmoid")(x)
    return x


def create_neural_network():
    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = convolutional_module(inputs, 8, 5)
    x = convolutional_module(x, 4, 5)
    x = layers.Flatten()(x)
    x = layers.Dense(49, use_bias=False)(x)
    x = layers.LeakyReLU(0.1)(x)
    outputs_encoder = layers.Dense(32, use_bias=False)(x)
    x = layers.LeakyReLU(0.1)(outputs_encoder)
    x = layers.Dense(49, use_bias=False)(x)
    x = layers.LeakyReLU(0.1)(x)
    x = tf.reshape(x, [-1, 7, 7, 1])
    x = deconvolutional_module(x, 4, 5)
    outputs = deconvolutional_module(x, 1, 5, final_layer=True)

    encoder = tf.keras.Model(inputs=inputs, outputs=outputs_encoder)
    autoencoder = tf.keras.Model(inputs=inputs, outputs=outputs)
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.MeanSquaredError())
    return encoder, autoencoder


@tf.function
def train_step_encoder(data, center):
    with tf.GradientTape() as tape:
        predictions = encoder(data, training=True)  # training definition necessary
        difference = tf.norm(tf.subtract(predictions, center), axis=1) ** 2
        loss = tf.reduce_sum(difference) / data.shape[0]
    gradients = tape.gradient(loss, encoder.trainable_variables)
    optimizer.apply_gradients(zip(gradients, encoder.trainable_variables))


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

train_filter = np.where(y_train == 1)
x_train, y_train = x_train[train_filter], y_train[train_filter]

x_train = x_train / 255.
x_test = x_test / 255.

encoder, autoencoder = create_neural_network()
autoencoder.fit(x_train, x_train, batch_size=128, epochs=100, shuffle=True)

center = encoder.predict(x_train)
center = np.mean(center, axis=0)

x_train = tf.data.Dataset.from_tensor_slices(x_train).shuffle(60000).batch(128)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

for i in range(100):
    for datapoints in x_train:
        train_step_encoder(datapoints, center)
    print(f'Epoch {i + 1}')

predictions = encoder.predict(x_test)
anomaly_score = tf.norm(tf.subtract(predictions, center), axis=1)

y_test = np.where(y_test == 1, 0, 1)
auc = metrics.roc_auc_score(y_test, anomaly_score)
print(auc)