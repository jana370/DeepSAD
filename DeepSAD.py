import keras.optimizers.optimizer_v1
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from matplotlib import pyplot as plt

#plt.gray()
#plt.imshow(image)
#plt.show()

def convolutional_module(input, filters, kernel_size, strides=1, padding="same"):
    x = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=False)(input)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.MaxPooling2D((2, 2))(x)
    return x

def deconvolutional_module(input, filters, kernel_size, strides=1, padding="same", layer=0):
    x = layers.LeakyReLU(alpha=0.1)(input)
    x = layers.UpSampling2D((2, 2))(x)
    if layer == 0:
        x = layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                   use_bias=False)(x)
        x = layers.BatchNormalization()(x)
    elif layer == 1:
        print(x.shape)
        #x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])
        x = tf.keras.layers.ZeroPadding2D(padding=1)(x)
        print(x.shape)
        x = layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding="same", output_padding=1,
                                   use_bias=False)(x)
        print(x.shape)
        x = layers.BatchNormalization()(x)
    else:
        x = layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                   use_bias=False, activation="softmax")(x)
    return x


def create_neural_network():
    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = convolutional_module(inputs, 8, 5)
    x = convolutional_module(x, 4, 5)
    x = layers.Flatten()(x)
    outputs_encoder = layers.Dense(32, use_bias=False)(x)
    x = layers.LeakyReLU(0.1)(outputs_encoder)
    x = tf.reshape(x, [-1, 4, 4, 2]) #komisch? nachprüfen ob [-1, 2, 4, 4] mehr Sinn macht
    x = deconvolutional_module(x, 4, 5)
    print(x.shape)
    x = deconvolutional_module(x, 8, 5, layer=1)
    print(x.shape)
    outputs = deconvolutional_module(x, 1, 5, layer=2)
    encoder = tf.keras.Model(inputs=inputs, outputs=outputs_encoder)
    autoencoder = tf.keras.Model(inputs=inputs, outputs=outputs)
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.MeanSquaredError())
    return encoder, autoencoder

def create_neural_network2():
    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = convolutional_module(inputs, 8, 5)
    x = convolutional_module(x, 4, 5)
    x = layers.Flatten()(x)
    outputs_encoder = layers.Dense(49, use_bias=False)(x)
    x = layers.LeakyReLU(0.1)(outputs_encoder)
    x = tf.reshape(x, [-1, 7, 7, 1]) #komisch? nachprüfen ob [-1, 2, 4, 4] mehr Sinn macht
    x = deconvolutional_module(x, 4, 5)
    outputs = deconvolutional_module(x, 1, 5, layer=2)
    encoder = tf.keras.Model(inputs=inputs, outputs=outputs_encoder)
    autoencoder = tf.keras.Model(inputs=inputs, outputs=outputs)
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.MeanSquaredError())
    return encoder, autoencoder


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train / 255.
x_test = x_test / 255.
#x_train = np.expand_dims(x_train, -1)
encoder, autoencoder = create_neural_network2()

autoencoder.fit(x_train, x_train, batch_size=128, epochs=50, shuffle=True, validation_data=(x_test, x_test))

a = autoencoder.predict(x_train[0])



plt.gray()
plt.imshow(a)
plt.show()