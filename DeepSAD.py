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

def deconvolutional_module(input, filters, kernel_size, strides=1, padding="same", final_layer=False):
    x = layers.LeakyReLU(alpha=0.1)(input)
    x = layers.UpSampling2D((2, 2))(x)
    if not final_layer:
        x = layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                   use_bias=False)(x)
        x = layers.BatchNormalization()(x)
    else:
        x = layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                   use_bias=False, activation="sigmoid")(x)
    return x


def create_neural_network():
    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = convolutional_module(inputs, 8, 5)
    x = convolutional_module(x, 4, 5)
    x = layers.Flatten()(x)
    x = layers.Dense(49, use_bias=False)(x)
    outputs_encoder = layers.Dense(32, use_bias=False)(x)
    x = layers.Dense(49, use_bias=False)(outputs_encoder)
    x = layers.LeakyReLU(0.1)(x)
    x = tf.reshape(x, [-1, 7, 7, 1]) #komisch? nachprüfen ob [-1, 2, 4, 4] mehr Sinn macht
    x = deconvolutional_module(x, 4, 5)
    outputs = deconvolutional_module(x, 1, 5, final_layer=True)
    encoder = tf.keras.Model(inputs=inputs, outputs=outputs_encoder)
    autoencoder = tf.keras.Model(inputs=inputs, outputs=outputs)
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.MeanSquaredError())
    return encoder, autoencoder

#@tf.function
def train_step(data, center):
    with tf.GradientTape() as tape:
        predictions = encoder(data, training=True)  # training definition necessary?
        squared_difference = tf.reduce_sum(tf.subtract(predictions, center) ** 2, axis=1)
        print(squared_difference.shape)
        loss = tf.reduce_sum(squared_difference) / data.shape[0]
    gradients = tape.gradient(loss, encoder.trainable_variables)
    optimizer.apply_gradient(zip(gradients, encoder.trainable_variables))


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train / 255.
x_test = x_test / 255.
print(max(y_train[0:500]))
#print(x_train[0].shape)
#x_train = np.expand_dims(x_train, -1)
#print(x_train[0].shape)
encoder, autoencoder = create_neural_network()

autoencoder.fit(x_train, x_train, batch_size=128, epochs=1, shuffle=True, validation_data=(x_test, x_test))

center = encoder.predict(x_train)


center = np.mean(center, axis=0)
print(center)
train_step(x_train, center)

#a = autoencoder.predict(np.expand_dims(x_train[0], 0))
#a = np.squeeze(a)



#plt.gray()
#plt.imshow(a)
#plt.show()