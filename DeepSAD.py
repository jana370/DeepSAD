import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import Preprocessing
from sklearn import metrics


def convolutional_module(input, filters, kernel_size, strides=1, padding="same"): # strides and padding could currently be removed
    x = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=False,
                      kernel_regularizer=tf.keras.regularizers.l2(1e-5))(input)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.MaxPooling2D((2, 2))(x)
    return x


def deconvolutional_module(input, filters, kernel_size, strides=1, padding="same", final_layer=False): # strides and padding could currently be removed
    x = layers.LeakyReLU(alpha=0.1)(input)
    x = layers.UpSampling2D((2, 2))(x)
    if not final_layer:
        x = layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                   use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
        x = layers.BatchNormalization()(x)
    else:
        x = layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding, use_bias=False,
                                   kernel_regularizer=tf.keras.regularizers.l2(1e-5), activation="sigmoid")(x)
    return x


def create_neural_network():
    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = convolutional_module(inputs, 8, 5)
    x = convolutional_module(x, 4, 5)
    x = layers.Flatten()(x)
    x = layers.Dense(49, use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
    x = layers.LeakyReLU(0.1)(x)
    outputs_encoder = layers.Dense(32, use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
    x = layers.LeakyReLU(0.1)(outputs_encoder)
    x = layers.Dense(49, use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
    x = layers.LeakyReLU(0.1)(x)
    x = tf.reshape(x, [-1, 7, 7, 1])
    x = deconvolutional_module(x, 4, 5)
    outputs = deconvolutional_module(x, 1, 5, final_layer=True)

    encoder = tf.keras.Model(inputs=inputs, outputs=outputs_encoder)
    autoencoder = tf.keras.Model(inputs=inputs, outputs=outputs)
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.MeanSquaredError())
    return encoder, autoencoder

def create_neural_network_cifar10():
    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = convolutional_module(inputs, 32, 5)
    x = convolutional_module(x, 64, 5)
    x = convolutional_module(x, 128, 5)
    x = layers.Flatten()(x)
    x = layers.Dense(64, use_bias=False, kernel_regularizer="l2")(x)
    x = layers.LeakyReLU(0.1)(x)
    outputs_encoder = layers.Dense(128, use_bias=False, kernel_regularizer="l2")(x)
    x = layers.LeakyReLU(0.1)(outputs_encoder)
    x = layers.Dense(64, use_bias=False, kernel_regularizer="l2")(x)
    x = layers.LeakyReLU(0.1)(x)
    x = tf.reshape(x, [-1, 8, 8, 1])
    x = deconvolutional_module(x, 4, 5)
    outputs = deconvolutional_module(x, 1, 5, final_layer=True)

    encoder = tf.keras.Model(inputs=inputs, outputs=outputs_encoder)
    autoencoder = tf.keras.Model(inputs=inputs, outputs=outputs)
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.MeanSquaredError())
    return encoder, autoencoder
def create_neural_network_fmnist():
    inputs = tf.keras.Input(shape=(28, 28, 1))
    print(inputs.shape)
    x = convolutional_module(inputs, 16, 5)
    x = convolutional_module(x, 32, 5)
    x = layers.Flatten()(x)
    x = layers.Dense(49, use_bias=False, kernel_regularizer="l2")(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.Dense(64, use_bias=False, kernel_regularizer="l2")(x)
    x = layers.LeakyReLU(0.1)(x)
    outputs_encoder = layers.Dense(32, use_bias=False, kernel_regularizer="l2")(x)
    x = layers.LeakyReLU(0.1)(outputs_encoder)
    x = layers.Dense(64, use_bias=False, kernel_regularizer="l2")(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.Dense(49, use_bias=False, kernel_regularizer="l2")(x)
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
        predictions = encoder(data[0], training=True)
        unlabeled_mask = tf.where(data[1] == 0)
        unlabeled_predictions = tf.gather(predictions, unlabeled_mask)
        unlabeled_differences = tf.norm(tf.subtract(unlabeled_predictions, center), axis=2) ** 2
        unlabeled_loss = tf.reduce_sum(unlabeled_differences) / data[0].shape[0]
        labeled_normal_mask = tf.where(data[1] == 1)
        labeled_normal_predictions = tf.gather(predictions, labeled_normal_mask)
        labeled_normal_differences = tf.norm(tf.subtract(labeled_normal_predictions, center), axis=2) ** 2
        labeled_normal_loss = tf.reduce_sum(labeled_normal_differences) / data[0].shape[0]
        labeled_outlier_mask = tf.where(data[1] == -1)
        labeled_outlier_predictions = tf.gather(predictions, labeled_outlier_mask)
        labeled_outlier_differences = 1 / (tf.norm(tf.subtract(labeled_outlier_predictions, center), axis=2) ** 2 + 1e-5
                                           )
        labeled_outlier_loss = tf.reduce_sum(labeled_outlier_differences) / data[0].shape[0]

        loss = unlabeled_loss + 1 * (labeled_normal_loss + labeled_outlier_loss)
    gradients = tape.gradient(loss, encoder.trainable_variables)
    optimizer.apply_gradients(zip(gradients, encoder.trainable_variables))
    train_loss(loss)

dataset = "cifar10"
Preprocessor = Preprocessing.PreProcessing(dataset, [1], [0], [2], ratio_known_outlier=0.1, ratio_known_normal=0.1)
(labeled_data, labeled_data_labels), (unlabeled_data, unlabeled_data_labels) = Preprocessor.get_train_data()
(test_data, test_data_labels) = Preprocessor.get_test_data()

if dataset == "mnist":
    encoder, autoencoder = create_neural_network()
elif dataset == "fmnist":
    encoder, autoencoder = create_neural_network_fmnist()
elif dataset == "cifar10":
    encoder, autoencoder = create_neural_network_cifar10()
autoencoder.fit(unlabeled_data, unlabeled_data, batch_size=128, epochs=100, shuffle=True)
print("Autoencoder training finished")

center = encoder.predict(unlabeled_data)  # combine unlabeled data and data labeled as normal?
center = np.mean(center, axis=0)

data = np.concatenate((labeled_data, unlabeled_data), axis=0)
labels = np.concatenate((labeled_data_labels, unlabeled_data_labels), axis=0)
data = tf.data.Dataset.from_tensor_slices((data, labels)).shuffle(60000).batch(128)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
train_loss = tf.keras.metrics.Mean()

for i in range(50):
    for datapoints in data:
        train_step_encoder(datapoints, center)
    print(f"Epoch: {i + 1}, Loss: {train_loss.result():.4f}")

print("Deep SAD training finished")
predictions = encoder.predict(test_data)
anomaly_score = tf.norm(tf.subtract(predictions, center), axis=1)

test_data_labels = np.where(test_data_labels == 1, 0, 1)  # in preprocessing?
auc = metrics.roc_auc_score(test_data_labels, anomaly_score)
print(auc)