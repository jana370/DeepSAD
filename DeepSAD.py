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


def create_neural_network_mnist():
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
    outputs_encoder = layers.Dense(128, use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
    x = layers.LeakyReLU(0.1)(outputs_encoder)
    x = tf.reshape(x, [-1, 4, 4, 8])
    x = deconvolutional_module(x, 64, 5)
    x = deconvolutional_module(x, 32, 5)
    outputs = deconvolutional_module(x, 1, 5, final_layer=True)

    encoder = tf.keras.Model(inputs=inputs, outputs=outputs_encoder)
    autoencoder = tf.keras.Model(inputs=inputs, outputs=outputs)
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.MeanSquaredError())
    return encoder, autoencoder


def create_neural_network_fmnist():
    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = convolutional_module(inputs, 16, 5)
    x = convolutional_module(x, 32, 5)
    x = layers.Flatten()(x)
    x = layers.Dense(49, use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.Dense(64, use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
    x = layers.LeakyReLU(0.1)(x)
    outputs_encoder = layers.Dense(32, use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
    x = layers.LeakyReLU(0.1)(outputs_encoder)
    x = layers.Dense(64, use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.Dense(49, use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
    x = layers.LeakyReLU(0.1)(x)
    x = tf.reshape(x, [-1, 7, 7, 1])
    x = deconvolutional_module(x, 16, 5)
    outputs = deconvolutional_module(x, 1, 5, final_layer=True)

    encoder = tf.keras.Model(inputs=inputs, outputs=outputs_encoder)
    autoencoder = tf.keras.Model(inputs=inputs, outputs=outputs)
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=tf.keras.losses.MeanSquaredError())
    return encoder, autoencoder


@tf.function
def big_train_step_encoder(data, center, mode):
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
        labeled_outlier_differences = (1 / tf.norm(tf.subtract(labeled_outlier_predictions, center), axis=2)) ** 2 + 1e-5

        labeled_outlier_loss = tf.reduce_sum(labeled_outlier_differences) / data[0].shape[0]

        if mode == "standard":
            loss = unlabeled_loss + labeled_normal_loss + labeled_outlier_loss
        if mode == "standard_normal":
            loss = unlabeled_loss + 3 * (labeled_normal_loss + labeled_outlier_loss)
        if mode == "extended":
            loss = unlabeled_loss + 2 * labeled_normal_loss + 4 * labeled_outlier_loss
    gradients = tape.gradient(loss, encoder.trainable_variables)
    optimizer.apply_gradients(zip(gradients, encoder.trainable_variables))
    train_loss(loss)


@tf.function
def small_train_step_encoder(data, center, mode, optimizer):
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
        labeled_outlier_differences = (1 / tf.norm(tf.subtract(labeled_outlier_predictions, center), axis=2)) ** 2 + 1e-5

        labeled_outlier_loss = tf.reduce_sum(labeled_outlier_differences) / data[0].shape[0]

        if mode == "standard":
            loss = unlabeled_loss + labeled_normal_loss + labeled_outlier_loss
        if mode == "standard_normal":
            loss = unlabeled_loss + 3 * (labeled_normal_loss + labeled_outlier_loss)
        if mode == "extended":
            loss = unlabeled_loss + 2 * labeled_normal_loss + 4 * labeled_outlier_loss
    gradients = tape.gradient(loss, encoder.trainable_variables)
    optimizer.apply_gradients(zip(gradients, encoder.trainable_variables))
    train_loss(loss)


dataset = "mnist"
mode = "standard"

categories_mnist = ((0, 6, 8, 9), (1, 4, 7), (2, 3, 5))
categories_fmnist = ((0, 2, 4, 6), (1, 3), (5, 7, 8, 9))
categories_cifar10 = ((0, 1, 8, 9), (3, 6), (2, 4, 5, 7))

if dataset == "mnist":
    encoder, autoencoder = create_neural_network_mnist()
    categories = categories_mnist
elif dataset == "fmnist":
    categories = categories_fmnist
    encoder, autoencoder = create_neural_network_fmnist()
elif dataset == "cifar10":
    categories = categories_cifar10
    encoder, autoencoder = create_neural_network_cifar10()

Preprocessor = Preprocessing.PreProcessing(dataset, categories[1], categories[2], ratio_known_outlier=0.05, ratio_known_normal=0.0, ratio_pollution=0.1, ratio_polluted_label_data=0.01)
(labeled_data, labeled_data_labels), (unlabeled_data, unlabeled_data_labels) = Preprocessor.get_train_data()
(test_data, test_data_labels) = Preprocessor.get_test_data()

normal_mask = np.where(labeled_data_labels == 0)
normal_data = labeled_data[normal_mask]
normal_data = np.concatenate((unlabeled_data, normal_data), axis=0)

print("Starting Autoencoder training")
autoencoder.fit(normal_data, normal_data, batch_size=128, epochs=150, shuffle=True)
print("Autoencoder training finished")

center = encoder.predict(normal_data)
center = np.mean(center, axis=0)

data = np.concatenate((labeled_data, unlabeled_data), axis=0)
labels = np.concatenate((labeled_data_labels, unlabeled_data_labels), axis=0)
data = tf.data.Dataset.from_tensor_slices((data, labels)).shuffle(60000).batch(128)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

train_loss = tf.keras.metrics.Mean()

print("Starting Deep SAD training")
for k in range(50):
    for datapoints in data:
        big_train_step_encoder(datapoints, center, mode)
    print(f"Epoch: {k + 1}, Loss: {train_loss.result():.4f}")
print("learning rate reduced to 0.00001")
optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
for k in range(50, 150):
    for datapoints in data:
        big_train_step_encoder(datapoints, center, mode)
    print(f"Epoch: {k + 1}, Loss: {train_loss.result():.4f}")

print("Deep SAD training finished")
predictions = encoder.predict(test_data)
anomaly_score = tf.norm(tf.subtract(predictions, center), axis=1)

auc = metrics.roc_auc_score(test_data_labels, anomaly_score)
print(auc)

