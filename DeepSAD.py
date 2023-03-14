import Preprocessing
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn import metrics


def convolutional_module(input, filters, kernel_size):
    """define convolutional modules used in encoder parts of NN"""
    x = layers.Conv2D(filters=filters, kernel_size=kernel_size, padding="same", use_bias=False,
                      kernel_regularizer=tf.keras.regularizers.l2(1e-5))(input)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.MaxPooling2D((2, 2))(x)
    return x


def deconvolutional_module(input, filters, kernel_size, final_layer=False):
    """define deconvolutional modules used in decoder parts of NN"""
    x = layers.LeakyReLU(alpha=0.1)(input)
    x = layers.UpSampling2D((2, 2))(x)
    if not final_layer:
        x = layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, padding="same", use_bias=False,
                                   kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
        x = layers.BatchNormalization()(x)
    else:
        x = layers.Conv2DTranspose(filters, kernel_size, padding="same", use_bias=False,
                                   kernel_regularizer=tf.keras.regularizers.l2(1e-5), activation="sigmoid")(x)
    return x


def create_neural_network_mnist():
    """define NN for MNIST"""
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

    # build autoencoder and encoder
    encoder = tf.keras.Model(inputs=inputs, outputs=outputs_encoder)
    autoencoder = tf.keras.Model(inputs=inputs, outputs=outputs)
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                        loss=tf.keras.losses.MeanSquaredError())
    return encoder, autoencoder


def create_neural_network_cifar10():
    """define NN for CIFAR-10"""
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

    # build autoencoder and encoder
    encoder = tf.keras.Model(inputs=inputs, outputs=outputs_encoder)
    autoencoder = tf.keras.Model(inputs=inputs, outputs=outputs)
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                        loss=tf.keras.losses.MeanSquaredError())
    return encoder, autoencoder


def create_neural_network_fmnist():
    """define NN for F-MNIST"""
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

    # build autoencoder and encoder
    encoder = tf.keras.Model(inputs=inputs, outputs=outputs_encoder)
    autoencoder = tf.keras.Model(inputs=inputs, outputs=outputs)
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                        loss=tf.keras.losses.MeanSquaredError())
    return encoder, autoencoder


@tf.function
def train_step_encoder(data, center, mode, weight, second_weight):
    """training step for Deep SAD"""
    with tf.GradientTape() as tape:
        predictions = encoder(data[0], training=True)

        # compute loss for unlabeled data
        unlabeled_mask = tf.where(data[1] == 0)
        unlabeled_predictions = tf.gather(predictions, unlabeled_mask)
        unlabeled_differences = tf.norm(tf.subtract(unlabeled_predictions, center), axis=2) ** 2
        unlabeled_loss = tf.reduce_sum(unlabeled_differences) / data[0].shape[0]

        # compute loss for labeled normal data
        labeled_normal_mask = tf.where(data[1] == 1)
        labeled_normal_predictions = tf.gather(predictions, labeled_normal_mask)
        labeled_normal_differences = tf.norm(tf.subtract(labeled_normal_predictions, center), axis=2) ** 2
        labeled_normal_loss = tf.reduce_sum(labeled_normal_differences) / data[0].shape[0]

        # compute loss for labeled anomalies
        labeled_outlier_mask = tf.where(data[1] == -1)
        labeled_outlier_predictions = tf.gather(predictions, labeled_outlier_mask)
        labeled_outlier_differences = (1 / tf.norm(tf.subtract(labeled_outlier_predictions, center), axis=2)) ** 2+1e-5
        labeled_outlier_loss = tf.reduce_sum(labeled_outlier_differences) / data[0].shape[0]

        # compute total loss based on Deep SAD variant used
        if mode == "standard":
            loss = unlabeled_loss + labeled_normal_loss + weight * labeled_outlier_loss
        if mode == "standard_normal":
            loss = unlabeled_loss + weight * (labeled_normal_loss + labeled_outlier_loss)
        if mode == "extended":
            loss = unlabeled_loss + weight * labeled_normal_loss + second_weight * labeled_outlier_loss

    # update NN
    gradients = tape.gradient(loss, encoder.trainable_variables)
    optimizer.apply_gradients(zip(gradients, encoder.trainable_variables))
    train_loss(loss)


if __name__ == "__main__":
    # Command Line-Interface
    parser = argparse.ArgumentParser(description="Run Deep SAD using the defined categories for one of the datasets")
    parser.add_argument("-d", "--dataset", metavar="", default="mnist", choices=["minst", "fmnist", "cifar10"],
                        help="choose the dataset which will be used; either \"mnist\", \"fmnist\", or \"cifar10\" can "
                             "be used; default is \"mnist\"")
    parser.add_argument("-m", "--mode", metavar="", default="standard",
                        choices=["standard", "standard_normal", "extended"],
                        help="choose the type of loss function, which will be used for Deep SAD; "
                             "\"standard\" will treat labeled normal data the same as unlabeled data and use the weight"
                             " only for labeled anomalies; "
                             "\"standard_normal\" will use the weight for both labeled normal data and labeled "
                             "anomalies; "
                             "\"extended\" will use the weight for the labeled normal data and the second weight for "
                             "the labeled anomalies; default is \"standard\"")
    parser.add_argument("-w", "--weight", metavar="", default=3, type=float,
                        help="choose the weight that will be used in the loss function; Note, that this only defines "
                             "the weight for the labeled normal data if the \"extended\" mode is used; default is 3")
    parser.add_argument("-sw", "--second_weight", metavar="", default=4, type=float,
                        help="choose the second weight that will be used for the labeled anomalies if the \"extended\" "
                             "mode is used; default is 4")
    parser.add_argument("-cn", "--category_normal", metavar="", default=0, type=int, choices=range(3),
                        help="choose category which will be used as the normal class, the following categories are"
                             "defined for each dataset: MNIST: 0: (0, 6, 8, and 9), 1: (1, 4, and 7), 2: (2, 3, and 5);"
                             "F-MNIST: 0: (T_shirt, Pullover, Coat, and Shirt), 1: (Trouser, and Dress), 2: (Sandal, " 
                             "Sneaker, Bag, and Ankleboot); CIFAR-10: 0: (plane, car, ship, and truck), 1: (bird, and "
                             "frog), 2: (cat, deer, dog, and horse); default is 0")
    parser.add_argument("-ca", "--category_anomaly", metavar="", default=1, type=int, choices=range(3),
                        help="choose category which will be used as the anomaly class, the following categories are"
                             "defined for each dataset: MNIST: 0: (0, 6, 8, and 9), 1: (1, 4, and 7), 2: (2, 3, and 5);"
                             "F-MNIST: 0: (T_shirt, Pullover, Coat, and Shirt), 1: (Trouser, and Dress), 2: (Sandal, "
                             "Sneaker, Bag, and Ankleboot); CIFAR-10: 0: (plane, car, ship, and truck), 1: (bird, and "
                             "frog), 2: (cat, deer, dog, and horse); default is 1")
    parser.add_argument("-ra", "--ratio_anomaly", metavar="", default=0.05, type=float,
                        help="choose the ratio of labeled anomalies that will be used; Note, that the value should "
                             "be between 0 and 1; default is 0.05")
    parser.add_argument("-rn", "--ratio_normal", metavar="", default=0.0, type=float,
                        help="choose the ratio of labeled normal data that will be used; Note, that the value should "
                             "be between 0 and 1; default is 0.0")
    parser.add_argument("-rpu", "--ratio_pollution_unlabeled", metavar="", default=0.1, type=float,
                        help="choose the ratio of pollution in the unlabeled data; Note, that the value should "
                             "be between 0 and 1; default is 0.1")
    parser.add_argument("-rpl", "--ratio_pollution_labeled", metavar="", default=0.0, type=float,
                        help="choose the ratio of pollution in the labeled anomalies; Note, that the value should "
                             "be between 0 and 1; default is 0.0")

    args = parser.parse_args()

    dataset = args.dataset.lower()
    mode = args.mode
    weight = args.weight
    second_weight = args.second_weight

    # define categories for datasets
    categories_mnist = ((0, 6, 8, 9), (1, 4, 7), (2, 3, 5))
    categories_fmnist = ((0, 2, 4, 6), (1, 3), (5, 7, 8, 9))
    categories_cifar10 = ((0, 1, 8, 9), (2, 6), (3, 4, 5, 7))

    # get NNs and categories for used dataset
    if dataset == "mnist":
        encoder, autoencoder = create_neural_network_mnist()
        categories = categories_mnist
    elif dataset == "fmnist":
        categories = categories_fmnist
        encoder, autoencoder = create_neural_network_fmnist()
    elif dataset == "cifar10":
        categories = categories_cifar10
        encoder, autoencoder = create_neural_network_cifar10()

    # get data
    Preprocessor = Preprocessing.PreProcessing(dataset, categories[args.category_normal],
                                               categories[args.category_anomaly],
                                               ratio_known_outlier=args.ratio_anomaly,
                                               ratio_known_normal=args.ratio_normal,
                                               ratio_pollution=args.ratio_pollution_unlabeled,
                                               ratio_polluted_label_data=args.ratio_pollution_labeled)
    (labeled_data, labeled_data_labels), (unlabeled_data, unlabeled_data_labels) = Preprocessor.get_train_data()
    (test_data, test_data_labels) = Preprocessor.get_test_data()

    # get all data considered normal (unlabeled and labeled normal)
    if labeled_data.size > 0 and unlabeled_data.size > 0:
        normal_mask = np.where(labeled_data_labels == 0, 1, 0)
        normal_data = labeled_data[normal_mask]
        normal_data = np.concatenate((unlabeled_data, normal_data), axis=0)
    elif labeled_data.size > 0:
        normal_mask = np.where(labeled_data_labels == 0, 1, 0)
        normal_data = labeled_data[normal_mask]
    else:
        normal_data = unlabeled_data

    print("Starting Autoencoder training")
    autoencoder.fit(normal_data, normal_data, batch_size=128, epochs=150, shuffle=True)
    print("Autoencoder training finished")

    # compute central point of hypersphere
    center = encoder.predict(normal_data)
    center = np.mean(center, axis=0)

    # concatenate labeled and unlabeled data
    if labeled_data.size > 0 and unlabeled_data.size > 0:
        data = np.concatenate((labeled_data, unlabeled_data), axis=0)
        labels = np.concatenate((labeled_data_labels, unlabeled_data_labels), axis=0)
    elif labeled_data.size > 0:
        data = labeled_data
        labels = labeled_data_labels
    else:
        data = unlabeled_data
        labels = unlabeled_data_labels

    data = tf.data.Dataset.from_tensor_slices((data, labels)).shuffle(60000).batch(128)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    train_loss = tf.keras.metrics.Mean()

    print("Starting Deep SAD training")
    for k in range(50):
        for datapoints in data:
            train_step_encoder(datapoints, center, mode, weight, second_weight)
        print(f"Epoch: {k + 1}, Loss: {train_loss.result():.4f}")
    print("Learning rate reduced to 0.00001")
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
    for k in range(50, 150):
        for datapoints in data:
            train_step_encoder(datapoints, center, mode, weight, second_weight)
        print(f"Epoch: {k + 1}, Loss: {train_loss.result():.4f}")
    print("Deep SAD training finished")

    # calculate anomaly score
    predictions = encoder.predict(test_data)
    anomaly_score = tf.norm(tf.subtract(predictions, center), axis=1)

    # calculate auc score
    auc = metrics.roc_auc_score(test_data_labels, anomaly_score)
    print(f"AUC score: {auc}")


