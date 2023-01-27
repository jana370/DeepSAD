import tensorflow as tf
from matplotlib import pyplot as plt


class mnistPreProcessing():
    
    def load_dataset(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        return (x_train, y_train), (x_test, y_test)

    def divide_dataset(self, outlier_number:int, normal_number:int):
        (x_train, y_train), (x_test, y_test) = self.load_dataset()
        # convert the data to tensorflow dataset
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        train_normal = train_dataset.filter(lambda x, y: tf.math.equal(y, normal_number))
        train_outlier = train_dataset.filter(lambda x, y: tf.math.equal(y, outlier_number))
        test_normal = test_dataset.filter(lambda x, y: tf.math.equal(y, normal_number))
        test_outlier = test_dataset.filter(lambda x, y: tf.math.equal(y, outlier_number)) 
        return train_normal, train_outlier, test_normal, test_outlier

