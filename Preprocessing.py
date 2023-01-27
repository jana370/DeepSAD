import tensorflow as tf
from matplotlib import pyplot as plt


class mnistPreProcessing():
    
    def load_dataset(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        return (x_train, y_train), (x_test, y_test)
