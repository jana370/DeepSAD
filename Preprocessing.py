import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


class PreProcessing():
    
    def load_dataset(self, datasetname: str):
        if datasetname == "mnist":
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        elif datasetname == "fmnist":
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        elif datasetname == "cifar10":
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        train_dataset = (x_train, y_train)
        test_dataset = (x_test, y_test)
        return train_dataset, test_dataset
    
    def make_data_semisupervised(self, dataset: tuple, normal_class: list, outlier_class: list, known_outlier_classes: list,
                                  ratio_known_normal, ratio_known_outlier, ratio_pollution, ratio_polluted_label_data):
                                

        index_normal = np.argwhere(np.isin(dataset[1], normal_class)).flatten()
        index_outlier = np.argwhere(np.isin(dataset[1], outlier_class)).flatten()
        index_known_outlier = np.argwhere(np.isin(dataset[1], known_outlier_classes)).flatten()

        n_normal = len(index_normal)
        
        # Solve system of linear equations to obtain respective number of samples
        a = np.array([[1, 1, 0, 0],
                    [(1-ratio_known_normal), -ratio_known_normal, -ratio_known_normal, -ratio_known_normal],
                    [-ratio_known_outlier, -ratio_known_outlier, -ratio_known_outlier, (1-ratio_known_outlier)],
                    [0, -ratio_pollution, (1-ratio_pollution), 0]])
        b = np.array([n_normal, 0, 0, 0])
        x = np.linalg.solve(a, b)

        # Get number of samples
        n_known_normal = int(x[0])
        n_unlabeled_normal = int(x[1])
        n_unlabeled_outlier = int(x[2])
        n_known_outlier = int(x[3])
        
        n_wrong_label_normal = int(ratio_polluted_label_data * n_known_outlier)
        n_wrong_label_outlier = int(ratio_polluted_label_data*n_known_outlier)            
        n_known_normal = int(n_known_normal - n_wrong_label_normal)
        n_known_outlier = int(n_known_outlier - n_wrong_label_outlier)
        #Grundgesamtheit: normal data + polluted data, known outlier ratio ist Prozent von dieser Gesamtheit
        #wenn polluted data and known outlier 0, dann gibt es keine outlier
        
        shuffled_normal = np.random.permutation(n_normal)
        shuffled_outlier = np.random.permutation(len(index_outlier))
        shuffled_known_outlier = np.random.permutation(len(index_known_outlier))

        idx_known_normal = index_normal[shuffled_normal[:n_known_normal]].tolist()
        idx_unlabeled_normal = index_normal[shuffled_normal[n_known_normal:n_known_normal+n_unlabeled_normal]].tolist()
        idx_wrong_labels_normal = index_normal[shuffled_normal[n_known_normal+n_unlabeled_normal:n_known_normal+n_unlabeled_normal+n_wrong_label_normal]].tolist()
        idx_unlabeled_outlier = index_outlier[shuffled_outlier[:n_unlabeled_outlier]].tolist()
        idx_known_outlier = index_known_outlier[shuffled_known_outlier[:n_known_outlier]].tolist()
        idx_wrong_labels_outlier = index_known_outlier[shuffled_known_outlier[n_known_outlier:n_known_outlier+n_wrong_label_outlier]].tolist()     
        
        labeled_data_list= []
        labeled_labels_list = []
        labeled_new_labels_list = []
        for index in idx_known_normal:
            labeled_data_list.append(dataset[0][index])
            labeled_labels_list.append(dataset[1][index])
            labeled_new_labels_list.append(1)
            
        for index in idx_known_outlier:
            labeled_data_list.append(dataset[0][index])
            labeled_labels_list.append(dataset[1][index])
            labeled_new_labels_list.append(-1)
            
        for index in idx_wrong_labels_normal:
            labeled_data_list.append(dataset[0][index])
            labeled_labels_list.append(dataset[1][index])
            labeled_new_labels_list.append(-1) 
            
        for index in idx_wrong_labels_outlier:
            labeled_data_list.append(dataset[0][index])
            labeled_labels_list.append(dataset[1][index])
            labeled_new_labels_list.append(1)
        
        labeled_data_array= np.array(labeled_data_list)
        labeled_labels_array = np.array(labeled_labels_list)
        labeled_new_labels_array = np.array(labeled_new_labels_list)
        labeled_data = (labeled_data_array, labeled_new_labels_array)
    
        unlabeled_data_list= []
        unlabeled_labels_list = []
        unlabeled_new_labels_list = []
        
        for index in idx_unlabeled_normal:
            unlabeled_data_list.append(dataset[0][index])
            unlabeled_labels_list.append(dataset[1][index])
            unlabeled_new_labels_list.append(0)
            
        for index in idx_unlabeled_outlier:
            unlabeled_data_list.append(dataset[0][index])
            unlabeled_labels_list.append(dataset[1][index])
            unlabeled_new_labels_list.append(0)   
            
        unlabeled_data_array= np.array(unlabeled_data_list)
        unlabeled_labels_array = np.array(unlabeled_labels_list)
        unlabeled_new_labels_array = np.array(unlabeled_new_labels_list)
        unlabeled_data = unlabeled_data_array
        #combine data: np.hstack((unlabeled_data_array, labeled_data_array))    
        return labeled_data, unlabeled_data
    
    def relable_test_data(self, dataset: tuple, normal_class: list):
        new_labels = np.empty(len(dataset[0]))
        for i, label in enumerate(dataset[1]):
            if label in normal_class:
                new_labels[i] = int(1)
            else:
                new_labels[i] = int(-1)
        test_data = (dataset[0], new_labels)
        return test_data   
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

