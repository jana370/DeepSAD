import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


class PreProcessing():
    
    def __init__(self, datasetname, normal_class:list, outlier_class: list, known_outlier_class: list, 
                 ratio_known_normal = 0.0, ratio_known_outlier = 0.01, ratio_pollution = 0.0, ratio_polluted_label_data = 0.0):
        self.datasetname = datasetname
        self.normal_class = normal_class
        self.outlier_class = outlier_class
        self.known_outlier_classes = known_outlier_class
        self.ratio_known_normal = ratio_known_normal
        self. ratio_known_outlier = ratio_known_outlier
        self.ratio_pollution = ratio_pollution
        self.ratio_polluted_label_data = ratio_polluted_label_data
    
    def load_dataset(self):
        if self.datasetname == "mnist":
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        elif self.datasetname == "fmnist":
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        elif self.datasetname == "cifar10":
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        train_dataset = (x_train, y_train)
        test_dataset = (x_test, y_test)
        return train_dataset, test_dataset
    
    def make_data_semisupervised(self, dataset: tuple):
                                

        index_normal = np.argwhere(np.isin(dataset[1], self.normal_class)).flatten()
        index_outlier = np.argwhere(np.isin(dataset[1], self.outlier_class)).flatten()
        index_known_outlier = np.argwhere(np.isin(dataset[1], self.known_outlier_classes)).flatten()

        n_normal = len(index_normal)
        
        # Solve system of linear equations to obtain respective number of samples
        a = np.array([[1, 1, 0, 0],
                    [(1-self.ratio_known_normal), -self.ratio_known_normal, -self.ratio_known_normal, -self.ratio_known_normal],
                    [-self.ratio_known_outlier, -self.ratio_known_outlier, -self.ratio_known_outlier, (1-self.ratio_known_outlier)],
                    [0, -self.ratio_pollution, (1-self.ratio_pollution), 0]])
        b = np.array([n_normal, 0, 0, 0])
        x = np.linalg.solve(a, b)

        # Get number of samples
        n_known_normal = int(x[0])
        n_unlabeled_normal = int(x[1])
        n_unlabeled_outlier = int(x[2])
        n_known_outlier = int(x[3])
        
        n_wrong_label_normal = int(self.ratio_polluted_label_data * n_known_outlier)
        n_wrong_label_outlier = int(self.ratio_polluted_label_data*n_known_outlier)            
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
        unlabeled_data = (unlabeled_data_array, unlabeled_new_labels_array)
        #combine data: np.hstack((unlabeled_data_array, labeled_data_array))    
        return labeled_data, unlabeled_data
    
    def relabel_test_data(self, dataset: tuple):
        new_labels = np.empty(len(dataset[0]))
        for i, label in enumerate(dataset[1]):
            if label in self.normal_class:
                new_labels[i] = int(0)
            else:
                new_labels[i] = int(1)
        test_data = (dataset[0], new_labels)
        return test_data
    
    def get_test_data(self):
        train_data, test_data = self.load_dataset()
        test_data = self.relabel_test_data(test_data)
        return test_data
    
    def get_train_data(self):
        train_data, test_data = self.load_dataset()
        labeled_data, unlabeled_data = self.make_data_semisupervised(train_data)
        #labeled data: tuple of an image array and an array with new labels
        #unlabeled data: array of images
        return labeled_data, unlabeled_data
        
if __name__ == "__main__":
    data = PreProcessing("mnist", [1], [2], [3])
    test_data = data.get_test_data()
    labeled, unlabeled = data.get_train_data()
    plt.imshow(unlabeled[3333], cmap='gray')
    plt.show()
    