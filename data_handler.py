from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets, DataSet
from tensorflow.python.framework import dtypes
from tensorflow.contrib.learn.python.learn.datasets import base

import numpy as np 

class DataHandler(object): 
    def __init__(self, dataset):
        self.set_dataset(dataset)
        
    def set_dataset(self, dataset): 
        if dataset == 'mnist':
            self.dataset = read_data_sets('./data/mnist/', one_hot=True)
    
    def get_dataset(self): 
        return self.dataset

    def split_dataset(self, dtype=dtypes.float32, reshape=True, seed=None, validation_size=7000): 
        labels = self.dataset.train.labels  

        # Find all training images/labels 1-4 
        train_labels_idx = np.nonzero(self.dataset.train.labels)[1]
        train_labels_idx = np.nonzero(train_labels_idx < 5)[0]
        train_labels = self.dataset.train.labels[train_labels_idx]
        train_images = self.dataset.train.images[train_labels_idx]

        # Find all testing images/labels 1-4 
        test_labels_idx = np.nonzero(self.dataset.test.labels)[1]
        test_labels_idx = np.nonzero(test_labels_idx < 5)[0]
        test_labels = self.dataset.test.labels[test_labels_idx] 
        test_images = self.dataset.test.images[test_labels_idx] 

        # Create validation/training groups 
        validation_images = train_images[:validation_size]
        validation_labels = train_labels[:validation_size]
        train_images = train_images[validation_size:]
        train_labels = train_labels[validation_size:]

        options = dict(dtype=dtype, reshape=False, seed=seed)

        # Define training, validation, and testing datasets  
        train = DataSet(train_images, train_labels, **options)
        validation = DataSet(validation_images, validation_labels, **options)
        test = DataSet(test_images, test_labels, **options)

        return base.Datasets(train=train, validation=validation, test=test)
