"""
Functions for explaining classifiers that use tabular data (matrices).
"""
#TODO
#from . import lime_base
#from . import explanation
from lime import lime_base
from lime import explanation
import numpy as np
import scipy as sp
import sklearn
import sklearn.preprocessing
import collections
import re
import itertools

class LimeTabularExplainer(object):
    """TODO"""
    def __init__(self, training_data, feature_types=None,
            feature_names=None, kernel_width=3,
            verbose=False, class_names=None, feature_selection='auto'):
        
        self.feature_types = feature_types;
        if self.feature_types is None:
            self.feature_types = {}
        if 'categorical' not in self.feature_types:
            self.feature_types['categorical'] = []
        if 'countable' not in self.feature_types:
            self.feature_types['countable'] = []

        kernel = lambda d: np.sqrt(np.exp(-(d**2) / kernel_width ** 2))
        self.base = lime_base.LimeBase(kernel, verbose)
        self.scaler = None
        self.feature_names = feature_names
        self.class_names = class_names
        self.scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
        self.scaler.fit(training_data)
        self.feature_values = {}
        self.feature_frequencies = {}
        for feature in feature_types['categorical']:
            feature_count = collections.defaultdict(lambda: 0.0)
            for value in training_data[:, feature]:
                feature_count[value] += 1
            values, frequencies = map(list, zip(*(feature_count.items())))
            self.feature_values[feature] = values
            self.feature_frequencies[feature] = (np.array(frequencies) /
                                                 sum(frequencies))
            self.scaler.mean_[feature] = 0
            self.scaler.scale_[feature] = 1
    def explain_instance(self, data_row, classifier_fn, labels=(1,),
            top_labels=None, num_features=10, num_samples=5000):
        data, inverse = self.data_inverse(data_row, num_samples)
        scaled_data = (data - self.scaler.mean_) / self.scaler.scale_
        distances = np.sqrt(np.sum((scaled_data - scaled_data[0]) ** 2, axis=1)) 
        yss = classifier_fn(inverse)
        if not self.class_names:
            self.class_names = [str(x) for x in range(yss[0].shape[0])]

    def data_inverse(self,
                       data_row,
                       num_samples):
        data = np.random.normal(0, 1, num_samples * data_row.shape[0]).reshape(
            num_samples, data_row.shape[0])
        data = data * self.scaler.scale_ + self.scaler.mean_
        for column in self.feature_types['countable']:
            data[:, column] = data[:, column].round()
        data[0] = data_row.copy()
        inverse = data.copy()
        for column in self.feature_types['categorical']:
            values = self.feature_values[column]
            freqs = self.feature_frequencies[column]
            inverse_column = np.random.choice(values, size=num_samples, replace=True, p=freqs)
            binary_column = np.array([1 if x == data[0, column] else 0 for x in inverse_column])
            binary_column[0] = 1
            inverse_column[0] = data[0, column]
            data[:, column] = binary_column
            inverse[:, column] = inverse_column
        return data, inverse 
