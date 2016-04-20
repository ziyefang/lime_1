"""
Functions for explaining classifiers that use tabular data (matrices).
"""
#TODO
from . import lime_base
from . import explanation
import numpy as np
import scipy as sp
import sklearn
import sklearn.preprocessing
import collections
import re
import itertools
import json
import copy

class TableDomainMapper(explanation.DomainMapper):
    def __init__(self, feature_names, feature_values, scaled_row, categorical_features):
        """Initializer.

        Args:
            indexed_string: lime_text.IndexedString, original string
        """
        self.feature_names = feature_names
        self.feature_values = feature_values
        self.scaled_row = scaled_row
        self.all_categorical = len(categorical_features) == len(scaled_row)
        self.categorical_features = categorical_features
    def map_exp_ids(self, exp):
        """Maps ids to words or word-position strings.

        Args:
            positions: if True, also return word positions

        Returns: 
        TODO TODO
            list of tuples (word, weight), or (word_positions, weight) if
            examples: ('bad', 1) or ('bad_3-6-12', 1)
        """
        return [(self.feature_names[x[0]], x[1]) for x in exp]
    def visualize_instance_html(self, exp, label, random_id,
            show_contributions=None, show_scaled=None, show_all=False):
        if show_contributions is None:
            show_contributions = not self.all_categorical
        if show_scaled is None:
            show_scaled = not self.all_categorical
        show_scaled = json.dumps(show_scaled)
        weights = [0] * len(self.feature_names)
        # TODO TODO 
        scaled_exp = []
        for i, value in exp:
            weights[i] = value * self.scaled_row[i]
            scaled_exp.append((i, value * self.scaled_row[i]))
        scaled_exp = json.dumps(self.map_exp_ids(scaled_exp))
        row = ['%.2f' % a if i not in self.categorical_features else 'N/A' for i, a in enumerate(self.scaled_row)]
        out_list = zip(self.feature_names, self.feature_values,
                row, weights)
        if not show_all:
            out_list = [out_list[x[0]] for x in exp]
        out = ''
        if show_contributions:
            out += '''<script>
                    var cur_width = parseInt(d3.select('#model%s').select('svg').style('width'));
                    console.log(cur_width);
                    var svg_contrib = d3.select('#model%s').append('svg');
                    exp.ExplainFeatures(svg_contrib, %d, %s, '%s', true);
                    cur_width = Math.max(cur_width, parseInt(svg_contrib.style('width'))) + 'px';
                    d3.select('#model%s').style('width', cur_width);
                    </script>
                    ''' % (random_id, random_id, label, scaled_exp, 'Feature contributions', random_id)

        out += '<div id="mytable%s"></div>' % random_id
        out += '''<script>
        var tab = d3.select('#mytable%s');
        exp.ShowTable(tab, %s, %d, %s);
        </script>
        ''' % (random_id, json.dumps(out_list), label, show_scaled)
        return out


class LimeTabularExplainer(object):
    """TODO"""
    def __init__(self, training_data, feature_types=None,
            feature_names=None, categorical_names=None, kernel_width=3,
            verbose=False, class_names=None, feature_selection='auto'):
        # categorical_names[id] = ['name1', 'name2', ...]
        
        self.categorical_names = categorical_names
        if self.categorical_names is None:
            self.categorical_names = {}
        self.feature_types = feature_types
        if self.feature_types is None:
            self.feature_types = {}
        if 'categorical' not in self.feature_types:
            self.feature_types['categorical'] = []
        if 'countable' not in self.feature_types:
            self.feature_types['countable'] = []

        kernel = lambda d: np.sqrt(np.exp(-(d**2) / kernel_width ** 2))
        self.feature_selection = feature_selection
        self.base = lime_base.LimeBase(kernel, verbose)
        self.scaler = None
        self.class_names = class_names
        self.feature_names = feature_names
        self.scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
        self.scaler.fit(training_data)
        self.feature_values = {}
        self.feature_frequencies = {}
        for feature in self.feature_types['categorical']:
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
        if self.class_names is None:
            self.class_names = [str(x) for x in range(yss[0].shape[0])]
        else:
            self.class_names = list(self.class_names)
        feature_names = copy.deepcopy(self.feature_names)
        if feature_names is None:
            feature_names = map(str, range(data_row.shape[0]))
        round_stuff = lambda x: ['%.2f' % a for a in x]
        values = round_stuff(data_row)
        for i in self.feature_types['categorical']:
            name = int(data_row[i])
            if i in self.categorical_names:
                name = self.categorical_names[i][name]
            feature_names[i] = '%s=%s' % (feature_names[i], name)
            values[i] = 'True'
        domain_mapper = TableDomainMapper(feature_names, values,
                scaled_data[0], categorical_features=self.feature_types['categorical'])
        ret_exp = explanation.Explanation(domain_mapper=domain_mapper,
                                          class_names=self.class_names)
        ret_exp.predict_proba = yss[0]
        #map_exp = lambda exp: [(indexed_string.word(x[0]), x[1]) for x in exp]
        if top_labels:
            labels = np.argsort(yss[0])[-top_labels:]
            ret_exp.top_labels = list(labels)
            ret_exp.top_labels.reverse()
        for label in labels:
            ret_exp.local_exp[label] = self.base.explain_instance_with_data(
                scaled_data, yss, distances, label, num_features,
                feature_selection=self.feature_selection)
            print [scaled_data[0,i[0]] for i in ret_exp.local_exp[label]]
            #print ret_exp.local_exp[label]
        return ret_exp

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
