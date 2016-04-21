"""
Functions for explaining classifiers that use tabular data (matrices).
"""
import collections
import json
import copy
import numpy as np
import sklearn
import sklearn.preprocessing
from . import lime_base
from . import explanation

class TableDomainMapper(explanation.DomainMapper):
    """Maps feature ids to names, generates table views, etc"""
    def __init__(self, feature_names, feature_values, scaled_row, categorical_features):
        """Init.

        Args:
            feature_names: list of feature names, in order
            feature_values: list of strings with the values of the original row
            scaled_row: scaled row
            categorical_featuers: list of categorical features ids (ints)
        """
        self.feature_names = feature_names
        self.feature_values = feature_values
        self.scaled_row = scaled_row
        self.all_categorical = len(categorical_features) == len(scaled_row)
        self.categorical_features = categorical_features
    def map_exp_ids(self, exp):
        """Maps ids to feature names.

        Args:
            exp: list of tuples [(id, weight), (id,weight)]

        Returns:
            list of tuples (feature_name, weight)
        """
        return [(self.feature_names[x[0]], x[1]) for x in exp]
    def visualize_instance_html(self,
                                exp,
                                label,
                                random_id,
                                show_table=True,
                                show_contributions=None,
                                show_scaled=None,
                                show_all=False):
        """Shows the current example in a table format.

        Args:
             exp: list of tuples [(id, weight), (id,weight)]
             label: label id (integer)
             random_id: random_id being used, appended to div ids and etc in
                html.
             show_table: if False, don't show table visualization.
             show_contributions: if True, add an aditional bar plot with weights
                multiplied by example. By default, this is true if there are any
                continuous features.
             show_scaled: if True, display scaled values in table.
             show_all: if True, show zero-weighted features in the table.
        """
        if show_contributions is None:
            show_contributions = not self.all_categorical
        if show_scaled is None:
            show_scaled = not self.all_categorical
        show_scaled = json.dumps(show_scaled)
        weights = [0] * len(self.feature_names)
        scaled_exp = []
        for i, value in exp:
            weights[i] = value * self.scaled_row[i]
            scaled_exp.append((i, value * self.scaled_row[i]))
        scaled_exp = json.dumps(self.map_exp_ids(scaled_exp))
        row = ['%.2f' % a if i not in self.categorical_features else 'N/A'
               for i, a in enumerate(self.scaled_row)]
        out_list = list(zip(self.feature_names, self.feature_values,
                       row, weights))
        if not show_all:
            out_list = [out_list[x[0]] for x in exp]
        out = u''
        if show_contributions:
            out += u'''<script>
                    var cur_width = parseInt(d3.select('#model%s').select('svg').style('width'));
                    console.log(cur_width);
                    var svg_contrib = d3.select('#model%s').append('svg');
                    exp.ExplainFeatures(svg_contrib, %d, %s, '%s', true);
                    cur_width = Math.max(cur_width, parseInt(svg_contrib.style('width'))) + 'px';
                    d3.select('#model%s').style('width', cur_width);
                    </script>
                    ''' % (random_id, random_id, label, scaled_exp,
                           'Feature contributions', random_id)

        if show_table:
            out += u'<div id="mytable%s"></div>' % random_id
            out += u'''<script>
            var tab = d3.select('#mytable%s');
            exp.ShowTable(tab, %s, %d, %s);
            </script>
            ''' % (random_id, json.dumps(out_list), label, show_scaled)
        return out


class LimeTabularExplainer(object):
    """Explains predictions on tabular (i.e. matrix) data.
    For numerical features, perturb them by sampling from a Normal(0,1) and
    doing the inverse operation of mean-centering and scaling, according to the
    means and stds in the training data. For categorical features, perturb by
    sampling according to the training distribution, and making a binary feature
    that is 1 when the value is the same as the instance being explained."""
    def __init__(self, training_data, feature_names=None, categorical_features=None,
                 categorical_names=None, kernel_width=3, verbose=False,
                 class_names=None, feature_selection='auto'):
        """Init function.

        Args:
            training_data: numpy 2d array
            feature_names: list of names (strings) corresponding to the columns
                in the training data.
            categorical_features: list of indices (ints) corresponding to the
                categorical columns. Everything else will be considered
                continuous.
            categorical_names: map from int to list of names, where
                categorical_names[x][y] represents the name of the yth value of
                column x.
            kernel_width: kernel width for the exponential kernel
            verbose: if true, print local prediction values from linear model
            class_names: list of class names, ordered according to whatever the
                classifier is using. If not present, class names will be '0',
                '1', ...
            feature_selection: feature selection method. can be
                'forward_selection', 'lasso_path', 'none' or 'auto'.
                See function 'explain_instance_with_data' in lime_base.py for
                details on what each of the options does.
        """

        self.categorical_names = categorical_names
        self.categorical_features = categorical_features
        if self.categorical_names is None:
            self.categorical_names = {}
        if self.categorical_features is None:
            self.categorical_features = []

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
        for feature in self.categorical_features:
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
        """Generates explanations for a prediction.

        First, we generate neighborhood data by randomly perturbing features from
        the instance (see __data_inverse). We then learn locally weighted linear
        models on this neighborhood data to explain each of the classes in an
        interpretable way (see lime_base.py).

        Args:
            data_row: 1d numpy array, corresponding to a row
            classifier_fn: classifier prediction probability function, which
                takes a string and outputs prediction probabilities.  For
                ScikitClassifiers , this is classifier.predict_proba.
            labels: iterable with labels to be explained.
            top_labels: if not None, ignore labels and produce explanations for
                the K labels with highest prediction probabilities, where K is
                this parameter.
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the linear model

        Returns:
            An Explanation object (see explanation.py) with the corresponding
            explanations.
        """
        data, inverse = self.__data_inverse(data_row, num_samples)
        scaled_data = (data - self.scaler.mean_) / self.scaler.scale_
        distances = np.sqrt(np.sum((scaled_data - scaled_data[0]) ** 2, axis=1))
        yss = classifier_fn(inverse)
        if self.class_names is None:
            self.class_names = [str(x) for x in range(yss[0].shape[0])]
        else:
            self.class_names = list(self.class_names)
        feature_names = copy.deepcopy(self.feature_names)
        if feature_names is None:
            feature_names = [str(x) for x in range(data_row.shape[0])]
        round_stuff = lambda x: ['%.2f' % a for a in x]
        values = round_stuff(data_row)
        for i in self.categorical_features:
            name = int(data_row[i])
            if i in self.categorical_names:
                name = self.categorical_names[i][name]
            feature_names[i] = '%s=%s' % (feature_names[i], name)
            values[i] = 'True'
        domain_mapper = TableDomainMapper(
            feature_names, values, scaled_data[0],
            categorical_features=self.categorical_features)
        ret_exp = explanation.Explanation(domain_mapper=domain_mapper,
                                          class_names=self.class_names)
        ret_exp.predict_proba = yss[0]
        if top_labels:
            labels = np.argsort(yss[0])[-top_labels:]
            ret_exp.top_labels = list(labels)
            ret_exp.top_labels.reverse()
        for label in labels:
            ret_exp.local_exp[label] = self.base.explain_instance_with_data(
                scaled_data, yss, distances, label, num_features,
                feature_selection=self.feature_selection)
        return ret_exp

    def __data_inverse(self,
                       data_row,
                       num_samples):
        """Generates a neighborhood around a prediction.

        For numerical features, perturb them by sampling from a Normal(0,1) and
        doing the inverse operation of mean-centering and scaling, according to
        the means and stds in the training data. For categorical features,
        perturb by sampling according to the training distribution, and making a
        binary feature that is 1 when the value is the same as the instance
        being explained.

        Args:
            data_row: 1d numpy array, corresponding to a row
            num_samples: size of the neighborhood to learn the linear model

        Returns:
            A tuple (data, inverse), where:
                data: dense num_samples * K matrix, where categorical features
                are encoded with either 0 (not equal to the corresponding value
                in data_row) or 1. The first row is the original instance.
                inverse: same as data, except the categorical features are not
                binary, but categorical (as the original data)
        """
        data = np.random.normal(0, 1, num_samples * data_row.shape[0]).reshape(
            num_samples, data_row.shape[0])
        data = data * self.scaler.scale_ + self.scaler.mean_
        data[0] = data_row.copy()
        inverse = data.copy()
        for column in self.categorical_features:
            values = self.feature_values[column]
            freqs = self.feature_frequencies[column]
            inverse_column = np.random.choice(values, size=num_samples, replace=True, p=freqs)
            binary_column = np.array([1 if x == data[0, column] else 0 for x in inverse_column])
            binary_column[0] = 1
            inverse_column[0] = data[0, column]
            data[:, column] = binary_column
            inverse[:, column] = inverse_column
        return data, inverse
