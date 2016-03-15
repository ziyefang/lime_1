"""
Functions for explaining text classifiers.
"""
from . import lime_base
from . import explanation
import numpy as np
import scipy as sp
import sklearn

class LimeTextExplainer(object):
    """Explains text classifiers.
       Currently, we are using an exponential kernel on cosine distance, and
       restricting explanations to words that are present in documents."""
    def __init__(self,
                 kernel_width=25,
                 verbose=False,
                 vocabulary=None,
                 class_names=None):
        """Init function.

        Args:
            kernel_width: kernel width for the exponential kernel
            verbose: if true, print local prediction values from linera model
            vocabulary: map from word to feature id. If present, explanations
                        will be returned with words, instead of feature ids.
            class_names: list of class names, ordered according to whatever the
                         classifier is using. If not present, class names will
                         be '0', '1', ...
        """
        # exponential kernel
        kernel = lambda d: np.sqrt(np.exp(-(d**2) / kernel_width ** 2))
        self.base = lime_base.LimeBase(kernel, verbose)
        self.class_names = class_names
        if vocabulary:
            terms = np.array(list(vocabulary.keys()))
            indices = np.array(list(vocabulary.values()))
            self.vocabulary = terms[np.argsort(indices)]
    def explain_instance(self,
                         instance,
                         classifier_fn,
                         labels=(1),
                         top_labels=None,
                         num_features=10,
                         num_samples=5000,
                         local_explanation=True,
                         top_words=True):
        """Generates explanations for a prediction.

        First, we generate neighborhood data by randomly hiding features from
        the instance (see __data_labels_distance_mapping). We then learn
        locally weighted linear models on this neighborhood data to explain each
        of the classes in an interpretable way (see lime_base.py).
        Args:
            instance: instance to be explained. Assumed to be a
                      sp.sparse.csr_matrix with one row.
            classifier_fn: classifier prediction probability function, which takes a
                        sparse matrix and outputs prediction probabilities.
                        For scikit-learn classifiers, this is
                        classifier.predict_proba.
            labels: iterable with labels to be explained.
            top_labels: if not None, ignore labels and produce explanations for
                        the K labels with highest prediction probabilities,
                        where K is this parameter.
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the linear model
            local_explanation: if true, learn a local explanation where features
                               can be positive or negative. This should provide
                               you with an understanding of how the classifier
                               behaves around the example.
            top_words: if true, also include in the returned object lists of the
                       most positive and most negative words towards the labels
                       of interest.

        Returns:
            An Explanation object (see explanation.py) with the corresponding
            explanations.
        """
        instance = sp.sparse.csr_matrix(instance)
        data, yss, distances, mapping = self.__data_labels_distances_mapping(
            instance, classifier_fn, num_samples)
        if not self.class_names:
            self.class_names = [str(x) for x in range(yss[0].shape[0])]
        ret_exp = explanation.Explanation(
            vocabulary=self.vocabulary, class_names=self.class_names)
        ret_exp.predict_proba = yss[0]
        map_exp = lambda exp: [(mapping[x[0]], x[1]) for x in exp]
        if top_labels:
            labels = np.argsort(yss[0])[-top_labels:]
        for label in labels:
            if local_explanation:
                ret_exp.local_exp[label] = map_exp(
                    self.base.explain_instance_with_data(
                        data, yss, distances, label, num_features))
            if top_words:
                exp = map_exp(self.base.explain_instance_with_data(
                    data, yss, distances, label, num_features, all_features=True))
                sign = lambda z: 1 if z > 0 else -1
                ret_exp.top_pos[label] = [x for x in exp if sign(x[1]) == 1][:num_features]
                ret_exp.top_neg[label] = [x for x in exp if sign(x[1]) == -1][:num_features]
        return ret_exp
    @staticmethod
    def __data_labels_distances_mapping(instance, classifier_fn, num_samples):
        """Generates a neighborhood around a prediction.

        Generates neighborhood data by randomly hiding features from
        the instance, and predicting with the classifier. Uses cosine distance
        to compute distances between original and perturbed instances.
        Args:
            instance: instance to be explained. Assumed to be a
                      sp.sparse.csr_matrix with one row.
            classifier_fn: classifier prediction probability function, which
                           takes a sparse matrix and outputs prediction
                           probabilities. For scikit-learn classifiers, this is
                           classifier.predict_proba.
            num_samples: size of the neighborhood to learn the linear model

        Returns:
            A tuple (data, labels, distances, mapping), where:
                data: dense num_samples * K binary matrix, where K is the
                      number of nonzero words in instance. The first row is
                      the original instance, and thus a row of ones.
                labels: num_samples * L matrix, where L is the number of target
                        labels
                distances: cosine distance between the original instance and
                           each perturbed instance, times 100.
                mapping: correspondance between the features in data and the
                         features in the original problem. mapping[0] is the
                         feature that corresponds to data[:, 0].
        """
        distance_fn = lambda x: sklearn.metrics.pairwise.cosine_distances(x[0], x)[0] * 100
        features = instance.nonzero()[1]
        vals = np.array(instance[instance.nonzero()])[0]
        doc_size = len(sp.sparse.find(instance)[2])
        sample = np.random.randint(1, doc_size, num_samples - 1)
        data = np.zeros((num_samples, len(features)))
        inverse_data = np.zeros((num_samples, len(features)))
        data[0] = np.ones(doc_size)
        inverse_data[0] = vals
        features_range = range(len(features))
        for i, size in enumerate(sample, start=1):
            active = np.random.choice(features_range, size, replace=False)
            data[i, active] = 1
            inverse_data[i, active] = vals[active]
        sparse_inverse = sp.sparse.lil_matrix(
            (inverse_data.shape[0], instance.shape[1]))
        sparse_inverse[:, features] = inverse_data
        sparse_inverse = sp.sparse.csr_matrix(sparse_inverse)
        mapping = features
        labels = classifier_fn(sparse_inverse)
        distances = distance_fn(sparse_inverse)
        return data, labels, distances, mapping
