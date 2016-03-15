"""
Contains abstract functionality for learning locally linear sparse model.
"""
import numpy as np
from sklearn import linear_model

class LimeBase(object):
    """Class for learning a locally linear sparse model from perturbed data"""
    def __init__(self,
                 kernel_fn,
                 verbose=False):
        """Init function

        Args:
            kernel_fn : function that transforms an array of distances into an
                        array of proximity values (floats).
            verbose: if true, print local prediction values from linear model.
        """
        self.kernel_fn = kernel_fn
        self.verbose = verbose
    @staticmethod
    def generate_lars_path(weighted_data, weighted_labels):
        """Generates the lars path for weighted data.

        Args:
            weighted_data: data that has been weighted by kernel
            weighted_label: labels, weighted by kernel

        Returns:
            (alphas, coefs), both are arrays corresponding to the regularization
            parameter and coefficients, respectively
        """
        x_vector = weighted_data
        alphas, _, coefs = linear_model.lars_path(x_vector, weighted_labels,
                                                  method='lasso',
                                                  verbose=False)
        return alphas, coefs
    def explain_instance_with_data(self,
                                   neighborhood_data,
                                   neighborhood_labels,
                                   distances,
                                   label,
                                   num_features,
                                   all_features=False):
        """Takes perturbed data, labels and distances, returns explanation.

        Args:
            neighborhood_data: perturbed data, 2d array. first element is
                               assumed to be the original data point.
            neighborhood_labels: corresponding perturbed labels. should have as
                                 many columns as the number of possible labels.
            distances: distances to original data point.
            label: label for which we want an explanation
            num_features: maximum number of features in explanation
            all_features: if true, ignore num_features and explain using all the
                          features.

        Returns:
            A sorted list of tuples, where each tuple (x,y) corresponds to the
            feature id (x) and the local weight (y). The list is sorted by
            decreasing absolute value of y.
        """
        weights = self.kernel_fn(distances)
        weighted_data = neighborhood_data * weights[:, np.newaxis]
        mean = np.mean(neighborhood_labels[:, label])
        shifted_labels = neighborhood_labels[:, label] - mean
        if self.verbose:
            print 'Explaining from mean=', mean
        weighted_labels = shifted_labels * weights
        used_features = range(weighted_data.shape[1])
        if not all_features:
            nonzero = used_features
            _, coefs = self.generate_lars_path(weighted_data,
                                               weighted_labels)
            for i in range(len(coefs.T) - 1, 0, -1):
                nonzero = coefs.T[i].nonzero()[0]
                if len(nonzero) <= num_features:
                    break
            used_features = nonzero

        debiased_model = linear_model.Ridge(alpha=0, fit_intercept=False)
        debiased_model.fit(weighted_data[:, used_features], weighted_labels)
        if self.verbose:
            local_pred = debiased_model.predict(
                neighborhood_data[0, used_features].reshape(1, -1)) + mean
            print 'Prediction_local', local_pred,
            print 'Right:', neighborhood_labels[0, label]
        return sorted(zip(used_features, debiased_model.coef_),
                      key=lambda x: np.abs(x[1]), reverse=True)
