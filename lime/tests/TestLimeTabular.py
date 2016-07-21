import unittest

import numpy as np
from sklearn.linear_model import Lasso, LogisticRegression

from lime.lime_tabular import LimeTabularExplainer
from lime.lime_text import LimeTextExplainer

__author__ = 'devgupta'


class TestLimeTabular(unittest.TestCase):
    def test_lime_explainer_bad_regressor(self):
        class_names = ['foo', 'bar']
        lasso = Lasso(alpha=1, fit_intercept=True)
        eye = np.eye(2, dtype=int)
        feature_names = ['1', '2']
        with self.assertRaises(ValueError):
            explainer = LimeTabularExplainer(training_data=eye, class_names=class_names, model_regressor=lasso, feature_names=feature_names)

    def test_lime_explainer_good_regressor(self):
        class_names = ['foo', 'bar']
        logit = LogisticRegression()
        eye = np.eye(2, dtype=int)
        feature_names = ['1', '2']
        explainer = LimeTabularExplainer(training_data=eye, class_names=class_names, model_regressor=logit, feature_names=feature_names)
