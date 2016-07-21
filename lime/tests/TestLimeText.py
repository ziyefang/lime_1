import unittest

from sklearn.linear_model import Lasso, LogisticRegression

from lime.lime_text import LimeTextExplainer

__author__ = 'devgupta'

class TestLimeText(unittest.TestCase):

    def test_lime_explainer_bad_regressor(self):
        class_names = ['foo', 'bar']
        lasso = Lasso(alpha = 1, fit_intercept=True)
        with self.assertRaises(ValueError):
            explainer = LimeTextExplainer(class_names=class_names, model_regressor=lasso)

    def test_lime_explainer_good_regressor(self):
        class_names = ['foo', 'bar']
        logit = LogisticRegression()
        explainer = LimeTextExplainer(class_names=class_names, model_regressor=logit)