import unittest

import numpy as np
from sklearn.datasets import load_iris, make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    # Deprecated in scikit-learn version 0.18, removed in 0.20
    from sklearn.cross_validation import train_test_split

from lime.lime_tabular import LimeTabularExplainer


class TestLimeTabular(unittest.TestCase):

    def setUp(self):
        iris = load_iris()

        self.feature_names = iris.feature_names
        self.target_names = iris.target_names

        (self.train,
         self.test,
         self.labels_train,
         self.labels_test) = train_test_split(iris.data, iris.target, train_size=0.80)

    def test_lime_explainer_bad_regressor(self):

        rf = RandomForestClassifier(n_estimators=500)
        rf.fit(self.train, self.labels_train)
        lasso = Lasso(alpha=1, fit_intercept=True)
        i = np.random.randint(0, self.test.shape[0])
        with self.assertRaises(TypeError):
            explainer = LimeTabularExplainer(self.train,
                                             mode="classification",
                                             feature_names=self.feature_names,
                                             class_names=self.target_names,
                                             discretize_continuous=True)
            exp = explainer.explain_instance(self.test[i],  # noqa:F841
                                             rf.predict_proba,
                                             num_features=2, top_labels=1,
                                             model_regressor=lasso)

    def test_lime_explainer_good_regressor(self):
        np.random.seed(1)

        rf = RandomForestClassifier(n_estimators=500)
        rf.fit(self.train, self.labels_train)
        i = np.random.randint(0, self.test.shape[0])

        explainer = LimeTabularExplainer(self.train,
                                         mode="classification",
                                         feature_names=self.feature_names,
                                         class_names=self.target_names,
                                         discretize_continuous=True)

        exp = explainer.explain_instance(self.test[i],
                                         rf.predict_proba,
                                         num_features=2,
                                         model_regressor=LinearRegression())

        self.assertIsNotNone(exp)
        keys = [x[0] for x in exp.as_list()]
        self.assertEqual(1,
                         sum([1 if 'petal width' in x else 0 for x in keys]),
                         "Petal Width is a major feature")
        self.assertEqual(1,
                         sum([1 if 'petal length' in x else 0 for x in keys]),
                         "Petal Length is a major feature")

    def test_lime_explainer_good_regressor_synthetic_data(self):
        X, y = make_classification(n_samples=1000,
                                   n_features=20,
                                   n_informative=2,
                                   n_redundant=2)

        rf = RandomForestClassifier(n_estimators=500)
        rf.fit(X, y)
        instance = np.random.randint(0, X.shape[0])
        feature_names = ["feature" + str(i) for i in range(20)]
        explainer = LimeTabularExplainer(X,
                                         feature_names=feature_names,
                                         discretize_continuous=True)

        exp = explainer.explain_instance(X[instance], rf.predict_proba)

        self.assertIsNotNone(exp)
        self.assertEqual(10, len(exp.as_list()))

    def test_lime_explainer_no_regressor(self):
        np.random.seed(1)

        rf = RandomForestClassifier(n_estimators=500)
        rf.fit(self.train, self.labels_train)
        i = np.random.randint(0, self.test.shape[0])

        explainer = LimeTabularExplainer(self.train,
                                         feature_names=self.feature_names,
                                         class_names=self.target_names,
                                         discretize_continuous=True)

        exp = explainer.explain_instance(self.test[i],
                                         rf.predict_proba,
                                         num_features=2)
        self.assertIsNotNone(exp)
        keys = [x[0] for x in exp.as_list()]
        self.assertEqual(1,
                         sum([1 if 'petal width' in x else 0 for x in keys]),
                         "Petal Width is a major feature")
        self.assertEqual(1,
                         sum([1 if 'petal length' in x else 0 for x in keys]),
                         "Petal Length is a major feature")

    def test_lime_explainer_entropy_discretizer(self):
        np.random.seed(1)

        rf = RandomForestClassifier(n_estimators=500)
        rf.fit(self.train, self.labels_train)
        i = np.random.randint(0, self.test.shape[0])

        explainer = LimeTabularExplainer(self.train,
                                         feature_names=self.feature_names,
                                         class_names=self.target_names,
                                         training_labels=self.labels_train,
                                         discretize_continuous=True,
                                         discretizer='entropy')

        exp = explainer.explain_instance(self.test[i],
                                         rf.predict_proba,
                                         num_features=2)
        self.assertIsNotNone(exp)
        keys = [x[0] for x in exp.as_list()]
        self.assertEqual(1,
                         sum([1 if 'petal width' in x else 0 for x in keys]),
                         "Petal Width is a major feature")
        self.assertEqual(1,
                         sum([1 if 'petal length' in x else 0 for x in keys]),
                         "Petal Length is a major feature")


if __name__ == '__main__':
    unittest.main()
