import numpy as np
import scipy as sp
from sklearn import linear_model
import sklearn.metrics.pairwise
import json
import matplotlib.pyplot as plt
import os
import os.path


class Explanation:
  def __init__(self, vocabulary=None, class_names=None):
    self.neighborhood_explanations = {}
    self.vocabulary = vocabulary
    self.class_names=class_names
    self.local_exp = {}
    self.top_pos = {}
    self.top_neg = {}
    self.predict_proba = None
    # if return_words:
    #   exp = [(self.vocabulary[x[0]], x[1]) for x in exp]
  def available_labels(self):
    return set(self.top_pos.keys() + self.top_neg.keys() + self.local_exp.keys())
  def as_list(self, label, explanation='local'):
    ''' can be local, pos or neg'''
    if explanation == 'local':
      exp = self.local_exp[label]
    elif explanation == 'pos':
      exp = self.top_pos[label]
    elif explanation == 'neg':
      exp = self.top_neg[label]
    if self.vocabulary is not None:
      exp = [(self.vocabulary[x[0]], x[1]) for x in exp]
    return exp
  def as_map(self, explanation='local'):
    '''local, pos or neg'''
    if explanation == 'local':
      return self.local_exp
    elif explanation == 'pos':
      return self.top_pos
    elif explanation == 'neg':
      return self.top_neg

  def as_pyplot_figure(self, label, explanation='local'):
    exp = self.as_list(label, explanation)
    fig = plt.figure()
    vals = [x[1] for x in exp]
    names = [x[0] for x in exp]
    vals.reverse()
    names.reverse()
    colors = map(lambda x: 'green' if x > 0 else 'red', vals)
    pos = np.arange(len(exp)) + .5
    plt.barh(pos,vals, align='center', color=colors)
    plt.yticks(pos, names)
    if explanation == 'local':
      plt.title('Local explanation for class %s' % self.class_names[label])
    elif explanation == 'pos':
      plt.title('Most positive towards %s' % self.class_names[label])
    elif explanation == 'neg':
      plt.title('Most negative towards %s' % self.class_names[label])
    return fig


  def as_html(self, label=1, include=['predict_proba', 'local', 'pos', 'neg']):
    this_dir, this_filename = os.path.split(__file__)
    d3 = open(os.path.join(this_dir, 'd3.min.js')).read()
    lodash = open(os.path.join(this_dir, 'lodash.js')).read()
    exp_js = open(os.path.join(this_dir, 'explanation.js')).read()
    out = '''<html><head><script>%s </script>
    <script>%s </script>
    <script>%s </script>
    </head>
    <body>
    ''' % (d3, lodash, exp_js)
    out += '''
    <div id="mychart%d" style="display:flex; justify-content:space-between;"></div>
    <script>
    var exp = new Explanation(%s);
    ''' % (label, json.dumps(self.class_names))

    if 'predict_proba' in include:
      out += '''
      var svg = d3.select('#mychart%d').append('svg');
      exp.PredictProba(svg, %s);
      ''' % (label, json.dumps(list(self.predict_proba)))
    if 'pos' in include:
      exp = json.dumps(self.as_list(label, 'pos'))
      out += '''
        var svg2 = d3.select('#mychart%d').append('svg');
        exp.ExplainFeatures(svg2, %d, %s, 'Most positive towards %s', false);
      ''' % (label, label, exp, self.class_names[label])
    if 'neg' in include:
      exp = json.dumps(self.as_list(label, 'neg'))
      out += '''
        var svg3 = d3.select('#mychart%d').append('svg');
        exp.ExplainFeatures(svg3, %d, %s, 'Most negative towards %s', false);
      ''' % (label, label, exp, self.class_names[label])
    if 'local' in include:
      exp = json.dumps(self.as_list(label, 'local'))
      out += '''
        var svg4 = d3.select('#mychart%d').append('svg');
        exp.ExplainFeatures(svg4, %d, %s, 'Local explanation at P(%s)=%.2f', true);
      ''' % (label, label, exp, self.class_names[label], self.predict_proba[label])

    out += '</script></body></html>'
    return out


    
class LimeTextExplainer:
  def __init__(self,
               kernel_width=25,
               verbose=False,
               vocabulary=None,
               class_names=None):
    # exponential kernel
    kernel = lambda d: np.sqrt(np.exp(-(d**2) / kernel_width ** 2))
    self.base = LimeBase(kernel, verbose)
    self.class_names = class_names
    if vocabulary:
      terms = np.array(list(vocabulary.keys()))
      indices = np.array(list(vocabulary.values()))
      self.vocabulary = terms[np.argsort(indices)]
  def explain_instance(self,
                       x,
                       classifier,
                       labels=[1],
                       top_labels=None,
                       num_features=10,
                       num_samples=5000,
                       local_explanation=True,
                       top_words=True):
    # classifier must implement predict_proba
    x = sp.sparse.csr_matrix(x)
    data, ys, distances, mapping = self.__data_labels_distances_mapping(x, classifier.predict_proba, num_samples)
    if not self.class_names:
      self.class_names = map(str, range(ys[0].shape[0]))
    ret_exp = Explanation(vocabulary=self.vocabulary, class_names=self.class_names)
    ret_exp.predict_proba = ys[0]
    map_exp = lambda exp: [(mapping[x[0]], x[1]) for x in exp]
    if top_labels:
      labels = np.argsort(ys[0])[-top_labels:]
    for label in labels:
      point = ys[0, label]
      if local_explanation:
        ret_exp.local_exp[label] = map_exp(self.base.explain_instance_with_data(data, ys, distances, label, num_features))
      if top_words:
        exp =   map_exp(self.base.explain_instance_with_data(data, ys, distances, label, num_features, all_features=True))
        sign = lambda z : 1 if z > 0 else -1
        #pos = 1 if np.argmax(ys[0]) == label else -1
        ret_exp.top_pos[label] = [x for x in exp if sign(x[1]) == 1][:num_features]
        ret_exp.top_neg[label] = [x for x in exp if sign(x[1]) == -1][:num_features]
    return ret_exp
  def __data_labels_distances_mapping(self, x, classifier_fn, num_samples):
      distance_fn = lambda x : sklearn.metrics.pairwise.cosine_distances(x[0],x)[0] * 100
      features = x.nonzero()[1]
      vals = np.array(x[x.nonzero()])[0]
      doc_size = len(sp.sparse.find(x)[2])                                    
      sample = np.random.randint(1, doc_size, num_samples - 1)                             
      data = np.zeros((num_samples, len(features)))    
      inverse_data = np.zeros((num_samples, len(features)))                                         
      data[0] = np.ones(doc_size)
      inverse_data[0] = vals
      features_range = range(len(features)) 
      for i, s in enumerate(sample, start=1):                                               
          active = np.random.choice(features_range, s, replace=False)                       
          data[i, active] = 1
          inverse_data[i, active] = vals[active]
      sparse_inverse = sp.sparse.lil_matrix((inverse_data.shape[0], x.shape[1]))
      sparse_inverse[:, features] = inverse_data
      sparse_inverse = sp.sparse.csr_matrix(sparse_inverse)
      mapping = features
      labels = classifier_fn(sparse_inverse)
      distances = distance_fn(sparse_inverse)
      return data, labels, distances, mapping

class LimeBase:
  def __init__(self,
               kernel_fn,
               verbose=True):
    # TODO
    self.kernel_fn = kernel_fn
    self.verbose = verbose
  def generate_lars_path(self, weighted_data, weighted_labels, positive=False):
    # Adding intercept columns
    #X = np.hstack((100 * np.ones((weighted_data.shape[0],1)), weighted_data))
    X = weighted_data
    alphas, active, coefs = linear_model.lars_path(X, weighted_labels, method='lasso', verbose=False, positive=positive)
    return alphas, coefs
  def explain_instance_with_data(self, data, labels, distances, label, num_features, positive=False, all_features=False):
    weights = self.kernel_fn(distances)
    weighted_data = data * weights[:, np.newaxis]
    mean = np.mean(labels[:, label]) 
    shifted_labels = labels[:, label] - mean
    if self.verbose:
      print 'Explaining from mean=', mean
    weighted_labels = shifted_labels * weights
    used_features = range(weighted_data.shape[1])
    if not all_features:
      nonzero = used_features
      alpha = 1
      alphas, coefs = self.generate_lars_path(weighted_data, weighted_labels, positive=positive)
      for i in range(len(coefs.T) - 1, 0, -1):
        nonzero = coefs.T[i].nonzero()[0]
        if len(nonzero) <= num_features:
            chosen_coefs = coefs.T[i]
            alpha = alphas[i]
            break
      used_features = nonzero

    debiased_model = linear_model.Ridge(alpha=0, fit_intercept=False)
    debiased_model.fit(weighted_data[:, used_features], weighted_labels)
    if self.verbose:
      print 'Prediction_local', debiased_model.predict(data[0, used_features].reshape(1, -1)) + mean, 'Right:', labels[0, label]
    return sorted(zip(used_features,
                  debiased_model.coef_),
                  key=lambda x:np.abs(x[1]), reverse=True)

