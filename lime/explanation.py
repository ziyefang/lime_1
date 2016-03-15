"""
Explanation class, with visualization functions.
"""
import os
import os.path
import json
import numpy as np
import matplotlib.pyplot as plt
import re

class Explanation(object):
    """Object returned by explainers."""
    def __init__(self, vocabulary=None, class_names=None):
        """Initializer.

        Args:
            vocabulary: map from word to feature id.
            class_names: list of class names
        """
        self.neighborhood_explanations = {}
        self.vocabulary = vocabulary
        self.class_names = class_names
        self.local_exp = {}
        self.top_pos = {}
        self.top_neg = {}
        self.predict_proba = None
    def available_labels(self):
        """Returns the list of labels for which we have any explanations."""
        return set(self.top_pos.keys() + self.top_neg.keys() + self.local_exp.keys())
    def as_list(self, label, explanation='local'):
        """Returns the explanation as a list.

        Args:
            label: desired label. If you ask for a label for which an
                   explanation wasn't computed, will throw an exception.
            explanation: can be 'local' (for local explanations), 'pos' (most
                         positive feature) or 'neg' (most negative features).
                         If you ask for an explanation that wasn't computed,
                         will throw an exception.

        Returns:
            list of tuples (feature_id, weight), or (word, weight) if vocabulary
            is defined.
        """
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
        """Returns the map of explanations.

        Args:
            explanation: can be 'local' (for local explanations), 'pos' (most
                         positive feature) or 'neg' (most negative features).
                         If you ask for an explanation that wasn't computed,
                         will throw an exception.

        Returns:
            Map from label to list of tuples (feature_id, weight).
        """
        if explanation == 'local':
            return self.local_exp
        elif explanation == 'pos':
            return self.top_pos
        elif explanation == 'neg':
            return self.top_neg

    def as_pyplot_figure(self, label, explanation='local'):
        """Returns the explanation as a figure.

        Args:
            label: desired label. If you ask for a label for which an
                   explanation wasn't computed, will throw an exception.
            explanation: can be 'local' (for local explanations), 'pos' (most
                         positive feature) or 'neg' (most negative features).
                         If you ask for an explanation that wasn't computed,
                         will throw an exception.

        Returns:
            pyplot figure (barchart).
        """
        exp = self.as_list(label, explanation)
        fig = plt.figure()
        vals = [x[1] for x in exp]
        names = [x[0] for x in exp]
        vals.reverse()
        names.reverse()
        colors = ['green' if x > 0 else 'red' for x in vals]
        pos = np.arange(len(exp)) + .5
        plt.barh(pos, vals, align='center', color=colors)
        plt.yticks(pos, names)
        if explanation == 'local':
            plt.title('Local explanation for class %s' % self.class_names[label])
        elif explanation == 'pos':
            plt.title('Most positive towards %s' % self.class_names[label])
        elif explanation == 'neg':
            plt.title('Most negative towards %s' % self.class_names[label])
        return fig


    def as_html(self, label=1, include=('predict_proba', 'local', 'pos', 'neg'), text=None):
        """Returns the explanation as an html page.

        Args:
            label: desired label. If you ask for a label for which an
                   explanation wasn't computed, will throw an exception.
            explanation: tuple with desired parts of visualization. Options:
                         'predict_proba': barchart with prediction probabilities
                                          for the top classes.
                         'local': barchart with local explanations
                         'pos': barchart with most positive features
                         'neg': barchart with most negative features
                         any combination of these is accepted.
            text: raw text data. If present, include the text in the HTML page,
                  with words in the explanation highlighted with the appropriate
                  color.

        Returns:
            code for an html page, including javascript includes.
        """
        this_dir, _ = os.path.split(__file__)
        dthree = open(os.path.join(this_dir, 'd3.min.js')).read()
        lodash = open(os.path.join(this_dir, 'lodash.js')).read()
        exp_js = open(os.path.join(this_dir, 'explanation.js')).read()
        out = '''<html><head><script>%s </script>
        <script>%s </script>
        <script>%s </script>
        </head>
        <body>
        ''' % (dthree, lodash, exp_js)
        out += '''
        <div id="mychart%d" style="display:flex; justify-content:space-between;"></div>
        ''' % (label)

        if text is not None:
            if 'pos' in include:
                for word, _ in self.as_list(label, 'pos'):
                    text = re.sub(r'(\W|^)(%s)(\W|$)' % word,
                                  r'\1<span class="pos">\2</span>\3',
                                  text)
            if 'neg' in include:
                for word, _ in self.as_list(label, 'neg'):
                    text = re.sub(r'(\W|^)(%s)(\W|$)' % word,
                                  r'\1<span class="neg">\2</span>\3',
                                  text)
            elif 'local' in include and 'pos' not in include:
                for word, val in self.as_list(label, 'local'):
                    class_ = 'pos' if val > 0 else 'neg'
                    text = re.sub(r'(\W|^)(%s)(\W|$)' % word,
                                  r'\1<span class="%s">\2</span>\3' % class_,
                                  text)
            text = re.sub('\n', '<br />', text)
            out += '<div id="mytext%d"><h3>Text with highlighted words</h3>%s</div>' % (label, text)
        out += '''
        <script>
        var exp = new Explanation(%s);
        ''' % (json.dumps(self.class_names))

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
        if text is not None:
            out += '''
            var text_div = d3.select('#mytext%d');
            exp.UpdateColors(text_div, %d);
            ''' % (label, label)

        out += '</script></body></html>'
        return out

