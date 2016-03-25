"""
Explanation class, with visualization functions.
"""
import os
import os.path
import json
import numpy as np
import re
import string
import itertools

def id_generator(size=15):
    """Helper function to generate random div ids. This is useful for embedding
    HTML into ipython notebooks."""
    chars = list(string.ascii_uppercase + string.digits)
    return ''.join(np.random.choice(chars, size, replace=True))

class Explanation(object):
    """Object returned by explainers."""
    def __init__(self, indexed_string, class_names=None):
        """Initializer.

        Args:
            indexed_string: lime_text.IndexedString, original string
            class_names: list of class names
        """
        self.indexed_string = indexed_string
        self.class_names = class_names
        self.local_exp = {}
        self.top_labels = None
        self.predict_proba = None
    def available_labels(self):
        """Returns the list of labels for which we have any explanations."""
        if self.top_labels:
            return self.top_labels
        return self.local_exp.keys()
    def as_list(self, label=1, positions=False):
        """Returns the explanation as a list.

        Args:
            label: desired label. If you ask for a label for which an
                explanation wasn't computed, will throw an exception.
            positions: if True, also return word positions

        Returns:
            list of tuples (word, weight), or (word, positions, weight) if
            positions=True. Word is a string, positions is a numpy array and
            weight is a float.
        """
        exp = self.local_exp[label]
        if positions:
            exp = [(self.indexed_string.word(x[0]),
                    self.indexed_string.string_position(x[0]),
                    x[1]) for x in exp]
        else:
            exp = [(self.indexed_string.word(x[0]), x[1]) for x in exp]
        return exp
    def as_map(self):
        """Returns the map of explanations.

        Returns:
            Map from label to list of tuples (feature_id, weight).
        """
        return self.local_exp

    def as_pyplot_figure(self, label=1):
        """Returns the explanation as a pyplot figure.

        Will throw an error if you don't have matplotlib installed
        Args:
            label: desired label. If you ask for a label for which an
                   explanation wasn't computed, will throw an exception.

        Returns:
            pyplot figure (barchart).
        """
        import matplotlib.pyplot as plt
        exp = self.as_list(label)
        fig = plt.figure()
        vals = [x[1] for x in exp]
        names = [x[0] for x in exp]
        vals.reverse()
        names.reverse()
        colors = ['green' if x > 0 else 'red' for x in vals]
        pos = np.arange(len(exp)) + .5
        plt.barh(pos, vals, align='center', color=colors)
        plt.yticks(pos, names)
        plt.title('Local explanation for class %s' % self.class_names[label])
        return fig


    def show_in_notebook(self, labels=None, predict_proba=True, text=True):
        """Shows html explanation in ipython notebook.

           See as_html for parameters.
           This will throw an error if you don't have IPython installed"""
        from IPython.core.display import display, HTML
        display(HTML(self.as_html(labels, predict_proba, text)))

    def save_to_file(self, file_path, labels=None, predict_proba=True, text=True):
        """Saves html explanation to file. See as_html for paramaters.

        Params:
            file_path: file to save explanations to
        """
        file_ = open(file_path, 'w')
        file_.write(self.as_html(labels, predict_proba, text))
        file_.close()


    def as_html(self, labels=None, predict_proba=True, text=True):
        """Returns the explanation as an html page.

        Args:
            labels: desired labels to show explanations for (as barcharts).
                If you ask for a label for which an explanation wasn't computed,
                will throw an exception. If None, will show explanations for all
                available labels.
            predict_proba: if true, add  barchart with prediction probabilities
                for the top classes.
            text: If True, include the text in the HTML page, with words in the
                explanation highlighted with the appropriate color. If more than
                one label is present, highlights the words according to the
                explanation of labels[0].

        Returns:
            code for an html page, including javascript includes.
        """
        if labels is None:
            labels = self.available_labels()
        this_dir, _ = os.path.split(__file__)
        dthree = open(os.path.join(this_dir, 'd3.min.js')).read()
        lodash = open(os.path.join(this_dir, 'lodash.js')).read()
        exp_js = open(os.path.join(this_dir, 'explanation.js')).read()
        # We embed random ids in the div and svg names, in case multiple of
        # these HTML pages are embedded into an ipython notebook (which would
        # cause interference if they all had the same name)
        random_id = id_generator()
        out = '''<html><head><script>%s </script>
        <script>%s </script>
        <script>%s </script>
        </head>
        <body>
        ''' % (dthree, lodash, exp_js)
        out += '''
        <div id="mychart%s" style="float:left"></div>
        ''' % (random_id)

        if text:
            text = self.indexed_string.raw_string().encode('ascii', 'xmlcharrefreplace')
            # removing < > and & to display in html
            text = re.sub(r'[<>&]', '|', text)
            exp = self.as_list(labels[0], positions=True)
            all_ocurrences = list(itertools.chain.from_iterable(
                [itertools.product([x[0]], x[1], [x[2]]) for x in exp]))
            sorted_ocurrences = sorted(all_ocurrences, key=lambda x: x[1])
            add_after = '</span>'
            added = 0
            for word, position, val in sorted_ocurrences:
                class_ = 'pos' if val > 0 else 'neg'
                add_before = '<span class="%s">' % class_
                idx0 = position + added
                idx1 = idx0 + len(word)
                text = '%s%s%s%s%s' % (text[:idx0],
                                       add_before,
                                       text[idx0:idx1],
                                       add_after,
                                       text[idx1:])
                added += len(add_before) + len(add_after)
            text = re.sub('\n', '<br />', text)
            out += ('<div id="mytext%s"><h3>Text with highlighted words</h3>'
                    '%s</div>' % (random_id, text))
        out += '''
        <script>
        var exp = new Explanation(%s);
        ''' % (json.dumps(self.class_names))

        if predict_proba:
            out += '''
            var svg = d3.select('#mychart%s').append('svg');
            exp.PredictProba(svg, %s);
            ''' % (random_id, json.dumps(list(self.predict_proba)))
        for i, label in enumerate(labels):
            exp = json.dumps(self.as_list(label))
            out += '''
                var svg%d = d3.select('#mychart%s').append('svg');
                exp.ExplainFeatures(svg%d, %d, %s, 'Local explanation', true);
            ''' % (i, random_id, i, label, exp)
        if text is not None:
            out += '''
            var text_div = d3.select('#mytext%s');
            exp.UpdateColors(text_div, %d);
            ''' % (random_id, labels[0])

        out += '</script></body></html>'
        return out

