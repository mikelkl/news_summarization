import numpy as np


class CoOccurrenceMatrix():
    """Co-occurrence Matrix"""

    def __init__(self, text, window_size=6, overspan_sents=True):
        """
        :param text: list
            A 2-dimension list containing preprocessed word,
            e.g.  [['cnnfor', 'time', 'papaci', 'pope', 'franci', 'announc', 'group',
                    'bishop', 'archbishop', 'set', 'cardin', 'world'], ...]
        :param window_size: int, optional
            Specifies the size of the window slided over the document.
        :param overspan_sents: boolean,
            True, an edge between two cooccurring words is created whether the two words belong to the same sentence or not.
            False, an edge between two cooccurring words is only created if the two words belong to the same sentence.
        """
        flat_text = [w for s in text for w in s]  # flatten 2-dimensional list
        if overspan_sents is True:
            self.text = flat_text
        else:
            self.text = text

        self.feature_names = list(set(flat_text))
        self.matrix = self.build_matrix(window_size, overspan_sents)


    def build_matrix(self, window_size, overspan_sents):
        """
        :param window_size: int, optional
            Specifies the size of the window slided over the document.
        :param overspan_sents: boolean,
            True, an edge between two cooccurring words is created whether the two words belong to the same sentence or not.
            False, an edge between two cooccurring words is only created if the two words belong to the same sentence.
        :return: numpy matrix
            A numpy matrix contains co-occurrence information.
        """
        n_features = len(self.feature_names)
        features_i = range(n_features)
        self.vocabulary = dict(zip(self.feature_names, features_i))  # store word-index pair for matrix building

        # build co-occurrence matrix
        matrix = np.zeros((n_features, n_features))

        def transverse_text(text):
            for i, w_s in enumerate(text):
                for w_in_window in text[i + 1:i + window_size]:
                    w_s_matrix_i = self.vocabulary[w_s]
                    w_in_window_i = self.vocabulary[w_in_window]
                    matrix[w_s_matrix_i][w_in_window_i] += 1

        if overspan_sents is True:
            transverse_text(self.text)
        else:
            for sent in self.text:
                transverse_text(sent)

        np.fill_diagonal(matrix, 0)  # fill the main diagonal of the matrix with 0 to avoid self loops
        return matrix
