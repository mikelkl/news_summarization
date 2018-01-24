import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


class Graph():
    """Co-occurrence network"""

    def __init__(self, features, matrix):
        """
        :param features: list
            A unique word list.
        :param matrix: numpy matrix
            A numpy matrix contains co-occurrence information.
        """
        self.featrues = features
        self.matrix = matrix
        self.G = self.build_graph()

    def build_graph(self):
        """
        :return: Networkx graph
            A word co-occurrence based networkx Graph.
        """
        G = nx.Graph()
        G.add_nodes_from(self.featrues)

        # construct weighted edge tuples, e.g. ('birthplac', 'men', 1.0)
        rows_i, cols_i = np.nonzero(self.matrix)  # get index of non-zero element in co-occurrence matrix
        vals = self.matrix[rows_i, cols_i]  # get value of non-zero element in co-occurrence matrix
        rows = [self.featrues[i] for i in rows_i]  # get 1st word based on row index
        cols = [self.featrues[i] for i in cols_i]  # get 2rd word based on column index
        weighted_edges = [tuple(x) for x in zip(rows, cols, vals)]  # build trigram tuple
        G.add_weighted_edges_from(weighted_edges)

        return G

    def plot_graph(self, G):
        """
        Visualize built graph.

        :param G: Networkx graph
        """
        nx.draw_networkx(G)
        plt.show()
