import matplotlib.pyplot as plt
import networkx as nx
from networkx.readwrite import json_graph
from nltk.tokenize import word_tokenize

from newssum.feature_extraction import CoOccurrenceMatrix
from newssum.models import Graph


class CoreRank():
    def __init__(self, parser, window_size=9, build_on_processed_text=True, overspan_sents=True,
                 weight='weight'):
        """
        :param parser: newssum.parser.StoryParser
        :param window_size: int, optional
            Specifies the size of the window slided over the document.
        :param build_on_processed_text: boolean, optional
            Whether the window should be slided over the (1) processed or the (2) unprocessed text.
        :param overspan_sents: boolean,
            True, an edge between two cooccurring words is created whether the two words belong to the same sentence or not.
            False, an edge between two cooccurring words is only created if the two words belong to the same sentence.
        :param weight: string or None, optional (default='weight')
           The edge attribute that holds the numerical value used as a weight. If None, then each edge has weight 1.
           The degree is the sum of the edge weights adjacent to the node.
        """
        if build_on_processed_text is True:
            self.words = parser.processed_words
        else:
            self.words = parser.unprocessed_words

        self.sents = parser.sents

        self.graph = self._build_graph(window_size, overspan_sents)
        self.scores = self._cal_keywords_score(weight)

    def _build_graph(self, window_size, overspan_sents):
        c_matrix = CoOccurrenceMatrix(self.words, window_size, overspan_sents)
        graph = Graph(c_matrix.feature_names, c_matrix.matrix)

        return graph.G

    def _cal_keywords_score(self, weight):
        """
        Calculate score for each keyword.

        :param weight: string or None
               The edge attribute that holds the numerical value used
               as a weight.  If None, then each edge has weight 1.
               The degree is the sum of the edge weights adjacent to the node.
        :return: dictionary
            A dictionary whose key is word, value is corresponding score,
            e.g. {'sentiment': 102, ...}
        """
        core_n = nx.core_number(self.graph,
                                weight=weight)  # get core number for each vertex considering the weight of each weight
        scores = {}
        for n, nbrs in self.graph.adj.items():
            score = 0
            for nbr in nbrs:
                score += core_n[nbr]
            scores.update({n: score})

        return scores

    def _top_n(self, dict, p):
        """
        Sort input dict and return the top p% items.

        :param dict: dict
        :param p: float, in between [0, 1]
        :return: list
            A list with tuple element which contains word and its score,
            e.g. [('cardin', 851), ...]
        """
        top_n = int(round(len(dict) * p))
        sorted_list = sorted(dict.items(), key=lambda x: x[1], reverse=True)  # sort dict by value

        return sorted_list[:top_n]

    # def get_best_sents(self, p=0.15, w_threshold=125, l=0.6, r=0.6):
    def get_best_sents(self, p=0.25, w_threshold=75, l=0.5, r=0.8):
        """
        Extract best sentences as output summary.

        :param p: float, in between [0, 1], optional
            Retain top p% words as keywords
        :param w_threshold: int, optional
            A soft constrain for output summary length.
        :param l: float, >= 0
            A trade-off parameter in objective function formula.
        :param r: float, > 0
            The scaling factor, adjusts for the fact that the objective function F and the cost of a sentence
            might be expressed in different units and thus not be directly comparable.
        :return: list
            Store output summary as a list of strings.
        """
        self.keywords = dict(self._top_n(self.scores, p))
        num_keywords = len(self.keywords)
        # print(self.scores)
        # print(self.keywords)

        # pre-calculate coverage score and keywords in each sents to improve efficiency
        coverage_score = {}
        keywords_in_sents = {}
        for v, s in enumerate(self.words):
            C = 0  # coverage function value
            keywords = []

            for w in s:
                if w in self.keywords:
                    C += self.keywords[w]  # calculate coverage function
                    keywords.append(w)
            keywords_in_sents.update({v: keywords})
            coverage_score.update({v: C})

        # extract best sentences
        # best_sents_i = [i for i, s in self._top_n(sents_score, 1)]
        # print(sents_score)
        # print(best_sents_i)
        # best_sents = extract_sents_below_threshold(self.sents, best_sents_i, w_threshold)

        best_sents = []  # current summary
        w_length = 0  # number of words in current summary
        F_old = 0  # objective function value of current summary
        C_old = 0  # coverage function value of current summary
        keywords_in_summary = []  # keywords in current summary
        coverage_score_copy = coverage_score.copy()

        # greedy algorithm for Budgeted maximization
        while coverage_score_copy:
            ratios = {}  # contains all the ratio of objective function gain to scaled cost in current iteration
            update_values = {}  # contains all the F_new values and C_new values in current iteration

            # iteratively selects the sentence that maximizes the ratio of objective function gain to scaled cost:
            for v, C in coverage_score_copy.items():
                if keywords_in_sents[v]:
                    keywords_in_temp_summary = list(set(keywords_in_summary + keywords_in_sents[
                        v]))  # number of (unique) keywords contained in the current temp summary
                    D = len(keywords_in_temp_summary) / num_keywords  # calculate diversity reward function
                    C_new = C_old + C  # update coverage function value for current temp summary
                    F_new = C_new + l * D  # calculate objective function
                    # TODO check if the cost is the length of keywords in a sent or length of original words of that sent
                    cost = len(word_tokenize(self.sents[v]))  # cost of sentence v
                    # cost = len(self.words[v])  # cost of sentence v
                    ratio = (F_new - F_old) / (cost ** r)
                    ratios.update({v: ratio})
                    update_values.update({v: [F_new, C_new]})
                else:
                    pass

            if ratios:
                max_v = max(ratios, key=lambda k: ratios[k])  # find the sent index with the max ratio
                keywords_in_summary += keywords_in_sents[max_v]  # update
                F_old = update_values[max_v][0]  # update
                C_old = update_values[max_v][1]  # update
                del coverage_score_copy[max_v]  # remove the current max sent from subsequent iterations

                # check whether the constrain (summary size) is violated
                best_sents.append(self.sents[max_v])
                tokenized_sent = word_tokenize(self.sents[max_v])
                w_length += len(tokenized_sent)
                if w_length >= w_threshold:
                    # print("w_length: " + str(w_length))
                    # print(max_v)
                    break
            else:
                # print("coverage_score_copy")
                # print(coverage_score_copy)
                # print("ratios")
                # print(ratios)
                break

        return best_sents

    def plot_graph(self, g=None):
        """
        Visualize graph.

        :param g: NetworkX graph
        """
        if g is None:
            g = self.graph
        labels = [node for node in g.nodes() if node in self.keywords]
        # for node in g.nodes():
        #     if node in self.keywords:
        #         # set the node name as the key and the label as its value
        #         labels[node] = node
        # set the argument 'with labels' to False so you have unlabeled graph
        nx.draw(g, with_labels=False, node_color="b", node_size=25, width=0.25)
        # Now only add labels to the nodes you require
        nx.draw_networkx(g, with_labels=False, nodelist=labels, width=0.25, font_size=16, font_color='r', node_size=50)
        plt.show()

    def plot_k_core(self, k=None):
        """
        Plot k-core subgraph.

        :param k: int, optional
            The order of the core. If not specified return the main core.
        """
        subgraph = nx.k_core(self.graph, k)
        self.plot_graph(subgraph)

    def graph_to_json(self):
        return json_graph.node_link_data(self.graph)
