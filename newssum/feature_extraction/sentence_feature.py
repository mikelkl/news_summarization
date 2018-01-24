import re
import string
import networkx as nx
import numpy as np
from nltk import word_tokenize
from evaluation import Rouge
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sumy.utils import get_stop_words


class SentenceFeature():

    def __init__(self, parser) -> None:
        self.paragrahs = parser.paragrahs
        self.sents = parser.sents
        self.sents_i = list(range(len(self.sents)))  # list contains index of each sentence
        self.refs = parser.highlights
        self.processed_words = parser.processed_words
        self.unprocessed_words = parser.unprocessed_words

    def label_sents(self):
        """
        Label each sentence set by greedily selecting the sentence with the maximum relative importance.

        :return:
        """
        concat_ref = " ".join(self.refs)  # concatenate reference summaries as a whole string

        S_i = []  # list contains (index, importance score) tuple for selected (candidate) summary sentence
        summary_length_constraint = len(self.refs)
        sents_i = self.sents_i.copy()

        # greedily add the optimum sentence to candidate summary
        while sents_i:
            s_max = (None, None)  # tuple stores max (index, relative importance) pair at current iteration of 2nd loop
            for s_i in sents_i:
                if S_i:
                    # action for 2nd and subsequent iteration

                    s_min = (None, None)  # tuple stores min (index, relative importance)
                    for s_prime_i, s_prime_importance in S_i:
                        # compute groundtruth of relative sentence importance
                        relative_importance = Rouge(self.sents[s_i] + self.sents[s_prime_i], concat_ref).get_rouge_2()[
                                                  "r"] - s_prime_importance

                        # select the sentence index with min (index, relative importance)
                        # pair at current iteration of 3rd loop
                        if s_min[1] is None or s_min[1] > relative_importance:
                            s_min = (s_i, relative_importance)

                    # select the sentence index with max (index, relative importance)
                    # pair at current iteration of 2nd loop
                    if s_max[1] is None or s_max[1] < s_min[1]:
                        s_max = s_min
                else:
                    # action for 1st iteration

                    # compute groundtruth of sentence importance
                    importance = Rouge(self.sents[s_i], concat_ref).get_rouge_2()["r"]

                    # select the sentence index with max (index, relative importance)
                    # pair at current iteration of 2nd loop
                    if s_max[1] is None or s_max[1] < importance:
                        s_max = (s_i, importance)

            s_max_i, importance = s_max

            # discard sentences does'nt contribute to summary
            if importance <= 0:
                break

            S_i.append(s_max)
            del sents_i[sents_i.index(s_max_i)]  # remove the current max sent from subsequent iterations

            # break the 1st loop when constrain is satisfied
            num_S = len(S_i)
            if num_S == summary_length_constraint:
                # retain sentences with same relative importance with the
                # last selected sentence even exceed length constrain
                if num_S >= 2 and S_i[-1][1] == S_i[-2][1]:
                    continue

                else:
                    break

        # for i, v in S_i:
        #     print("{}: {}: {}".format(i, v, sents[i]))

        pos_sents_i = [i for i, _ in S_i]
        neg_sents_i = [i for i in sents_i]
        return (pos_sents_i, neg_sents_i)

    # def label_sents(self, sents, refs):
    #     """
    #     Label each sentence set by ranking sentence with the maximum importance.
    #
    #     :param sents:
    #     :param refs:
    #     :return:
    #     """
    #     concat_ref = " ".join(refs)
    #     rouge_2_precisions = {}
    #     for i, sent in enumerate(sents):
    #         rouge = Rouge(sent, concat_ref)
    #         rouge_2_precision = rouge.get_rouge_2()["p"]
    #         rouge_2_precisions.update({i:rouge_2_precision})
    #
    #     sorted_list = sorted(rouge_2_precisions.items(), key=lambda x: x[1], reverse=True)  # sort dict by value
    #     print(sorted_list)
    #     for k,v in sorted_list:
    #         print("{}: {}\n".format(sents[k], v))

    def _get_doc_first(self, sent_i):
        """
        :param sent_i: int
            Index of a sentence
        :return: int
            1, input sentence is the first sentence of a document.
            0, input sentence is not the first sentence of a document.
        """
        return int(sent_i == 0)

    def _get_para_first(self, sent_i):
        """
        :param sent_i: int
            Index of a sentence
        :return: int
            1, input sentence is the first sentence of a paragraph
            0, input sentence is not the first sentence of a paragraph.
        """
        for paragraph in self.paragrahs:
            if self.sents[sent_i] == paragraph[0]:
                return 1
            else:
                return 0

    def _get_length(self, sent_i):
        """
        :param sent_i: int
            Index of a sentence
        :return: int
            The number of words in a sentence
        """
        return len(self.unprocessed_words[sent_i])

    def _get_quote(self, sent_i):
        """
        :param sent_i: int
            Index of a sentence
        :return: int
            The number of quoted words in input sentence.
        """

        # find the paragraph containing the input sentence, concatenate the paragraph as a singe string
        in_paragraph = None
        sent = self.sents[sent_i]
        for paragraph in self.paragrahs:
            if sent in paragraph:
                in_paragraph = " ".join(paragraph)
                break

        # get the position of input sentence in the paragraph
        sent_start_i_in_paragraph = in_paragraph.find(sent)
        sent_end_i_in_paragraph = sent_start_i_in_paragraph + len(sent)

        quote = 0  # stores number of quoted words in input sentence.
        quotes_i = [m.start() for m in re.finditer('"', in_paragraph)]  # get indices of all quotes in target paragraph
        if quotes_i:
            quotes_i = iter(quotes_i)  # facilitate iterating two elements at same iteration
            for quote_i in quotes_i:  # # get position of 1st quote of a pair of quotes
                next_quote_i = next(quotes_i)  # get position of 2nd quote of a pair of quotes

                # 6 kind of partition using quotes for a sentence
                if quote_i <= sent_start_i_in_paragraph:
                    # 1. " ---------- "
                    if next_quote_i >= sent_end_i_in_paragraph:
                        quote += len(self.unprocessed_words[sent_i])
                    # 2. " -----"-----
                    elif next_quote_i > sent_start_i_in_paragraph:
                        quote_part = in_paragraph[sent_start_i_in_paragraph:next_quote_i].translate(
                            str.maketrans('', '', string.punctuation))
                        quote += len(word_tokenize(quote_part))
                    # 3. " " ----------
                    else:
                        quote += 0
                elif quote_i < sent_end_i_in_paragraph:
                    # 4. -----"----- "
                    if next_quote_i >= sent_end_i_in_paragraph:
                        quote_part = in_paragraph[quote_i:sent_end_i_in_paragraph].translate(
                            str.maketrans('', '', string.punctuation))
                        quote += len(word_tokenize(quote_part))
                    # 5. ---"----"---
                    else:
                        quote_part = in_paragraph[quote_i:next_quote_i].translate(
                            str.maketrans('', '', string.punctuation))
                        quote += len(word_tokenize(quote_part))
                # 6. ---------- " "
                else:
                    quote += 0
        else:
            quote += 0

        return quote

    def get_surface_features(self, sents_i=None):
        """
        Surface features are based on structure of documents or sentences.

        :param sents_i: list or int, optional
            list contains multiple sentence indices
            int indicate a single sentence index
        :return: list
            1-dimensional list consists of position, doc_first, para_first, length and quote features for int sents_i parameter
            2-dimensional list consists of position, doc_first, para_first, length and quote features for list sents_i parameter
        """

        # solely get surface features for unlabeled data
        if sents_i is None:
            sents_i = self.sents_i

        def get_features(sent_i):
            position = sent_i  # get 1/sentence no
            doc_first = self._get_doc_first(sent_i)
            para_first = self._get_para_first(sent_i)
            length = self._get_length(sent_i)
            quote = self._get_quote(sent_i)
            return [position, doc_first, para_first, length, quote]

        surface_features = []
        if type(sents_i) is list:
            # get surface features for multiple samples for labeled data
            for sent_i in sents_i:
                surface_feature = get_features(sent_i)
                surface_features.append(surface_feature)
        else:
            # get surface features for single sample for labeled data
            surface_features = get_features(sents_i)

        return surface_features

    def _get_stopwords_ratio(self, sent_i):
        """
        :param sent_i: int
            Index of a sentence
        :return: float, in between [0, 1]
            Stop words ratio of s
        """
        words_num = len(self.unprocessed_words[sent_i])
        non_stopwords_num = len(self.processed_words[sent_i])
        stopwords_ratio = (words_num - non_stopwords_num) / words_num
        return stopwords_ratio

    def _get_avg_term_freq(self, sent_i, vectorizer, X):
        """
        :param sent_i: int
            Index of a sentence
        :param vectorizer: sklearn.feature_extraction.text.CountVectorizer
        :param X: array, [n_samples, n_features]
            Document-term matrix.
        :return: float
            Average Term Frequency
        """
        X_array = X.toarray()
        unprocessed_words = self.unprocessed_words[sent_i]
        total_TF = 0
        count = 0

        for w in unprocessed_words:
            w_i_in_array = vectorizer.vocabulary_.get(w)  # map from word to column index
            if w_i_in_array:
                total_TF += np.sum(X_array[:, w_i_in_array])
                count += 1

        avg_TF = total_TF / count

        return avg_TF

    def _get_avg_doc_freq(self, sent_i, vectorizer, X):
        """
        :param sent_i: int
            Index of a sentence
        :param vectorizer: sklearn.feature_extraction.text.CountVectorizer
        :param X: array, [n_samples, n_features]
            Document-term matrix.
        :return: float
            Average Document Frequency
        """
        X_array = X.toarray()
        unprocessed_words = self.unprocessed_words[sent_i]
        total_DF = 0
        count = 0

        for w in unprocessed_words:
            w_i_in_array = vectorizer.vocabulary_.get(w)  # map from word to column index
            if w_i_in_array:
                total_DF += np.count_nonzero(X_array[:, w_i_in_array])
                count += 1

        avg_DF = total_DF / count

        return avg_DF

    def get_content_features(self, sents_i, vectorizer, X):
        # solely get content features for unlabeled data
        if sents_i is None:
            sents_i = self.sents_i

        def get_features(sent_i):
            stop = self._get_stopwords_ratio(sent_i)
            TF = self._get_avg_term_freq(sent_i, vectorizer, X)
            DF = self._get_avg_doc_freq(sent_i, vectorizer, X)
            return [stop, TF, DF]

        content_features = []
        if type(sents_i) is list:
            # get surface features for multiple samples for labeled data
            for sent_i in sents_i:
                content_feature = get_features(sent_i)
                content_features.append(content_feature)
        else:
            # get surface features for single sample for labeled data
            content_features = get_features(sents_i)

        return content_features

    def _cal_cosine_similarity(self, documents):
        """
        :param documents: list
        :return: float, in between [0, 1]
        """
        tfidf_vectorizer = TfidfVectorizer()
        try:
            tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
            similarity = cosine_similarity(tfidf_matrix[0, :], tfidf_matrix[1, :])[0][0]
        except ValueError:
            if documents[0] == documents[1]:
                similarity = 1.0
            else:
                similarity = 0.0

        return similarity

    def _get_first_rel_doc(self, sent_i):
        """
        :param sent_i: int
            Index of a sentence
        :return: float
            Similarity with the first sentence in the document
        """
        first_sent_doc = self.sents[0]
        sent = self.sents[sent_i]

        relevance = self._cal_cosine_similarity([first_sent_doc, sent])

        return relevance

    def _get_first_rel_para(self, sent_i):
        """
        :param sent_i: int
            Index of a sentence
        :return: float
            Similarity with the first sentence in the paragraph
        """

        # find the paragraph containing the input sentence, concatenate the paragraph as a singe string
        first_sent_para = None
        sent = self.sents[sent_i]
        for paragraph in self.paragrahs:
            if sent in paragraph:
                first_sent_para = paragraph[0]
                break

        relevance = self._cal_cosine_similarity([first_sent_para, sent])

        return relevance

    def _page_rank_rel(self, thres=0.1):
        """
        PageRank value of the sentence based on the sentence map

        :param thres: int
            Every two sentences are regarded relevant if their similarity is above a threshold.
        :return: dict
            Dictionary of index nodes with PageRank as value.
        """
        G = nx.Graph()

        # Build a sentence map.
        # Every two sentences are regarded relevant if their similarity is above a threshold.
        # Every two relevant sentences are connected with a unidirectional link.
        for i in self.sents_i[:-2]:
            for j in self.sents_i[i + 1:]:
                sim = sent_feature._cal_cosine_similarity([sent_feature.sents[i], sent_feature.sents[j]])
                if sim > thres:
                    G.add_edge(i, j)

        pr = nx.pagerank(G)

        return pr

    def get_relevance_features(self, sents_i):
        """
        Relevance features are incorporated to exploit inter-sentence relationships.

        :param sents_i: list or int, optional
            list contains multiple sentence indices
            int indicate a single sentence index
        :return: list
            1-dimensional list consists of first_rel_doc, first_rel_para and page_rank_rel features for int sents_i parameter
            2-dimensional list consists of first_rel_doc, first_rel_para and page_rank_rel features for list sents_i parameter
        """

        # solely get relevance features for unlabeled data
        if sents_i is None:
            sents_i = self.sents_i

        try:
            self.pr
        except AttributeError:
            self.pr = self._page_rank_rel()

        def get_features(sent_i):
            first_rel_doc = self._get_first_rel_doc(sent_i)
            first_rel_para = self._get_first_rel_para(sent_i)
            page_rank_rel = self.pr.get(sent_i, 0)
            return [first_rel_doc, first_rel_para, page_rank_rel]

        relevance_features = []
        if type(sents_i) is list:
            # get surface features for multiple samples for labeled data
            for sent_i in sents_i:
                relevance_feature = get_features(sent_i)
                relevance_features.append(relevance_feature)
        else:
            # get surface features for single sample for labeled data
            relevance_features = get_features(sents_i)

        return relevance_features

    def get_all_features(self, vectorizer, X, sents_i=None):
        # get all feature for unlabeled data
        if sents_i is None:
            sents_i = self.sents_i

        all_features = []
        for sent_i in sents_i:
            # surface_features = self.get_surface_features(sent_i)
            content_features = self.get_content_features(sent_i, vectorizer, X)
            # relevance_features = self.get_relevance_features(sent_i)
            pass
            # all_feature = surface_features + content_features + event_features + relevance_features
            # all_features.append(all_feature)

        return all_features

    @staticmethod
    def get_global_term_freq(parsers):
        """
        :param parsers: newssum.parser.StoryParser
        :return: tuple, (vectorizer, X)
            vectorizer, sklearn.feature_extraction.text.CountVectorizer.
            X, Document-term matrix.
        """
        vectorizer = CountVectorizer(stop_words=get_stop_words("english"))
        if type(parsers) is list:
            corpus = [parsers.body for parser in parsers]
        else:
            corpus = [parsers.body]
        X = vectorizer.fit_transform(corpus)

        return (vectorizer, X)


if __name__ == "__main__":
    from parsers import StoryParser

    data_dir = 'C:/KangLong/Data/cnn/stories/data/train/'
    parser = StoryParser.from_file(data_dir + "c7af94074d86535e5c02e1199946ac722585b0ac.story")
    vectorizer, X = SentenceFeature.get_global_term_freq(parser)
    sent_feature = SentenceFeature(parser)
    # labeled_data = sent_feature.label_sents()
    # print(labeled_data)
    sent_feature.get_all_features(vectorizer, X)
    # for i in sent_feature.sents_i[:-2]:
    #     for j in sent_feature.sents_i[i+1:]:
    #         sim = sent_feature._cal_cosine_similarity([sent_feature.sents[i], sent_feature.sents[j]])
    #         print(sim)
    # pr = sent_feature._page_rank_rel()
    # print(len(pr))
    # print(len(sent_feature.sents_i))