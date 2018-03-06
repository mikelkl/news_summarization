import numpy as np
import pathos.multiprocessing as multiprocessing
import random
from sklearn import preprocessing, svm
from sklearn.utils import shuffle
from newssum.evaluation import Rouge
from newssum.feature_extraction.sentence_feature import SentenceFeature
from nltk import word_tokenize


class InfoFilter():
    def __init__(self, X, y, scaler, vectorizer, matrix) -> None:
        """
        :param X: {array-like, sparse matrix}, shape (n_samples, n_features)
        :param y: array-like, shape (n_samples,)
        :param scaler: sklearn.preprocessing.*Scaler
        :param vectorizer: sklearn.feature_extraction.text.CountVectorizer
        :param matrix: Document-term matrix
        """
        self.clf = self.fit(X, y)
        self.scaler = scaler
        self.vectorizer = vectorizer
        self.matrix = matrix

    def fit(self, X, y):
        """
        :param X: {array-like, sparse matrix}, shape (n_samples, n_features)
        :param y: array-like, shape (n_samples,)
        :return: sklearn.svm.LinearSVC
        """
        print("Training...")
        clf = svm.LinearSVC()
        clf.fit(X, y)

        return clf

    def evaluate_classifier(self, X, y):
        """
        :param X: {array-like, sparse matrix}, shape (n_samples, n_features)
        :param y: array-like, shape (n_samples,)
        """
        print("Evaluating classifier...")
        acc = self.clf.score(X, y)

        print("Classifier accuracy: {}".format(acc))

    def get_best_sents(self, parser, **kwargs):
        """
        :param parser: newssum.parser.StoryParser
        :param kwargs:
            See below

        :Keyword Arguments:
            * *w_threshod* (``int``)
                word length threshold
            * *s_threshod* (``int``)
                sentence length threshold
        :return: list or null string
        """
        sent_feature = SentenceFeature(parser)
        X = sent_feature.get_all_features(self.vectorizer, self.matrix)
        X = self.scaler.transform(X)
        # local_norm = ["length", "quote", "core_rank_score", "page_rank_rel"]
        # global_norm = [0, 1, 2, 5, 6, 7, 9, 10]
        # X = sent_feature.doc_level_normalize(X, local_norm)
        # X_global = X[:, global_norm]
        # X_global = self.scaler.transform(X_global)
        # X_local = np.delete(X, global_norm, axis=1)
        # X = np.concatenate((X_local, X_global), axis=1)
        y = self.clf.predict(X)
        pos_i = np.where(y == 1)[0]
        if pos_i.size != 0:
            selected_sents = []
            if "w_threshod" in kwargs:
                w_threshold = kwargs["w_threshod"]
                w_length = 0
                for i in pos_i:
                    sent = parser.sents[i]
                    selected_sents.append(sent)
                    tokenized_sent = word_tokenize(sent)
                    w_length += len(tokenized_sent)
                    if w_length >= w_threshold:
                        break
            elif "s_threshod" in kwargs:
                s_threshod = kwargs["s_threshod"]
                for i in pos_i[:s_threshod]:
                    sent = parser.sents[i]
                    selected_sents.append(sent)
        else:
            selected_sents = ""

        return selected_sents

    def evaluate_summarizer(self, parsers, **kwargs):
        """
        :param parsers: list
            List stores newssum.parser.StoryParser.
        :param kwargs:
            See below

        :Keyword Arguments:
            * *w_threshod* (``int``)
                word length threshold
            * *s_threshod* (``int``)
                sentence length threshold
        """
        print("Evaluating summarizer...")
        p = multiprocessing.ProcessingPool(multiprocessing.cpu_count() - 2)

        def get_rouges(parser):
            print("Get rouges...")
            print(parser)
            selected_sents = self.get_best_sents(parser, **kwargs)
            rouge = Rouge(selected_sents, parser.highlights).get_rouge()

            return rouge

        rouges = p.map(get_rouges, parsers)
        # p.close()
        # p.join()

        avg_rouge = Rouge.cal_avg_rouge(rouges)
        Rouge.print("InfoFilter", avg_rouge)

    @staticmethod
    def get_train_val_test_data(train_parsers, val_parsers, test_parsers):
        """
        :param train_parsers: list
            List stores newssum.parser.StoryParser for train data.
        :param val_parsers: list
            List stores newssum.parser.StoryParser for validation data.
        :param test_parsers: list
            List stores newssum.parser.StoryParser for test data.
        :return: (array, array, array, array, array, array,
                    sklearn.preprocessing.*Scaler, sklearn.feature_extraction.text.CountVectorizer,
                    array)
        """
        scaler = preprocessing.StandardScaler()
        vectorizer, matrix = SentenceFeature.get_global_term_freq(train_parsers)

        p = multiprocessing.ProcessingPool(multiprocessing.cpu_count() - 2)
        # local_norm = ["length", "quote", "core_rank_score", "page_rank_rel"]
        # global_norm = [0, 1, 2, 5, 6, 7, 9, 10]

        def get_data(parser):
            print("Get data...")
            print(parser)
            sent_feature = SentenceFeature(parser)
            y = sent_feature.label_sents()
            X = sent_feature.get_all_features(vectorizer, matrix)
            # X = sent_feature.doc_level_normalize(X, local_norm)

            # X_initial = sent_feature.get_all_features(vectorizer, matrix)
            # pos_sents_i, neg_sents_i = sent_feature.label_sents()
            # X =[]
            # y=[]
            # for i in pos_sents_i:
            #     X.append(X_initial[i][:])
            #     y.append(1)
            # for i in neg_sents_i:
            #     # X += X_initial[i][:]
            #     X.append(X_initial[i][:])
            #     y.append(-1)
            return (X, y)

        def preprocess(parsers, mode="train"):
            print("Preprocess for {} data".format(mode))
            data = p.map(get_data, parsers)
            # p.close()
            # p.join()

            X_initial = []
            y_initial = []
            for X, y in data:
                X_initial.append(X)
                y_initial.append(y)
            X_initial = np.concatenate(X_initial)
            y_initial = np.concatenate(y_initial)
            if mode == "train":
                X_initial = scaler.fit_transform(X_initial)
                # X_global = X_initial[:,global_norm]
                # X_global = scaler.fit_transform(X_global)
                # X_local = np.delete(X_initial, global_norm, axis=1)
                # X = np.concatenate((X_local, X_global), axis=1)
            else:
                # X_initial = X_initial[3:]
                X_initial = scaler.transform(X_initial)
                # X_global = X_initial[:, global_norm]
                # X_global = scaler.transform(X_global)
                # X_local = np.delete(X_initial, global_norm, axis=1)
                # X = np.concatenate((X_local, X_global), axis=1)
                # y_initial = y_initial[3:]
            # X_train_bool = X_train[:, [1, 2]]
            # X_train_non_bool = np.delete(X_train, [1, 2], axis=1)
            # X_train_non_bool = scaler.fit_transform(X_train_non_bool)
            # X_train = np.concatenate((X_train_bool, X_train_non_bool), axis=0)

            # randomly select same size of negative samples
            # as positive samples to avoid imbalance
            X = X_initial.copy()
            y = y_initial.copy()
            y_pos_i = np.where(y == 1)[0]
            y_pos = y[y_pos_i]
            y = np.delete(y, y_pos_i)
            X_pos = X[y_pos_i]
            X = np.delete(X, y_pos_i, axis=0)

            num_pos = len(y_pos_i)
            X, y = shuffle(X, y, random_state=0)
            y_neg = y[:num_pos]
            X_neg = X[:num_pos]

            y = np.concatenate((y_pos, y_neg), axis=0)
            X = np.concatenate((X_pos, X_neg), axis=0)
            X, y = shuffle(X, y, random_state=1)

            return X, y

        X_train, y_train = preprocess(train_parsers)
        X_val, y_val = preprocess(val_parsers, mode="val")
        X_test, y_test = preprocess(test_parsers, mode="test")
        np.savetxt("X.csv", X_train, delimiter=",")
        np.savetxt("y.csv", y_train, delimiter=",")

        return (X_train, y_train, X_val, y_val, X_test, y_test, scaler, vectorizer, matrix)


if __name__ == "__main__":
    # Debug code in development phrase,
    # will remove in formal version.
    from newssum.parsers import StoryParser
    from newssum.definitions import ROOT_DIR
    import pickle as pk
    import os

    OUTPUT_DIR = os.path.join(os.path.dirname(ROOT_DIR), "data")


    def read_file(data_dir, file):
        files = os.listdir(data_dir)
        files = shuffle(files, random_state=0)
        parsers = []
        for f in files[:200]:
            parser = StoryParser.from_file(data_dir + f)
            parsers.append(parser)
        pk.dump(parsers, open(os.path.join(OUTPUT_DIR, file), "wb"))

        return parsers


    print("Reading training files...")
    # train_parsers = read_file('C:/KangLong/Data/cnn/stories/data/train/', "mini_train.pk")
    train_parsers = pk.load(open(os.path.join(OUTPUT_DIR, "mini_train.pk"), "rb"))
    print("Reading val files...")
    val_parsers = pk.load(open(os.path.join(OUTPUT_DIR, "mini_val.pk"), "rb"))
    # read_file('C:/KangLong/Data/cnn/stories/data/validation/', "mini_val.pk")
    print("Reading test files...")
    # read_file('C:/KangLong/Data/cnn/stories/data/test/', "mini_test.pk")
    test_parsers = pk.load(open(os.path.join(OUTPUT_DIR, "mini_test.pk"), "rb"))
    data = InfoFilter.get_train_val_test_data(train_parsers, val_parsers, test_parsers)
    pk.dump(data, open(os.path.join(OUTPUT_DIR, "mini_data.pk"), "wb"))
    X_train, y_train, X_val, y_val, X_test, y_test, scaler, vectorizer, matrix = data
    # X_train, y_train, X_val, y_val, X_test, y_test, scaler, vectorizer, matrix = pk.load(
    #     open(os.path.join(OUTPUT_DIR, "mini_data.pk"), "rb"))
    filter = InfoFilter(X_train, y_train, scaler, vectorizer, matrix)
    filter.evaluate_classifier(X_test, y_test)
    filter.evaluate_summarizer(test_parsers, s_threshod=3)

    # test lead-3 method
    # rouges = []
    # for parser in test_parsers:
    #     lead_3 = parser.sents[:3]
    #     rouge = Rouge(lead_3, parser.highlights).get_rouge()
    #     rouges.append(rouge)
    #
    # avg_rouge = Rouge.cal_avg_rouge(rouges)
    # Rouge.print("Lead-3", avg_rouge)
