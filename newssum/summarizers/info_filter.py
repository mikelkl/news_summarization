import numpy as np
import pathos.multiprocessing as multiprocessing
from sklearn import preprocessing, svm

from newssum.evaluation import Rouge
from newssum.feature_extraction.sentence_feature import SentenceFeature


class InfoFilter():
    def __init__(self, X, Y, scaler, vectorizer, matrix) -> None:
        self.clf = self.fit(X, Y)
        self.scaler = scaler
        self.vectorizer = vectorizer
        self.matrix = matrix

    def fit(self, X, Y):
        print("training")
        lin_clf = svm.LinearSVC()
        lin_clf.fit(X, Y)

        return lin_clf

    def evaluate(self, parsers):
        print("Evaluating...")
        p = multiprocessing.ProcessingPool(multiprocessing.cpu_count() - 2)

        def get_rouges(parser):
            print("Get rouges...")
            print(parser)
            sent_feature = SentenceFeature(parser)
            X = sent_feature.get_all_features(self.vectorizer, self.matrix)
            X = np.asarray(X)
            X = self.scaler.transform(X)
            Y = self.clf.predict(X)
            pos_i = np.nonzero(Y)[0]
            if pos_i.size != 0:
                selected_sents = [parser.sents[i] for i in pos_i]
            else:
                selected_sents = ""
            rouge = Rouge(selected_sents, parser.highlights).get_rouge()

            return rouge

        rouges = p.map(get_rouges, parsers)
        p.close()
        p.join()

        avg_rouge = Rouge.cal_avg_rouge(rouges)
        Rouge.print("InfoFilter", avg_rouge)

    @staticmethod
    def get_train_data(parsers):
        print("Get train data...")
        min_max_scaler = preprocessing.MinMaxScaler()
        vectorizer, matrix = SentenceFeature.get_global_term_freq(parsers)

        p = multiprocessing.ProcessingPool(multiprocessing.cpu_count() - 2)

        def get_data(parser):
            print("Get data...")
            print(parser)
            sent_feature = SentenceFeature(parser)
            Y = sent_feature.label_sents()
            X = sent_feature.get_all_features(vectorizer, matrix)

            return (X, Y)

        data = p.map(get_data, parsers)
        p.close()
        p.join()

        X_train = []
        Y_train = []
        for X,Y in data:
            X_train += X
            Y_train += Y
        X_train = np.asarray(X_train)
        Y_train = np.asarray(Y_train)
        X_train = min_max_scaler.fit_transform(X_train)

        return (X_train, Y_train, min_max_scaler, vectorizer, matrix)


if __name__ == "__main__":
    from newssum.definitions import ROOT_DIR
    import pickle as pk
    import os

    OUTPUT_DIR = os.path.join(os.path.dirname(ROOT_DIR), "data")
    print("Reading training files...")
    train_parsers = pk.load(open(os.path.join(OUTPUT_DIR, "mini_train.pk"), "rb"))
    # print("Reading val files...")
    # val_parsers = pk.load(open(os.path.join(OUTPUT_DIR, "mini_val.pk"), "rb"))
    print("Reading test files...")
    test_parsers = pk.load(open(os.path.join(OUTPUT_DIR, "test.pk"), "rb"))
    data = InfoFilter.get_train_data(train_parsers)
    # pk.dump(data, open(os.path.join(OUTPUT_DIR, "data.pk"), "wb"))
    X_train, Y_train, min_max_scaler, vectorizer, matrix = data
    # X_train, Y_train, min_max_scaler, vectorizer, matrix = pk.load(open(os.path.join(OUTPUT_DIR, "mini_data.pk"), "rb"))
    filter = InfoFilter(X_train, Y_train, min_max_scaler, vectorizer, matrix)
    filter.evaluate(test_parsers)
