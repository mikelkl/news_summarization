from newssum.feature_extraction.sentence_feature import SentenceFeature
from sklearn import preprocessing, svm
from newssum.evaluation import Rouge
import numpy as np


class InfoFilter():
    def __init__(self, X, Y) -> None:
        self.clf = self.fit(X, Y)

    def fit(self, X, Y):
        print("training")
        lin_clf = svm.LinearSVC()
        lin_clf.fit(X, Y)

        return lin_clf

    def evaluate(self, X_test, Y_test, parsers):
        print("evaluating")
        rouges = []
        start = 0
        for parser in parsers:
            sent_num = len(parser.sents)
            next = start + sent_num
            X = X_test[start:next, :]
            Y = self.clf.predict(X)
            pos_i = np.nonzero(Y)[0]
            selected_sents = [parser.sents[i] for i in pos_i]
            rouge = Rouge(selected_sents, parser.highlights).get_rouge()
            rouges.append(rouge)
            start = next
        avg_rouge = Rouge.cal_avg_rouge(rouges)
        Rouge.print("InfoFilter", avg_rouge)

    @staticmethod
    def get_train_val_test(train_parsers):
        min_max_scaler = preprocessing.MinMaxScaler()
        vectorizer, matrix = SentenceFeature.get_global_term_freq(train_parsers)

        def get_data(parsers, mode="train"):
            X = []
            Y = []
            total = len(parsers)
            counter = 1
            for parser in parsers:
                sent_feature = SentenceFeature(parser)
                y = sent_feature.label_sents()
                Y += y
                x = sent_feature.get_all_features(vectorizer, matrix)
                X += x
                print("{}: {}/{}".format(mode, counter, total))
                counter += 1
            X = np.asarray(X)
            Y = np.asarray(Y)
            if mode == "train":
                X = min_max_scaler.fit_transform(X)
            else:
                X = min_max_scaler.transform(X)
            return (X, Y)

        X_train, Y_train = InfoFilter.get_data(train_parsers, vectorizer, matrix)
        X_val, Y_val = get_data(val_parsers, mode="val")
        X_test, Y_test = get_data(test_parsers, mode="test")

        return (X_train, Y_train, X_val, Y_val, X_test, Y_test, min_max_scaler)


if __name__ == "__main__":
    from parsers import StoryParser
    from definitions import ROOT_DIR
    import pickle as pk
    import os

    OUTPUT_DIR = os.path.join(os.path.dirname(ROOT_DIR), "data")
    # print("Reading files...")
    # try:
    #     parsers = pk.load(open("../data/data.pk", "rb"))
    # except FileNotFoundError:
    #     TRAIN_DIR = 'C:/KangLong/Data/cnn/stories/data/train/'
    #     VAL_DIR = 'C:/KangLong/Data/cnn/stories/data/validation/'
    #     TEST_DIR = 'C:/KangLong/Data/cnn/stories/data/test/'
    #
    #
    #
    #     def read(data_dir, target):
    #         files = os.listdir(data_dir)
    #
    #         parsers = []
    #         for f in files:
    #             try:
    #                 parser = StoryParser.from_file(data_dir + f)
    #                 parsers.append(parser)
    #             except ValueError:
    #                 print(f)
    #                 os.remove(data_dir + f)
    #                 continue
    #
    #         target = os.path.join(OUTPUT_DIR, target)
    #         pk.dump(parsers, open(target, "wb"))
    #         return parsers
    #
    #
    #     train_parsers = read(TRAIN_DIR, "train.pk")
    #     val_parsers = read(VAL_DIR, "val.pk")
    #     test_parsers = read(TEST_DIR, "test.pk")

    # try:
    #     train_parsers = pk.load(open(os.path.join(OUTPUT_DIR, "train.pk"), "rb"))[:100]
    #     pk.dump(train_parsers, open(os.path.join(OUTPUT_DIR, "mini_train.pk"), "wb"))
    #     val_parsers = pk.load(open(os.path.join(OUTPUT_DIR, "val.pk"), "rb"))[:10]
    #     pk.dump(val_parsers, open(os.path.join(OUTPUT_DIR, "mini_val.pk"), "wb"))
    #     test_parsers = pk.load(open(os.path.join(OUTPUT_DIR, "test.pk"), "rb"))[:10]
    #     pk.dump(test_parsers, open(os.path.join(OUTPUT_DIR, "mini_test.pk"), "wb"))
    #     print("Splitting...")
    #     data = InfoFilter.get_train_val_test(train_parsers, val_parsers, test_parsers)
    #     pk.dump(data, open(os.path.join(OUTPUT_DIR, "mini_data.pk"), "wb"))
    # except FileNotFoundError:
    #     pass
    # try:
    #     train_parsers = pk.load(open(os.path.join(OUTPUT_DIR, "mini_train.pk"), "rb"))
    #     val_parsers = pk.load(open(os.path.join(OUTPUT_DIR, "mini_val.pk"), "rb"))
    #     test_parsers = pk.load(open(os.path.join(OUTPUT_DIR, "mini_test.pk"), "rb"))
    #     print("Splitting...")
    #     data = InfoFilter.get_train_val_test(train_parsers, val_parsers, test_parsers)
    #     pk.dump(data, open(os.path.join(OUTPUT_DIR, "mini_data.pk"), "wb"))
    # except FileNotFoundError:
    #     pass
    print("Reading training files...")
    train_parsers = pk.load(open(os.path.join(OUTPUT_DIR, "train.pk"), "rb"))
    print("Reading val files...")
    val_parsers = pk.load(open(os.path.join(OUTPUT_DIR, "val.pk"), "rb"))
    print("Reading test files...")
    test_parsers = pk.load(open(os.path.join(OUTPUT_DIR, "test.pk"), "rb"))
    print("Splitting...")
    data = InfoFilter.get_train_val_test(train_parsers, val_parsers, test_parsers)
    pk.dump(data, open(os.path.join(OUTPUT_DIR, "data.pk"), "wb"))
    X_train, Y_train, X_val, Y_val, X_test, Y_test, min_max_scaler = data
    # X_train, Y_train, X_val, Y_val, X_test, Y_test, min_max_scaler = pk.load(open(os.path.join(OUTPUT_DIR, "mini_data.pk"), "rb"))
    filter = InfoFilter(X_train, Y_train)
    filter.evaluate(X_test, Y_test, test_parsers)
