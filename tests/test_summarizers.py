import os
import unittest
import pathos.multiprocessing as multiprocessing
import matplotlib.pyplot as plt
import numpy as np
import pickle as pk
from prettytable import PrettyTable
from sumy.nlp.stemmers import Stemmer
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.kl import KLSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.random import RandomSummarizer
from sumy.summarizers.sum_basic import SumBasicSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.utils import get_stop_words
from newssum.evaluation import Rouge
from newssum.parsers import StoryParser, TweetsParser
from newssum.summarizers import CoreRank
from newssum.utils import extract_sents_below_threshold, read_tweets

data_dir = 'C:/KangLong/Data/cnn/stories/'


def cal_avg_rouge(dict, n):
    dict["f"] /= n
    dict["p"] /= n
    dict["r"] /= n


def cal_rouge(output_sum, ref_sum, avg_rouge):
    rouge = Rouge(output_sum, ref_sum)
    rouge = rouge.get_rouge()

    for k in rouge:
        avg_rouge[k]["f"] += rouge[k]["f"]
        avg_rouge[k]["p"] += rouge[k]["p"]
        avg_rouge[k]["r"] += rouge[k]["r"]


def execute_baseline_summarizer(summarizer, doc, SENTENCES_COUNT, w_threshold):
    best_sents = summarizer(doc, SENTENCES_COUNT)
    best_sents = [str(s) for s in best_sents]
    best_sents = extract_sents_below_threshold(best_sents, range(len(best_sents)), w_threshold)

    return best_sents


def plot(x, y, labels, filename, xlabel):
    # visualization
    fig, big_axes = plt.subplots(figsize=(15.0, 15.0), nrows=3, ncols=1, sharey=True)
    for row, big_ax in enumerate(big_axes, start=1):
        if row == len(big_axes):
            big_ax.set_title("ROUGE-l \n", fontsize=16)
        else:
            big_ax.set_title("ROUGE-%s \n" % row, fontsize=16)

        # Turn off axis lines and ticks of the big subplot
        # obs alpha is 0 in RGBA string!
        big_ax.tick_params(labelcolor=(1., 1., 1., 0.0), top='off', bottom='off', left='off', right='off')
        # removes the white frame
        big_ax._frameon = False

    titles = ['Recall', 'Precision', 'F1-score', 'Recall', 'Precision', 'F1-score', 'Recall', 'Precision', 'F1-score']
    k1s = ['rouge-1', 'rouge-1', 'rouge-1', "rouge-2", "rouge-2", "rouge-2", "rouge-l", "rouge-l", "rouge-l"]
    k2s = ["r", "p", "f", "r", "p", "f", "r", "p", "f"]

    def configure_plot(i):
        ax = fig.add_subplot(3, 3, i + 1)
        k1 = k1s[i]
        k2 = k2s[i]
        temp = ax.plot(np.array([[j[k1][k2] for j in i] for i in y]))
        ax.legend(temp, labels, loc=1)
        ax.set_title(titles[i])
        ax.set_xticks(range(len(x)))
        ax.set_xticklabels(x)

    for i in range(0, 9):
        configure_plot(i)

    fig.set_facecolor('w')
    plt.xlabel(xlabel)
    plt.tight_layout()
    # plt.show()
    fig.savefig(filename, dpi=fig.dpi)


class MyTestCase(unittest.TestCase):
    def test_cnn(self):
        print("Reading files...")

        try:
            parsers = pk.load(open("../data/test.pk", "rb"))
        except FileNotFoundError:
            data_dir = 'C:/KangLong/Data/cnn/stories/data/test/'
            files = os.listdir(data_dir)

            parsers = []
            for f in files:
                try:
                    parser = StoryParser.from_file(data_dir + f)
                    parsers.append(parser)
                except ValueError:
                    print(f)
                    os.remove(data_dir + f)
                    continue
            pk.dump(parsers, open("../data/test.pk", "wb"))

        print("Summarizing...")
        LANGUAGE = "english"
        SENTENCES_COUNT = 10
        w_thres = 75
        stemmer = Stemmer(LANGUAGE)
        stop_words = get_stop_words(LANGUAGE)

        kl_summarizer = KLSummarizer(stemmer)
        kl_summarizer.stop_words = stop_words
        lr_summarizer = LexRankSummarizer(stemmer)
        lr_summarizer.stop_words = stop_words
        lsa_summarizer = LsaSummarizer(stemmer)
        lsa_summarizer.stop_words = stop_words
        l_summarizer = LuhnSummarizer(stemmer)
        l_summarizer.stop_words = stop_words
        r_summarizer = RandomSummarizer(stemmer)
        r_summarizer.stop_words = stop_words
        sb_summarizer = SumBasicSummarizer(stemmer)
        sb_summarizer.stop_words = stop_words
        tr_summarizer = TextRankSummarizer(stemmer)
        tr_summarizer.stop_words = stop_words

        baseline_summarizers = [kl_summarizer, lr_summarizer, lsa_summarizer, l_summarizer, r_summarizer,
                                sb_summarizer, tr_summarizer]

        labels = ["KL-Sum", "LexRank", "Latent Semantic Analysis", "Luhn", "Random", "SumBasic", "TextRank", "CoreRank"]

        n_files = len(parsers)
        avg_rouges = [{"rouge-1": {"f": 0, "p": 0, "r": 0},
                       "rouge-2": {"f": 0, "p": 0, "r": 0},
                       "rouge-l": {"f": 0, "p": 0, "r": 0}} for i in range(8)]

        p = multiprocessing.ProcessingPool(multiprocessing.cpu_count())

        def summarize(parser):
            rouges = []

            p_parser = PlaintextParser.from_string(parser.body, Tokenizer(LANGUAGE))

            for i, baseline_summarizer in enumerate(baseline_summarizers):
                output_sum = execute_baseline_summarizer(baseline_summarizer, p_parser.document, SENTENCES_COUNT,
                                                         w_thres)
                cal_rouge(output_sum, parser.highlights, avg_rouges[i])
                rouge = Rouge(output_sum, parser.highlights)
                rouge = rouge.get_rouge()
                rouges.append(rouge)

            cr_summarizer = CoreRank(parser)
            cr_best_sents = cr_summarizer.get_best_sents(w_threshold=w_thres)
            rouge = Rouge(cr_best_sents, parser.highlights)
            rouge = rouge.get_rouge()
            rouges.append(rouge)

            return rouges

        all_rouges = p.map(summarize, parsers)
        p.close()
        p.join()

        for rouges in all_rouges:
            for i, rouge in enumerate(rouges):
                for k in rouge:
                    avg_rouges[i][k]["f"] += rouge[k]["f"]
                    avg_rouges[i][k]["p"] += rouge[k]["p"]
                    avg_rouges[i][k]["r"] += rouge[k]["r"]

        for avg_rouge in avg_rouges:
            for k, v in avg_rouge.items():
                cal_avg_rouge(v, n_files)

        print("Rouge-1")
        t1 = PrettyTable(['Summarizer', 'F1-score', 'Precision', 'Recall'])
        for i, v in enumerate(labels):
            t1.add_row([v] + list(avg_rouges[i]["rouge-1"].values()))
        print(t1)

        print("Rouge-2")
        t2 = PrettyTable(['Summarizer', 'F1-score', 'Precision', 'Recall'])
        for i, v in enumerate(labels):
            t2.add_row([v] + list(avg_rouges[i]["rouge-2"].values()))
        print(t2)

        print("Rouge-L")
        tl = PrettyTable(['Summarizer', 'F1-score', 'Precision', 'Recall'])
        for i, v in enumerate(labels):
            tl.add_row([v] + list(avg_rouges[i]["rouge-l"].values()))
        print(tl)

    # def test_tweets(self):
    #     print("Reading files...")
    #     train_data = 'C:/KangLong/Data/tweets/twitter-2016train-BD.txt'
    #     tweets = read_tweets(train_data)
    #     parsers = [TweetsParser(t, s) for t, s in tweets.items()]
    #     print("Summarizing...")
    #     for parser in parsers:
    #         cr_summarizer = CoreRank(parser, overspan_sents=False)
    #         cr_best_sents = cr_summarizer.get_best_sents()
    #         print("{}:".format(parser.topic))
    #         for s in cr_best_sents:
    #             print("\t" + s)


if __name__ == '__main__':
    unittest.main()
