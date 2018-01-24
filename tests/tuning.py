import os
import shutil
import random
import pathos.multiprocessing as multiprocessing
import numpy as np
import pickle as pk
from newssum.parsers import StoryParser
from newssum.summarizers import CoreRank
from newssum.evaluation.rouge import Rouge
from test_summarizers import data_dir, cal_rouge, cal_avg_rouge, plot
from datetime import datetime


# def split_train_dev_test(data, dev_percentage=0.015, test_percentage=0.015):
#     data_shuffled = data.copy()
#     random.shuffle(data_shuffled)
#     dev_index = -1 * int(dev_percentage * len(data)) - 1 * int(test_percentage * len(data))
#     test_index = -1 * int(test_percentage * len(data))
#     train, dev, test = data_shuffled[:dev_index - 1], data_shuffled[dev_index - 1:test_index], data_shuffled[
#                                                                                                test_index:]
#
#     def move_files(files, dst):
#         for f in files:
#             shutil.move(data_dir + f, data_dir + dst)
#
#     move_files(train, "train/")  # change to your destination train directory
#     move_files(dev, "dev/")  # change to your destination dev directory
#     move_files(test, "test/")  # change to your destination test directory



def read_files():
    print("Reading files...")
    try:
        parsers = pk.load(open("../data/dev.pk", "rb"))
    except FileNotFoundError:
        data_dir = 'C:/KangLong/Data/cnn/stories/data/validation/'
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
        pk.dump(parsers, open("../data/validation.pk", "wb"))

    return parsers


def grid_search():
    start = datetime.now()
    w_thresholds = np.arange(50, 175, 25)
    ps = np.arange(0.15, 0.45, 0.1)
    win_sizes = np.arange(3, 10, 1)
    ls = np.arange(0.5, 7, 0.1)
    rs = np.arange(0.5, 1.2, 0.1)
    max_f1 = {"paras": None, "f": 0}

    parsers = read_files()
    n_files = len(parsers)

    print("Summarizing...")
    process = multiprocessing.ProcessingPool(multiprocessing.cpu_count() - 2)
    for r in rs:
        for l in ls:
            for w_threshold in w_thresholds:
                for p in ps:
                    for win_size in win_sizes:
                        avg_rouge = [{"rouge-1": {"f": 0, "p": 0, "r": 0},
                                      "rouge-2": {"f": 0, "p": 0, "r": 0},
                                      "rouge-l": {"f": 0, "p": 0, "r": 0}}]

                        def summarize(parser):
                            cr_summarizer = CoreRank(parser, window_size=win_size)
                            cr_best_sents = cr_summarizer.get_best_sents(p=p, w_threshold=w_threshold, l=l, r=r)

                            rouge = Rouge(cr_best_sents, parser.highlights)
                            rouge = rouge.get_rouge()
                            return rouge

                        all_rouges = process.map(summarize, parsers)

                        for rouge in all_rouges:
                            for k in rouge:
                                avg_rouge[0][k]["f"] += rouge[k]["f"]
                                avg_rouge[0][k]["p"] += rouge[k]["p"]
                                avg_rouge[0][k]["r"] += rouge[k]["r"]

                        for v in avg_rouge[0].values():
                            cal_avg_rouge(v, n_files)
                        paras = (win_size, p, w_threshold, l, r)
                        print("{}: {}".format(paras, avg_rouge))

                        if max_f1["f"] < avg_rouge[0]["rouge-1"]["f"]:
                            max_f1["paras"] = paras
                            max_f1["f"] = avg_rouge[0]["rouge-1"]["f"]
                            with open("../output/tuning_parameters.txt", "a") as file:
                                file.write("New optimum parameters combination:\n{}\n".format(max_f1))
    process.close()
    process.join()

    print("Grid search finished!")
    print("Best parameters: {}\n{}".format(max_f1["paras"], max_f1["f"]))
    print("Runing time: {}".format(datetime.now() - start))


def run(x, y, para):
    parsers = read_files()
    n_files = len(parsers)

    print("Summarizing...")
    p = multiprocessing.ProcessingPool(multiprocessing.cpu_count())
    for s in x:
        avg_rouge = [{"rouge-1": {"f": 0, "p": 0, "r": 0},
                      "rouge-2": {"f": 0, "p": 0, "r": 0},
                      "rouge-l": {"f": 0, "p": 0, "r": 0}}]

        def summarize(parser):
            if para == "w_threshold":
                cr_summarizer = CoreRank(parser)
                cr_best_sents = cr_summarizer.get_best_sents(w_threshold=s)
            elif para == "p":
                cr_summarizer = CoreRank(parser)
                cr_best_sents = cr_summarizer.get_best_sents(p=s)
            elif para == "win_size":
                cr_summarizer = CoreRank(parser, window_size=s)
                cr_best_sents = cr_summarizer.get_best_sents()
            elif para == "l":
                cr_summarizer = CoreRank(parser)
                cr_best_sents = cr_summarizer.get_best_sents(l=s)
            elif para == "r":
                cr_summarizer = CoreRank(parser)
                cr_best_sents = cr_summarizer.get_best_sents(r=s)

            rouge = Rouge(cr_best_sents, parser.highlights)
            rouge = rouge.get_rouge()
            return rouge

        all_rouges = p.map(summarize, parsers)
        print("{} finished!".format(s))

        for rouge in all_rouges:
            for k in rouge:
                avg_rouge[0][k]["f"] += rouge[k]["f"]
                avg_rouge[0][k]["p"] += rouge[k]["p"]
                avg_rouge[0][k]["r"] += rouge[k]["r"]

        for v in avg_rouge[0].values():
            cal_avg_rouge(v, n_files)

        y.append(avg_rouge)
    p.close()
    p.join()


def tune_w_threshold():
    start = datetime.now()
    x = np.arange(50, 300, 25)
    y = []
    labels = ("CoreRank")

    run(x, y, "w_threshold")

    plot(x, y, labels, "../figures/Performance of tuning w_threshold on CNN.png", 'summary size (words)')
    print("w_threshold finished!")
    print("Runing time: {}".format(datetime.now() - start))


def tune_p():
    start = datetime.now()
    x = np.arange(0.05, 0.95, 0.1)
    y = []
    labels = ("CoreRank")

    run(x, y, "p")

    plot(x, y, labels, "../figures/Performance of tuning p on CNN.png", "P (%)")
    print("p finished!")
    print("Runing time: {}".format(datetime.now() - start))


def tune_win_size():
    start = datetime.now()

    x = np.arange(2, 10, 1)
    y = []
    labels = ("CoreRank")

    run(x, y, "win_size")
    plot(x, y, labels, "../figures/Performance of tuning window_size on CNN.png", "window size (words)")
    print("win_size finished!")
    print("Runing time: {}".format(datetime.now() - start))


def tune_l():
    start = datetime.now()

    x = np.arange(0, 7, 0.1)
    y = []
    labels = ("CoreRank")

    run(x, y, "l")

    plot(x, y, labels, "../figures/Performance of tuning l on CNN.png", "l")
    print("l finished!")
    print("Runing time: {}".format(datetime.now() - start))


def tune_r():
    start = datetime.now()

    x = np.arange(0, 2, 0.1)
    y = []
    labels = ("CoreRank")

    run(x, y, "r")

    plot(x, y, labels, "../figures/Performance of tuning r on CNN.png", "r")
    print("r finished!")
    print("Runing time: {}".format(datetime.now() - start))


if __name__ == "__main__":
    # split_train_dev_test(os.listdir(data_dir))
    read_files()
    # tune_w_threshold()
    # tune_p()
    # tune_win_size()
    # tune_l()
    # tune_r()
    # grid_search()
