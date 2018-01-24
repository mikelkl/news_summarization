import unittest
import os
from newssum.parsers import StoryParser
from newssum.summarizers import CoreRank

data_dir = 'C:/KangLong/Data/cnn/stories/'


class MyTestCase(unittest.TestCase):
    def test_core_rank_visualization(self):
        print("Reading files...")
        files = os.listdir(data_dir)
        parsers = []
        for f in files[:1]:
            try:
                parser = StoryParser.from_file(data_dir + f)
                parsers.append(parser)
            except ValueError:
                print(f)
                os.remove(data_dir + f)
                continue

        print("Summarizing...")
        for parser in parsers:
            cr_summarizer = CoreRank(parser)
            cr_summarizer.get_best_sents()
            # cr_summarizer.plot_graph()
            # cr_summarizer.plot_k_core()

if __name__ == '__main__':
    unittest.main()
