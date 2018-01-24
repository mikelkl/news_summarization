import unittest
import os
import pickle as pk
from newssum.feature_extraction import SentenceImportanceDetector
from newssum.parsers import StoryParser


class MyTestCase(unittest.TestCase):
    def test_sentence_importance_extraction(self):
        data_dir = 'C:/KangLong/Data/cnn/stories/dev/'
        files = os.listdir(data_dir)
        i = files.index("1405a31fa3ddc589e72cbf2a9fbbd507c6d7c241.story")
        out_dir = '../output/dev' + str(i)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        parser = StoryParser.from_file(data_dir + files[i])
        detector = SentenceImportanceDetector(parser.sents, parser.highlights)
        detector.label_sents(out_dir)


if __name__ == '__main__':
    unittest.main()
