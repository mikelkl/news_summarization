import os
import unittest

from newssum.parsers import StoryParser, TweetsParser
from newssum.utils import read_tweets


class TestParser(unittest.TestCase):
    # def test_cnn_parser(self):
    #     data_dir = 'C:/KangLong/Data/cnn/stories/'
    #     files = os.listdir(data_dir)
    #     n_files = 0
    #     avg_reference_summary_sent_length = 0
    #     avg_reference_summary_word_length = 0
    #     for f in files:
    #         try:
    #             parser = StoryParser.from_file(data_dir + f)
    #             avg_reference_summary_sent_length += parser.get_reference_summary_sent_length()
    #             avg_reference_summary_word_length += parser.get_reference_summary_word_length()
    #             n_files += 1
    #         except ValueError:
    #             print(f)
    #             os.remove(data_dir + f)
    #             continue
    #
    #     avg_reference_summary_sent_length /= n_files
    #     avg_reference_summary_word_length /= n_files
    #
    #     print("avg reference summary sent length {}".format(avg_reference_summary_sent_length))
    #     print("avg reference summary word length {}".format(avg_reference_summary_word_length))

    def test_tweets_parser(self):
        train_data = 'C:/KangLong/Data/tweets/twitter-2016train-BD.txt'
        tweets = read_tweets(train_data)
        parsers = [TweetsParser(t, s) for t,s in tweets.items()]
        for parser in parsers:
            sub_topics = parser.topic_modeling(parser.processed_words)
            print(parser.topic)
            print(sub_topics)

if __name__ == '__main__':
    unittest.main()
