from nltk.tokenize import word_tokenize
import pandas as pd


def extract_sents_below_threshold(sents, best_sents_i, w_threshold):
    """
    :param sents: list
        A sentence list.
    :param best_sents_i: list
        A list contains index of sent.
    :param w_threshold: int
            A soft constrain for output summary length.
    :return: list
        A sents list with highest score under the constrain of w_threshold.
    """
    w_length = 0
    best_sents = []
    for i in best_sents_i:
        best_sents.append(sents[i])
        tokenized_sent = word_tokenize(sents[i])
        w_length += len(tokenized_sent)
        if w_length >= w_threshold:
            break

    # print(best_sents)
    # Post-processing
    # rearrange best sents by their original order
    num_of_best_sents = len(best_sents)
    selected_best_sents_i = best_sents[:num_of_best_sents]
    best_sents = [s for _, s in sorted(zip(selected_best_sents_i, best_sents))]
    # print(best_sents)

    return best_sents


def read_tweets(file_path, group_by_sentiment_under_topic=True):
    """
    Read and group raw tweets data.
    Tweets format E.g.
    id                  topic   sentiment   body
    641638855477731328	ac/dc	positive	I wanna go see AC/DC so bad on the 15th.
    634842103588233217	ac/dc	negative	Promised to take my niece to AC/DC tomorrow.  Soooo not in the mood now.

    :param file_path: string
    :return: dict
        Key is topic,
        Value is compositional body under that topic.
    """
    # with open(file_path, encoding="utf8") as file:
        # tweets = {}
        # sents = []
        # pre_topic = None
        # test = file.readlines()
        # for line in file:
        #     if line:
        #         cols = line.split("\t")
        #         current_topic = cols[1]
        #         body = cols[3].rstrip()
        #
        #         # exclude initial None pre_topic before comparing
        #         if pre_topic is None:
        #             pre_topic = current_topic
        #
        #         # group tweets by topic
        #         if pre_topic != current_topic:
        #             tweets.update({pre_topic: sents})
        #             sents = []
        #             pre_topic = current_topic
        #
        #         sents.append(body)
        # tweets.update({pre_topic: sents})
        # return tweets

    # Read data from file
    tweets = pd.read_csv(file_path, sep='\t',
                         names=['id', 'topic', 'sentiment', 'body'])  # Change here to your own column name
    tweets_groups = {}
    if group_by_sentiment_under_topic is True:
        tweets_groups_temp = tweets.groupby(['topic', 'sentiment'])  # group by multiple columns
    else:
        tweets_groups_temp = tweets.groupby("topic")  # group by topic column only

    # loop over the grouped object
    for k in tweets_groups_temp.groups:
        v = tweets_groups_temp.get_group(k)["body"].tolist()
        tweets_groups.update({k: v})

    return tweets_groups
