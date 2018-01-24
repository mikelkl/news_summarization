from .parser import BaseParser
from nltk.tokenize import TweetTokenizer
from nltk import pos_tag
from sumy.utils import get_stop_words
import re


class TweetsParser(BaseParser):
    """
    Parse Tweets data.
    """

    def __init__(self, topic, sents, pos_tagger=pos_tag, keep_only_n_and_adj=True, remove_stopwords=True,
                 stemming_mode="stemming"):
        self.sents = []
        for s in sents:
            s = re.sub("http\S+", "", s)  # Removing URLS
            s = re.sub("#\S+", "", s)  # Removing Hashtag
            s = re.sub("@\w*", "", s)  # Removing @...
            s = re.sub("RT: ", "", s)  # Removing retweet label
            s = re.sub("[ ]{2,}", " ", s)  # Removing continuous space
            self.sents.append(s.strip())
        self.topic = topic
        tokenizer = TweetTokenizer().tokenize
        super().__init__(tokenizer, get_stop_words("english"), pos_tagger, keep_only_n_and_adj, remove_stopwords,
                         stemming_mode)

    def topic_modeling(self, doc):
        import gensim
        from gensim import corpora
        # Creating the term dictionary of our courpus, where every unique term is assigned an index.
        dictionary = corpora.Dictionary(doc)

        # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above
        doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc]

        # Creating the object for LDA model using gensim library
        Lda = gensim.models.ldamodel.LdaModel

        # Running and Trainign LDA model on the document term matrix.
        ldamodel = Lda(doc_term_matrix, num_topics=16, id2word=dictionary, passes=5000, random_state=1, alpha='auto',
                       eta='auto')

        return ldamodel.print_topics(num_words=5)
