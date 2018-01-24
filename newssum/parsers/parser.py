import string
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

class BaseParser():
    def __init__(self, word_tokenize, stopwords, pos_tagger, keep_only_n_and_adj, remove_stopwords,
                 stemming_mode):
        self.unprocessed_words = self.tokenize_words(self.sents, word_tokenize)
        processed_words = self.unprocessed_words

        if keep_only_n_and_adj is True:
            processed_words_with_pos = self.keep_only_n_and_adj(processed_words, pos_tagger)
            if stemming_mode == "lemmatization":
                processed_words = self.lemmatize_words(processed_words_with_pos)
            else:
                processed_words = [[t for t, pos in s] for s in processed_words_with_pos]
            # processed_words = self.keep_only_n_and_adj(processed_words, pos_tagger)
        if remove_stopwords is True:
            processed_words = self.remove_stopwords(processed_words, stopwords)
        if stemming_mode == "stemming":
            processed_words = self.stem_words(processed_words)
        if keep_only_n_and_adj or remove_stopwords or stemming_mode:
            self.processed_words = processed_words

    def tokenize_words(self, sents, tokenizer):
        sents = list(map(lambda x: x.translate(str.maketrans('', '', string.punctuation)), sents))  # remove punctuation
        return [[t.lower() for t in tokenizer(sent)] for sent in sents]

    def keep_only_n_and_adj(self, words, pos_tagger):
        processed_words_with_pos = []
        for s in words:
            temp_list = []
            for t, pos in pos_tagger(s):
                if pos.startswith("NN"):
                    temp_list.append((t, "n"))
                elif pos.startswith("JJ"):
                    temp_list.append((t, "a"))
            processed_words_with_pos.append(temp_list)
        return processed_words_with_pos

    def remove_stopwords(self, words, stopwords):
        return [[t for t in s if t not in stopwords] for s in words]

    def lemmatize_words(self, words):
        wordnet_lemmatizer = WordNetLemmatizer()
        return [[wordnet_lemmatizer.lemmatize(w, pos=pos) for w, pos in s] for s in words]

    def stem_words(self, words):
        ps = PorterStemmer()
        return [[ps.stem(w) for w in s] for s in words]

    def get_reference_summary_sent_length(self):
        return len(self.highlights)

    def get_reference_summary_word_length(self):
        return sum([len(sent.split()) for sent in self.highlights])
