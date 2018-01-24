from .parser import BaseParser
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
from sumy.utils import get_stop_words


class StoryParser(BaseParser):
    """
    Parse cnn stories data.

    Please refer cnn dataset at: http://cs.nyu.edu/~kcho/DMQA/
    """

    def __init__(self, text, pos_tagger, keep_only_n_and_adj=True, remove_stopwords=True, stemming_mode="stemming"):
        highlight_i = text.index('@highlight')
        if highlight_i:
            self.body = text[:highlight_i].strip()
            paragraphs = self.body.split("\n")
            self.paragraphs, self.sents = self.tokenize_sents_paras(paragraphs)
            self.highlights = [h.strip() + "." for h in text[highlight_i + len('@highlight'):].split('@highlight')]
            super().__init__(word_tokenize, get_stop_words("english"), pos_tagger, keep_only_n_and_adj,
                             remove_stopwords, stemming_mode)
        else:
            raise ValueError('Not completed story file! Please check input file')

    def tokenize_sents_paras(self, paragraphs):
        """
        :param paragrahs: list
            Each element is a paragraph string.
        :return: tuple, (paras, sents)
            paras, 2-d list, 1st d is paragraphs, 2nd d is sentences in each paragraph
            sents, list of all sentences
        """
        paras = []
        sents = []
        for paragrah in paragraphs:
            sents_in_para = sent_tokenize(paragrah)
            paras.append(sents_in_para)
            sents += sents_in_para

        return (paras, sents)

    @classmethod
    def from_file(cls, file_path, pos_tagger=pos_tag):
        with open(file_path, encoding="utf8") as file:
            text = ""
            for line in file:
                if line != "\n":
                    text += line
            return cls(text, pos_tagger=pos_tagger)
