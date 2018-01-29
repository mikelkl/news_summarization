from rouge import Rouge as R
from prettytable import PrettyTable

class Rouge:
    def __init__(self, evaluated_sentences, reference_sentences):
        rouge = R()
        eval_sents = None
        ref_sents = None
        if type(evaluated_sentences) is list:
            eval_sents = ' '.join(evaluated_sentences)
        else:
            eval_sents = evaluated_sentences
        if type(reference_sentences) is list:
            ref_sents = ' '.join(reference_sentences)
        else:
            ref_sents = reference_sentences
        self._score = rouge.get_scores(eval_sents, ref_sents)[0]

    def get_rouge(self):
        return self._score

    def get_rouge_1(self):
        return self._score["rouge-1"]

    def get_rouge_2(self):
        return self._score["rouge-2"]

    def get_rouge_l(self):
        return self._score["rouge-l"]

    @staticmethod
    def cal_avg_rouge(rouges):
        """
        :param rouges: list
            List of dict.
        :return: dict
        """
        avg_rouge = rouges[0]
        n = len(rouges)
        if n > 1:
            for rouge in rouges[1:]:
                for k in rouge:
                    avg_rouge[k]["f"] += rouge[k]["f"]
                    avg_rouge[k]["p"] += rouge[k]["p"]
                    avg_rouge[k]["r"] += rouge[k]["r"]
            for v in avg_rouge.values():
                v["f"] /= n
                v["p"] /= n
                v["r"] /= n

        return avg_rouge

    @staticmethod
    def print(labels, rouges):
        if type(labels) is not list:
            labels = [labels]
        if type(rouges) is not list:
            rouges = [rouges]
        for k in rouges[0]:
            print(k)
            t = PrettyTable(['Summarizer', 'F1-score', 'Precision', 'Recall'])
            for i, v in enumerate(labels):
                t.add_row([v] + list(rouges[i][k].values()))
            print(t)