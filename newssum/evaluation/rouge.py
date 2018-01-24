from rouge import Rouge as R


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
