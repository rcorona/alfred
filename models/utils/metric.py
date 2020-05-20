import numpy as np
import re
import string
import collections
import pdb
import editdistance

def normalize_answer(s):
    """
    Lower text and remove punctuation, articles and extra whitespace.
    """

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    #pdb.set_trace()
    # return int(normalize_answer(a_gold) == normalize_answer(a_pred))
    if len(a_gold) != len(a_pred):
        return 0.0
    for g, p in zip(a_gold, a_pred):
        if g != p:
            return 0
    return 1


def compute_f1(gold_toks, pred_toks):
    # gold_toks = get_tokens(a_gold)
    # pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def compute_edit_distance(gold_toks, pred_toks):
    # gold_toks = get_tokens(a_gold)
    # pred_toks = get_tokens(a_pred)
    return editdistance.eval(gold_toks, pred_toks)
