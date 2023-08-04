#!/usr/bin/env python3
"""
Calculates the unigram BLEU score for a sentence
"""
import numpy as np


def uni_bleu(references, sentence):
    """
    Calculates the unigram BLEU score for a sentence.

    Args:
        references: list of reference translations
            - Each reference translataion is a list of the words in the
              translation
        sentence: list containing the model proposed sentnece

    Returns:
        the unigram BLEU score
    """
    w_t = len(sentence)
    m = 0
    corpus = []

    for reference in references:
        for word in sentence:
            if word in reference and word not in corpus:
                corpus.append(word)

    m = len(corpus)
    P = m / w_t

    ref_len = min(len(reference) for reference in references)

    if w_t < ref_len:
        BP = np.exp(1-(ref_len/w_t))
    else:
        BP = 1

    return P * BP
