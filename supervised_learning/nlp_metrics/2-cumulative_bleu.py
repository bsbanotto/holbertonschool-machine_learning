#!/usr/bin/env python3
"""
Calculates the cumulative n=gram BLEU score of a sentence
"""
import numpy as np
from scipy.stats import gmean


ngram_bleu = __import__('1-ngram_bleu').ngram_bleu


def cumulative_bleu(references, sentence, n):
    """
    Args:
        references: list of reference translations
            - Each reference translataion is a list of the words in the
              translation
        sentence: list containing the model proposed sentence
        n: size of the largest n-gram to use for evaluation

    Returns:
        the unigram BLEU score
    """
    n_gram_scores = []
    for i in range(1, n + 1):
        n_gram_scores.append(ngram_bleu(references, sentence, i))
    return(gmean(n_gram_scores))
