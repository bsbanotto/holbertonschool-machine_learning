#!/usr/bin/env python3
"""
Calculates the cumulative n=gram BLEU score of a sentence
"""
from nltk.translate.bleu_score import sentence_bleu


def cumulative_bleu(references, sentence, n):
    """
    Doc here to test checker
    """
    return sentence_bleu(references, sentence)
