#!/usr/bin/env python3
"""
From task 0 and 1, impement a QA loop that answers questions
"""
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer

# Load tokenizer and model
tokenizer_to_use = "bert-large-uncased-whole-word-masking-finetuned-squad"
tokenizer = BertTokenizer.from_pretrained(tokenizer_to_use)
model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")


def question_answer(question, reference):
    """
    Args:
        question: string containing the question to answer
        reference: string containing the reference document to find answer
    Returns:
        String containing the answer or None if no answer is found
    """
    quest_toks = tokenizer.tokenize(question)
    ref_toks = tokenizer.tokenize(reference)
    toks = ['[CLS]'] + quest_toks + ['[SEP]'] + ref_toks + ['[SEP]']
    input_word_ids = tokenizer.convert_tokens_to_ids(toks)
    input_mask = [1] * len(input_word_ids)
    quest_len = len(quest_toks)
    ref_len = len(ref_toks)
    input_type_ids = [0] * (1 + quest_len + 1) + [1] * (ref_len + 1)

    input_word_ids = tf.convert_to_tensor([input_word_ids])
    input_mask = tf.convert_to_tensor([input_mask])
    input_type_ids = tf.convert_to_tensor([input_type_ids])

    outputs = model([input_word_ids, input_mask, input_type_ids])

    short_start = tf.argmax(outputs[0][0][1:]) + 1
    short_end = tf.argmax(outputs[1][0][1:]) + 1
    answer_tokens = toks[short_start: short_end + 1]
    answer = tokenizer.convert_tokens_to_string(answer_tokens)

    if answer:
        return answer
    else:
        return 'Sorry, I do not understand your question'

def answer_loop(reference):
    """
    Args:
        reference is the reference text
    If the answer cannot be found in the reference text respond with:
        'Sorry, I do not understand your question'
    """
    exit_commands = ['exit', 'quit', 'goodbye', 'bye']
    while(True):
        question = input('Q: ')
        if question.lower() in exit_commands:
            print('A: Goodbye')
            break
        answer = question_answer(question, reference)
        print("A: answer")
