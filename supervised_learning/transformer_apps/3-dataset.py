#!/usr/bin/env python3
"""
Transformer Applications Project
"""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """
    Loads and prepares a dataset for machine translation
    """
    def __init__(self, batch_size, max_len):
        """
        Class constructor for Dataset class
        """
        def filter_max_length(pt, en):
            """
            Checks the length of an item in a tuple vs max_length
            """
            return tf.logical_and(tf.size(pt) <= self.max_len,
                                  tf.size(en) <= self.max_len)
        self.batch_size = batch_size
        self.max_len = max_len
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='train',
                                    as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='validation',
                                    as_supervised=True)
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
            )
        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

        self.data_train = self.data_train.filter(filter_max_length)
        self.data_train = self.data_train.cache()
        # Stupid variable declaration to keep next line pycode compliant
        T = True
        self.data_train = self.data_train.shuffle(2**15,
                                                  reshuffle_each_iteration=T)
        self.data_train = self.data_train.padded_batch(batch_size)
        # Stupid variable declaration to keep next line pycode compliant
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        self.data_train = self.data_train.prefetch(AUTOTUNE)

        self.data_valid = self.data_valid.filter(filter_max_length)
        self.data_valid = self.data_valid.padded_batch(batch_size)

    def tokenize_dataset(self, data):
        """
        Creates sub-word tokenizers for the dataset
        Args:
            data: tf.data.Dataset whose examples are formatted as a tuple
                    (pt, en)
                pt: tf.Tensor containing the Portuguese sentence
                en: tf.Tensor containing the corresponding English sentence
        Maximum Vocab size should be set to 2**15
        Returns:
            tokenizer_pt: The Portuguese tokenizer
            tokenizer_en: The English tokenizer
        """
        token_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, _ in data),
            target_vocab_size=2**15
            )
        token_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for _, en in data),
            target_vocab_size=2**15
            )

        return token_pt, token_en

    def encode(self, pt, en):
        """
        Encodes a translation into tokens
        Args:
            pt: the tf.Tensor containing the Portuguese sentence
            en: the tf.Tensore containing the corresponding English sentence

        The tokenized sentences should contain the start end end tokens
        The start token should be indexed as vocab_size
        The end token should be indexed as vocab_size + 1
        Returns:
            pt_tokens: np.ndarray containing the Portuguese tokens
            en_tokens: np.ndarray containing the English tokens
        """
        pt_start = [self.tokenizer_pt.vocab_size]
        pt_end = [self.tokenizer_pt.vocab_size + 1]
        en_start = [self.tokenizer_en.vocab_size]
        en_end = [self.tokenizer_en.vocab_size + 1]

        pt_tokens = pt_start + self.tokenizer_pt.encode(pt.numpy()) + pt_end
        en_tokens = en_start + self.tokenizer_en.encode(en.numpy()) + en_end

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        pt_examples, en_examples = tf.py_function(
            self.encode, [pt, en], [tf.int64, tf.int64]
            )
        pt_examples.set_shape([None])
        en_examples.set_shape([None])

        return pt_examples, en_examples


if __name__ == "__main__":
    import tensorflow as tf

    tf.compat.v1.set_random_seed(0)
    data = Dataset(32, 40)
    for pt, en in data.data_train.take(1):
        print(pt, en)
    for pt, en in data.data_valid.take(1):
        print(pt, en)
