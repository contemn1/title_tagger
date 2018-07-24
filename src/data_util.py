from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals


import io
import logging
import sys
from collections import Counter


def read_file(file_path, pre_process=lambda x: x, encoding="utf-8"):
    try:
        with io.open(file_path, encoding=encoding) as file:
            for sentence in file:
                yield (pre_process(sentence))

    except IOError as err:
        logging.error("Failed to open file {0}".format(err))
        sys.exit(1)


def output_iterator(file_path, iterator,
                    post_process=lambda x: x[0] + "\t" + str(x[1]),
                    encoding="utf-8"):
    try:
        with io.open(file_path, mode="w+", encoding=encoding) as file:
            for element in iterator:
                file.write(post_process(element) + "\n")

    except IOError as err:
        logging.error("Failed to open file {0}".format(err))
        sys.exit(1)


def build_dict_from_iterator(sentence_iter):
    counter = Counter()
    for ele in sentence_iter:
        if len(ele) != 2:
            continue

        tags = ele[0] * 2
        words = ele[1]
        counter.update(tags)
        counter.update(words)

    return counter


def build_word_index_mapping(word_dict, min_freq):
    word_to_idx = {"PAD": 0, "SOS": 1, "EOS": 2, "OOV": 3}
    index_to_word = ["PAD", "SOS", "EOS", "OOV"]
    for key, value in word_dict.items():
        if value >= min_freq and key not in word_to_idx:
            word_to_idx[key] = len(word_to_idx)
            index_to_word.append(key)
    return word_to_idx, index_to_word


def word_to_id(word_list, word_to_index):
    word_indices = []
    oov_words = []
    for word in word_list:
        if word in word_to_index:
            word_indices.append(word_to_index[word])
        else:
            word_indices.append(word_to_index["OOV"])
            oov_words.append(word)
    return word_indices, oov_words
