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
        if not key or not key.strip():
            continue

        if value >= min_freq and key not in word_to_idx:
            word_to_idx[key] = len(word_to_idx)
            index_to_word.append(key)

    return word_to_idx, index_to_word


def word_to_id(word_list, word_to_index):
    word_indices = []
    oov_dict = {}
    initial = len(word_to_index)
    for word in word_list:
        if word in word_to_index:
            word_indices.append(word_to_index[word])
        else:
            oov_index = initial + len(oov_dict)
            word_indices.append(oov_index)
            oov_dict[word] = oov_index

    return word_indices, oov_dict


def tag_to_id(tag_list, word_to_index, oov_dict):
    tag_indices = []
    for tag in tag_list:
        if tag in word_to_index:
            tag_indices.append(word_to_index[tag])
        elif tag in oov_dict:
            tag_indices.append(oov_dict[tag])
        else:
            tag_indices.append(word_to_index["OOV"])
    return tag_indices

def restore_word_index_mapping(file_path):
    word_index_iter = read_file(file_path, lambda x: x.strip().split("\t"))
    word_index_iter = (ele for ele in word_index_iter if ele and len(ele) == 2)
    word_to_index_iter = ((key, int(value)) for key, value in word_index_iter)
    word_to_index = dict(word_to_index_iter)
    index_to_word = sorted(word_to_index.items(), key=lambda x: x[1])
    index_to_word = [key for key, value in index_to_word]
    return word_to_index, index_to_word
