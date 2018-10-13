from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from typing import List, Dict
import io
import logging
import sys
from collections import Counter
from src.constants import BOS_WORD, EOS_WORD, UNK_WORD, PAD_WORD


def read_file(file_path, pre_process=lambda x: x, encoding="utf-8"):
    """

    """
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


def build_dict_from_iterator(sentence_iter, min_freq):
    counter = Counter()
    for ele in sentence_iter:
        counter.update(ele)

    counter = {key: value for key, value in counter.items() if value >= min_freq}
    return counter


def build_word_index_mapping(word_dict, min_freq):
    word_to_idx = {PAD_WORD: 0, BOS_WORD: 1, EOS_WORD: 2, UNK_WORD: 3}
    index_to_word = [PAD_WORD, BOS_WORD, EOS_WORD, UNK_WORD]
    for key, value in word_dict.items():
        if not key or not key.strip():
            continue
        if value >= min_freq and key not in word_to_idx:
            word_to_idx[key] = len(word_to_idx)
            index_to_word.append(key)

    return word_to_idx, index_to_word


def word_to_id(word_list, word_to_index):
    def get_value(key):
        return word_to_index[key] if key in word_to_index else word_to_index[
            UNK_WORD]

    word_indices = [get_value(word) for word in word_list]
    return word_indices


def word_to_id_ext(word_list, vocabulary):
    """
    :type word_list: List[str]
    :type vocabulary: Dict[str, int]
    """
    word_indices = []
    oov_dict = {}
    initial = len(vocabulary)
    for word in word_list:
        if word in vocabulary:
            word_indices.append(vocabulary[word])
        elif word in oov_dict:
            continue
        else:
            oov_index = initial + len(oov_dict)
            word_indices.append(oov_index)
            oov_dict[word] = oov_index

    return word_indices, oov_dict


def restore_word_index_mapping(file_path):
    word_index_iter = read_file(file_path, lambda x: x.strip().split("\t"))
    word_index_iter = (ele for ele in word_index_iter if ele and len(ele) == 2)
    word_to_index_iter = ((key, int(value)) for key, value in word_index_iter)
    word_to_index = dict(word_to_index_iter)
    index_to_word = sorted(word_to_index.items(), key=lambda x: x[1])
    index_to_word = [key for key, value in index_to_word]
    return word_to_index, index_to_word


def build_mapping_from_dict(word_dict, tag_dict):
    word_to_index = {PAD_WORD: 0, BOS_WORD: 1, EOS_WORD: 2, UNK_WORD: 3}
    tag_to_index = {PAD_WORD: 0, BOS_WORD: 1, EOS_WORD: 2, UNK_WORD: 3}
    initial_length = len(word_to_index)
    shared_words = set()
    exclusive_words = set()
    exclusive_tags = set()
    for word in word_dict:
        if word in tag_dict:
            shared_words.add(word)
        else:
            exclusive_words.add(word)

    for tag in tag_dict:
        if tag not in word_dict:
            exclusive_tags.add(tag)

    for index, word in enumerate(shared_words):
        word_to_index[word] = index + initial_length
        tag_to_index[word] = index + initial_length

    for unique_word in exclusive_words:
        word_to_index[unique_word] = len(word_to_index)

    for unique_tag in exclusive_tags:
        tag_to_index[unique_tag] = len(tag_to_index)

    index_to_word = [word for word, _ in
                     sorted(word_to_index.items(), key=lambda x: x[1])]
    index_to_tag = [tag for tag, _ in sorted(tag_to_index.items(), key=lambda x: x[1])]

    return (word_to_index, tag_to_index, index_to_word,
            index_to_tag, len(shared_words))
