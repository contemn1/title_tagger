from __future__ import absolute_import

import numpy as np
from typing import List, Dict
import torch
from torch.utils.data import Dataset

from src.data_util import word_to_id, word_to_id_ext
from src.constants import PAD, EOS_WORD


class TextIndexDataset(Dataset):
    def __init__(self, word_sequence, tag_sequence, word_to_index, tag_to_index):
        # type: (List[List[str]], List[List[str]], Dict[str, int], Dict[str, int]) -> None
        """
        :param word_sequence: sequence of tokenized video titles
        :param tag_sequence: sequence of tags
        :param word_to_index: a dict maps each word into a unique index
        :param tag_to_index: a dict maps each tag into a unique index
        """
        self.data_x = word_sequence
        self.data_y = tag_sequence
        self.word_to_index = word_to_index
        self.tag_to_index = tag_to_index

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, index):
        sentence = self.data_x[index]
        tags = self.data_y[index]
        tags.append(EOS_WORD)
        word_indices = word_to_id(sentence, self.word_to_index)
        word_indices_ext, oov_dict = word_to_id_ext(sentence, self.tag_to_index)
        index_to_oov = {value: key for key, value in oov_dict.items()}
        tag_indices = word_to_id(tags, self.tag_to_index)
        tag_indices_ext = [get_new_index(idx, tag, oov_dict) for tag, idx in
                           zip(tags, tag_indices)]

        return (sentence, tags, word_indices, tag_indices,
                word_indices_ext, tag_indices_ext, index_to_oov)

    def collate_fn_one2one(self, batches):
        words_list = [batch[0] for batch in batches]
        tags_list = [batch[1] for batch in batches]
        word_indices_list = [batch[2] for batch in batches]
        tag_indices_list = [batch[3] for batch in batches]
        word_indices_ext = [batch[4] for batch in batches]
        tag_indices_ext = [batch[5] for batch in batches]
        index_to_oov_list = [batch[6] for batch in batches]
        # pad word and tag indices per line to max length
        padded_word_indices, word_length = pad(word_indices_list, PAD)
        padded_tag_indices, _ = pad(tag_indices_list, PAD)
        word_length = torch.from_numpy(np.array(word_length, dtype=np.int64))
        padded_word_indices_ext, _ = pad(word_indices_ext, PAD)
        padded_tag_indices_ext, _ = pad(tag_indices_ext, PAD)

        return (padded_word_indices, padded_word_indices_ext, word_length,
                padded_tag_indices, padded_tag_indices_ext, words_list,
                tags_list, index_to_oov_list)


def pad(sequence_raw, pad_id):
    # type: (List[int], int) -> tuple
    """
    :param sequence_raw: original sequence
    :param pad_id: a int number representing id of PAD character
    :return: paded sequence, actual length of each line in padded sequence
    """
    def pad_per_line(index_list, max_length):
        return np.concatenate(
            (index_list, [pad_id] * (max_length - len(index_list))))

    sequence_length = [len(x_) for x_ in sequence_raw]
    max_seq_length = max(sequence_length)
    padded_sequence = np.array(
        [pad_per_line(x_, max_seq_length) for x_ in sequence_raw],
        dtype=np.int64)

    padded_sequence = torch.from_numpy(padded_sequence)

    assert padded_sequence.size(1) == max_seq_length

    return padded_sequence, sequence_length


def get_new_index(old_index, old_word, oov_dict):
    if old_word not in oov_dict:
        return old_index
    else:
        return oov_dict[old_word]