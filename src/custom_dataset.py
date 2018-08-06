from __future__ import absolute_import

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data_util import word_to_id, tag_to_id


class TextIndexDataset(Dataset):
    def __init__(self, word_to_index, word_sequence, tag_sequence):
        self.word_to_index = word_to_index
        self.data_x = word_sequence
        self.data_y = tag_sequence

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, index):
        current_sentence = self.data_x[index]
        current_tags = self.data_y[index]
        word_indices, oov_dict = word_to_id(current_sentence,
                                            self.word_to_index)
        tag_indices = tag_to_id(current_tags, self.word_to_index, oov_dict)
        return current_sentence, current_tags, word_indices, tag_indices

    def collate_fn_one2one(self, batches):
        '''
        Puts each data field into a tensor with outer dimension batch size"
        '''
        pad_id = self.word_to_index["PAD"]
        oov_id = self.word_to_index["OOV"]
        max_vocab_id = len(self.word_to_index) - 1

        word_indices_ext_list = []
        tag_indices_ext_list = []
        tags_per_batch = []
        words_per_batch = []
        for current_sentence, current_tags, word_indices, tag_indices in batches:
            word_indices_ext_list.append(word_indices)
            tag_indices_ext_list.append(tag_indices)
            tags_per_batch.append(current_tags)

        padded_word_indices_ext, word_length = pad(word_indices_ext_list,
                                                   pad_id)
        word_mask_tensor = torch.zeros_like(padded_word_indices_ext) + oov_id
        padded_word_indices = torch.where(
            padded_word_indices_ext > max_vocab_id,
            word_mask_tensor,
            padded_word_indices_ext)

        max_oov_number = torch.sum(padded_word_indices == oov_id,
                                   dim=1).max().item()

        word_length = torch.from_numpy(np.array(word_length, dtype=np.int64))
        padded_tag_indices_ext, _ = pad(tag_indices_ext_list, pad_id)
        tag_mask_tensor = torch.zeros_like(padded_tag_indices_ext) + oov_id

        padded_tag_indices = torch.where(padded_tag_indices_ext > max_vocab_id,
                                         tag_mask_tensor,
                                         padded_tag_indices_ext)

        return padded_word_indices, padded_word_indices_ext, word_length, \
               padded_tag_indices, padded_tag_indices_ext, max_oov_number, tags_per_batch, words_per_batch


def pad(sequence_raw, pad_id):
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
