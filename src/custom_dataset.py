from __future__ import absolute_import

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data_util import word_to_id


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
        word_indices, oov_words = word_to_id(current_sentence,
                                             self.word_to_index)
        tag_indices, _ = word_to_id(current_tags, self.word_to_index)
        return current_sentence, current_tags, word_indices, tag_indices, oov_words

    def collate_fn_one2one(self, batches):
        '''
        Puts each data field into a tensor with outer dimension batch size"
        '''
        pad_id = self.word_to_index["PAD"]
        vocab_size = len(self.word_to_index)
        oov_words_batch = list(set([word for ele in batches for word in ele[4]]))
        oov_dict = {value: index + vocab_size for index, value in
                    enumerate(oov_words_batch)}

        word_indices_ext_list = []
        tag_indices_ext_list = []
        word_indices_list = []
        tag_indices_list = []
        for current_sentence, current_tags, word_indices, tag_indices, _ in batches:
            word_indices_list.append(word_indices)
            tag_indices_list.append(tag_indices)
            word_indices_ext = [value if key not in oov_dict else oov_dict[key]
                                for key, value in
                                zip(current_sentence, word_indices)]
            tag_indices_ext = [value if key not in oov_dict else oov_dict[key]
                               for key, value in
                               zip(current_tags, word_indices)]

            word_indices_ext_list.append(word_indices_ext)
            tag_indices_ext_list.append(tag_indices_ext)

        padded_word_indices, word_length = pad(word_indices_list, pad_id)
        padded_word_indices_ext, _ = pad(word_indices_ext_list, pad_id)
        padded_tag_indices, _ = pad(tag_indices_list, pad_id)
        padded_tag_indices_ext, _ = pad(tag_indices_ext_list, pad_id)
        return padded_word_indices, padded_word_indices_ext, word_length, \
               padded_tag_indices, padded_tag_indices_ext, oov_words_batch


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