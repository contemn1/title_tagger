import numpy as np
import torch
from torch.utils.data import Dataset
from data_util import word_to_id
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data_util import read_file, build_dict_from_iterator, build_word_index_mapping
from collections import namedtuple
from model import Seq2SeqLSTMAttention


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
        word_indices, oov_words = word_to_id(current_sentence, self.word_to_index)
        tag_indices, _ = word_to_id(current_tags, self.word_to_index)
        return current_sentence, current_tags, word_indices, tag_indices, oov_words

    def collate_fn_one2one(self, batches):
        '''
        Puts each data field into a tensor with outer dimension batch size"
        '''
        pad_id = self.word_to_index["PAD"]
        vocab_size = len(self.word_to_index)
        oov_words_batch = [word for ele in batches for word in ele[4]]
        oov_dict = {}
        for word in oov_words_batch:
            if word not in oov_dict:
                oov_dict[word] = vocab_size + len(oov_dict)

        word_indices_ext_list = []
        tag_indices_ext_list = []
        word_indices_list = []
        tag_indices_list = []
        for current_sentence, current_tags, word_indices, tag_indices, _ in batches:
            word_indices_list.append(word_indices)
            tag_indices_list.append(tag_indices)
            word_indices_ext = [value if key not in oov_dict else oov_dict[key]
                                for key, value in zip(current_sentence, word_indices)]
            tag_indices_ext = [value if key not in oov_dict else oov_dict[key]
                               for key, value in zip(current_tags, word_indices)]

            word_indices_ext_list.append(word_indices_ext)
            tag_indices_ext_list.append(tag_indices_ext)

        padded_word_indices, word_length = pad(word_indices_list, pad_id)
        padded_word_indices_ext, _ = pad(word_indices_ext_list, pad_id)
        padded_tag_indices, _ = pad(tag_indices_list, pad_id)
        padded_tag_indices_ext, _ = pad(tag_indices_ext_list, pad_id)
        return padded_word_indices, padded_word_indices_ext, word_length, padded_tag_indices, \
               padded_tag_indices_ext, oov_words_batch


def pad(sequence_raw, pad_id):
    def pad_per_line(index_list, max_length):
        return np.concatenate((index_list, [pad_id] * (max_length - len(index_list))))

    sequence_length = [len(x_) for x_ in sequence_raw]
    max_seq_length = max(sequence_length)
    padded_sequence = np.array([pad_per_line(x_, max_seq_length) for x_ in sequence_raw],
                               dtype=np.int64)

    padded_sequence = torch.from_numpy(padded_sequence)

    assert padded_sequence.size(1) == max_seq_length

    return padded_sequence, sequence_length


if __name__ == '__main__':
    path = "/Users/zxj/Downloads/sorted_result_test.txt"
    file_iter = read_file(path, pre_process=lambda x: x.strip().split("\t"))
    file_list = [(ele[0].split("$$"), ele[1].strip().split("\002")) for ele in file_iter if len(ele) == 2]
    tag_list = [tags for tags, _ in file_list]
    word_list = [words for _, words in file_list]
    word_dict = build_dict_from_iterator(file_list)
    word_index_map, index_word_map = build_word_index_mapping(word_dict, min_freq=4)

    text_dataset = TextIndexDataset(word_index_map, word_list, tag_list)

    batch_size_train = 32
    data_loader = DataLoader(text_dataset, batch_size=batch_size_train, shuffle=True,
                             collate_fn=text_dataset.collate_fn_one2one)
    Params = namedtuple("Parameters", ["vocab_size", "word_vec_size", "bidirectional", "rnn_size",
                                      "batch_size", "enc_layers", "dec_layers", "dropout",
                                      "attention_mode", "input_feeding", "copy_attention",
                                      "copy_mode", "copy_input_feeding", "reuse_copy_attn",
                                       "must_teacher_forcing"],
                       verbose=False)

    opt = Params(vocab_size=len(word_index_map), word_vec_size=300, bidirectional=True,
                 rnn_size=300, batch_size=batch_size_train, enc_layers=1, dec_layers=1,
                 dropout=0.5, attention_mode="general", input_feeding=False, copy_input_feeding=True,
                 copy_attention=True, copy_mode="general", reuse_copy_attn=True,
                 must_teacher_forcing=True)

    size = 0
    model = Seq2SeqLSTMAttention(opt)
    for batch in data_loader:
        padded_word_indices, padded_word_indices_ext, word_length, \
        padded_tag_indices, padded_tag_indices_ext, oov_words_batch = batch
        res = model.forward(padded_word_indices, word_length, input_trg=padded_tag_indices,
                            input_src_ext=padded_word_indices_ext,
                            oov_lists=oov_words_batch)
