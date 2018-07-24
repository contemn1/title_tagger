from __future__ import absolute_import
import logging
import os
import time
from collections import namedtuple

import torch
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.model import Seq2SeqLSTMAttention, EOS, PAD_WORD
from src.data_util import read_file, build_dict_from_iterator
from src.data_util import build_word_index_mapping
from src.custom_dataset import TextIndexDataset


def init_model(opt):
    logging.info(
        '======================  Model Parameters  =========================')

    if opt.copy_attention:
        logging.info('Train a Seq2Seq model with Copy Mechanism')
    else:
        logging.info('Train a normal Seq2Seq model')

    model = Seq2SeqLSTMAttention(opt)

    if opt.train_from:
        logging.info("loading previous checkpoint from %s" % opt.train_from)
        # load the saved the meta-model and override the current one
        model = torch.load(
            open(os.path.join(opt.model_path, opt.exp, '.initial.model'), 'wb')
        )

        if torch.cuda.is_available():
            checkpoint = torch.load(open(opt.train_from, 'rb'))
        else:
            checkpoint = torch.load(
                open(opt.train_from, 'rb'),
                map_location=lambda storage, loc: storage
            )
        # some compatible problems, keys are started with 'module.'
        checkpoint = dict(
            [(k[7:], v) if k.startswith('module.') else (k, v) for k, v in
             checkpoint.items()])
        model.load_state_dict(checkpoint)
    else:
        # dump the meta-model
        torch.save(
            model.state_dict(),
            open(os.path.join(opt.train_from[: opt.train_from.find('.epoch=')],
                              'initial.model'), 'wb')
        )

    if torch.cuda.is_available():
        model = model.cuda()

    return model


def forward_ml(model, word_indices, word_length, tag_indices,
               word_indices_ext, tag_indices_ext, oov_words_batch):
    decoder_log_probs, _, _ = model.forward(word_indices, word_length,
                                            tag_indices, word_indices_ext,
                                            oov_words_batch)
    return decoder_log_probs, 1.0


def calculate_reward(predicted_indices, target_indices):
    max_length = target_indices.size(1)
    target_mask = (target_indices != 0).data.numpy()
    predicted_seq_length = (predicted_indices == EOS).data.numpy()
    predicted_seq_length = np.argmax(predicted_seq_length, axis=1)
    predicted_mask = [np.concatenate((np.ones(ele), np.zeros(max_length - ele)))
                      for ele in predicted_seq_length]
    predicted_mask = np.array(predicted_mask)
    true_positive = (predicted_indices == target_indices).data.numpy() \
                    * target_mask * predicted_mask

    true_positive = np.sum(true_positive, axis=1)
    target_seq_length = np.sum(target_mask, axis=1)
    precision = true_positive / predicted_seq_length
    recall = true_positive / target_seq_length
    f1_score = 2 * precision * recall / (precision + recall)

    return np.average(f1_score)


def forward_rl(model, word_indices, word_length, tag_indices,
               word_indices_ext, tag_indices_ext, oov_words_batch):
    sampling_log_probs, sampling_indices, _ = model.forward(word_indices,
                                                            word_length,
                                                            tag_indices,
                                                            word_indices_ext,
                                                            oov_words_batch,
                                                            "categorical")

    greedy_log_probs, greedy_indices, _ = model.forward(word_indices,
                                                        word_length,
                                                        tag_indices,
                                                        word_indices_ext,
                                                        oov_words_batch,
                                                        "greedy")

    baseline_reward = calculate_reward(sampling_indices, tag_indices_ext)
    sampling_reward = calculate_reward(greedy_indices, tag_indices_ext)
    return sampling_log_probs, sampling_reward - baseline_reward


def train_one_batch(data_batch, model, optimizer, custom_forward,
                    criterion, opt):
    """
    :type data_batch: tuple
    :type model: Seq2SeqLSTMAttention
    :type optimizer: Adam
    """

    word_indices, word_indices_ext, word_length, tag_indices, \
    tag_indices_ext, oov_words_batch = data_batch

    max_oov_number = len(oov_words_batch)
    if torch.cuda.is_available():
        word_indices = word_indices.cuda()
        word_indices_ext = word_indices_ext.cuda()
        word_length = word_length.cuda()
        tag_indices = tag_indices.cuda()
        tag_indices_ext = tag_indices_ext.cuda()

    optimizer.zero_grad()

    decoder_log_probs, reward = custom_forward(model, word_indices, word_length,
                                               tag_indices, word_indices_ext,
                                               tag_indices_ext,
                                               oov_words_batch)

    # simply average losses of all the predicitons
    # IMPORTANT, must use logits instead of probs to compute the loss, otherwise it's super super slow at the beginning (grads of probs are small)!
    start_time = time.time()

    if not opt.copy_attention:
        loss = criterion(
            decoder_log_probs.contiguous().view(-1, opt.vocab_size),
            tag_indices.contiguous().view(-1)
        )
    else:
        if decoder_log_probs.size(1) != tag_indices_ext.size(1):
            print("predicted sequence length {0}".format(decoder_log_probs.size(1)))
            print("actual sequence length {0}".format(tag_indices_ext.size(1)))

        loss = criterion(
            decoder_log_probs.contiguous().view(-1,
                                                opt.vocab_size + max_oov_number),
            tag_indices_ext.contiguous().view(-1)
        )

    logging.info(
        "--loss calculation- {0} seconds -- ".format(time.time() - start_time))

    loss *= reward
    print(loss)
    start_time = time.time()
    loss.backward()
    logging.info("--backward- {0} seconds -- ".format(time.time() - start_time))

    if opt.max_grad_norm > 0:
        pre_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                  opt.max_grad_norm)
        after_norm = (sum(
            [p.grad.data.norm(2) ** 2 for p in model.parameters() if
             p.grad is not None])) ** (1.0 / 2)
        logging.info('clip grad (%f -> %f)' % (pre_norm, after_norm))

    optimizer.step()

    return loss.item(), decoder_log_probs


def init_optimizer_criterion(model, opt):
    """
    mask the PAD <pad> when computing loss, before we used weight matrix, but not handy for copy-model, change to ignore_index
    :param model:
    :param opt:
    :return:
    """
    '''
    if not opt.copy_attention:
        weight_mask = torch.ones(opt.vocab_size).cuda() if torch.cuda.is_available() else torch.ones(opt.vocab_size)
    else:
        weight_mask = torch.ones(opt.vocab_size + opt.max_unk_words).cuda() if torch.cuda.is_available() else torch.ones(opt.vocab_size + opt.max_unk_words)
    weight_mask[opt.word2id[pykp.IO.PAD_WORD]] = 0
    criterion = torch.nn.NLLLoss(weight=weight_mask)

    optimizer = Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=opt.learning_rate)
    # optimizer = torch.optim.Adadelta(model.parameters(), lr=0.1)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=0.1)
    '''
    criterion = torch.nn.NLLLoss(ignore_index=PAD_WORD)

    if opt.train_ml:
        optimizer_ml = Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=opt.learning_rate)
    else:
        optimizer_ml = None

    if opt.train_rl:
        optimizer_rl = Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=opt.learning_rate_rl)
    else:
        optimizer_rl = None

    if torch.cuda.is_available():
        criterion = criterion.cuda()

    return optimizer_ml, optimizer_rl, criterion


if __name__ == '__main__':
    path = "/Users/zxj/Downloads/sorted_result_test.txt"
    file_iter = read_file(path, pre_process=lambda x: x.strip().split("\t"))
    file_list = [(ele[0].split("$$"), ele[1].strip().split("\002")) for ele in
                 file_iter if
                 len(ele) == 2]
    tag_list = [tags for tags, _ in file_list]
    word_list = [words for _, words in file_list]
    word_dict = build_dict_from_iterator(file_list)
    word_index_map, index_word_map = build_word_index_mapping(word_dict,
                                                              min_freq=4)

    text_dataset = TextIndexDataset(word_index_map, word_list, tag_list)

    batch_size_train = 64
    data_loader = DataLoader(text_dataset, batch_size=batch_size_train,
                             shuffle=True,
                             collate_fn=text_dataset.collate_fn_one2one)
    Params = namedtuple("Parameters",
                        ["vocab_size", "word_vec_size", "bidirectional",
                         "rnn_size",
                         "batch_size", "enc_layers", "dec_layers", "dropout",
                         "attention_mode", "input_feeding", "copy_attention",
                         "copy_mode", "copy_input_feeding", "reuse_copy_attn",
                         "must_teacher_forcing", "train_ml", "train_rl",
                         "learning_rate", "learning_rate_rl", "max_grad_norm"],
                        verbose=False)

    opt = Params(vocab_size=len(word_index_map), word_vec_size=300,
                 bidirectional=True,
                 rnn_size=300, batch_size=batch_size_train, enc_layers=1,
                 dec_layers=1,
                 dropout=0.5, attention_mode="general", input_feeding=False,
                 copy_input_feeding=True,
                 copy_attention=True, copy_mode="general", reuse_copy_attn=True,
                 must_teacher_forcing=False, train_ml=True, train_rl=True,
                 learning_rate=0.0001, learning_rate_rl=0.0001,
                 max_grad_norm=4.0)

    size = 0
    model = Seq2SeqLSTMAttention(opt)
    optimizer_ml, optimizer_rl, criterion = init_optimizer_criterion(model, opt)

    for batch in data_loader:
        print(batch[3].size(1) == batch[4].size(1))
        train_one_batch(batch, model, optimizer_ml, forward_ml, criterion, opt)