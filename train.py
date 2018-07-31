from __future__ import absolute_import

import argparse
import copy
import logging
import multiprocessing
import os
import time

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.custom_dataset import TextIndexDataset
from src.data_util import build_word_index_mapping
from src.data_util import read_file, build_dict_from_iterator
from src.model import Seq2SeqLSTMAttention, EOS, PAD_WORD, BOS


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
    predicted_seq_length = np.where(predicted_seq_length == 0, max_length,
                                    predicted_seq_length)

    predicted_mask = [np.concatenate((np.ones(ele), np.zeros(max_length - ele)))
                      for ele in predicted_seq_length]
    predicted_mask = np.array(predicted_mask)
    true_positive = (predicted_indices == target_indices).data.numpy() \
                    * target_mask * predicted_mask

    true_positive = np.sum(true_positive, axis=1)
    target_seq_length = np.sum(target_mask, axis=1)
    precision = true_positive / predicted_seq_length
    recall = true_positive / target_seq_length
    numerator = 2 * precision * recall
    denominator = precision + recall
    f1_score = np.where(numerator == 0, 0, numerator / denominator)

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


def prepare_data(data_batch):
    word_indices, word_indices_ext, word_length, tag_indices, \
    tag_indices_ext, oov_words_batch = data_batch
    batch_size = tag_indices.size(0)
    sos_padding = np.full((batch_size, 1), BOS, dtype=np.int64)
    sos_padding = torch.from_numpy(sos_padding)
    tag_indices = torch.cat((sos_padding, tag_indices), dim=1)

    if torch.cuda.is_available():
        word_indices = word_indices.cuda()
        word_indices_ext = word_indices_ext.cuda()
        tag_indices = tag_indices.cuda()
        tag_indices_ext = tag_indices_ext.cuda()
        word_length = word_length.cuda()
        oov_words_batch = oov_words_batch.cuda()

    return word_indices, word_indices_ext, tag_indices, tag_indices_ext, \
           word_length, oov_words_batch


def inference_one_batch(data_batch, model, criterion):
    word_indices, word_indices_ext, word_length, tag_indices, \
    tag_indices_ext, oov_words_batch = data_batch

    max_oov_number = oov_words_batch.size(0)

    word_indices, word_indices_ext, tag_indices, tag_indices_ext, \
    word_length, oov_words_batch = prepare_data(data_batch)

    with torch.no_grad():
        decoder_log_probs, _, _ = model.forward(word_indices, word_length,
                                                tag_indices, word_indices_ext,
                                                oov_words_batch, "greedy")

        if not opt.copy_attention:
            loss = criterion(
                decoder_log_probs.contiguous().view(-1, opt.vocab_size),
                tag_indices.contiguous().view(-1)
            )
        else:
            loss = criterion(
                decoder_log_probs.contiguous().view(-1,
                                                    opt.vocab_size + max_oov_number),
                tag_indices_ext.contiguous().view(-1)
            )

    return loss.item()


def train_one_batch(data_batch, model, optimizer, custom_forward,
                    criterion, opt):
    """
    :type data_batch: tuple
    :type model: Seq2SeqLSTMAttention
    :type optimizer: Adam
    """

    word_indices, word_indices_ext, tag_indices, tag_indices_ext, \
    word_length, oov_words_batch = prepare_data(data_batch)

    max_oov_number = oov_words_batch.size(0)

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
        loss = criterion(
            decoder_log_probs.contiguous().view(-1,
                                                opt.vocab_size + max_oov_number),
            tag_indices_ext.contiguous().view(-1)
        )

    logging.info(
        "--loss calculation- {0} seconds -- ".format(time.time() - start_time))

    if loss != 1.0:
        loss *= reward

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
    loss_output = loss.data.item()
    del decoder_log_probs, loss

    return loss_output


def train_model(model, optimizer, criterion,
                train_data_loader, valid_data_loader, opt):
    valid_history_losses = []
    best_loss = 1000000.0  # for f-score
    stop_increasing = 0
    best_model = model
    best_optimizer = optimizer

    early_stop_flag = False
    total_batch = 0
    for epoch in range(opt.start_epoch, opt.num_epoches):
        if early_stop_flag:
            break
        epoch += 1
        train_ml_losses = []
        for batch_i, batch in enumerate(train_data_loader):
            model.train()
            total_batch += 1
            loss_ml = train_one_batch(batch, model,
                                      optimizer,
                                      forward_ml, criterion,
                                      opt)
            print("Loss for batch {0} is: {1}".format(batch_i, loss_ml))
            train_ml_losses.append(loss_ml)

            if total_batch > 1 and total_batch % opt.run_valid_every == 0:
                valid_loss_epoch = []
                for batch_valid in valid_data_loader:
                    loss_valid = inference_one_batch(batch_valid,
                                                     model, criterion)
                    valid_loss_epoch.append(loss_valid)

                loss_epoch_mean = np.mean(valid_loss_epoch)
                valid_history_losses.append(loss_epoch_mean)

                if loss_epoch_mean < best_loss:
                    best_model = copy.deepcopy(model)
                    best_optimizer = copy.deepcopy(optimizer)
                    best_loss = loss_epoch_mean
                    stop_increasing = 0
                else:
                    stop_increasing += 1

                if total_batch > 1 and (
                        total_batch % opt.save_model_every == 0):
                    save_model(opt.model_path, epoch, batch_i, best_model,
                               best_optimizer)

                if stop_increasing >= opt.early_stop_tolerance:
                    message = "Have not increased for {0} epoches, early stop training"
                    logging.info(message.format(epoch))
                    early_stop_flag = True
                    break

            if total_batch > 1 and (total_batch % opt.save_model_every == 0):
                save_model(opt.model_path, epoch, batch_i, best_model,
                           best_optimizer)


def save_model(model_directory, epoch, batch, model, optimizer):
    model_name = "video_tagger_checkpoint_epoch{0}_batch_{1}".format(epoch,
                                                                     batch)
    model_path = os.path.join(model_directory, model_name)
    logging.info("save model to {0}".format(model_path))
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }
    torch.save(state, model_path)


def beam_search_one_batch(data_batch, model):
    word_indices, word_indices_ext, word_length, tag_indices, \
    tag_indices_ext, oov_words_batch = data_batch

    batch_size = tag_indices.size(0)
    sos_padding = np.full((batch_size, 1), BOS, dtype=np.int64)
    sos_padding = torch.from_numpy(sos_padding)
    tag_indices = torch.cat((sos_padding, tag_indices), dim=1)

    if torch.cuda.is_available():
        word_indices = word_indices.cuda()
        word_indices_ext = word_indices_ext.cuda()
        tag_indices = tag_indices.cuda()

    with torch.no_grad():
        all_hypos, all_scores = model.beam_search(word_indices, word_length,
                                                  tag_indices, word_indices_ext,
                                                  oov_words_batch, 3, n_best=2)
    return all_hypos, all_scores


def init_optimizer_criterion(model, opt):
    """
    mask the PAD <pad> when computing loss, before we used weight matrix, but not handy for copy-model, change to ignore_index
    :param model:
    :param opt:
    :return:
    """

    criterion = torch.nn.NLLLoss(ignore_index=PAD_WORD)

    if opt.train_ml:
        optimizer_ml = Adam(
            params=filter(lambda p: p.requires_grad, model.parameters()),
            lr=opt.learning_rate)
    else:
        optimizer_ml = None

    if opt.train_rl:
        optimizer_rl = Adam(
            params=filter(lambda p: p.requires_grad, model.parameters()),
            lr=opt.learning_rate_rl)
    else:
        optimizer_rl = None

    if torch.cuda.is_available():
        criterion = criterion.cuda()

    return optimizer_ml, optimizer_rl, criterion


def init_argument_parser():
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument("--training-dir", type=str, metavar="N",
                        help="path of training directory")

    parser.add_argument("--training-file", type=str, metavar="N",
                        help="name of training file")
    parser.add_argument("--batch-size", type=int, default=64, metavar="N",
                        help="input batch size for training (default: 64)")

    parser.add_argument("--word-vec-size", type=int, default=300, metavar="N",
                        help="dimension of word embedding")

    parser.add_argument("--enc-layers", type=int, default=1, metavar="N",
                        help="number of layers of encoder RNNs")

    parser.add_argument("--dec-layers", type=int, default=1, metavar="N",
                        help="number of layers of decoder RNNs")

    parser.add_argument("--dropout", type=float, default=0.5, metavar="N",
                        help="dropout rate")

    parser.add_argument("--model-path", type=str, metavar="N",
                        help="path to save model")

    parser.add_argument("--rnn-size", type=int, default=300, metavar="N",
                        help="hidden size of RNN cell")

    parser.add_argument("--bidirectional", type=bool, default=True, metavar="N",
                        help="whether to use bidirectional RNN")

    parser.add_argument("--attention-mode", type=str, metavar="N",
                        default="general",
                        help="type of attention general, dot or concat")

    parser.add_argument("--copy-attention", type=bool, metavar="N",
                        default=True, help="type of attention")

    parser.add_argument("--input-feeding", type=bool, default=False,
                        metavar="N",
                        help="whether to use input feeding")

    parser.add_argument("--copy-input-feeding", type=bool, default=True,
                        metavar="N",
                        help="whether to use copy input feeding")

    parser.add_argument("--copy-mode", type=str, default="general",
                        metavar="N",
                        help="same as attention mode")

    parser.add_argument("--reuse-copy-attn", type=bool, default=True,
                        metavar="N",
                        help="whether to use intra decoder attention as copy attention")

    parser.add_argument("--train-ml", type=bool, default=True,
                        metavar="N",
                        help="whether to use maximum log likelihood in loss function")

    parser.add_argument("--train-rl", type=bool, default=True,
                        metavar="N",
                        help="whether to use self crictic in loss function")

    parser.add_argument("--learning-rate", type=float, default=0.001,
                        metavar="N",
                        help="learning rate for maximum likelihood")

    parser.add_argument("--learning-rate-rl", type=float, default=0.001,
                        metavar="N",
                        help="learning rate for reinforcement learning")

    parser.add_argument("--max-grad-norm", type=float, default=4.0, metavar="N",
                        help="dropout rate")

    parser.add_argument("--num-epoches", type=int, default=100, metavar="N",
                        help="number of epoches")

    parser.add_argument("--start-epoch", type=int, default=1, metavar="N",
                        help="number of start epoches")

    parser.add_argument("--run-valid-every", type=int, default=10000,
                        metavar="N",
                        help="number of epochs to run validation set")

    parser.add_argument("--save-model-every", type=int, default=20000,
                        metavar="N",
                        help="number of epochs to run validation set")

    parser.add_argument("--early-stop-tolerance", type=int, default=20,
                        metavar="N",
                        help="number of epochs to run validation set")

    parser.add_argument("--min-word-freq", type=int, default=15,
                        metavar="N", help="minimum word frequency")

    return parser.parse_args()


if __name__ == '__main__':
    opt = init_argument_parser()
    path = os.path.join(opt.training_dir, opt.training_file)
    file_iter = read_file(path, pre_process=lambda x: x.strip().split("\t"))
    file_list = [(ele[0].split("$$"), ele[1].strip().split("\002")) for ele in
                 file_iter if
                 len(ele) == 2]
    training_size = int(len(file_list) * 0.8)
    tag_list = [tags for tags, _ in file_list]
    word_list = [words for _, words in file_list]
    word_dict = build_dict_from_iterator(file_list)
    word_index_map, index_word_map = build_word_index_mapping(
        word_dict, min_freq=opt.min_word_freq)

    print("Number of words {0}".format(len(word_index_map)))
    num_threads = multiprocessing.cpu_count()
    text_dataset_train = TextIndexDataset(word_index_map,
                                          word_list[:training_size],
                                          tag_list[:training_size])
    text_dataset_valid = TextIndexDataset(word_index_map,
                                          word_list[training_size:],
                                          tag_list[training_size:])

    train_loader = DataLoader(text_dataset_train, batch_size=opt.batch_size,
                              shuffle=True,
                              collate_fn=text_dataset_train.collate_fn_one2one,
                              num_workers=num_threads,
                              pin_memory=torch.cuda.is_available())

    valid_loader = DataLoader(text_dataset_valid, batch_size=opt.batch_size,
                              shuffle=True,
                              collate_fn=text_dataset_valid.collate_fn_one2one,
                              num_workers=num_threads,
                              pin_memory=torch.cuda.is_available())

    opt.vocab_size = len(word_index_map)

    size = 0
    model = Seq2SeqLSTMAttention(opt)
    if torch.cuda.is_available():
        model = model.cuda() if torch.cuda.device_count() == 1 else \
            nn.DataParallel(model)

    optimizer_ml, optimizer_rl, criterion = init_optimizer_criterion(model, opt)

    train_model(model, optimizer_ml, criterion, train_loader,
                valid_loader, opt)
