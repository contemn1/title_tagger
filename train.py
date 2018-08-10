from __future__ import absolute_import

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

from src.argument_parser import init_argument_parser
from src.constants import EOS, PAD, BOS
from src.custom_dataset import TextIndexDataset
from src.data_util import restore_word_index_mapping, build_mapping_from_dict
from src.data_util import read_file, build_dict_from_iterator
from src.model import Seq2SeqLSTMAttention
from src.sampling import teacher_forcing_sampler, greedy_sampler, random_sampler


def forward_ml(model, word_indices, word_length, tag_indices,
               word_indices_ext, max_oov_number, tag_indices_ext):
    """
    :type model: Seq2SeqLSTMAttention
    """
    decoder_log_probs, predicted_indices = model.forward(word_indices,
                                                         word_length,
                                                         tag_indices,
                                                         word_indices_ext,
                                                         max_oov_number,
                                                         sampler=teacher_forcing_sampler)
    return decoder_log_probs, 1.0, predicted_indices


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
    true_positive = (predicted_indices == target_indices).data.numpy()
    true_positive = true_positive * target_mask * predicted_mask

    true_positive = np.sum(true_positive, axis=1)
    target_seq_length = np.sum(target_mask, axis=1)
    precision = true_positive / predicted_seq_length
    recall = true_positive / target_seq_length
    numerator = 2 * precision * recall
    denominator = precision + recall
    f1_score = np.where(numerator == 0, 0, numerator / denominator)

    return np.average(f1_score)


def forward_rl(model, word_indices, word_length, tag_indices,
               word_indices_ext, max_oov_number, tag_indices_ext):
    sampling_log_probs, sampling_indices = model.forward(word_indices,
                                                         word_length,
                                                         tag_indices,
                                                         word_indices_ext,
                                                         max_oov_number,
                                                         random_sampler)

    with torch.no_grad():
        greedy_log_probs, greedy_indices = model.forward(word_indices,
                                                         word_length,
                                                         tag_indices,
                                                         word_indices_ext,
                                                         max_oov_number,
                                                         greedy_sampler)

    baseline_reward = calculate_reward(sampling_indices, tag_indices_ext)
    sampling_reward = calculate_reward(greedy_indices, tag_indices_ext)
    return sampling_log_probs, sampling_reward - baseline_reward, sampling_indices


def prepare_data(data_batch):
    word_indices, word_indices_ext, word_length, tag_indices, tag_indices_ext, *_ = data_batch
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

    return (word_indices, word_indices_ext, tag_indices,
            tag_indices_ext, word_length)


def inference_one_batch(data_batch, model, criterion, sampler, vocab_size):
    word_indices, word_indices_ext, tag_indices, tag_indices_ext, \
    word_length = prepare_data(data_batch)
    batch_size, seq_length = tag_indices_ext.size()
    oov_per_batch = torch.sum(word_indices_ext >= vocab_size,
                              dim=1).max().item()
    with torch.no_grad():
        decoder_log_probs, predicted_indices = model.forward(word_indices,
                                                             word_length,
                                                             tag_indices,
                                                             word_indices_ext,
                                                             oov_per_batch,
                                                             sampler)
        loss = criterion(
            decoder_log_probs.contiguous().view(batch_size * seq_length, -1),
            tag_indices_ext.contiguous().view(-1)
        )
        loss_item = loss.item()
        del loss

    return loss_item, predicted_indices


def train_one_batch(data_batch, model, optimizer,
                    custom_forward, criterion, opt):
    vocab_size = opt.vocab_size_decoder
    word_indices, word_indices_ext, tag_indices, tag_indices_ext, \
    word_length = prepare_data(data_batch)
    oov_per_batch = torch.sum(word_indices_ext >= vocab_size,
                              dim=1).max().item()
    batch_size, seq_length = tag_indices_ext.size()
    optimizer.zero_grad()
    decoder_log_probs, reward, pred_indices = custom_forward(model,
                                                             word_indices,
                                                             word_length,
                                                             tag_indices,
                                                             word_indices_ext,
                                                             oov_per_batch,
                                                             tag_indices_ext)

    # simply average losses of all the predicitons
    # IMPORTANT, must use logits instead of probs to compute the loss, otherwise it's super super slow at the beginning (grads of probs are small)!
    start_time = time.time()
    loss = criterion(
        decoder_log_probs.contiguous().view(batch_size * seq_length, -1),
        tag_indices_ext.contiguous().view(-1))

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
    best_loss = 100000.0  # for f-score
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

            if total_batch % opt.print_loss_every == 0:
                print("Training loss in batch {0} is {1:.2f}".format(
                    total_batch, loss_ml
                ))

            train_ml_losses.append(loss_ml)

            if total_batch > 1 and total_batch % opt.run_valid_every == 0:
                valid_loss_epoch = []
                model.eval()
                for batch_valid in valid_data_loader:
                    loss_valid, predicted_indices = inference_one_batch(
                        batch_valid, model, criterion,
                        teacher_forcing_sampler, opt.vocab_size_decoder)

                    valid_loss_epoch.append(loss_valid)

                loss_epoch_mean = np.mean(valid_loss_epoch)
                print("Loss on valid set for batch {0} is {1:.2f}".format(
                    total_batch, loss_epoch_mean
                ))
                valid_history_losses.append(loss_epoch_mean)

                if loss_epoch_mean < best_loss:
                    best_model = copy.deepcopy(model)
                    best_optimizer = copy.deepcopy(optimizer)
                    best_loss = loss_epoch_mean
                    stop_increasing = 0
                else:
                    stop_increasing += 1

                if stop_increasing >= opt.early_stop_tolerance:
                    message = "Have not increased for {0} epoches, early stop training"
                    print(message.format(epoch))
                    early_stop_flag = True
                    break

            if total_batch > 1 and (total_batch % opt.save_model_every == 0):
                save_model(opt.model_path, epoch, batch_i, best_model,
                           best_optimizer)

        average_epoch_loss = np.mean(train_ml_losses)
        save_model(opt.model_path, epoch, total_batch, model, optimizer)
        print("Loss for epoch {0} is: {1}".format(epoch, average_epoch_loss))


def save_model(model_directory, epoch, batch, model, optimizer):
    model_name = "video_tagger_checkpoint_epoch{0}_batch_{1}.pt".format(epoch,
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

    criterion = torch.nn.NLLLoss(ignore_index=PAD)

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


def read_training_data(opt):
    def remove_empty(input_list):
        return [ele for ele in input_list if ele and ele.strip()]

    path = os.path.join(opt.training_dir, opt.training_file)
    file_iter = read_file(path, pre_process=lambda x: x.strip().split("\t"))
    file_list = [(ele[0].split("$$"), ele[1].strip().split("\001")) for ele in
                 file_iter if
                 len(ele) == 2]
    tag_list = [remove_empty(tags) for tags, _ in file_list]
    word_list = [remove_empty(words) for _, words in file_list]
    return word_list, tag_list


def load_pretrained_model(model_path, model, optimizer, opt):
    if os.path.exists(model_path):
        check_point = torch.load(model_path, lambda storage, location: storage)
        model.load_state_dict(check_point["model_state_dict"])
        optimizer.load_state_dict(check_point["optimizer_state_dict"])
        opt.start_epoch = check_point["epoch"] + 1


def construct_mapping(word_list, tag_list, opt):
    word_freq = build_dict_from_iterator(word_list, opt.min_word_freq)
    tag_freq = build_dict_from_iterator(tag_list, opt.min_tag_freq)
    return build_mapping_from_dict(word_freq, tag_freq)


def restore_mapping(opt):
    word_path = os.path.join(opt.model_path, opt.word_index_map_name)
    word_index_map, index_word_map = restore_word_index_mapping(word_path)
    tag_path = os.path.join(opt.model_path, opt.tag_index_map_name)
    tag_index_map, index_tag_map = restore_word_index_mapping(tag_path)
    num_shared = 0
    for tag in tag_index_map:
        num_shared += 1 if tag in word_index_map else 0
    return word_index_map, tag_index_map, index_word_map, index_tag_map, num_shared


def main():
    opt = init_argument_parser()
    num_threads = multiprocessing.cpu_count()
    num_threads = min(num_threads, 4)

    word_list, tag_list = read_training_data(opt)
    mappings = restore_mapping(opt) if opt.restore_model \
        else construct_mapping(word_list, tag_list, opt)

    (word_index_map, tag_index_map, index_word_map,
     index_tag_map, num_shared_words) = mappings

    training_size = int(len(word_list) * 0.8)

    print("Number of words {0}, tags {1} an d shared_words {2}".format(
        len(word_index_map),
        len(tag_index_map),
        num_shared_words))
    text_dataset_train = TextIndexDataset(word_list[:training_size],
                                          tag_list[:training_size],
                                          word_index_map,
                                          tag_index_map)

    text_dataset_valid = TextIndexDataset(word_list[training_size:],
                                          tag_list[training_size:],
                                          word_index_map,
                                          tag_index_map)

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

    vocab_size, vocab_size_decoder = len(word_index_map), len(tag_index_map)
    opt.vocab_size = vocab_size
    opt.vocab_size_decoder = vocab_size_decoder
    model = Seq2SeqLSTMAttention(opt, vocab_size, vocab_size_decoder)

    if torch.cuda.is_available():
        model = model.cuda() if torch.cuda.device_count() == 1 else \
            nn.parallel.DataParallel(model.cuda())

    optimizer_ml, optimizer_rl, criterion = init_optimizer_criterion(model, opt)
    if opt.restore_model:
        model_path = os.path.join(opt.model_path, opt.model_name)
        load_pretrained_model(model_path, model, optimizer_ml, opt)

    train_model(model, optimizer_ml, criterion, train_loader,
                valid_loader, opt)


if __name__ == '__main__':
    main()
