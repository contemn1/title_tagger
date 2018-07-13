import torch
import logging
from model import Seq2SeqLSTMAttention
import os
import time


def init_model(opt):
    logging.info('======================  Model Parameters  =========================')

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
                open(opt.train_from, 'rb'), map_location=lambda storage, loc: storage
            )
        # some compatible problems, keys are started with 'module.'
        checkpoint = dict([(k[7:],v) if k.startswith('module.') else (k,v) for k,v in checkpoint.items()])
        model.load_state_dict(checkpoint)
    else:
        # dump the meta-model
        torch.save(
            model.state_dict(),
            open(os.path.join(opt.train_from[: opt.train_from.find('.epoch=')], 'initial.model'), 'wb')
        )

    if torch.cuda.is_available():
        model = model.cuda()


    return model


def train_ml(data_batch, model, optimizer, criterion, opt):
    src, src_len, trg, trg_target, trg_copy_target, src_oov, oov_lists = data_batch
    max_oov_number = max([len(oov) for oov in oov_lists])

    print("src size - ", src.size())
    print("target size - ", trg.size())

    if torch.cuda.is_available():
        src = src.cuda()
        trg = trg.cuda()
        trg_target = trg_target.cuda()
        trg_copy_target = trg_copy_target.cuda()
        src_oov = src_oov.cuda()

    optimizer.zero_grad()

    decoder_log_probs, _, _ = model.forward(src, src_len, trg, src_oov, oov_lists)

    # simply average losses of all the predicitons
    # IMPORTANT, must use logits instead of probs to compute the loss, otherwise it's super super slow at the beginning (grads of probs are small)!
    start_time = time.time()

    if not opt.copy_attention:
        loss = criterion(
            decoder_log_probs.contiguous().view(-1, opt.vocab_size),
            trg_target.contiguous().view(-1)
        )
    else:
        loss = criterion(
            decoder_log_probs.contiguous().view(-1, opt.vocab_size + max_oov_number),
            trg_copy_target.contiguous().view(-1)
        )
    loss = loss * (1 - opt.loss_scale)
    print("--loss calculation- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    loss.backward()
    print("--backward- %s seconds ---" % (time.time() - start_time))

    if opt.max_grad_norm > 0:
        pre_norm = torch.nn.utils.clip_grad_norm(model.parameters(), opt.max_grad_norm)
        after_norm = (sum([p.grad.data.norm(2) ** 2 for p in model.parameters() if p.grad is not None])) ** (1.0 / 2)
        logging.info('clip grad (%f -> %f)' % (pre_norm, after_norm))

    optimizer.step()

    return loss.item(), decoder_log_probs
