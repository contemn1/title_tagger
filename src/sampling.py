from __future__ import absolute_import
from src.constants import UNK
import torch


def teacher_forcing_sampler(final_distribution, trg_inputs, step,
                            **unused_args):
    _, top_idx = final_distribution.topk(1, dim=-1)

    max_step = trg_inputs.size(1) - 1
    next_step = step + 1 if step < max_step else max_step
    next_input = trg_inputs[:, next_step].unsqueeze(1)
    return top_idx.squeeze(2), next_input


def greedy_sampler(final_distribution, *unused_args, **unused_kargs):
    top_v, top_idx = final_distribution.topk(1, dim=-1)
    predicted_index = top_idx.squeeze(2)
    return predicted_index, top_idx


def random_sampler(final_distribution, *unused_args, **unused_kargs):
    categorical_distribution = torch.distributions.categorical.Categorical(
        logits=final_distribution)
    top_idx = categorical_distribution.sample()
    trg_input = top_idx.unsqueeze(1)
    return top_idx, trg_input