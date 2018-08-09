from src.constants import UNK
import torch
import copy
from src.constants import UNK
import torch


def teacher_forcing_sampler(final_distribution, trg_inputs, step,
                            **unused_args):
    _, candidate_index = final_distribution.topk(2, dim=-1)
    top_idx = candidate_index[:, :, 0]
    top_idx = torch.where(top_idx == UNK, candidate_index[:, :, 1], top_idx)

    max_step = trg_inputs.size(1) - 1
    next_step = step + 1 if step < max_step else max_step
    next_input = trg_inputs[:, next_step].unsqueeze(1)
    return top_idx, next_input


def greedy_sampler(final_distribution, **unused_args):
    top_v, top_idx = final_distribution.topk(1, dim=-1)
    predicted_index = top_idx.squeeze(2)
    return predicted_index, top_idx


def random_sampler(final_distribution, **unused_args):
    categorical_distribution = torch.distributions.categorical.Categorical(
        logits=final_distribution)
    top_idx = categorical_distribution.sample()
    trg_input = top_idx.unsqueeze(1)
    return top_idx, trg_input