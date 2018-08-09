import numpy as np
import torch


def merge_copy_generation(copy_probs, gen_probs, src_seq, n_vocab):
    batch_size, max_length, _ = copy_probs.size()

    def expand_dim(input_tensor):
        src_len = input_tensor.size(1)
        expanded_input = input_tensor.unsqueeze(1).expand(batch_size,
                                                          max_length,
                                                          src_len)
        return expanded_input.contiguous().view(batch_size * max_length, -1)

    expanded_src_seq = expand_dim(src_seq)
    expanded_gen_probs = gen_probs.contiguous().view(batch_size * max_length, -1)
    oov_each_line = torch.sum(src_seq >= n_vocab, dim=1).cpu().numpy()
    max_oov_number = int(np.max(oov_each_line))
    flat_copy_prbs = copy_probs.view(batch_size * max_length, -1)
    if max_oov_number > 0:
        oov_mask = create_padding_mask(oov_each_line, max_oov_number)
        oov_mask = expand_dim(oov_mask)
        expanded_gen_probs = torch.cat([expanded_gen_probs, oov_mask], dim=1)

    result = expanded_gen_probs.scatter_add_(1, expanded_src_seq, flat_copy_prbs)
    return result.view(batch_size, max_length, -1)


def create_padding_mask(oov_each_line, max_oov_number):
    padding_mask = []
    for oov in oov_each_line:
        padding_line = np.concatenate((np.zeros(oov),
                                       np.full(max_oov_number - oov,
                                               float('-inf'))))
        padding_mask.append(padding_line)
    return torch.from_numpy(np.array(padding_mask, dtype=np.float32))
