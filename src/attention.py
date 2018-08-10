import torch
from torch import nn
import numpy as np


class Attention(nn.Module):
    def __init__(self, key_dim, query_dim, method='general',
                 scale=False, normalize=False):
        """
        :param key_dim: dimension of key
        :param query_dim: dimension of value
        :param method: type of attention
        :param scale: whether to scale attention energies, See
        :param normalize: whether to normalize attention over time steps
        """

        super(Attention, self).__init__()
        self.method = method

        if self.method == 'general':
            self.attn = nn.Linear(key_dim, query_dim)
        elif self.method == 'concat':
            self.attn = nn.Linear(key_dim + query_dim, query_dim)
            self.v = nn.Linear(query_dim, 1)

        self.softmax = nn.Softmax(dim=2)
        self.scale = scale
        self.normalize = normalize

        # input size is enc_dim + trg_dim as it's a concatenation of both context vectors and target hidden state
        # for Dot Attention, context vector has been converted to trg_dim first

        self.tanh = nn.Tanh()

    def score(self, query, key):
        '''
        :param query: (batch, trg_len, trg_hidden_dim)
        :param key: (batch, src_len, src_hidden_dim)
        :return: energy score (batch, trg_len, src_len)
        '''
        if self.method == 'dot':
            # hidden (batch, trg_len, trg_hidden_dim) * encoder_outputs (batch, src_len, src_hidden_dim).transpose(1, 2) -> (batch, trg_len, src_len)
            energies = torch.bmm(query,
                                 key.transpose(1,
                                               2))  # (batch, trg_len, src_len)
        elif self.method == 'general':
            energies = self.attn(
                key)  # (batch, src_len, trg_hidden_dim)
            # hidden (batch, trg_len, trg_hidden_dim) * encoder_outputs (batch, src_len, src_hidden_dim).transpose(1, 2) -> (batch, trg_len, src_len)
            energies = torch.bmm(query, energies.transpose(1,
                                                           2))  # (batch, trg_len, src_len)
        elif self.method == 'concat':
            energies = []
            batch_size = key.size(0)
            src_len = key.size(1)
            encoder_outputs_reshaped = key.contiguous().view(-1,
                                                             key.size(
                                                                 2))
            for trg_i in range(query.size(1)):
                expanded_hidden = query[:, trg_i, :].unsqueeze(1).expand(-1,
                                                                         src_len,
                                                                         -1)  # (batch, src_len, trg_hidden_dim)
                expanded_hidden = expanded_hidden.contiguous().view(-1,
                                                                    expanded_hidden.size(
                                                                        2))  # (batch * src_len, trg_hidden_dim)
                concated = torch.cat(
                    (expanded_hidden, encoder_outputs_reshaped),
                    1)  # (batch_size * src_len, dec_hidden_dim + enc_hidden_dim)
                # W_a * concated -> (batch_size * src_len, dec_hidden_dim)
                energy = self.tanh(self.attn(concated))
                # (batch_size * src_len, dec_hidden_dim) * (dec_hidden_dim, 1) -> (batch_size * src_len, 1)
                energy = self.v(energy)
                energies.append(energy.view(batch_size, src_len).unsqueeze(
                    0))  # (1, batch_size, src_len)

            energies = torch.cat(energies, dim=0).permute(1, 0,
                                                          2)  # (trg_len, batch_size, src_len) -> (batch_size, trg_len, src_len)


        return energies.contiguous()

    def forward(self, hidden, encoder_outputs,
                previous_attn=None):
        """
        Compute the attention and h_tilde, inputs/outputs must be batch first
        :param hidden: (batch_size, trg_len, trg_hidden_dim)
        :param encoder_outputs: (batch_size, src_len, trg_hidden_dim), if this is dot attention, you have to convert enc_dim to as same as trg_dim first
        :param previous_attn: (batch_size, trg_len, src_len)
        :return:
            h_tilde (batch_size, trg_len, trg_hidden_dim)
            attn_weights (batch_size, trg_len, src_len)
            attn_energies  (batch_size, trg_len, src_len): the attention energies before softmax
        """

        # hidden (batch_size, trg_len, trg_hidden_dim) * encoder_outputs (batch, src_len, src_hidden_dim).transpose(1, 2) -> (batch, trg_len, src_len)
        attn_energies = self.score(hidden, encoder_outputs)
        if self.scale:
            attn_energies = attn_energies / np.power(hidden.size(2), 0.5)

        if self.normalize and previous_attn:
            attn_energies = torch.exp(attn_energies)
            previous_attn_tensor = torch.cat(previous_attn, dim=1)
            previous_attn_sum = torch.sum(previous_attn_tensor, 1, True)
            normalized_attn_energies = attn_energies / previous_attn_sum

            normalized_attn_sum = torch.sum(normalized_attn_energies, 2, True)

            attn_weights = normalized_attn_energies / normalized_attn_sum

        else:
            attn_weights = self.softmax(attn_energies)

        # reweighting context, attn (batch_size, trg_len, src_len) * encoder_outputs (batch_size, src_len, src_hidden_dim) = (batch_size, trg_len, src_hidden_dim)
        weighted_context = torch.bmm(attn_weights, encoder_outputs)

        # return h_tilde (batch_size, trg_len, trg_hidden_dim), attn (batch_size, trg_len, src_len) and energies (before softmax)
        return weighted_context, attn_weights, attn_energies
