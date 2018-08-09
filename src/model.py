import torch
from torch import nn

from src.attention import Attention
from src.constants import PAD
from src.util import merge_copy_generation
from src.constants import UNK


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, bidirectional, hidden_size,
                 n_layers, dropout=None):
        """
        :param vocab_size: size of encoder vocabulary
        :param embedding_size: dimension of word embedding
        :param bidirectional: whether to use bidirectional LSTM
        :param hidden_size: dimension of hidden state of LSTM
        :param n_layers: number of layers of LSTM network
        :param dropout: dropout rate between LSTM layers, this parameter will work
        when number of layers >= 1
        """
        super(Encoder, self).__init__()
        self.pad_token = PAD
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_size,
            self.pad_token
        )
        self.num_directions = 2 if bidirectional else 1
        self.hidden_dim = hidden_size

        self.rnn = nn.LSTM(input_size=embedding_size,
                           hidden_size=hidden_size,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           batch_first=True)
        if n_layers > 1:
            assert (dropout is not None)
            self.rnn.dropout = dropout

    def init_encoder_state(self, encoder_input):
        """Get cell states and hidden states at first time step."""
        batch_size = encoder_input.size(0) \
            if self.rnn.batch_first else encoder_input.size(1)

        h0_encoder = torch.zeros(
            self.rnn.num_layers * self.num_directions,
            batch_size,
            self.hidden_dim)

        c0_encoder = torch.zeros(
            self.rnn.num_layers * self.num_directions,
            batch_size,
            self.hidden_dim
        )

        if torch.cuda.is_available(): return h0_encoder.cuda(), c0_encoder.cuda()

        return h0_encoder, c0_encoder

    def forward(self, input_src, input_src_len):
        """
        Propogate input through the network.
        """
        # initial encoder state, two zero-matrix as h and c at time=0
        # (self.encoder.num_layers * self.num_directions, batch_size, self.src_hidden_dim)
        initial_hidden, initial_c = self.init_encoder_state(input_src)

        total_length = input_src.size(1)
        embeddings = self.embedding(input_src)

        # sort input by length in descending order
        sent_len_sorted, idx_sort = torch.sort(input_src_len, descending=True)
        _, idx_unsort = torch.sort(idx_sort)
        embeddings = embeddings.index_select(0, idx_sort)
        embeddings = nn.utils.rnn.pack_padded_sequence(embeddings,
                                                       sent_len_sorted,
                                                       batch_first=True)

        # src_h (batch_size, seq_len, hidden_size * num_directions): outputs (h_t) of all the time steps
        # src_h_t, src_c_t (num_layers * num_directions, batch, hidden_size): hidden and cell state at last time step
        src_h, (src_h_t, src_c_t) = self.rnn(
            embeddings, (initial_hidden, initial_c)
        )

        src_h, _ = nn.utils.rnn.pad_packed_sequence(src_h, batch_first=True,
                                                    total_length=total_length)

        src_h = src_h.index_select(0, idx_unsort)

        # concatenate to (batch_size, hidden_size * num_directions)
        if self.rnn.bidirectional:
            h_t = torch.cat((src_h_t[-1], src_h_t[-2]), 1)
            c_t = torch.cat((src_c_t[-1], src_c_t[-2]), 1)
        else:
            h_t = src_h_t[-1]
            c_t = src_c_t[-1]

        return src_h, (h_t, c_t)


class CopyDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, encoder_hidden,
                 bidirectional, decoder_hidden, n_layers, dropout=None,
                 attention_mode="general", input_feeding=False,
                 normalize=False):
        """
        :param vocab_size: size of decoder vocabulary
        :param embedding_size: dimension of word embedding
        :param bidirectional: whether to use bidirectional LSTM
        :param encoder_hidden: dimension of hidden state of encoder LSTM
        :param decoder_hidden: dimension of hidden state of decoder LSTM
        :param n_layers: number of layers of decoder LSTM network
        :param dropout: dropout rate between LSTM layers, this parameter will
        work when number of layers >= 1
        :param attention_mode: attention_mode to choose(dot, general, or concat)
        :param input_feeding: whether to use input_feeding
        :param normalize: whether to normalize encoder_decoder attention over
        time steps, set this parameter True if you want to mitigate repetition
        """

        super(CopyDecoder, self).__init__()
        self.pad_token = PAD
        self.vocab_size = vocab_size
        self.input_feeding = input_feeding
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_size,
            self.pad_token
        )

        self.num_directions = 2 if bidirectional else 1
        self.hidden_dim = decoder_hidden
        self.decoder_hidden = decoder_hidden
        self.rnn = nn.LSTM(input_size=embedding_size,
                           hidden_size=decoder_hidden,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           batch_first=True)

        if n_layers > 1:
            assert (dropout is not None)
            self.rnn.dropout = dropout

        self.enc_dec_attn = Attention(
            encoder_hidden, decoder_hidden, method=attention_mode, scale=True,
            normalize=normalize
        )

        self.dec_self_attn = Attention(
            decoder_hidden, decoder_hidden, method="dot", scale=True,
            normalize=False
        )

        self.softmax = nn.LogSoftmax(dim=-1)
        self.tanh = nn.Tanh()

        self.decoder2vocab = nn.Linear(decoder_hidden * 3, self.vocab_size)
        self.copy_switch = nn.Sequential(nn.Linear(self.hidden_dim * 3, 1),
                                         nn.Sigmoid())

        if self.input_feeding:
            self.dec_input_bridge = nn.Linear(decoder_hidden * 3,
                                              decoder_hidden)

    def init_weights(self, init_range=1.0):
        """Initialize weights."""
        self.embedding.weight.data.uniform_(-init_range, init_range)
        # fill with fixed numbers for debugging
        # self.embedding.weight.data.fill_(0.01)
        self.encoder2decoder_hidden.bias.data.fill_(0)
        self.encoder2decoder_cell.bias.data.fill_(0)
        self.decoder2vocab.bias.data.fill_(0)

    def merge_decoder_inputs(self, decoder_embedding, enc_dec_attn,
                             dec_self_attn):
        if not self.input_feeding:
            return decoder_embedding

        decoder_input = torch.cat((decoder_embedding, enc_dec_attn,
                                   dec_self_attn), 2)

        return self.dec_input_bridge(decoder_input)

    def forward(self, dec_inputs, enc_input_ext,
                enc_output, enc_hidden, sampler):
        """
         The differences of copy model from normal seq2seq here are:
         1. The size of decoder_logits is (batch_size, trg_seq_len, vocab_size + max_oov_number).
         2. Return the copy_attn_weights as well. the weights are same to attn_weights as it reuse the original attention
         3. Very important: as we need to merge probs of copying and generative part, thus we have to operate with probs instead of logits. Thus here we return the probs not logits. Respectively, the loss criterion outside is NLLLoss but not CrossEntropyLoss any more.

        :param dec_inputs: decoder inputs (batch_size, seq_length)
        :param enc_input_ext:
        :param enc_output: encoder outputs (batch_size, seq_length, 2*rnn_size)
        :param enc_hidden: hidden and cell state of encoder's last time step
        (1, batch_size, sequence_length)
        :param sampler: sampler function to predict index of current tags and
        decide input of next time step
        :return: log prob distributions of decoder, index of prediction
        """
        batch_size = dec_inputs.size(0)
        max_length = dec_inputs.size(1) - 1
        trg_input = dec_inputs[:, 0].unsqueeze(1)
        enc_dec_attention = torch.zeros(batch_size, 1, self.hidden_dim)
        dec_self_attention = torch.zeros(batch_size, 1, self.hidden_dim)
        if torch.cuda.is_available():
            enc_dec_attention = enc_dec_attention.cuda()
            dec_self_attention = dec_self_attention.cuda()

        dec_hidden = (enc_hidden[0].unsqueeze(0), enc_hidden[1].unsqueeze(0))
        decoder_outputs = []
        decoder_log_probs = []
        predicted_indices_batch = []
        previous_encoder_decoder_attn = []

        for di in range(max_length):
            trg_emb = self.embedding(trg_input)  # (batch_size, 1, embed_dim)
            decoder_input = self.merge_decoder_inputs(trg_emb,
                                                      enc_dec_attention,
                                                      dec_self_attention)

            decoder_output, dec_hidden = self.rnn(decoder_input, dec_hidden)

            enc_dec_attention, _, enc_dec_logit = self.enc_dec_attn(
                decoder_output, enc_output, previous_encoder_decoder_attn)
            previous_encoder_decoder_attn.append(enc_dec_logit)

            if decoder_outputs:
                decoder_context = torch.cat(decoder_outputs, dim=1)
                dec_self_attention, _, dec_self_logit = self.dec_self_attn(
                    decoder_output, decoder_context)

            decoder_outputs.append(decoder_output)

            decoder_logit = torch.cat((decoder_output, enc_dec_attention,
                                       dec_self_attention), 2)
            generation_prob_dist = self.decoder2vocab(decoder_logit)

            copy_prob = self.copy_switch(decoder_logit)
            generation_prob_dist *= (1 - copy_prob)
            copy_prob_dist = copy_prob * enc_dec_logit

            final_distribution = merge_copy_generation(copy_prob_dist,
                                                       generation_prob_dist,
                                                       enc_input_ext,
                                                       self.vocab_size)

            final_distribution = self.softmax(final_distribution)
            predicted_index, trg_input = sampler(final_distribution, dec_inputs,
                                                 di)
            trg_input[trg_input >= self.vocab_size] = UNK
            decoder_log_probs.append(final_distribution)
            predicted_indices_batch.append(predicted_index)

        decoder_log_probs = torch.cat(decoder_log_probs, 1)
        predicted_indices_batch = torch.cat(predicted_indices_batch, 1)

        return decoder_log_probs, predicted_indices_batch


class Seq2SeqLSTMAttention(nn.Module):
    """Container module with an encoder, deocder, embeddings."""

    def __init__(self, opt, vocab_size, vocab_size_decoder, shared_words=0):
        """Initialize model."""
        super(Seq2SeqLSTMAttention, self).__init__()
        self.encoder = Encoder(vocab_size=vocab_size,
                               embedding_size=opt.word_vec_size,
                               hidden_size=opt.rnn_size,
                               bidirectional=True,
                               n_layers=opt.enc_layers,
                               dropout=opt.dropout)

        self.decoder = CopyDecoder(vocab_size=vocab_size_decoder,
                                   embedding_size=opt.word_vec_size,
                                   encoder_hidden=opt.rnn_size * 2,
                                   decoder_hidden=opt.rnn_size * 2,
                                   bidirectional=False,
                                   n_layers=opt.dec_layers,
                                   dropout=opt.dropout,
                                   attention_mode=opt.attention_mode,
                                   input_feeding=opt.input_feeding)
        if shared_words > 0:
            self.encoder.embedding.weight[:shared_words] = self.decoder.embedding.weight[:shared_words]

    def forward(self, input_src, input_src_len,
                input_trg, input_src_ext, sampler):
        enc_output, enc_hidden = self.encoder.forward(input_src, input_src_len)
        decoder_log_probs, predicted_indices_batch = self.decoder.forward(
            input_trg, input_src_ext, enc_output, enc_hidden,
            sampler
        )
        return decoder_log_probs, predicted_indices_batch
