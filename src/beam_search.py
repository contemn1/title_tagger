import torch

PAD_WORD = 0
BOS = 1
EOS = 2
UNK = 3


class Beam(object):
    """Ordered beam of candidate outputs."""

    def __init__(self, size, cuda=torch.cuda.is_available(), n_best=1,
                 minimum_length=1):
        """Initialize params."""
        self.size = size
        self.eos_top = False
        self.pad = PAD_WORD
        self.bos = BOS
        self.eos = EOS
        self.tt = torch.cuda if cuda else torch
        self.n_best = n_best

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        self.all_scores = []

        # Time and k pair for finished.
        self.finished = []

        self.minimum_length = minimum_length
        # The backpointers at each time-step.
        self.previous_paths = []

        # The outputs at each time-step.
        self.next_inputs = [self.tt.LongTensor(size).fill_(self.eos)]

        self.next_inputs[0][0] = BOS

        # The attentions (matrix) for each time.

    # Get the outputs for the current timestep.
    def get_current_state(self):
        """Get state of beam."""
        return self.next_inputs[-1]

    # Get the backpointers for the current timestep.
    def get_current_origin(self):
        """Get the backpointer to the beam at this step."""
        return self.previous_paths[-1]

    #  Given prob over words for every last beam `wordLk` and attention
    #   `attnOut`: Compute and update the beam search.
    #
    # Parameters:
    #
    #     * `wordLk`- probs of advancing from the last step (K x words)
    #     * `attnOut`- attention at the last step
    #
    # Returns: True if beam search is complete.

    def advance(self, word_lk):
        """Advance the beam."""
        num_words = word_lk.size(1)

        # Sum the previous scores.
        if len(self.previous_paths) > 0:
            word_lk[:, 0] += -1e10
            beam_lk = word_lk + self.scores.unsqueeze(1).expand_as(word_lk)

            for i in range(self.next_inputs[-1].size(0)):
                if self.next_inputs[-1][i] == self.eos:
                    beam_lk[i] = -1e10

        else:
            beam_lk = word_lk

        flat_beam_lk = beam_lk.contiguous().view(-1)
        best_scores, best_scores_id = flat_beam_lk.topk(self.size, 0, True, True)
        self.all_scores.append(self.scores)
        self.scores = best_scores

        prev_k = best_scores_id / num_words
        self.previous_paths.append(prev_k)
        self.next_inputs.append(best_scores_id - prev_k * num_words)

        for i in range(self.next_inputs[-1].size(0)):
            if self.next_inputs[-1][i] == self.eos:
                s = self.scores[i]

                if len(self.next_inputs) - 1 >= self.minimum_length:
                    self.finished.append((s, len(self.next_inputs) - 1, i))

        if self.next_inputs[-1][0] == EOS:
            self.eos_top = True


    def sort_best(self):
        """Sort the beam."""
        return torch.sort(self.scores, 0, True)

    # Get the score of the best in the beam.
    def get_best(self):
        """Get the most likely candidate."""
        scores, ids = self.sort_best()
        return scores[1], ids[1]

    def done(self):
        return self.eos_top and len(self.finished) >= self.n_best

    def beam_update(self, state, idx):
        positions = self.get_current_origin()
        for e in state:
            a, br, d = e.size()
            e = e.view(a, self.size, br // self.size, d)
            sent_states = e[:, :, idx]
            sent_states.data.copy_(sent_states.data.index_select(1, positions))

    def sort_finished(self, minimum=None):
        if minimum is not None:
            i = 0
            # Add from beam until we have minimum outputs.
            while len(self.finished) < minimum:
                s = self.scores[i]
                self.finished.append((s, len(self.next_inputs) - 1, i))
            i += 1

        self.finished.sort(key=lambda a: -a[0])
        scores = [sc for sc, _, _ in self.finished]
        ks = [(t, k) for _, t, k in self.finished]
        return scores, ks

    def get_hypothesis(self, timestep, k):
        """
        Walk back to construct the full hypothesis.
        """
        hyp = []
        for j in range(len(self.previous_paths[:timestep]) - 1, -1, -1):
            hyp.append(self.next_inputs[j + 1][k])
            k = self.previous_paths[j][k]

        return hyp[::-1]
