import os
import random
import tqdm
import collections

from typing import List

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from vocab import Vocab

from torch.distributions import Poisson

from models.model.base import BaseModule, move_dict_to_cuda
from models.nn import vnn

from torch_struct import SemiMarkovCRF

from models.model.instruction_chunker_subgoal import SubgoalChunker

from models.model.base import embed_packed_sequence
from models.model.instruction_chunker import compute_acc
from models.utils.helper_utils import safe_zip
from models.utils.metric import compute_f1, compute_exact, compute_edit_distance

BIG_NEG = -1e9

def sliding_sum(inputs, k):
    # inputs: b x T x c
    # sums sliding windows along the T dim, of length k
    batch_size = inputs.size(0)
    assert k > 0
    if k == 1:
        return inputs
    sliding_windows = F.unfold(inputs.unsqueeze(1),
                               kernel_size=(k, 1),
                               padding=(k, 0)).reshape(batch_size, k, -1, inputs.size(-1))
    sliding_summed = sliding_windows.sum(dim=1)
    ret = sliding_summed[:, k:-1, :]
    assert ret.shape == inputs.shape
    return ret

def log_hsmm(transition, emission_scores, init, length_scores, lengths, add_eos,
             all_batched=False, allowed_ends_per_instance=None):
    """
    Convert HSMM to a linear chain.
    Parameters (if all_batched = False):
        transition: C X C
        emission_scores: b x N x C
        init: C
        length_scores: K x C
        add_eos: bool, whether to augment with an EOS class (with index C) which can only appear in the final timestep
    OR, if all_batched = True:
        transition: b x C X C
        emission_scores: b x N x C
        init: b x C
        length_scores: b x K x C
        add_eos: bool, whether to augment with an EOS class (with index C) which can only appear in the final timestep

        all_batched: if False, emission_scores is the only tensor with a batch dimension
    Returns:
        edges: b x (N-1) x C x C if not add_eos, or b x (N) x (C+1) x (C+1) if add_eos
    """
    b, N_1, C_1 = emission_scores.shape
    if all_batched:
        _, K, _C = length_scores.shape
        assert C_1 == _C
    else:
        K, _C = length_scores.shape
        assert C_1 == _C
        transition = transition.unsqueeze(0).expand(b, C_1, C_1)
        length_scores = length_scores.unsqueeze(0).expand(b, K, C_1)
        init = init.unsqueeze(0).expand(b, C_1)
        # emission_scores is already batched

    if K > N_1:
        K = N_1
        length_scores = length_scores[:, :K]
    # assert N_1 >= K
    # need to add EOS token
    if add_eos:
        N = N_1 + 1
        C = C_1 + 1
    else:
        N = N_1
        C = C_1
    if add_eos:
        transition_augmented = torch.full((b, C, C), BIG_NEG, device=transition.device)
        transition_augmented[:, :C_1, :C_1] = transition
        if allowed_ends_per_instance is None:
            # can transition from anything to EOS
            transition_augmented[:, C_1, :] = 0
        else:
            # can transition from any of allowed_ends to EOS
            for i, allowed_ends in enumerate(allowed_ends_per_instance):
                assert len(allowed_ends) > 0
                transition_augmented[i, C_1, allowed_ends] = 0

        init_augmented = torch.full((b, C), BIG_NEG, device=init.device)
        init_augmented[:, :C_1] = init

        length_scores_augmented = torch.full((b, K, C), BIG_NEG, device=length_scores.device)
        length_scores_augmented[:, :, :C_1] = length_scores
        # EOS must be length 1, although I don't think this is checked in the dp
        if length_scores_augmented.size(1) > 1:
            length_scores_augmented[:, 1, C_1] = 0
        else:
            # oops
            length_scores_augmented[:, 0, C_1] = 0

        emission_augmented = torch.full((b, N, C), BIG_NEG, device=emission_scores.device)
        for i, length in enumerate(lengths):
            assert emission_augmented[i, :length, :C_1].size() == emission_scores[i, :length].size()
            emission_augmented[i, :length, :C_1] = emission_scores[i, :length]
            emission_augmented[i, length, C_1] = 0
        # emission_augmented[:, :N_1, :C_1] = emission_scores
        # emission_augmented[:, lengths, C_1] = 0
        # emission_augmented[:, N_1, C_1] = 0

        lengths_augmented = lengths + 1

    else:
        transition_augmented = transition

        init_augmented = init

        length_scores_augmented = length_scores

        emission_augmented = emission_scores

        lengths_augmented = lengths

    scores = torch.zeros(b, N - 1, K, C, C, device=emission_scores.device).type_as(emission_scores)
    scores[:, :, :, :, :] += transition_augmented.view(b, 1, 1, C, C)
    # transition scores should include prior scores at first time step
    scores[:, 0, :, :, :] += init_augmented.view(b, 1, 1, C)
    scores[:, :, :, :, :] += length_scores_augmented.view(b, 1, K, 1, C)
    # add emission scores
    # TODO: progressive adding
    for k in range(1, K):
        # scores[:, :, k, :, :] += sliding_sum(emission_augmented, k).view(b, N, 1, C)[:, :N - 1]
        # scores[:, N - 1 - k, k, :, :] += emission_augmented[:, N - 1].view(b, C, 1)
        summed = sliding_sum(emission_augmented, k).view(b, N, 1, C)
        for i in range(b):
            length = lengths_augmented[i]
            scores[i, :length - 1, k, :, :] += summed[i, :length - 1]
            scores[i, length - 1 - k, k, :, :] += emission_augmented[i, length - 1].view(C, 1)

    return scores

class SubgoalChunkerSemiMarkov(SubgoalChunker):
    MAX_SEGMENT_LENGTH = 65

    def add_tag_params(self):
        self.tag_init = nn.Parameter(torch.zeros((len(self.SUBMODULE_NAMES),)), requires_grad=True)
        self.tag_transitions = nn.Parameter(torch.zeros((len(self.SUBMODULE_NAMES), len(self.SUBMODULE_NAMES))), requires_grad=True)
        self.log_rates = nn.Parameter(torch.zeros((len(self.SUBMODULE_NAMES),)).float(), requires_grad=True)
        self.length_scale = nn.Parameter(torch.ones((1,)).float(), requires_grad=True)
        self.length_bias = nn.Parameter(torch.zeros((1,)).float(), requires_grad=True)

    def add_pred_layer(self):
        self.chunk_pred_layer = nn.Linear(self.args.dhid*2, len(self.SUBMODULE_NAMES))

    def length_unaries(self):
        C = len(self.SUBMODULE_NAMES)
        time_steps = torch.arange(self.MAX_SEGMENT_LENGTH, device=self.tag_init.device).float()
        time_steps = time_steps.unsqueeze(-1).expand(self.MAX_SEGMENT_LENGTH, C)
        poissons = Poisson(torch.exp(self.log_rates))
        length_unaries = self.length_scale * poissons.log_prob(time_steps) + self.length_bias
        return length_unaries

    def make_potentials(self, tag_unaries, lengths):
        # tag_unaries: batch x N x c
        # lengths: batch
        batch, N, C = tag_unaries.size()
        assert C == len(self.SUBMODULE_NAMES)
        length_unaries = self.length_unaries()
        return log_hsmm(self.tag_transitions, tag_unaries, self.tag_init, length_unaries,
                        lengths, add_eos=False, all_batched=False)

    @classmethod
    def featurize(cls, ex, args, test_mode):
        feat = {}
        cls.serialize_lang_action(ex, test_mode)

        lang_instr = ex['num']['lang_instr']
        feat['lang_instr'] = lang_instr
        feat['lang_instr_len'] = len(lang_instr)
        sub_instr_lengths = ex['num']['sub_instr_lengths']
        sub_instr_high_indices = ex['num']['sub_instr_high_indices']
        assert sum(sub_instr_lengths) == len(lang_instr)
        chunk_tags = torch.full((len(lang_instr),), -1, dtype=torch.long)
        pos = 0
        for high_idx, length in safe_zip(sub_instr_high_indices, sub_instr_lengths):
            submodule_name = ex['plan']['high_pddl'][high_idx]['discrete_action']['action']
            chunk_tags[pos] = cls.SUBMODULE_NAMES.index(submodule_name)
            # chunk_tags[pos+1:pos+length] = -1
            pos += length
        assert pos == len(lang_instr)
        feat['chunk_tags'] = chunk_tags

        has_no_op = cls.SUBMODULE_TO_BEGIN_INDICES['NoOp'] in feat['chunk_tags']
        if has_no_op:
            assert (feat['lang_instr'][-1] == 34) # 34 is index of <<stop>>

        return feat

    @classmethod
    def chunk_instruction_into_sentences(cls, instruction, chunk_tags):
        assert len(instruction) == len(chunk_tags)
        all_chunks = []
        this_chunk = []
        this_label = None
        all_chunk_labels = []
        for word, tag in zip(instruction, chunk_tags):
            if tag != -1: # in cls.BEGIN_INDICES_TO_SUBMODULE:
                assert tag < len(cls.SUBMODULE_NAMES)
                if this_chunk:
                    all_chunks.append(this_chunk)
                    all_chunk_labels.append(this_label)
                    this_chunk = []
                    assert this_label is not None, "sequence did not start with a begin label"
                this_label = cls.SUBMODULE_NAMES[tag]
            # elif tag in cls.INSIDE_INDICES_TO_SUBMODULE:
            #     label = cls.INSIDE_INDICES_TO_SUBMODULE[tag]
            #     assert this_label == label, "invalid B I sequence: {} -> {}".format(this_label, label)
            this_chunk.append(word)
        if this_chunk:
            all_chunks.append(this_chunk)
            assert this_label is not None, "sequence did not start with a begin label"
            all_chunk_labels.append(this_label)
        assert len(all_chunks) == len(all_chunk_labels)
        return all_chunks, all_chunk_labels

    def make_debug(self, preds, data):
        '''
        readable output generator for debugging
        '''
        debug = {}
        assert data
        for ex, feat in data:
            key = self.get_instance_key(ex)
            lang_instr_flat = [
                self.vocab['word'].index2word(index)
                for index in ex['num']['lang_instr']
            ]
            pred_chunk_tags = preds[key]['chunk_tags']
            gold_chunk_tags = feat['chunk_tags']

            gold_chunk_tag_indices = gold_chunk_tags.cpu().tolist()

            debug['--'.join(map(str,key))] = {
                'lang_instr': lang_instr_flat,
                'pred_chunk_tag_indices': pred_chunk_tags,
                'gold_chunk_tag_indices': gold_chunk_tag_indices,
                'pred_chunk_tags': [self.index_to_tag(index) for index in pred_chunk_tags],
                'gold_chunk_tags': [self.index_to_tag(index) for index in gold_chunk_tag_indices],
            }
        return debug

    def index_to_tag(self, index):
        if index < 0:
            return "I"
        else:
            return self.SUBMODULE_NAMES[index]

    def forward(self, feat, max_decode=None):
        # Move everything onto gpu if needed.
        if self.args.gpu:
            move_dict_to_cuda(feat)

        enc_lang = self.encode_lang(feat)
        unaries = self.chunk_pred_layer(enc_lang)

        edge_potentials = self.make_potentials(unaries, feat['lang_instr_len'])
        dist = SemiMarkovCRF(edge_potentials, lengths=feat['lang_instr_len'])
        feat.update({'out_chunk_dist': dist})
        return feat

    def subtask_sequence(self, preds, ensure_noop_at_end=False):
        chunk_tag_indices: List[int] = preds['chunk_tags']
        labels = []
        for tag_index in chunk_tag_indices:
            if tag_index > 0:
                labels.append(self.SUBMODULE_NAMES[tag_index])
            else:
                assert tag_index == -1
        if ensure_noop_at_end and labels[-1] != "NoOp":
            labels.append("NoOp")
        return labels

    def extract_preds(self, out, batch, feat):
        pred = {}

        chunk_dist = feat['out_chunk_dist']
        pred_labels, extra = chunk_dist.struct.from_parts(chunk_dist.argmax)

        assert len(batch) == pred_labels.size(0) == feat['chunk_tags'].size(0)

        for ex_ix, ex in enumerate(batch):
            key = self.get_instance_key(ex)

            instr_len = feat['lang_instr_len'][ex_ix]
            pred_tags = pred_labels[ex_ix][:instr_len].cpu().tolist()

            assert len(pred_tags) == instr_len

            pred[key] = {
                'chunk_tags': pred_tags,
                'chunk_tag_names': [self.index_to_tag(index) for index in pred_tags],
            }
        return pred

    def compute_loss(self, out, batch, feat):
        losses = dict()

        C = len(self.SUBMODULE_NAMES)
        K = min(feat['lang_instr_len'].max(), self.MAX_SEGMENT_LENGTH)

        gold_parts = SemiMarkovCRF.struct.to_parts(
            feat['chunk_tags'],
            (C, K),
            lengths=feat['lang_instr_len']
        )
        log_likelihoods = feat['out_chunk_dist'].log_prob(gold_parts)
        log_likelihood = log_likelihoods.mean()
        if log_likelihood < -1e7:
            print("warning: check potential construction; huge loss {}".format(-log_likelihood))
            batch_keys = [(ex['task_id'], ex['repeat_idx']) for ex in batch]
            print(zip(batch_keys, log_likelihoods))
        losses['chunk_tags'] = -log_likelihood

        return losses
