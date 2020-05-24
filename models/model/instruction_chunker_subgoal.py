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

from models.model.base import BaseModule, move_dict_to_cuda
from models.nn import vnn

from torch_struct import LinearChainCRF

from models.model.base import embed_packed_sequence
from models.model.instruction_chunker import compute_acc
from models.utils.metric import compute_f1, compute_exact, compute_edit_distance


def make_indices(label_vocab, prefix, submodule_names):
    d = {
        submodule_name: label_vocab.word2index('{}_{}'.format(prefix, submodule_name))
        for submodule_name in submodule_names
    }
    return d, {v: k for k, v in d.items()}

BIG_NEG = -1e9

class SubgoalChunker(BaseModule):
    PAD_TOKEN = '<<pad>>'
    PAD_ID = 0

    SUBMODULE_NAMES = [
        'GotoLocation',
        'PickupObject',
        'PutObject',
        'CoolObject',
        'HeatObject',
        'CleanObject',
        'SliceObject',
        'ToggleObject',
        'NoOp'
    ]

    SUBMODULE_BEGINS = ['B_' + module for module in SUBMODULE_NAMES]
    SUBMODULE_INSIDES = ['I_' + module for module in SUBMODULE_NAMES]

    LABEL_VOCAB = Vocab(words=[PAD_TOKEN] + SUBMODULE_BEGINS + SUBMODULE_INSIDES)

    # dict containing e.g. {'GotoLocation': LABEL_VOCAB.word2index('B_GotoLocation')}
    SUBMODULE_TO_BEGIN_INDICES, BEGIN_INDICES_TO_SUBMODULE = make_indices(LABEL_VOCAB, 'B', SUBMODULE_NAMES)
    # dict containing e.g. {'GotoLocation': LABEL_VOCAB.word2index('I_GotoLocation')}
    SUBMODULE_TO_INSIDE_INDICES, INSIDE_INDICES_TO_SUBMODULE = make_indices(LABEL_VOCAB, 'I', SUBMODULE_NAMES)

    BEGIN_INDICES = sorted(SUBMODULE_TO_BEGIN_INDICES.values())
    INSIDE_INDICES = sorted(SUBMODULE_TO_INSIDE_INDICES.values())

    @staticmethod
    def add_arguments(parser):
        # hyper parameters
        parser.add_argument('--demb', help='language embedding size', default=100, type=int)
        parser.add_argument('--dhid', help='hidden layer size', default=128, type=int)

        # dropouts
        parser.add_argument('--lang_dropout', help='dropout rate for language instr', default=0., type=float)

    def __init__(self, args, vocab):
        super().__init__(args)
        self.vocab = vocab

        # emb
        self.emb_word = nn.Embedding(len(vocab['word']), args.demb)

        # encoder and self-attention
        self.enc = nn.LSTM(args.demb, args.dhid, bidirectional=True, batch_first=True)

        self.lang_dropout = nn.Dropout(args.lang_dropout, inplace=True)

        self.bce_with_logits = torch.nn.BCEWithLogitsLoss(reduction='none')

        self.chunk_pred_layer = nn.Linear(args.dhid*2, len(self.LABEL_VOCAB))

        self.tag_init = nn.Parameter(torch.zeros((len(self.SUBMODULE_NAMES),)), requires_grad=True)
        self.tag_transitions = nn.Parameter(torch.zeros((len(self.SUBMODULE_NAMES), len(self.SUBMODULE_NAMES))), requires_grad=True)

        # paths
        self.root_path = os.getcwd()

    def make_potentials(self, tag_unaries, lengths):
        # following https://github.com/harvardnlp/pytorch-struct/blob/b6816a4d436136c6711fe2617995b556d5d4d300/torch_struct/linearchain.py#L137
        # tag_scores: batch x N x c
        # lengths: batch
        batch, N, C = tag_unaries.size()
        assert C == len(self.LABEL_VOCAB)

        # to, from
        transition_mat = torch.full((C, C), BIG_NEG, device=tag_unaries.device)
        # can transition from anything to PAD
        transition_mat[self.PAD_ID,:] = 0
        # can transition from B to I of the same subgoal
        for b, i in zip(self.BEGIN_INDICES, self.INSIDE_INDICES):
            transition_mat[i, b] = 0
        for i in self.INSIDE_INDICES:
            transition_mat[i, i] = 0
        # scores from transitioning from B to B of each subgoal type
        transition_mat[np.ix_(self.BEGIN_INDICES,self.BEGIN_INDICES)] = self.tag_transitions
        # scores from transitioning from I to B of each subgoal type
        transition_mat[np.ix_(self.BEGIN_INDICES,self.INSIDE_INDICES)] = self.tag_transitions

        # batch x N x C(to) x C(from)
        edge_potentials = torch.zeros((batch, N-1, C, C), device=tag_unaries.device)
        for ix, t in enumerate(lengths):
            # padding iff < the sequence length
            edge_potentials[ix,:t-1,self.PAD_ID,:] = BIG_NEG
            edge_potentials[ix,t-1:,:,:] = BIG_NEG
            edge_potentials[ix,t-1:,self.PAD_ID,:] = 0

        edge_potentials[:,0,:,self.BEGIN_INDICES] += self.tag_init.view(1,1,self.tag_init.size(-1))
        edge_potentials[:,0,:,self.INSIDE_INDICES] = BIG_NEG
        edge_potentials[:,:,:,:] += transition_mat.view(1,1,C,C)

        edge_potentials[:,:,:,:] += tag_unaries.view(batch, N, C, 1)[:, 1:]
        edge_potentials[:,0,:,:] += tag_unaries.view(batch, N, 1, C)[:, 0]

        return edge_potentials

    @classmethod
    def featurize(cls, ex, args, test_mode):
        feat = {}
        cls.serialize_lang_action(ex, test_mode)

        lang_instr = ex['num']['lang_instr']
        feat['lang_instr'] = lang_instr
        feat['lang_instr_len'] = len(lang_instr)
        sub_instr_lengths = ex['num']['sub_instr_lengths']
        assert sum(sub_instr_lengths) == len(lang_instr)
        chunk_tags = torch.full((len(lang_instr),), cls.PAD_ID, dtype=torch.long)
        pos = 0
        for ix, length in enumerate(sub_instr_lengths):
            submodule_name = ex['plan']['high_pddl'][ix]['discrete_action']['action']
            chunk_tags[pos] = cls.SUBMODULE_TO_BEGIN_INDICES[submodule_name]
            chunk_tags[pos+1:pos+length] = cls.SUBMODULE_TO_INSIDE_INDICES[submodule_name]
            pos += length
            if ix == len(sub_instr_lengths) - 1:
                break
        assert pos == len(lang_instr)
        feat['chunk_tags'] = chunk_tags

        return feat

    @classmethod
    def chunk_instruction_into_sentences(cls, instruction, chunk_tags):
        assert len(instruction) == len(chunk_tags)
        all_chunks = []
        this_chunk = []
        this_label = None
        all_chunk_labels = []
        for word, tag in zip(instruction, chunk_tags):
            if tag in cls.BEGIN_INDICES_TO_SUBMODULE:
                if this_chunk:
                    all_chunks.append(this_chunk)
                    all_chunk_labels.append(this_label)
                    this_chunk = []
                    assert this_label is not None, "sequence did not start with a begin label"
                this_label = cls.BEGIN_INDICES_TO_SUBMODULE[tag]
            elif tag in cls.INSIDE_INDICES_TO_SUBMODULE:
                label = cls.INSIDE_INDICES_TO_SUBMODULE[tag]
                assert this_label == label, "invalid B I sequence: {} -> {}".format(this_label, label)
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
            key = (ex['task_id'], ex['repeat_idx'])
            lang_instr_flat = [
                self.vocab['word'].index2word(index)
                for index in ex['num']['lang_instr']
            ]
            pred_chunk_tags = preds[key]['chunk_tags']
            gold_chunk_tags = feat['chunk_tags']

            gold_chunk_tag_indices = gold_chunk_tags.cpu().tolist()

            debug['{}--{}'.format(*key)] = {
                'lang_instr': lang_instr_flat,
                'pred_chunk_tag_indices': pred_chunk_tags,
                'gold_chunk_tag_indices': gold_chunk_tag_indices,
                'pred_chunk_tags': self.LABEL_VOCAB.index2word(pred_chunk_tags),
                'gold_chunk_tags': self.LABEL_VOCAB.index2word(gold_chunk_tag_indices),
            }
        return debug

    def encode_lang(self, feat):
        '''
        encode instr language
        '''
        # TODO: this is just copied from instruction_chunker
        packed_lang_instr = pack_padded_sequence(feat['lang_instr'], feat['lang_instr_len'], batch_first=True, enforce_sorted=False)
        emb_lang_instr = embed_packed_sequence(self.emb_word, packed_lang_instr)
        # apply dropout in-place
        self.lang_dropout(emb_lang_instr.data)
        emb_lang_instr, _ = self.enc(emb_lang_instr)
        emb_lang_instr, _ = pad_packed_sequence(emb_lang_instr, batch_first=True)
        self.lang_dropout(emb_lang_instr)

        return emb_lang_instr

    def forward(self, feat, max_decode=None):
        # Move everything onto gpu if needed.
        if self.args.gpu:
            move_dict_to_cuda(feat)

        enc_lang = self.encode_lang(feat)
        unaries = self.chunk_pred_layer(enc_lang)

        edge_potentials = self.make_potentials(unaries, feat['lang_instr_len'])
        dist = LinearChainCRF(edge_potentials, lengths=feat['lang_instr_len'])
        feat.update({'out_chunk_dist': dist})
        return feat

    def subtask_sequence(self, preds, ensure_noop_at_end=False):
        chunk_tag_indices: List[int] = preds['chunk_tags']
        labels = []
        for tag_index in chunk_tag_indices:
            if tag_index in SubgoalChunker.BEGIN_INDICES_TO_SUBMODULE:
                labels.append(SubgoalChunker.BEGIN_INDICES_TO_SUBMODULE[tag_index])
            else:
                assert tag_index in SubgoalChunker.INSIDE_INDICES_TO_SUBMODULE or tag_index == SubgoalChunker.PAD_ID, "invalid tag index {}".format(tag_index)
        if ensure_noop_at_end and labels[-1] != "NoOp":
            labels.append("NoOp")
        return labels

    def extract_preds(self, out, batch, feat):
        pred = {}

        chunk_dist = feat['out_chunk_dist']
        pred_labels, extra = chunk_dist.struct.from_parts(chunk_dist.argmax)

        assert len(batch) == pred_labels.size(0) == feat['chunk_tags'].size(0)

        for ex_ix, ex in enumerate(batch):
            key = ex['task_id'], ex['repeat_idx']

            instr_len = feat['lang_instr_len'][ex_ix]
            pred_tags = pred_labels[ex_ix][:instr_len].cpu().tolist()

            pred[key] = {
                'chunk_tags': pred_tags,
                'chunk_tag_names': self.LABEL_VOCAB.index2word(pred_tags),
            }
        return pred

    def compute_loss(self, out, batch, feat):
        losses = dict()

        gold_parts = LinearChainCRF.struct.to_parts(
            feat['chunk_tags'], len(self.LABEL_VOCAB), lengths=feat['lang_instr_len']
        )
        log_likelihoods = feat['out_chunk_dist'].log_prob(gold_parts)
        log_likelihood = log_likelihoods.mean()
        if log_likelihood < -1e7:
            print("warning: check potential construction; huge loss {}".format(-log_likelihood))
            batch_keys = [(ex['task_id'], ex['repeat_idx']) for ex in batch]
            print(zip(batch_keys, log_likelihoods))
        losses['chunk_tags'] = -log_likelihood

        return losses

    def compute_metric(self, preds, data):
        m = collections.defaultdict(list)
        for ex, feat in data:
            key = (ex['task_id'], ex['repeat_idx'])
            gold_tags = feat['chunk_tags'].cpu().tolist()
            pred_tags = preds[key]['chunk_tags']
            assert len(gold_tags) == len(pred_tags)
            m['chunk_tag_acc'].append(compute_acc(gold_tags, pred_tags))

            flat_instr = feat['lang_instr']

            gold_chunks, gold_labels = self.chunk_instruction_into_sentences(flat_instr, gold_tags)
            pred_chunks, pred_labels = self.chunk_instruction_into_sentences(flat_instr, pred_tags)
            m['num_gold_chunks'].append(len(gold_chunks))
            m['num_pred_chunks'].append(len(pred_chunks))
            m['gold_chunk_length'].extend(len(chunk) for chunk in gold_chunks)
            m['pred_chunk_length'].extend(len(chunk) for chunk in pred_chunks)

            m['action_high_f1'].append(compute_f1(gold_labels, pred_labels))
            m['action_high_em'].append(compute_exact(gold_labels, pred_labels))
            m['action_high_gold_length'].append(len(gold_labels))
            m['action_high_pred_length'].append(len(pred_labels))
            m['action_high_edit_distance'].append(compute_edit_distance(gold_labels, pred_labels))

        return {k: sum(v)/len(v) for k, v in m.items()}