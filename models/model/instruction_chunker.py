import os
import random
import tqdm
import collections

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from models.model.base import BaseModule, move_dict_to_cuda, embed_packed_sequence
from models.nn import vnn

def compute_acc(gold, pred):
    assert len(gold) == len(pred), (gold, pred)
    return sum(1.0 if g == p else 0.0 for g, p in zip(gold, pred)) / len(gold)

class Chunker(BaseModule):
    PAD_ID = 0
    BEGIN_CHUNK_ID = 1
    INSIDE_CHUNK_ID = 2

    OUTPUT_IDS = [PAD_ID, BEGIN_CHUNK_ID, INSIDE_CHUNK_ID]

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

        self.chunk_pred_layer = nn.Linear(args.dhid*2, len(self.OUTPUT_IDS))

        # paths
        self.root_path = os.getcwd()

    @classmethod
    def featurize(cls, ex, args, test_mode):
        feat = {}
        cls.serialize_lang_action(ex, test_mode)

        lang_instr = ex['num']['lang_instr']
        feat['lang_instr'] = lang_instr
        feat['lang_instr_len'] = len(lang_instr)
        if not test_mode:
            sub_instr_lengths = ex['num']['sub_instr_lengths']
            assert sum(sub_instr_lengths) == len(lang_instr)
            chunk_tags = torch.full((len(lang_instr),), cls.INSIDE_CHUNK_ID, dtype=torch.long)
            pos = 0
            for ix, length in enumerate(sub_instr_lengths):
                chunk_tags[pos] = cls.BEGIN_CHUNK_ID
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
        for word, tag in zip(instruction, chunk_tags):
            if tag == cls.BEGIN_CHUNK_ID and this_chunk:
                all_chunks.append(this_chunk)
                this_chunk = []
            this_chunk.append(word)
        if this_chunk:
            all_chunks.append(this_chunk)
        return all_chunks

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

            # took these out for now since they're slow to generate, and we can reconstruct from lang_instr and the tags
            # pred_lang_instr_chunked = self.chunk_instruction_into_sentences(
            #     lang_instr_flat, pred_chunk_tags
            # )
            # gold_lang_instr_chunked = [
            #     [word.strip().lower() for word in desc]
            #     for desc in ex['ann']['instr']
            # ]
            # gold_lang_instr_chunked_ = self.chunk_instruction_into_sentences(
            #     gold_lang_instr_flat, gold_chunk_tags
            # )
            # assert gold_lang_instr_chunked_ == gold_lang_instr_chunked

            debug['{}--{}'.format(*key)] = {
                'lang_instr': lang_instr_flat,
                # 'gold_lang_instr_chunked': gold_lang_instr_chunked,
                # 'pred_lang_instr_chunked': pred_lang_instr_chunked,
                'pred_chunk_tags': pred_chunk_tags,
                'gold_chunk_tags': gold_chunk_tags.cpu().tolist(),
            }
        return debug

    def encode_lang(self, feat):
        '''
        encode instr language
        '''
        if self.args.gpu:
            move_dict_to_cuda(feat)

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
        chunk_tag_logits = self.chunk_pred_layer(enc_lang)
        feat.update({'out_chunk_tags': chunk_tag_logits})
        return feat

    def extract_preds(self, out, batch, feat):
        pred = {}
        assert len(batch) == feat['out_chunk_tags'].size(0)
        chunk_tag_mask = (feat['chunk_tags'] == self.PAD_ID)
        assert len(batch) == feat['out_chunk_tags'].size(0) == feat['chunk_tags'].size(0)
        for ex_ix, ex in enumerate(batch):
            key = ex['task_id'], ex['repeat_idx']

            pred_tag_logits = feat['out_chunk_tags'][ex_ix]
            true_tags = feat['chunk_tags'][ex_ix]
            instr_len = feat['lang_instr_len'][ex_ix]
            pred_tags = pred_tag_logits[:instr_len].argmax(dim=-1).cpu().tolist()

            pred[key] = {
                'chunk_tags': pred_tags,
            }
        return pred

    def compute_loss(self, out, batch, feat):
        losses = dict()
        p_tags = out['out_chunk_tags'].view(-1, len(self.OUTPUT_IDS))
        l_tags = feat['chunk_tags'].view(-1)

        pad_valid = (l_tags != self.PAD_ID)
        tag_loss = F.cross_entropy(p_tags, l_tags, reduction='none')
        tag_loss *= pad_valid.float()
        tag_loss = tag_loss.sum() / pad_valid.sum()

        losses['chunk_tags'] = tag_loss

        return losses

    def compute_metric(self, preds, data):
        m = collections.defaultdict(list)
        for ex, feat in data:
            key = (ex['task_id'], ex['repeat_idx'])
            gold_tags = feat['chunk_tags']
            pred_tags = preds[key]['chunk_tags']
            assert len(gold_tags) == len(pred_tags)
            m['chunk_tag_acc'].append(compute_acc(gold_tags, pred_tags))

            flat_instr = feat['lang_instr']

            gold_chunks = self.chunk_instruction_into_sentences(flat_instr, gold_tags)
            pred_chunks = self.chunk_instruction_into_sentences(flat_instr, pred_tags)
            m['num_gold_chunks'].append(len(gold_chunks))
            m['num_pred_chunks'].append(len(pred_chunks))
            m['gold_chunk_length'].extend(len(chunk) for chunk in gold_chunks)
            m['pred_chunk_length'].extend(len(chunk) for chunk in pred_chunks)

        return {k: sum(v)/len(v) for k, v in m.items()}
