import os

import torch
import numpy as np
import nn.vnn as vnn
import collections
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from model.seq2seq import Module as Base

from models.nn.vnn import MLP
from models.utils.metric import compute_f1, compute_exact, compute_edit_distance
from gen.utils.image_util import decompress_mask
import pdb
import tqdm

from models.model.base import embed_packed_sequence, move_dict_to_cuda


class Module(Base):
    def __init__(self, args, vocab):
        '''
        Seq2Seq agent
        '''
        super().__init__(args, vocab)

        # encoder and self-attention
        self.enc = nn.LSTM(args.demb, args.dhid, bidirectional=True, batch_first=True)
        self.enc_att = vnn.SelfAttn(args.dhid*2)

        # subgoal monitoring
        self.subgoal_monitoring = (self.args.pm_aux_loss_wt > 0 or self.args.subgoal_aux_loss_wt > 0)

        # frame mask decoder
        decoder = vnn.ConvFrameMaskDecoderProgressMonitor if self.subgoal_monitoring else vnn.ConvFrameMaskDecoder
        self.dec = decoder(self.emb_action_low, args.dframe, 2*args.dhid,
                           pframe=args.pframe,
                           attn_dropout=args.attn_dropout,
                           hstate_dropout=args.hstate_dropout,
                           actor_dropout=args.actor_dropout,
                           input_dropout=args.input_dropout,
                           teacher_forcing=args.dec_teacher_forcing)

        # encoder -- decoder transformation
        connection_type = vars(self.args).get('encoder_decoder_transform', 'identity')
        if connection_type == 'identity':
            self.enc_to_dec = nn.Sequential()
        elif connection_type == 'linear':
            self.enc_to_dec = nn.Linear(args.dhid*2, args.dhid*2)
        elif connection_type == 'mlp':
            # TODO: this is massive! but I'm worried about the low-rank if we bottleneck; maybe overparam is good
            self.enc_to_dec = MLP(args.dhid*2, args.dhid*2, args.dhid*2, num_linears=2)
        elif connection_type == 'zero':
            self.enc_to_dec = None
        else:
            raise ValueError("invalid --encoder_decoder_transform {}".format(connection_type))

        learn_cell_init = vars(self.args).get('learn_cell_init', False)
        if learn_cell_init:
            self.cell_init = nn.Parameter(torch.zeros(args.dhid*2), requires_grad=True)
        else:
            self.cell_init = None

        # dropouts
        self.vis_dropout = nn.Dropout(args.vis_dropout)
        self.lang_dropout = nn.Dropout(args.lang_dropout, inplace=True)
        self.input_dropout = nn.Dropout(args.input_dropout)

        # TODO: this is only used by leaderboard
        self.test_mode = False

        # bce reconstruction loss
        self.bce_with_logits = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.mse_loss = torch.nn.MSELoss(reduction='none')

        # paths
        self.root_path = os.getcwd()

        # params
        self.max_subgoals = 25

        # reset model
        self.reset()

    @classmethod
    def decompress_mask(self, compressed_mask):
        '''
        decompress mask from json files
        '''
        mask = np.array(decompress_mask(compressed_mask))
        mask = np.expand_dims(mask, axis=0)
        return mask

    @classmethod
    def featurize(cls, ex, args, test_mode, load_mask=True, load_frames=True):
        feat = super().featurize(ex, args, test_mode, load_mask=load_mask, load_frames=load_frames)
        if not test_mode:
            # low-level action mask
            if load_mask:
                feat['action_low_mask'] = [cls.decompress_mask(a['mask']) for a in ex['num']['action_low'] if a['mask'] is not None]
        return feat

    def forward(self, feat, max_decode=300):
        if self.args.gpu:
            move_dict_to_cuda(feat)
        cont_lang, enc_lang = self.encode_lang(feat)

        state_0 = self.create_decoder_init_state(cont_lang)
        frames = self.vis_dropout(feat['frames'])
        res = self.dec(enc_lang, frames, max_decode=max_decode, gold=feat['action_low'], state_0=state_0)
        feat.update(res)
        return feat


    def encode_lang(self, feat):
        '''
        encode goal+instr language
        '''
        enc_lang_goal_instr = self.encode_lang_base(feat)
        cont_lang_goal_instr = self.enc_att(enc_lang_goal_instr)

        return cont_lang_goal_instr, enc_lang_goal_instr


    def reset(self):
        '''
        reset internal states (used for real-time execution during eval)
        '''
        self.r_state = {
            'state_t': None,
            'e_t': None,
            'cont_lang': None,
            'enc_lang': None
        }

    def create_decoder_init_state(self, cont_lang):
        learn_cell_init = vars(self.args).get('learn_cell_init', False)
        if learn_cell_init:
            c = self.cell_init.repeat(cont_lang.size(0), 1)
            assert c.size() == cont_lang.size()
        else:
            c = torch.zeros_like(cont_lang)
        connection_type = vars(self.args).get('encoder_decoder_transform', 'identity')
        if connection_type in {'identity', 'linear', 'mlp'}:
            h = self.enc_to_dec(cont_lang)
        elif connection_type == 'zero':
            h = torch.zeros_like(cont_lang)
        else:
            raise ValueError("invalid --encoder_decoder_transform {}".format(connection_type))
        return h, c

    def step(self, feat, prev_action=None):
        '''
        forward the model for a single time-step (used for real-time execution during eval)
        '''

        # encode language features
        if self.r_state['cont_lang'] is None and self.r_state['enc_lang'] is None:
            self.r_state['cont_lang'], self.r_state['enc_lang'] = self.encode_lang(feat)

        # initialize embedding and hidden states
        if self.r_state['e_t'] is None and self.r_state['state_t'] is None:
            self.r_state['e_t'] = self.dec.go.repeat(self.r_state['enc_lang'].size(0), 1)
            # self.r_state['state_t'] = self.r_state['cont_lang'], torch.zeros_like(self.r_state['cont_lang'])
            self.r_state['state_t'] = self.create_decoder_init_state(self.r_state['cont_lang'])

        # previous action embedding
        e_t = self.embed_action(prev_action) if prev_action is not None else self.r_state['e_t']

        # decode and save embedding and hidden states
        out_action_low, out_action_low_mask, state_t, *_ = self.dec.step(self.r_state['enc_lang'], feat['frames'][:, 0], e_t=e_t, state_tm1=self.r_state['state_t'])

        # save states
        self.r_state['state_t'] = state_t
        self.r_state['e_t'] = self.dec.emb(out_action_low.max(1)[1])

        # output formatting
        feat['out_action_low'] = out_action_low.unsqueeze(0)
        feat['out_action_low_mask'] = out_action_low_mask.unsqueeze(0)
        return feat


    def extract_preds(self, out, batch, feat, clean_special_tokens=True, return_masks=False):
        '''
        output processing
        '''
        pred = {}
        for ex, alow, alow_mask in zip(batch, feat['out_action_low'].max(2)[1].tolist(), feat['out_action_low_mask']):
            # remove padding tokens
            if self.pad in alow:
                pad_start_idx = alow.index(self.pad)
                alow = alow[:pad_start_idx]
                alow_mask = alow_mask[:pad_start_idx]

            if clean_special_tokens:
                # remove <<stop>> tokens
                if self.stop_token in alow:
                    stop_start_idx = alow.index(self.stop_token)
                    alow = alow[:stop_start_idx]
                    alow_mask = alow_mask[:stop_start_idx]

            # index to API actions
            words = self.vocab['action_low'].index2word(alow)

            # sigmoid preds to binary mask
            alow_mask = F.sigmoid(alow_mask)
            p_mask = [(alow_mask[t] > 0.5).cpu().numpy() for t in range(alow_mask.shape[0])]

            key = self.get_instance_key(ex)
            assert key not in pred

            # print(len(p_mask))

            pred[key] = {
                'action_low_names': words,
            }
            if return_masks:
                pred[key]['action_low_mask'] = p_mask

        return pred


    def embed_action(self, action):
        '''
        embed low-level action
        '''
        device = torch.device('cuda') if self.args.gpu else torch.device('cpu')
        action_num = torch.tensor(self.vocab['action_low'].word2index(action), device=device)
        action_emb = self.dec.emb(action_num).unsqueeze(0)
        return action_emb


    def compute_loss(self, out, batch, feat):
        '''
        loss function for Seq2Seq agent
        '''
        losses = dict()

        # GT and predictions
        p_alow = out['out_action_low'].view(-1, len(self.vocab['action_low']))
        l_alow = feat['action_low'].view(-1)
        p_alow_mask = out['out_action_low_mask']
        valid = feat['action_low_valid_interact']

        # action loss
        pad_valid = (l_alow != self.pad)

        alow_loss = F.cross_entropy(p_alow, l_alow, reduction='none')

        alow_loss *= pad_valid.float()
        alow_loss = alow_loss.mean()
        losses['action_low'] = alow_loss * self.args.action_loss_wt

        # mask loss
        valid_idxs = valid.view(-1).nonzero().view(-1)
        flat_p_alow_mask = p_alow_mask.view(p_alow_mask.shape[0]*p_alow_mask.shape[1], *p_alow_mask.shape[2:])[valid_idxs]
        flat_alow_mask = torch.cat(feat['action_low_mask'], dim=0)
        if self.args.gpu:
            flat_alow_mask = flat_alow_mask.cuda()

        if len(flat_alow_mask) > 0:
            alow_mask_loss = self.weighted_mask_loss(flat_p_alow_mask, flat_alow_mask)
            losses['action_low_mask'] = alow_mask_loss * self.args.mask_loss_wt

        # subgoal completion loss
        if self.args.subgoal_aux_loss_wt > 0:
            p_subgoal = feat['out_subgoal'].squeeze(2)
            l_subgoal = feat['subgoals_completed']
            sg_loss = self.mse_loss(p_subgoal, l_subgoal)
            sg_loss = sg_loss.view(-1) * pad_valid.float()
            subgoal_loss = sg_loss.mean()
            losses['subgoal_aux'] = self.args.subgoal_aux_loss_wt * subgoal_loss

        # progress monitoring loss
        if self.args.pm_aux_loss_wt > 0:
            p_progress = feat['out_progress'].squeeze(2)
            l_progress = feat['subgoal_progress']
            pg_loss = self.mse_loss(p_progress, l_progress)
            pg_loss = pg_loss.view(-1) * pad_valid.float()
            progress_loss = pg_loss.mean()
            losses['progress_aux'] = self.args.pm_aux_loss_wt * progress_loss

        return losses


    def weighted_mask_loss(self, pred_masks, gt_masks):
        '''
        mask loss that accounts for weight-imbalance between 0 and 1 pixels
        '''
        bce = self.bce_with_logits(pred_masks, gt_masks)

        flipped_mask = self.flip_tensor(gt_masks)
        inside = (bce * gt_masks).sum() / (gt_masks).sum()
        outside = (bce * flipped_mask).sum() / (flipped_mask).sum()

        return inside + outside


    def flip_tensor(self, tensor, on_zero=1, on_non_zero=0):
        '''
        flip 0 and 1 values in tensor
        '''
        res = tensor.clone()
        res[tensor == 0] = on_zero
        res[tensor != 0] = on_non_zero
        return res


    def compute_metric(self, preds, data):
        '''
        compute f1 and extract match scores for output
        '''
        m = collections.defaultdict(list)
        for ex, feat in tqdm.tqdm(data, ncols=80, desc='compute_metric'):
            key = self.get_instance_key(ex)
            label_actions = [a['discrete_action']['action'] for a in ex['plan']['low_actions']]
            pred_actions = preds[key]['action_low_names']
            m['action_low_f1'].append(compute_f1(label_actions, pred_actions))
            m['action_low_em'].append(compute_exact(label_actions, pred_actions))
            m['action_low_gold_length'].append(len(label_actions))
            m['action_low_pred_length'].append(len(pred_actions))
            m['action_low_edit_distance'].append(compute_edit_distance(label_actions, pred_actions))
        return {k: sum(v)/len(v) for k, v in m.items()}
