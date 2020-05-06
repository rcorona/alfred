import os
import torch
import numpy as np
import nn.vnn as vnn
import collections
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from model.seq2seq import Module as Base
from models.utils.metric import compute_f1, compute_exact
from gen.utils.image_util import decompress_mask
import pdb
from vocab.vocab import Vocab

class Module(Base):

    # Static variables. 
    max_subgoals = 25

    # List for submodule names. 
    submodule_names = [
        'GotoLocation', 
        'PickupObject', 
        'PutObject', 
        'CoolObject', 
        'HeatObject', 
        'CleanObject', 
        'SliceObject', 
        'ToggleObject',
        'STOP'
    ]

    def __init__(self, args, vocab):
        '''
        Seq2Seq agent
        '''
        super().__init__(args, vocab)

        self.args = args

        # TODO remove hardcoding and base on list of module names or something. 
        n_modules = 8

        # Add high level vocab.
        self.vocab['high_level'] = Vocab()
        self.vocab['high_level'].word2index(self.submodule_names, train=True)

        # encoder and self-attention for starting state modules and high-level controller.
        self.enc = nn.LSTM(args.demb, args.dhid, bidirectional=True, batch_first=True)
        self.enc_att = nn.ModuleList([vnn.SelfAttn(args.dhid*2) for i in range(2)]) # One for submodules and one for controller. 

        # subgoal monitoring
        self.subgoal_monitoring = (self.args.pm_aux_loss_wt > 0 or self.args.subgoal_aux_loss_wt > 0)

        # frame mask decoder
        decoder = vnn.ConvFrameMaskDecoderProgressMonitor if self.subgoal_monitoring else vnn.ConvFrameMaskDecoderModular
        self.dec = decoder(self.emb_action_low, args.dframe, 2*args.dhid,
                           pframe=args.pframe,
                           attn_dropout=args.attn_dropout,
                           hstate_dropout=args.hstate_dropout,
                           actor_dropout=args.actor_dropout,
                           input_dropout=args.input_dropout,
                           teacher_forcing=args.dec_teacher_forcing)

        # dropouts
        self.vis_dropout = nn.Dropout(args.vis_dropout)
        self.lang_dropout = nn.Dropout(args.lang_dropout, inplace=True)
        self.input_dropout = nn.Dropout(args.input_dropout)

        # internal states
        self.state_t = None
        self.e_t = None
        self.test_mode = False

        # bce reconstruction loss
        self.bce_with_logits = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.mse_loss = torch.nn.MSELoss(reduction='none')

        # paths
        self.root_path = os.getcwd()

        # reset model
        self.reset()

    @classmethod
    def get_transitions(cls, sequence, first_subgoal=False):
        """
        Get one-hot vector with ones at the inflection points between subgoal segments. 
        """

        transition_one_hot = np.zeros((len(sequence,)))

        # Include value for first-subgoal if desired. 
        if first_subgoal:
            transition_one_hot[0] = 1

        curr_subgoal = sequence[0]

        for i in range(len(transition_one_hot)): 
            if sequence[i] != curr_subgoal: 
                transition_one_hot[i] = 1
                curr_subgoal = sequence[i]

        return transition_one_hot

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
        '''
        tensorize and pad batch input
        '''
        feat = super().featurize(ex, args, test_mode, load_mask=load_mask, load_frames=load_frames)
        #########
        # outputs
        #########

        if load_mask:
            feat['action_low_mask'] = [cls.decompress_mask(a['mask']) for a in ex['num']['action_low'] if a['mask'] is not None]

        # Ground truth high-idx for modular model.
        high_idxs = [a['high_idx'] for a in ex['num']['action_low']]
        module_names = [ex['plan']['high_pddl'][idx]['discrete_action']['action'] for idx in high_idxs]
        module_names = [a if a != 'NoOp' else 'GotoLocation' for a in module_names] # No-op action will not be used, use index we actually have submodule for.
        module_idxs = [cls.submodule_names.index(name) for name in module_names]

        feat['module_idxs'] = np.array(module_idxs)

        # Get indexes of transitions between subgoals.
        transition_one_hot = cls.get_transitions(feat['module_idxs'])

        # Get indexes of transition time steps.
        transition_idxs = np.nonzero(transition_one_hot)[0]

        # Inject STOP action at each transition point.
        feat['action_low'] = np.insert(feat['action_low'], transition_idxs, 2)
        feat['action_low'][-1] = cls.pad

        # Get submodule idxs right before transition points.
        vals = feat['module_idxs'][transition_idxs - 1]

        # Extend each submodule to account for STOP action.
        feat['module_idxs'] = np.insert(feat['module_idxs'], transition_idxs, vals)

        # Add High-level STOP action to high-level controller.
        feat['module_idxs'][-1] = 8

        # Attention masks for high level controller.
        attn_mask = np.zeros((len(feat['module_idxs']), 9))
        attn_mask[np.arange(len(feat['module_idxs'])),feat['module_idxs']] = 1.0
        feat['controller_attn_mask'] = attn_mask

        # Used to mask loss for high-level controller.
        feat['controller_loss_mask'] = np.ones((len(feat['module_idxs']),))

        # Copy image frames to account for STOP action additions.
        new_frames = []

        for i in range(len(feat['frames'])):

            # If in transition, additionally add the last frame from the last time step.
            if transition_one_hot[i] == 1:
                new_frames.append(feat['frames'][i-1])

            new_frames.append(feat['frames'][i])

        feat['frames'] = torch.stack(new_frames)

        # Add invalid interactions to account for stop action.
        feat['action_low_valid_interact'] = np.insert(feat['action_low_valid_interact'], transition_idxs, 0)

        # Get transition mask for training attention mechanism with STOP action in mind.
        feat['transition_mask'] = cls.get_transitions(feat['module_idxs'], first_subgoal=True)

        return feat


    def forward(self, feat, max_decode=300):
        
        # Finish vectorizing language. 
        pad_seq, seq_lengths = feat['lang_goal_instr']
        
        if self.args.gpu: 
            pad_seq = pad_seq.cuda()
        
        embed_seq = self.emb_word(pad_seq)
        packed_input = pack_padded_sequence(embed_seq, seq_lengths, batch_first=True, enforce_sorted=False)
        feat['lang_goal_instr'] = packed_input

        # Move everything onto gpu if needed.
        if self.args.gpu: 
            for k in feat:
                if hasattr(feat[k], 'cuda'):
                    feat[k] = feat[k].cuda()

        cont_lang, enc_lang = self.encode_lang(feat)

        # Each module will have its own cell, but all will share a hidden state. TODO should we only have one cell state too? 
        state_0 = cont_lang[0], torch.zeros_like(cont_lang[0])
        controller_state_0 = cont_lang[1], torch.zeros_like(cont_lang[1])
        frames = self.vis_dropout(feat['frames'])
        
        # Get ground truth attention if provided. 
        if 'controller_attn_mask' in feat: 
            controller_mask = feat['controller_attn_mask']
        else: 
            controller_mask = None

        res = self.dec(enc_lang, frames, max_decode=max_decode, gold=feat['action_low'], state_0=state_0, controller_state_0=controller_state_0, controller_mask=controller_mask)
        feat.update(res)
        return feat

    def encode_lang(self, feat):
        '''
        encode goal+instr language
        '''
        emb_lang_goal_instr = feat['lang_goal_instr']
        self.lang_dropout(emb_lang_goal_instr.data)
        enc_lang_goal_instr, _ = self.enc(emb_lang_goal_instr)
        enc_lang_goal_instr, _ = pad_packed_sequence(enc_lang_goal_instr, batch_first=True)
        self.lang_dropout(enc_lang_goal_instr)
        cont_lang_goal_instr = [enc_att(enc_lang_goal_instr) for enc_att in self.enc_att]
        
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
            self.r_state['state_t'] = self.r_state['cont_lang'][0], torch.zeros_like(self.r_state['cont_lang'][0])
            self.r_state['controller_state_t'] = self.r_state['cont_lang'][1], torch.zeros_like(self.r_state['cont_lang'][1])

        # previous action embedding
        e_t = self.embed_action(prev_action) if prev_action is not None else self.r_state['e_t']

        # decode and save embedding and hidden states
        out_action_low, out_action_low_mask, state_t, controller_state_t, *_ = self.dec.step(self.r_state['enc_lang'], feat['frames'][:, 0], e_t=e_t, state_tm1=self.r_state['state_t'], controller_state_tm1=self.r_state['controller_state_t'])

        # save states
        self.r_state['state_t'] = state_t
        self.r_state['controller_state_t'] = controller_state_t
        self.r_state['e_t'] = self.dec.emb(out_action_low.max(1)[1])

        # output formatting
        feat['out_action_low'] = out_action_low.unsqueeze(0)
        feat['out_action_low_mask'] = out_action_low_mask.unsqueeze(0)
        return feat

    def extract_preds(self, out, batch, feat, clean_special_tokens=True):
        '''
        output processing
        '''
        pred = {}

        for ex, alow, alow_mask, controller_attn in zip(batch, feat['out_action_low'].max(2)[1].tolist(), feat['out_action_low_mask'],
                feat['out_module_attn_scores'].max(2)[1].tolist()):
            """
            # remove padding tokens
            if self.pad in alow:
                pad_start_idx = alow.index(self.pad)
                alow = alow[:pad_start_idx]
                alow_mask = alow_mask[:pad_start_idx]
            """

            if clean_special_tokens:

                # Stop each trajectory after high-level controller stops.
                if 8 in controller_attn:
                    stop_start_idx = controller_attn.index(8)
                    alow = alow[:stop_start_idx]
                    controller_attn = controller_attn[:stop_start_idx]
                    alow_mask = alow_mask[:stop_start_idx]

                """
                # remove <<stop>> tokens
                if self.stop_token in alow:
                    stop_start_idx = alow.index(self.stop_token)
                    alow = alow[:stop_start_idx]
                    alow_mask = alow_mask[:stop_start_idx]
                """

            # index to API actions
            #words = self.vocab['action_low'].index2word(alow)

            # sigmoid preds to binary mask
            alow_mask = F.sigmoid(alow_mask)
            p_mask = [(alow_mask[t] > 0.5).cpu().numpy() for t in range(alow_mask.shape[0])]

            key = (ex['task_id'], ex['repeat_idx'])

            pred[key] = {
                'action_low': alow,
                'action_low_mask': p_mask,
                'controller_attn': controller_attn
            }

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

        # Controller submodule attention loss.
        #attn_loss = F.mse_loss(feat['out_module_attn_scores'].float(), feat['controller_attn_mask'].float(), reduction='none').sum(-1).view(-1)
        attn_loss = F.cross_entropy(feat['out_module_attn_scores'].float().view(-1,9), feat['module_idxs'].view(-1).long(), reduction='none')

        # Mask attention loss both based on each sequences length as well as subgoal transition points in each trajectory. 
        attn_loss = (attn_loss * feat['transition_mask'].view(-1,).float()).mean()
        losses['controller_attn'] = attn_loss

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
        for ex, feat in data:
            # if 'repeat_idx' in ex: ex = self.load_task_json(self.args, ex, None)[0]
            key = (ex['task_id'], ex['repeat_idx'])
            # feat should already contain the following, since all AlfredDataset s which are fed into this function have test_mode=False
            # feat = self.featurize(ex, self.args, False, load_mask=True, load_frames=True)

            # Evaluate low level actions.
            label = ' '.join(self.vocab['action_low'].index2word(feat['action_low'].tolist()))
            pred = ' '.join(self.vocab['action_low'].index2word(preds[key]['action_low']))

            m['action_low_f1'].append(compute_f1(label.lower(), pred.lower()))
            m['action_low_em'].append(compute_exact(label.lower(), pred.lower()))

            # Evaluate high-level controller.
            # Get indexes of predicted transitions.
            stop_idxs = np.argwhere(np.array(preds[key]['action_low'])[:-1] == 2).flatten()
            high_idxs = np.append([0], stop_idxs + 1).astype(np.int32)

            # Get predicted submodule transitions
            pred_high_idx = np.array(preds[key]['controller_attn'])[high_idxs]
            label_high_idx = feat['module_idxs'][np.nonzero(feat['transition_mask'])]

            pred = ' '.join(self.vocab['high_level'].index2word(pred_high_idx.tolist()))
            label = ' '.join(self.vocab['high_level'].index2word(label_high_idx.tolist()))

            m['action_high_f1'].append(compute_f1(label.lower(), pred.lower()))
            m['action_high_em'].append(compute_exact(label.lower(), pred.lower()))

        return {k: sum(v)/len(v) for k, v in m.items()}
