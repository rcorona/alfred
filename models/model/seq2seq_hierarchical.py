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

class Module(Base):

    # Static variables. 
    max_subgoals = 25
    feat_pt = 'feat_conv.pt'

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
        'NoOp'
    ]



    def __init__(self, args, vocab):
        '''
        Seq2Seq agent
        '''
        super().__init__(args, vocab)

        self.args = args

        # TODO remove hardcoding and base on list of module names or something. 
        n_modules = 8

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
    def featurize(cls, ex, args, test_mode, load_mask=True, load_frames=True):
        '''
        tensorize and pad batch input
        '''
        device = torch.device('cuda') if args.gpu else torch.device('cpu')
        feat = {}

        ###########
        # auxillary
        ###########

        if not test_mode:
            # subgoal completion supervision
            if args.subgoal_aux_loss_wt > 0:
                feat['subgoals_completed'] = np.array(ex['num']['low_to_high_idx']) / cls.max_subgoals

            # progress monitor supervision
            if args.pm_aux_loss_wt > 0:
                num_actions = len([a for sg in ex['num']['action_low'] for a in sg])
                subgoal_progress = [(i+1)/float(num_actions) for i in range(num_actions)]
                feat['subgoal_progress'] = subgoal_progress

        #########
        # inputs
        #########

        # serialize segments
        cls.serialize_lang_action(ex, test_mode)

        # goal and instr language
        lang_goal, lang_instr = ex['num']['lang_goal'], ex['num']['lang_instr']

        # zero inputs if specified
        lang_goal = cls.zero_input(lang_goal) if args.zero_goal else lang_goal
        lang_instr = cls.zero_input(lang_instr) if args.zero_instr else lang_instr

        # append goal + instr
        lang_goal_instr = lang_goal + lang_instr
        feat['lang_goal_instr'] = lang_goal_instr

        # load Resnet features from disk
        if load_frames and not test_mode:
            root = cls.get_task_root(ex, args)
            im = torch.load(os.path.join(root, cls.feat_pt))
            keep = [None] * len(ex['plan']['low_actions'])
            for i, d in enumerate(ex['images']):
                # only add frames linked with low-level actions (i.e. skip filler frames like smooth rotations and dish washing)
                if keep[d['low_idx']] is None:
                    keep[d['low_idx']] = im[i]
            keep.append(keep[-1])  # stop frame
            feat['frames'] = torch.stack(keep, dim=0)

        #########
        # outputs
        #########

        if not test_mode:
            
            # Ground trutch high-idx for modular model.
            high_idxs = [a['high_idx'] for a in ex['num']['action_low']]
            module_names = [ex['plan']['high_pddl'][idx]['discrete_action']['action'] for idx in high_idxs]
            module_names = [a if a != 'NoOp' else 'GotoLocation' for a in module_names] # No-op action will not be used, use index we actually have submodule for. 
            module_idxs = [cls.submodule_names.index(name) for name in module_names]

            feat['module_idxs'] = module_idxs

            # Attention masks for high level controller. 
            attn_mask = np.zeros((len(module_idxs), 8))
            attn_mask[np.arange(len(module_idxs)),module_idxs] = 1.0
            feat['controller_attn_mask'] = attn_mask

            # low-level action
            feat['action_low'] = [a['action'] for a in ex['num']['action_low']]

            # low-level action mask
            if load_mask:
                feat['action_low_mask'] = [cls.decompress_mask(a['mask']) for a in ex['num']['action_low'] if a['mask'] is not None]

            # low-level valid interact
            feat['action_low_valid_interact'] = [a['valid_interact'] for a in ex['num']['action_low']]
        
        return feat

    @classmethod
    def serialize_lang_action(cls, feat, test_mode):
        '''
        append segmented instr language and low-level actions into single sequences
        ''' 
        try: 
            is_serialized = not isinstance(feat['num']['lang_instr'][0], list)
            if not is_serialized:
                feat['num']['lang_instr'] = [word for desc in feat['num']['lang_instr'] for word in desc]
                if not test_mode:
                    feat['num']['action_low'] = [a for a_group in feat['num']['action_low'] for a in a_group]
        except: 
            pass#pdb.set_trace()

    @classmethod
    def decompress_mask(cls, compressed_mask):
        '''
        decompress mask from json files
        '''
        mask = np.array(decompress_mask(compressed_mask))
        mask = np.expand_dims(mask, axis=0)
        return mask

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

            pred[ex['task_id']] = {
                'action_low': ' '.join(words),
                'action_low_mask': p_mask,
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
        attn_loss = F.cross_entropy(feat['out_module_attn_scores'].float().view(-1,8), feat['module_idxs'].view(-1).long(), reduction='none')
        attn_loss = (attn_loss * pad_valid.float()).mean()
        losses['controller_attn'] = attn_loss

        # mask loss
        valid_idxs = valid.view(-1).nonzero().view(-1)
        flat_p_alow_mask = p_alow_mask.view(p_alow_mask.shape[0]*p_alow_mask.shape[1], *p_alow_mask.shape[2:])[valid_idxs]
        flat_alow_mask = torch.cat(feat['action_low_mask'], dim=0)
            
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
        pred_masks = pred_masks.cuda()
        gt_masks = gt_masks.cuda()
        
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
            if 'repeat_idx' in ex: ex = self.load_task_json(self.args, ex, None)[0]
            i = ex['task_id']
            label = ' '.join([a['discrete_action']['action'] for a in ex['plan']['low_actions']])
            m['action_low_f1'].append(compute_f1(label.lower(), preds[i]['action_low'].lower()))
            m['action_low_em'].append(compute_exact(label.lower(), preds[i]['action_low'].lower()))
        return {k: sum(v)/len(v) for k, v in m.items()}
