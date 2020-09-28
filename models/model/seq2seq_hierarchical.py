import os
import torch
import numpy as np
import nn.vnn as vnn
import collections
import tqdm
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from model.seq2seq import Module as Base

from models.utils.helper_utils import dropout_mask_like
from models.utils.metric import compute_f1, compute_exact, compute_edit_distance
from gen.utils.image_util import decompress_mask
import pdb
from vocab.vocab import Vocab

from models.model.base import embed_packed_sequence, move_dict_to_cuda


class Module(Base):

    # Static variables. 
    #max_subgoals = 25
    #feat_pt = 'feat_conv.pt'
    # TODO Are these still needed? 

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

    noop_index = submodule_names.index('NoOp')

    def __init__(self, args, vocab):
        '''
        Seq2Seq agent
        '''
        super().__init__(args, vocab)

        self.args = args

        self.controller_type = vars(args).get('hierarchical_controller', 'attention')
        assert self.controller_type in {'attention', 'chunker'}, "invalid --hierarchical_controller {}".format(self.controller_type)

        # TODO remove hardcoding and base on list of module names or something.
        # n_modules = 8

        # Add high level vocab.
        self.vocab['high_level'] = Vocab()
        self.vocab['high_level'].word2index(self.submodule_names, train=True)

        # encoder and self-attention for starting state modules and high-level controller.
        self.enc = nn.LSTM(args.demb, args.dhid, bidirectional=True, batch_first=True)
        if self.controller_type == 'attention':
            self.enc_att = nn.ModuleList([vnn.SelfAttn(args.dhid*2) for i in range(2)]) # One for submodules and one for controller.
        else:
            self.enc_att = nn.ModuleList([vnn.SelfAttn(args.dhid*2)]) # just for submodules

        # subgoal monitoring
        self.subgoal_monitoring = (self.args.pm_aux_loss_wt > 0 or self.args.subgoal_aux_loss_wt > 0)

        cloned_module_initialization = vars(args).get("cloned_module_initialization", False)
        init_model_path = vars(args).get("init_model_path", None)
        modularize_actor_mask = vars(args).get('modularize_actor_mask', None)

        model_dict = self.state_dict()
        # Load pretrained parameters if desired.
        if init_model_path: 
        
            # Load model parameters first. 
            params = torch.load(init_model_path, map_location='cpu')['model']

            # Filter out only the paramters we need. 
            loading_params = {k: params[k] for k in params if 'emb_' in k or 'enc.' in k}
           
            for i in range(len(self.enc_att)): 
                for k in params.keys(): 
                    if 'enc_att' in k:

                        # Hack for getting the number inside the key. 
                        new_k = k.split('.')
                        new_k = [new_k[0]] + [str(i)] + new_k[1:]
                        new_k = '.'.join(new_k)

                        assert new_k in model_dict

                        # Clone encoder attention. 
                        loading_params[new_k] = params[k]

            # Load parameters. 
            model_dict.update(loading_params)
            self.load_state_dict(model_dict)

        # frame mask decoder
        if self.subgoal_monitoring:
            decoder = vnn.ConvFrameMaskDecoderProgressMonitor
            
        elif args.indep_modules:
            print("Indep Modules!")
            decoder = vnn.ConvFrameMaskDecoderModularIndependent
        else:
            decoder = vnn.ConvFrameMaskDecoderModular

        assert not (args.hstate_dropout != 0 and args.variational_hstate_dropout != 0)

        self.dec = decoder(self.emb_action_low, args.dframe, 2*args.dhid,
                           pframe=args.pframe,
                           attn_dropout=args.attn_dropout,
                           hstate_dropout=args.hstate_dropout,
                           actor_dropout=args.actor_dropout,
                           input_dropout=args.input_dropout,
                           teacher_forcing=args.dec_teacher_forcing,
                           controller_type=self.controller_type,
                           cloned_module_initialization=cloned_module_initialization, # TODO(dfried): add controller type for ConvFrameMaskDecoderProgressMonitor
                           init_model_path=init_model_path,
                           modularize_actor_mask=modularize_actor_mask,
                           h_translation=args.hierarchical_h_translation,
                           c_translation=args.hierarchical_c_translation,
                           )
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
    def featurize(cls, ex, args, test_mode, load_mask=True, load_frames=True):
        '''
        tensorize and pad batch input
        '''
        feat = super().featurize(ex, args, test_mode, load_mask=load_mask, load_frames=load_frames)
        
        #########
        # outputs
        #########
        
        # Only get ground truth high level pddl plan. 
        module_names_per_subgoal = [subgoal['discrete_action']['action'] for subgoal in ex['plan']['high_pddl']]
        feat['module_idxs_per_subgoal'] = [cls.submodule_names.index(name) for name in module_names_per_subgoal]

        if not test_mode:
            if load_mask:
                feat['action_low_mask'] = [cls.decompress_mask(a['mask']) for a in ex['num']['action_low'] if a['mask'] is not None]

            # Ground truth high-idx. 
            high_idxs = [a['high_idx'] for a in ex['num']['action_low']]
            module_names = [ex['plan']['high_pddl'][idx]['discrete_action']['action'] for idx in high_idxs]
            for i, name in enumerate(module_names):
                if name == 'NoOp':
                    assert i == len(module_names) - 1, 'NoOp found before end of the high level sequence'
            #NoOp has been added to module list, should resolve to correct index through name. 
            #module_names = [a if a != 'NoOp' else 'GotoLocation' for a in module_names] # No-op action will not be used, use index we actually have submodule for.
            module_idxs = [cls.submodule_names.index(name) for name in module_names]

            feat['module_idxs'] = np.array(module_idxs)

            # Get indexes of transitions between subgoals.
            transition_one_hot = cls.get_transitions(feat['module_idxs'])

            # Get indexes of transition time steps.
            transition_idxs = np.nonzero(transition_one_hot)[0]

            # Inject STOP action at each transition point.
            feat['action_low'] = np.insert(feat['action_low'], transition_idxs, 2)
            feat['action_low'][-1] = 2

            # Get submodule idxs right before transition points.
            vals = feat['module_idxs'][transition_idxs - 1]

            # Extend each submodule to account for STOP action.
            feat['module_idxs'] = np.insert(feat['module_idxs'], transition_idxs, vals)

            # NoOP added back above, don't need this? 
            # Add High-level STOP action to high-level controller. 
            #if not args.subgoal_pairs: 
                #feat['module_idxs'][-1] = cls.noop_index

            # Attention masks for high level controller.
            attn_mask = np.zeros((len(feat['module_idxs']), 9))
            attn_mask[np.arange(len(feat['module_idxs'])),feat['module_idxs']] = 1.0
            feat['controller_attn_mask'] = attn_mask

            # Used to mask loss for high-level controller.
            feat['controller_loss_mask'] = np.ones((len(feat['module_idxs']),))

            # Copy image frames to account for STOP action additions.
            new_frames = []

            for i in range(len(transition_one_hot)):

                # If in transition, additionally add the last frame from the last time step.
                if transition_one_hot[i] == 1:
                    new_frames.append(feat['frames'][i-1])

                # Determine which frame to add (accounting for rare SliceObject discrepancy). 
                if i >= len(feat['frames']): 
                    frame = feat['frames'][-1]
                else: 
                    frame = feat['frames'][i]

                new_frames.append(frame)

            feat['frames'] = np.stack(new_frames)
            # feat['frames'] = torch.stack(new_frames)

            # low-level action mask was here, but is now created in the super class

            # low-level valid interact
            # action_low_valid_interact is now initially created in the super class
            # Add invalid interactions to account for stop action.
            
            feat['action_low_valid_interact'] = np.insert(feat['action_low_valid_interact'], transition_idxs, 0)

            # Get transition mask for training attention mechanism with STOP action in mind.
            feat['transition_mask'] = cls.get_transitions(feat['module_idxs'], first_subgoal=True) 

            # Evaluate low level actions.
            #label_actions = self.vocab['action_low'].index2word(feat['action_low'].tolist())
            #label_actions_no_stop = [ac for ac in label_actions if ac != '<<stop>>' and ac != '<<pad>>']
            #assert label_actions_no_stop == [a['discrete_action']['action'] for a in ex['plan']['low_actions']] 

        return feat

    @classmethod
    def serialize_lang_action(cls, feat, test_mode):
        '''
        append segmented instr language and low-level actions into single sequences
        '''
        try:
            # TODO: process lang_instr_subgoal_mask, see base.py
            is_serialized = not isinstance(feat['num']['lang_instr'][0], list)
            if not is_serialized:
                feat['num']['lang_instr'] = [word for desc in feat['num']['lang_instr'] for word in desc]
                if not test_mode:
                    feat['num']['action_low'] = [a for a_group in feat['num']['action_low'] for a in a_group]
        except: 
            pass#pdb.set_trace()

    def tensorize(self, feat): 
        """
        Tensorisation method for evaluation. 
        """
        # TODO Need to refactor this with AlfredDataset in seq2seq.py somehow. 

        # tensorization and padding
        for k, v in feat.items():
            if k in {'lang_goal_instr'}:
                # language embedding and padding
                seq = torch.tensor(v)
                seq_length = len(v)
                feat[k] = (seq, seq_length)

            else:
                # default: tensorize and pad sequence
                seq = torch.tensor(v, dtype=torch.long)
                feat[k] = seq

        return feat

    @classmethod
    def decompress_mask(cls, compressed_mask):
        '''
        decompress mask from json files
        '''
        mask = np.array(decompress_mask(compressed_mask))
        mask = np.expand_dims(mask, axis=0)
        return mask

    def tensorize_lang(self, feat):
        # TODO: this is no longer used, was moved to AlfredDataset

        # Finish vectorizing language. 
        pad_seq, seq_lengths = feat['lang_goal_instr']
        
        if self.args.gpu: 
            pad_seq = pad_seq.cuda()
    
        embed_seq = self.emb_word(pad_seq)

        # Singleton corner case. 
        if len(embed_seq.shape) == 2: 
            embed_seq = embed_seq.unsqueeze(0)

        if type(seq_lengths) == int: 
            seq_lengths = torch.Tensor([seq_lengths]).long()

            if self.args.gpu: 
                seq_lengths = seq_lengths.cuda()

        packed_input = pack_padded_sequence(embed_seq, seq_lengths, batch_first=True, enforce_sorted=False)
        feat['lang_goal_instr'] = packed_input

    def forward(self, feat, max_decode=300):
        if self.args.gpu:
            move_dict_to_cuda(feat)
        cont_lang, enc_lang = self.encode_lang(feat)

        # Each module will have its own cell, but all will share a hidden state. TODO should we only have one cell state too? 
        state_0_transitioned = h, c = cont_lang[0], torch.zeros_like(cont_lang[0])
        # hacky check: use nans here to ensure that we're taking from transitioned for the initial state
        state_0 = torch.full_like(state_0_transitioned[0], float("nan")), torch.full_like(state_0_transitioned[1], float("nan"))
        if self.args.variational_hstate_dropout != 0.0 and self.training:
            hstate_dropout_mask = dropout_mask_like(h, self.args.variational_hstate_dropout)
        else:
            hstate_dropout_mask = torch.full_like(h, 1.0)

        if self.controller_type == 'attention':
            controller_state_0 = cont_lang[1], torch.zeros_like(cont_lang[1])
        else:
            controller_state_0 = None
        frames = self.vis_dropout(feat['frames'])
        
        # Get ground truth attention if provided.
        # this is provided both in train and in valid_seen and unseen
        if 'controller_attn_mask' in feat: 
            controller_mask = feat['controller_attn_mask']
        else: 
            controller_mask = None

        transition_mask = feat['transition_mask']

        res = self.dec(
            enc_lang, frames, max_decode=max_decode, gold=feat['action_low'],
            state_0=state_0, state_0_transitioned=state_0_transitioned,
            controller_state_0=controller_state_0, controller_mask=controller_mask,
            transition_mask=transition_mask, hstate_dropout_mask=hstate_dropout_mask
        )
        feat.update(res)
        return feat

    def encode_lang(self, feat):
        '''
        encode goal+instr language
        '''
        enc_lang_goal_instr = self.encode_lang_base(feat)
        cont_lang_goal_instr = [enc_att(enc_lang_goal_instr) for enc_att in self.enc_att]
        
        return cont_lang_goal_instr, enc_lang_goal_instr


    def reset(self):
        '''
        reset internal states (used for real-time execution during eval)
        '''
        self.r_state = {
            'state_t': None,
            'state_transitioned_t': None,
            'e_t': None,
            'cont_lang': None,
            'enc_lang': None,
            'subgoal': None,
            'subgoal_counter': 0
        }

    def step(self, feat, prev_action=None, oracle=False, module_idxs_per_subgoal=None, allow_submodule_stop=True, force_submodule_stop=False):
        '''
        forward the model for a single time-step (used for real-time execution during eval)
        '''

        # encode language features
        if self.r_state['cont_lang'] is None and self.r_state['enc_lang'] is None:
            self.r_state['cont_lang'], self.r_state['enc_lang'] = self.encode_lang(feat)

        # initialize embedding and hidden states
        if self.r_state['e_t'] is None and self.r_state['state_t'] is None:
            self.r_state['e_t'] = self.dec.go.repeat(self.r_state['enc_lang'].size(0), 1)

            h, c = self.r_state['cont_lang'][0], torch.zeros_like(self.r_state['cont_lang'][0])
            if self.args.variational_hstate_dropout != 0.0 and self.training:
                hstate_dropout_mask = dropout_mask_like(h, self.args.variational_hstate_dropout)
            else:
                hstate_dropout_mask = torch.full_like(h, 1.0)
            self.r_state['state_transitioned_t'] = h, c
            self.r_state['state_t'] = torch.full_like(h, float("nan")), torch.full_like(c, float("nan"))
            self.r_state['hstate_dropout_mask'] = hstate_dropout_mask
            if self.controller_type == 'attention':
                self.r_state['controller_state_t'] = self.r_state['cont_lang'][1], torch.zeros_like(self.r_state['cont_lang'][1])
            else:
                self.r_state['controller_state_t'] = None

        if oracle:
            assert module_idxs_per_subgoal is None, "can't pass both module_idxs_per_subgoal and oracle=True"
            # get rid of the batch dimension
            module_idxs_per_subgoal = feat['module_idxs_per_subgoal'].squeeze(0)
            assert module_idxs_per_subgoal.dim() == 1
            module_idxs_per_subgoal = module_idxs_per_subgoal.tolist() # prevent indexing errors

        if self.r_state['subgoal_counter'] == 0:

            # Force subgoal module selection if using oracle. 
            if module_idxs_per_subgoal is not None:
                controller_mask = torch.zeros(1, 9).float()
                controller_mask[:,module_idxs_per_subgoal[0]] = 1.0

                if self.args.gpu:
                    controller_mask = controller_mask.cuda()

                self.r_state['subgoal'] = controller_mask

            self.r_state['subgoal_counter'] += 1

        # previous action embedding
        e_t = self.embed_action(prev_action) if prev_action is not None else self.r_state['e_t']

        # decode and save embedding and hidden states
        out_action_low, out_action_low_mask, state_t, state_transitioned_t, controller_state_t, _, controller_attn_logits, out_controller_attn = self.dec.step(
            self.r_state['enc_lang'], feat['frames'][:, 0], e_t=e_t,
            state_tm1=self.r_state['state_t'], state_transitioned_tm1=self.r_state['state_transitioned_t'],
            controller_state_tm1=self.r_state['controller_state_t'],
            controller_mask=self.r_state['subgoal'],
            hstate_dropout_mask=self.r_state['hstate_dropout_mask'],
        )
        # out_controller_attn will always be a one-hot distribution. if self.r_state['subgoal'] = None, this will be
        # the argmax of the distribution predicted by the attention-based controller. Otherwise, it will equal to self.r_state['subgoal']
        # So, if we're running in an oracle setting or with module_idxs_per_subgoal, this will be the index of the module that was
        # used for this step.

        # Get selected low-level action
        max_action_low = out_action_low.max(1)[1]

        # If current subgoal predicted stop, then change subgoal module.
        if force_submodule_stop or (allow_submodule_stop and max_action_low == self.stop_token):
            self.r_state['subgoal'] = None

        # Select next subgoal module to pay attention to.
        if self.r_state['subgoal'] is None:

            # Either use max over attention or oracle subgoal selection. 
            if module_idxs_per_subgoal is not None:
                if self.r_state['subgoal_counter'] < len(module_idxs_per_subgoal):
                    max_subgoal = module_idxs_per_subgoal[self.r_state['subgoal_counter']]
                else:
                    max_subgoal = module_idxs_per_subgoal[-1]
                    print("warning: advanced past the end of module_idxs_per_subgoal; using last subgoal {}".format(max_subgoal))
            else:
                # Only pay attention to a single module.
                max_subgoal = controller_attn_logits.max(2)[1].squeeze()
                
            module_attn = torch.zeros_like(out_controller_attn).view(1,-1)
            module_attn[:,max_subgoal] = 1.0

            self.r_state['subgoal'] = module_attn
            self.r_state['subgoal_counter'] += 1

        # save states
        self.r_state['state_t'] = state_t
        self.r_state['state_transitioned_t'] = state_transitioned_t
        self.r_state['controller_state_t'] = controller_state_t
        self.r_state['e_t'] = self.dec.emb(max_action_low)

        # output formatting
        feat['out_action_low'] = out_action_low.unsqueeze(0)
        feat['out_action_low_mask'] = out_action_low_mask.unsqueeze(0)
        feat['out_module_attn_scores'] = out_controller_attn.view(1,1,9)
        feat['modules_used'] = out_controller_attn.view(1,1,9)

        return feat

    def extract_preds(self, out, batch, feat, clean_special_tokens=True, return_masks=False, allow_stop=True):
        '''
        output processing
        '''
        pred = {}

        if not allow_stop:
            raise NotImplementedError("allow_stop=False")

        for ix, (ex, alow, alow_mask) in enumerate(zip(
            batch, feat['out_action_low'].max(2)[1].tolist(), feat['out_action_low_mask'],
        )):
            """
            # remove padding tokens
            if self.pad in alow:
                pad_start_idx = alow.index(self.pad)
                alow = alow[:pad_start_idx]
                alow_mask = alow_mask[:pad_start_idx]
            """

            # feat['modules_used']: batch x time x num_modules (different shape than out_module_attn_scores!)
            assert feat['modules_used'].size(2) == len(self.submodule_names)
            modules_used = feat['modules_used'].max(2)[1].tolist()[ix]
            if clean_special_tokens:
                # Stop after high-level controller stops.
                if self.noop_index in modules_used:
                    stop_start_idx = modules_used.index(self.noop_index)
                    modules_used = modules_used[:stop_start_idx+1]

            if self.controller_type == 'attention':
                # TODO: out_module_attn_scores means different things when produced by dec.forward() and dec.step()
                controller_attn = feat['out_module_attn_scores'].max(2)[1].tolist()[ix]
                if clean_special_tokens:

                    # Stop each trajectory after high-level controller stops.
                    if self.noop_index in controller_attn:
                        stop_start_idx = controller_attn.index(self.noop_index)
                        alow = alow[:stop_start_idx]
                        controller_attn = controller_attn[:stop_start_idx+1]
                        alow_mask = alow_mask[:stop_start_idx+1]

                    """
                    # remove <<stop>> tokens
                    if self.stop_token in alow:
                        stop_start_idx = alow.index(self.stop_token)
                        alow = alow[:stop_start_idx]
                        alow_mask = alow_mask[:stop_start_idx]
                    """
            else:
                pass
                # TODO(dfried): truncate alow after seeing n stops, where n is the number of subgoals

            # index to API actions
            #words = self.vocab['action_low'].index2word(alow)

            # sigmoid preds to binary mask
            alow_mask = F.sigmoid(alow_mask)
            p_mask = [(alow_mask[t] > 0.5).cpu().numpy() for t in range(alow_mask.shape[0])]

            key = self.get_instance_key(ex)

            words = self.vocab['action_low'].index2word(alow)

            # print(len(p_mask))
            pred[key] = {
                'action_low_names': words,
                'action_low_idxs': alow,
                'modules_used': modules_used,
            }
            if self.controller_type == 'attention':
                pred[key]['controller_attn'] = controller_attn # I think this has one element for each timestep, plus NoOp (stop)
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

        if self.controller_type == 'attention':
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
        for ex, feat in tqdm.tqdm(data, ncols=80, desc='compute_metric'):
            key = self.get_instance_key(ex)
            # feat should already contain the following, since all AlfredDataset s which are fed into this function have test_mode=False
            # feat = self.featurize(ex, self.args, False, load_mask=True, load_frames=True)

            # Evaluate low level actions.
            label_actions = self.vocab['action_low'].index2word(feat['action_low'].tolist())
            label_actions_no_stop = [ac for ac in label_actions if ac != '<<stop>>' and ac != '<<pad>>']
            #assert label_actions_no_stop == [a['discrete_action']['action'] for a in ex['plan']['low_actions']]
            pred_actions = preds[key]['action_low_names']
            pred_actions_no_stop = [ac for ac in pred_actions if ac != '<<stop>>' and ac != '<<pad>>']

            m['action_low_f1'].append(compute_f1(label_actions_no_stop, pred_actions_no_stop))
            m['action_low_em'].append(compute_exact(label_actions_no_stop, pred_actions_no_stop))
            m['action_low_gold_length'].append(len(label_actions_no_stop))
            m['action_low_pred_length'].append(len(pred_actions_no_stop))
            m['action_low_edit_distance'].append(compute_edit_distance(label_actions_no_stop, pred_actions_no_stop))

            # Evaluate high-level controller.
            # Get indexes of predicted transitions.
            # self.stop_token is actually an index into the vocabulary
            stop_idxs = np.argwhere(np.array(preds[key]['action_low_idxs'])[:-1] == self.stop_token).flatten()
            high_idxs = np.append([0], stop_idxs + 1).astype(np.int32)

            # Get predicted submodule transitions
            if self.controller_type == 'attention':
                pred_high_idx = np.array(preds[key]['controller_attn'])[high_idxs]
                label_high_idx = feat['module_idxs'][np.nonzero(feat['transition_mask'])]

                m['action_high_f1'].append(compute_f1(label_high_idx, pred_high_idx))
                m['action_high_em'].append(compute_exact(label_high_idx, pred_high_idx))

                m['action_high_gold_length'].append(len(label_high_idx))
                m['action_high_pred_length'].append(len(pred_high_idx))
                m['action_high_edit_distance'].append(compute_edit_distance(label_high_idx, pred_high_idx))

        return {k: sum(v)/len(v) for k, v in m.items()}
