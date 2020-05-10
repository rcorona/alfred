import os
import torch
import numpy as np
from torch import nn
from tqdm import trange
import tqdm
from torch.utils.data import Dataset, DataLoader
import pdb
from copy import deepcopy
import pickle
import time
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from models.model.base import BaseModule, embed_packed_sequence
from transformers import BertModel, BertTokenizer

class Module(BaseModule):

    # static sentinel tokens
    pad = 0
    seg = 1

    # Static variables.
    feat_pt = 'feat_conv.pt'

    max_subgoals = 25

    def __init__(self, args, vocab):
        '''
        Base Seq2Seq agent with common train and val loops
        '''
        super().__init__(args)

        self.vocab = vocab

        # Word embeddings. 
        if args.lang_model == 'default': 
            self.emb_word = nn.Embedding(len(vocab['word']), args.demb)
        
        elif args.lang_model == 'bert': 
            
            # Load BERT tokenizer and model. 
            self.__class__.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.emb_word = BertModel.from_pretrained('bert-base-uncased')
            self.lang_projection = nn.Sequential(nn.Linear(768, args.dhid * 2), nn.ReLU())

        # Low-level actions.  
        self.emb_action_low = nn.Embedding(len(vocab['action_low']), args.demb)

        # end tokens
        self.stop_token = self.vocab['action_low'].word2index("<<stop>>", train=False)
        self.seg_token = self.vocab['action_low'].word2index("<<seg>>", train=False)

    def forward(self, feat, max_decode=100):
        raise NotImplementedError()

    @classmethod
    def post_process_lang(cls, ex, args): 
        
        # Default partitioning. 
        if args.lang_model == 'default':
            goal, instr = ex['num']['lang_goal'], ex['num']['lang_instr']

        elif args.lang_model == 'bert': 

            # Get strings for goal and low-level instruction. 
            goal = ''.join(['[SEP]'] + ex['ann']['goal'][:-1] + ['[SEP]'])
            instr = ''.join(['[CLS]'] + [word for desc in ex['ann']['instr'] for word in desc])

            # Run them through tokenizer and concatenate. 
            goal = cls.tokenizer.encode(goal)
            instr = cls.tokenizer.encode(instr)

        return goal, instr

    def encode_lang_base(self, feat): 

        lang_goal_instr = feat['lang_goal_instr'].long()
        seq_lengths = torch.from_numpy(np.array(list(map(len, lang_goal_instr)))).long()

        # Place on GPU if needed. 
        if self.args.gpu: 
            lang_goal_instr = lang_goal_instr.cuda()
            seq_lengths = seq_lengths.cuda()

        if self.args.lang_model == 'default': 
            embed_seq = self.emb_word(lang_goal_instr)
            emb_lang_goal_instr = pack_padded_sequence(embed_seq, seq_lengths, batch_first=True, enforce_sorted=False)
            self.lang_dropout(emb_lang_goal_instr.data)
            enc_lang_goal_instr, _ = self.enc(emb_lang_goal_instr)
            enc_lang_goal_instr, _ = pad_packed_sequence(enc_lang_goal_instr, batch_first=True)

        elif self.args.lang_model == 'bert': 
            
            # Attention mask to ignore padding tokens. 
            attention_mask = lang_goal_instr != 0
            enc_lang_goal_instr = self.emb_word(lang_goal_instr, attention_mask)[0]

            # Project language embedding to the expected dimensionality by rest of model. 
            enc_lang_goal_instr = self.lang_projection(enc_lang_goal_instr)

        self.lang_dropout(enc_lang_goal_instr)

        return enc_lang_goal_instr

    @classmethod
    def featurize(cls, ex, args, test_mode, load_mask=True, load_frames=True):
        '''
        tensorize and pad batch input
        '''
        feat = {}

        ###########
        # auxillary
        ###########

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

        # goal and instr language. 
        lang_goal, lang_instr = cls.post_process_lang(ex, args)

        # zero inputs if specified
        lang_goal = cls.zero_input(lang_goal) if args.zero_goal else lang_goal
        lang_instr = cls.zero_input(lang_instr) if args.zero_instr else lang_instr

        # append goal + instr
        if args.lang_model == 'default': 
            lang_goal_instr = lang_goal + lang_instr
        elif args.lang_model == 'bert': 
            lang_goal_instr = lang_instr + lang_goal

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
            assert all(x is not None for x in keep)
            keep.append(keep[-1])  # stop frame
            feat['frames'] = torch.stack(keep, dim=0)

        #########
        # outputs
        #########

        if not test_mode:
            # low-level action
            feat['action_low'] = [a['action'] for a in ex['num']['action_low']]

            ## this warning prints for < 10 examples (out of 20K) in the training set,
            ## and it's due to the segment merging in merge_last_two_low_actions in preprocess.py
            # if len(feat['action_low']) != feat['frames'].size(0):
            #     key = ex['task_id'], ex['repeat_idx'], ex['subgoal_idx']
            #     print("warning: number of actions {} does not match number of frames {} in instance {}".format(
            #         len(feat['action_low']), feat['frames'].size(0), key
            #     ))

            # low-level valid interact
            feat['action_low_valid_interact'] = np.array([a['valid_interact'] for a in ex['num']['action_low']])

        return feat

    def make_debug(self, preds, data):
        '''
        readable output generator for debugging
        '''
        debug = {}
        for ex, feat in tqdm.tqdm(data, ncols=80, desc='make_debug'):
            # if 'repeat_idx' in ex: ex = self.load_task_json(ex, None)[0]
            key = self.get_instance_key(ex)
            this_debug = {
                'lang_goal': ex['turk_annotations']['anns'][ex['ann']['repeat_idx']]['task_desc'],
                'action_low': [a['discrete_action']['action'] for a in ex['plan']['low_actions']],
                'p_action_low': preds[key]['action_low']
            }
            if 'controller_attn' in preds[key]:
                this_debug['p_action_high'] = preds[key]['controller_attn']
            debug['{}--{}'.format(*key)] = this_debug
        return debug

    @classmethod
    def zero_input(cls, x, keep_end_token=True):
        '''
        pad input with zeros (used for ablations)
        '''
        end_token = [x[-1]] if keep_end_token else [cls.pad]
        return list(np.full_like(x[:-1], cls.pad)) + end_token

    def zero_input_list(self, x, keep_end_token=True):
        '''
        pad a list of input with zeros (used for ablations)
        '''
        end_token = [x[-1]] if keep_end_token else [pad]
        lz = [list(np.full_like(i, pad)) for i in x[:-1]] + end_token
        return lz

    @classmethod
    def has_interaction(cls, action):
        '''
        check if low-level action is interactive
        '''
        non_interact_actions = ['MoveAhead', 'Rotate', 'Look', '<<stop>>', '<<pad>>', '<<seg>>']
        if any(a in action for a in non_interact_actions):
            return False
        else:
            return True
