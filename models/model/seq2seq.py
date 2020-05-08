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
from models.model.base import BaseModule

class AlfredDataset(Dataset): 

    def __init__(self, args, data, model_class, test_mode): 
        self.data = data
        self.model_class = model_class
        self.args = args
        self.test_mode = test_mode

    def __getitem__(self, idx): 

        # Load task from dataset. 
        task = self.model_class.load_task_json(self.args, self.data[idx], None)[0] 

        # Create dict of features from dict. 
        feat = self.model_class.featurize(task, self.args, self.test_mode)

        return (task, feat)

    def __len__(self): 
        return len(self.data)

    def collate_fn(batch): 
      
        tasks = [e[0] for e in batch]
        feats = [e[1] for e in batch]
        batch = tasks # Stick to naming convention. 

        pad = 0

        # Will hold vectorized features. 
        feat = {}

        # Make batch out feature dicts.
        for k in feats[0].keys(): 
            feat[k] = [element[k] for element in feats]

        # tensorization and padding
        for k, v in feat.items():
            if k in {'lang_goal_instr'}:
                # language embedding and padding
                seqs = [torch.tensor(vv) for vv in v]
                pad_seq = pad_sequence(seqs, batch_first=True, padding_value=pad)
                seq_lengths = torch.from_numpy(np.array(list(map(len, v)))).long()
                feat[k] = (pad_seq, seq_lengths)
                #embed_seq = self.emb_word(pad_seq)
                #packed_input = pack_padded_sequence(embed_seq, seq_lengths, batch_first=True, enforce_sorted=False)
                #feat[k] = packed_input
            elif k in {'action_low_mask'}:
                # mask padding
                seqs = [torch.tensor(vv, dtype=torch.float) for vv in v]
                feat[k] = seqs
            elif k in {'subgoal_progress', 'subgoals_completed'}:
                # auxillary padding
                seqs = [torch.tensor(vv, dtype=torch.float) for vv in v]
                pad_seq = pad_sequence(seqs, batch_first=True, padding_value=pad)
                feat[k] = pad_seq
            
            else:
                # default: tensorize and pad sequence
                seqs = [torch.tensor(vv, dtype=torch.float if ('frames' in k) else torch.long) for vv in v]
                pad_seq = pad_sequence(seqs, batch_first=True, padding_value=pad)
                feat[k] = pad_seq

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

        # emb modules
        self.emb_word = nn.Embedding(len(vocab['word']), args.demb)
        self.emb_action_low = nn.Embedding(len(vocab['action_low']), args.demb)

        # end tokens
        self.stop_token = self.vocab['action_low'].word2index("<<stop>>", train=False)
        self.seg_token = self.vocab['action_low'].word2index("<<seg>>", train=False)

    def forward(self, feat, max_decode=100):
        raise NotImplementedError()

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
            # low-level action
            feat['action_low'] = [a['action'] for a in ex['num']['action_low']]

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
            key = (ex['task_id'], ex['repeat_idx'])
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
