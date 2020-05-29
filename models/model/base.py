import collections
import json
import os
import pickle
import pprint
import random
import time
import copy
from copy import deepcopy
import math

from typing import Set

import numpy as np
import torch
import tqdm
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, PackedSequence
from torch.utils.data import Dataset, DataLoader
from torch.optim.optimizer import Optimizer
from tqdm import trange
from tensorboardX import SummaryWriter
import pdb
from pytorch_transformers import WarmupConstantSchedule

from collections.abc import MutableMapping

# data utilities
from models.utils.helper_utils import safe_zip


def embed_packed_sequence(embeddings: nn.Embedding, packed_sequence: PackedSequence):
    
    return PackedSequence(embeddings(packed_sequence.data),
                          batch_sizes=packed_sequence.batch_sizes,
                          sorted_indices=packed_sequence.sorted_indices,
                          unsorted_indices=packed_sequence.unsorted_indices)

def move_dict_to_cuda(dictionary):
    for k, v in dictionary.items():
        if hasattr(v, 'cuda'):
            dictionary[k] = v.cuda()
    return dictionary

def tensorize(vv, name):
    if isinstance(vv, torch.Tensor):
        return vv
    else:
        return torch.tensor(vv, dtype=torch.float if ('frames' in name) else torch.long)

class AdamW(Optimizer):
    r"""Implements AdamW algorithm.

    The original Adam algorithm was proposed in `Adam: A Method for Stochastic Optimization`_.
    The AdamW variant was proposed in `Decoupled Weight Decay Regularization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(AdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Perform stepweight decay
                p.data.mul_(1 - group['lr'] * group['weight_decay'])

                # Perform optimization step
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss

class AlfredDataset(Dataset):
    @staticmethod
    def extract_subgoal_indices(task_data, subgoal_names: Set[str]=None, keep_noop=False):
        """
        :param data:
        :param subgoal_names: if None, return all subgoals
        :param keep_noop:
        :return:
        """
        valid_idx = set()

        high_indices_with_actions = set()
        for actions in task_data['num']['action_low']:
            for action in actions:
                high_indices_with_actions.add(action['high_idx'])

        for subgoal in task_data['plan']['high_pddl']:

            curr_subgoal = subgoal['discrete_action']['action']
            if curr_subgoal == 'NoOp' and not keep_noop:
                continue

            # Keep the subgoal we're looking for.
            if subgoal_names is None or curr_subgoal in subgoal_names:
                high_idx = subgoal['high_idx']
                if high_idx in high_indices_with_actions:
                    valid_idx.add(high_idx)

            # TODO: this wasn't being used
            # As well as the final NoOp operation, we will copy this to the end of every subgoal example.
            # elif curr_subgoal == 'NoOp':
            #     noop_idx = subgoal['high_idx']
        return valid_idx

    @staticmethod
    def load_task_json_unsplit(args, task):
        # TODO: unuglify this
        json_path = os.path.join(args.data, task['task'], '%s' % args.pp_folder, 'ann_%d.json' % task['repeat_idx'])
        with open(json_path) as f:
            data = json.load(f)

        data['repeat_idx'] = task['repeat_idx']
        # None means this contains all subgoals; will be set in copies if filter_subgoal_index is called
        data['subgoal_idx'] = None
        return data

    @staticmethod
    def load_task_json(args, task, subgoal=None, split_into_subtrajectories=False,
                       add_stop_in_subtrajectories=True, filter_instructions=True):
        '''
        load preprocessed json from disk
        '''
        data = AlfredDataset.load_task_json_unsplit(args, task)

        if subgoal is not None or split_into_subtrajectories:
            if subgoal is not None:
                subgoals_to_keep = {subgoal}
            else:
                subgoals_to_keep = None # means keep all
            # Will return list of subgoal datapoints.
            subgoal_indices = AlfredDataset.extract_subgoal_indices(data, subgoals_to_keep)
            return [
                AlfredDataset.filter_subgoal_index(data, subgoal_ix, add_stop_in_subtrajectories,
                                                   filter_instructions=filter_instructions)
                for subgoal_ix in sorted(subgoal_indices)
            ]

        # Otherwise return list with singleton item.
        return [data]

    @staticmethod
    def filter_subgoal_index(data, subgoal_index, add_stop_in_subtrajectories=True, filter_instructions=True):
        '''
        Split a loaded json to only contain the subtrajectory specified by subgoal_index
        if add_stop_in_subtrajectories, then add a NoOp high-level action, with a <<stop>> low-level action, a copy of the last image, and the <<stop>> instruction
        '''

        # Make subgoal pairs and single subgoal compatible with each other.
        if type(subgoal_index) == int:
            subgoal_index = {subgoal_index}
        elif type(subgoal_index) == tuple: 
            subgoal_index_ = set(list(subgoal_index))
            assert len(subgoal_index) == len(subgoal_index_)
            subgoal_index = subgoal_index_

        # Use for counting dataset examples.
        #self.subgoal_count += len(valid_idx) - 1
        #self.trajectories_count += 1
        #for subgoal in data['plan']['high_pddl']: self.subgoal_set.add(subgoal['discrete_action']['action'])
        #return

        # Create an example from each instance of the subgoal.
        examples = []

        # this fails on the training data
        # for ix, act_low in enumerate(data['num']['action_low']):
        #     for act in act_low:
        #         assert act['high_idx'] == ix

        # this is true for <10 trials in the training data
        # if len(data['plan']['high_pddl']) != len(data['num']['lang_instr']):
        #     print("number of language instructions doesn't match number of high indices in {}: {} != {}".format(
        #         data['task_id'], len(data['num']['lang_instr']), len(data['plan']['high_pddl'])))

        # Copy data to make individual example.
        # data_cp = deepcopy(data)
        data_cp = data.copy()


        # Filter language instructions.
        data_cp['num'] = data_cp['num'].copy()
        if filter_instructions:
            high_indices = []
            lang_instrs = []
            for instr, a in safe_zip(data_cp['num']['lang_instr'], data_cp['num']['action_low']):
                high_idx = a[0]['high_idx']
                if high_idx not in subgoal_index:
                    continue
                lang_instrs.append(instr)
                high_indices.append(high_idx)
            data_cp['num']['lang_instr'] = lang_instrs
            data_cp['num']['sub_instr_high_indices'] = high_indices
            if add_stop_in_subtrajectories:
                # pull <<stop>> from the end of the original instruction
                assert len(data['num']['lang_instr'][-1]) == 1
                data_cp['num']['lang_instr'].append(
                    data['num']['lang_instr'][-1]
                )

            # Language instruction masks. 
            lengths = [len(instr) for instr in data_cp['num']['lang_instr']]
            total_len = sum(lengths)

            data_cp['num']['lang_instr_subgoal_mask'] = np.full((total_len,), 1, dtype=np.bool).tolist()
        else:
            data_cp['num']['lang_instr_subgoal_mask'] = [
                np.full((len(instr),), 1 if a[0]['high_idx'] in subgoal_index else 0, dtype=np.bool)
                for instr, a in safe_zip(data_cp['num']['lang_instr'], data_cp['num']['action_low'])
            ]

        # Filter low level actions.
        data_cp['plan'] = data_cp['plan'].copy()
        data_cp['plan']['low_actions'] = [a for a in data_cp['plan']['low_actions'] if a['high_idx'] in subgoal_index]

        # Filter images.
        data_cp['images'] = [img for img in data_cp['images'] if img['high_idx'] in subgoal_index]

        # Fix image idx.
        low_idxs = sorted(list(set([img['low_idx'] for img in data_cp['images']])))
        new_images = []
        for img in data_cp['images']:
            img = img.copy()
            img['low_idx'] = low_idxs.index(img['low_idx'])
            new_images.append(img)
        if add_stop_in_subtrajectories:
            new_images.append(img)
        data_cp['images'] = new_images

        # TODO: handle ['num']['low_to_high_idx'], for progress monitors

        # Filter action-low.
        # for i in range(len(data_cp['num']['action_low'])):
        #     data_cp['num']['action_low'][i] = [a for a in data_cp['num']['action_low'][i] if a['high_idx'] == idx]
        data_cp['num']['action_low'] = [
            [a for a in data_cp['num']['action_low'][i] if a['high_idx'] in subgoal_index]
            for i in range(len(data_cp['num']['action_low']))
            ]

        data_cp['num']['action_low'] = [a for a in data_cp['num']['action_low'] if len(a) > 0]

        # assert len(data_cp['num']['action_low']) <= 1

        # if not len(data_cp['num']['action_low']) == 1:
        #     continue

        assert(len(data_cp['num']['action_low']) == 1 or len(data_cp['num']['action_low']) == 2)

        # old non-functional form
        # Extend low-level actions with stop action.
        # data_cp['num']['action_low'][0].append(data['num']['action_low'][-1][0])

        # old form that concatenates stop, rather than adding as a separate subtrajectory
        # if add_stop_in_subtrajectories:
        #     data_cp['num']['action_low'] = [
        #         data_cp['num']['action_low'][0] + [data['num']['action_low'][-1][0]]
        #     ]
        if add_stop_in_subtrajectories:
            # pull the stop action from the end. the high_idx will stay unchanged, but we're not updating those
            assert len(data['num']['action_low'][-1]) == 1
            data_cp['num']['action_low'].append(data['num']['action_low'][-1])

        data_cp['subgoal_idx'] = subgoal_index

        return data_cp

    def __init__(self, args, data, model_class, test_mode, featurize=True):
        self._data = data
        self.model_class = model_class
        self.args = args
        self.test_mode = test_mode
        self.featurize = featurize

    def __getitem__(self, idx):
        # Load task from dataset.
        task = AlfredDataset.load_task_json_unsplit(self.args, self._data[idx])

        # Create dict of features from dict.
        if self.featurize:
            feat = self.model_class.featurize(task, self.args, self.test_mode)
        else:
            feat = None

        return (task, feat)

    def __len__(self):
        return len(self._data)

    @staticmethod
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
            if k in {'lang_goal_instr', 'lang_instr'}:
                # language embedding and padding
                seqs = [torch.tensor(vv) for vv in v]
                pad_seq = pad_sequence(seqs, batch_first=True, padding_value=pad)
                # Singleton corner case.
                if pad_seq.dim() == 1:
                    pad_seq = pad_seq.unsqueeze(0)
                assert pad_seq.dim() == 2
                # pack the sequences for now; we can embed them later
                # feat[k] = (pad_seq, seq_lengths)
                feat[k] = pad_seq
                #feat['{}_seq_lengths'.format(k)] = seq_lengths
            elif k in {'action_low_mask'}:
                # mask padding
                seqs = [torch.tensor(vv, dtype=torch.float) for vv in v]
                feat[k] = seqs
            elif k in {'subgoal_progress', 'subgoals_completed'}:
                # auxillary padding
                seqs = [torch.tensor(vv, dtype=torch.float) for vv in v]
                pad_seq = pad_sequence(seqs, batch_first=True, padding_value=pad)
                feat[k] = pad_seq
            elif k in {'lang_goal_instr_len', 'lang_instr_len'}:
                feat[k] = torch.tensor(v, dtype=torch.long)
            elif k in {'lang_instr_subgoal_mask', 'lang_goal_instr_subgoal_mask'}:
                seqs = [torch.tensor(vv, dtype=torch.bool) for vv in v]
                pad_seq = pad_sequence(seqs, batch_first=True, padding_value=False)
                feat[k] = pad_seq
            else:
                # default: tensorize and pad sequence
                seqs = [tensorize(vv, k) for vv in v]
                pad_seq = pad_sequence(seqs, batch_first=True, padding_value=pad)
                feat[k] = pad_seq


        ## due to the segment merging in merge_last_two_low_actions in preprocess.py, in some very rare
        ## cases, ['action_low'] has more timesteps than frames. There is code various other places (e.g. Decoder.forward())
        ## to handle this, but it still breaks if we get unlucky and an entire batch consists of these examples,
        ## as it does with subtrajectories filtered to the SliceGoal subgoal
        if 'action_low' in feat:
            bsz, t = feat['action_low'].size()
            bsz_, t_, *_ = feat['frames'].size()
            assert bsz == bsz_

            while t != t_:
                if t > t_: 
                    feat['frames'] = torch.cat((
                        feat['frames'], feat['frames'][:,-1].unsqueeze(1)
                    ),dim=1)
                    t_ = feat['frames'].size(1)
                else: 
                    feat['frames'] = feat['frames'][:,:-1]
                    t_ = feat['frames'].size(1)

        return (batch, feat)

class AlfredSubtrajectoryDataset(AlfredDataset):
    def __init__(self, args, data, model_class, test_mode, featurize=True, subgoal_names:Set[str]=None,
                 add_stop_in_subtrajectories=True,
                 filter_instructions=True, subgoal_pairs=False):
        """
        :param args:
        :param data:
        :param model_class:
        :param test_mode:
        :param featurize:
        :param subgoals:  if None, load all sub-trajectory types. otherwise, a set of sub-trajectories to filter to
        """
        super().__init__(args, data, model_class, test_mode, featurize=featurize)
        self.subgoal_names = subgoal_names
        self._task_and_indices = []
        self.add_stop_in_subtrajectories = add_stop_in_subtrajectories
        self.filter_instructions = filter_instructions
        self.subgoal_pairs = subgoal_pairs

        if len(data) > 1000:
            iter = tqdm.tqdm(data, ncols=80, desc='dataset: getting subgoals')
        else:
            iter = data

        for datum in iter:
            task = AlfredDataset.load_task_json_unsplit(self.args, datum)
            
            # Either extract contiguous subgoal pairs. 
            if subgoal_pairs: 
        
                # Get last high-level index. 
                last_idx = task['plan']['high_pddl'][-1]['high_idx']

                # valid_indices = list(sorted(AlfredDataset.extract_subgoal_indices(
                #     task, keep_noop=True
                # )))
                # if valid_indices != list(range(0, last_idx+1)):
                #     print("only subset of indices valid: {},{}: {}".format(task['task_id'],task['repeat_idx'],valid_indices))
                # TODO: consider using `pairs = list(zip(valid_indices, valid_indices[1:]))`

                # Get all contiguous pairs of subgoals.
                pairs = [p for p in zip(range(0, last_idx), range(1, last_idx + 1))]
           
                # Add to dataset. 
                for pair in pairs: 
                    self._task_and_indices.append((task, pair))

            # Or extract all subgoals of a particular type.
            else: 
                subgoal_indices = AlfredDataset.extract_subgoal_indices(
                    task, subgoal_names=subgoal_names, keep_noop=False
                )
                for ix in subgoal_indices:
                    self._task_and_indices.append((task, ix))

    def __getitem__(self, idx):
        
        task, subgoal_index = self._task_and_indices[idx]
        
        old_task = json.dumps(task)
        task = AlfredDataset.filter_subgoal_index(
            task,
            subgoal_index,
            add_stop_in_subtrajectories=self.add_stop_in_subtrajectories,
            filter_instructions=self.filter_instructions
        )
        assert json.dumps(self._task_and_indices[idx][0]) == old_task

        # Create dict of features from dict.
        if self.featurize:
            feat = self.model_class.featurize(task, self.args, self.test_mode)
        else:
            feat = None

        """ Used for debugging. 
        pdb.set_trace()

        if task['plan']['high_pddl'][-1]['high_idx'] in subgoal_index: 
            pdb.set_trace()
        """

        return (task, feat)

    def __len__(self):
        return len(self._task_and_indices)

class BaseModule(nn.Module):
    """inheritable by seq2seq and instruction chunker"""
    @classmethod
    def serialize_lang_action(cls, feat, test_mode):
        '''
        append segmented instr language and low-level actions into single sequences
        '''
        is_serialized = not isinstance(feat['num']['lang_instr'][0], list)
        print("is_serialized: {}".format(is_serialized))
        if is_serialized:
            print(feat['num']['lang_instr'][0])
        if not is_serialized:
            if 'sub_instr_high_indices' not in feat['num']:
                # this value will be set by filter_subgoal_index if filter_instructions=True, i.e. if the instructions
                # were filtered down to some particular high indices. otherwise, they're just the sequential ordering
                feat['num']['sub_instr_high_indices'] = list(range(len(feat['num']['lang_instr'])))
            feat['num']['sub_instr_lengths'] = [len(desc) for desc in feat['num']['lang_instr']]
            feat['num']['lang_instr'] = [word for desc in feat['num']['lang_instr'] for word in desc]
            if 'lang_instr_subgoal_mask' in feat['num']:
                # currently this is only added by the subtrajectory dataset
                
                # TODO hack for compatibility with subtrajectories, fix. 
                if type(feat['num']['lang_instr_subgoal_mask'][0]) == bool: 
                    mask = feat['num']['lang_instr_subgoal_mask']
                else: 
                    mask = [x for xs in feat['num']['lang_instr_subgoal_mask'] for x in xs]
    
                feat['num']['lang_instr_subgoal_mask'] = mask
                # check types
                # took these out because the goal is prepended
                # assert mask[0] == False or mask[0] == True
                # feat['num']['lang_instr_subgoal_start'] = mask.index(True)
                # feat['num']['lang_instr_subgoal_end'] = len(mask)-1 - mask[::-1].index(True)
        if not test_mode:
            feat['num']['action_low'] = [a for a_group in feat['num']['action_low'] for a in a_group]

    @classmethod
    def get_task_root(cls, ex, args):
        '''
        returns the folder path of a trajectory
        '''
        return os.path.join(args.data, ex['split'], *(ex['root'].split('/')[-2:]))

    @staticmethod
    def adjust_lr(optimizer, init_lr, epoch, decay_epoch=5):
        '''
        decay learning rate every decay_epoch
        '''
        lr = init_lr * (0.1 ** (epoch // decay_epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    @classmethod
    def load(cls, fsave):
        '''
        load pth model from disk
        '''
        save = torch.load(fsave)
        args = save['args']
        # some models first load parameters from a separate path, if init_model_path is specified, so set it to None
        args.init_model_path = None
        model = cls(args, save['vocab'])
        model.load_state_dict(save['model'])
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer.load_state_dict(save['optim'])
        return model, optimizer

    @classmethod
    def featurize(cls, ex, args, test_mode):
        raise NotImplementedError()

    def __init__(self, args):
        super().__init__()
        self.args = args

        # set random seed (Note: this is not the seed used to initialize THOR object locations)
        random.seed(a=args.seed)

        # summary self.writer
        self.summary_writer = None

        # Will hold json datasets (train, val, etc.) in memory.
        self.datasets = {}

        # Used for counting dataset examples.
        #self.subgoal_count = 0
        #self.trajectories_count = 0
        #self.subgoal_set = set()

    def get_instance_key(self, instance):
        # return a unique identifier for the instance in the dataset
        # TODO: move this to AlfredDataset / AlfredSubtrajectoryDataset
        subgoal_idx = None if instance['subgoal_idx'] is None else tuple(instance['subgoal_idx'])
        return instance['task_id'], instance['repeat_idx'], subgoal_idx

    def forward(self, feat, max_decode=100):
        raise NotImplementedError()

    def make_debug(self, preds, data):
        '''
        readable output generator for debugging
        '''
        raise NotImplementedError()

    def extract_preds(self, out, batch, feat):
        raise NotImplementedError()

    def compute_loss(self, out, batch, feat):
        raise NotImplementedError()

    def compute_metric(self, preds, data):
        raise NotImplementedError()

    def load_dataset(self, data, subgoal, dataset_type, data_path):

        # Where to save/load dataset.
        path = '{}_{}.pkl'.format(data_path, dataset_type)

        # Either load prepared dataset or craft it.
        if os.path.isfile(path):
            self.datasets[dataset_type] = pickle.load(open(path, 'rb'))

        else:
            # First read all json files.
            self.datasets[dataset_type] = []

            # Add all examples extracted from trajectory as individual data points to dataset.
            i = 0

            for task in data:
                self.datasets[dataset_type].extend(self.load_task_json(task, subgoal))
                i += 1

                print('{}% loaded.'.format(float(i) / float(len(data))), end='\r')

            # Save dataset.
            pickle.dump(self.datasets[dataset_type], open(path, 'wb'))

        return self.datasets[dataset_type]

    def run_train(self, splits, args=None, optimizer=None):
        '''
        training loop
        '''

        # args
        args = args or self.args

        # splits
        train = splits['train']
        valid_seen = splits['valid_seen']
        valid_unseen = splits['valid_unseen']

        # debugging: chose a small fraction of the dataset
        if self.args.dataset_fraction > 0:
            small_train_size = int(self.args.dataset_fraction * 0.7)
            small_valid_size = int((self.args.dataset_fraction * 0.3) / 2)
            train = train[:small_train_size]
            valid_seen = valid_seen[:small_valid_size]
            valid_unseen = valid_unseen[:small_valid_size]

        # debugging: use to check if training loop works without waiting for full epoch
        if self.args.fast_epoch:
            train = train[:16]
            valid_seen = valid_seen[:16]
            valid_unseen = valid_unseen[:16]

        # Pre-load json files into memory.
        if args.preloaded_dataset:
            train = self.load_dataset(train, args.subgoal, 'train', args.preloaded_dataset)
            valid_seen = self.load_dataset(valid_seen, args.subgoal, 'val_seen', args.preloaded_dataset)
            valid_unseen = self.load_dataset(valid_unseen, args.subgoal, 'val_unseen', args.preloaded_dataset)

        # subsample the training data for the purposes of evaluation
        train_subset = copy.copy(train)
        random_state = random.Random(1)
        random_state.shuffle(train_subset)
        train_subset = train_subset[::20]

        # this isn't implemented for instruction_chunker
        if vars(args).get('train_on_subtrajectories', False) or vars(args).get('train_on_subtrajectories_full_instructions', False):
            if args.subgoal:
                subgoal_names = {args.subgoal}
            else:
                subgoal_names = None

            subgoal_pairs = args.subgoal_pairs
            add_stop_in_subtrajectories = args.add_stop_in_subtrajectories

            full_instructions = vars(args).get('train_on_subtrajectories_full_instructions', False)
            dataset_constructor = lambda tasks: AlfredSubtrajectoryDataset(
                args, tasks, self.__class__, False, subgoal_names=subgoal_names,
                filter_instructions = not full_instructions,
                add_stop_in_subtrajectories = add_stop_in_subtrajectories,
                subgoal_pairs = subgoal_pairs
            )
        else:
            dataset_constructor = lambda tasks: AlfredDataset(args, tasks, self.__class__, False)

        # Put dataset splits into wrapper class for parallelizing data-loading.
        train = dataset_constructor(train)
        valid_seen = dataset_constructor(valid_seen)
        valid_unseen = dataset_constructor(valid_unseen)
        train_subset = dataset_constructor(train_subset)

        # setting this to True didn't seem to give a speedup
        pin_memory = False

        # DataLoaders
        train_loader = DataLoader(train, batch_size=args.batch, shuffle=True, num_workers=args.num_workers, collate_fn=AlfredDataset.collate_fn, pin_memory=pin_memory)
        valid_seen_loader = DataLoader(valid_seen, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, collate_fn=AlfredDataset.collate_fn, pin_memory=pin_memory)
        valid_unseen_loader = DataLoader(valid_unseen, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, collate_fn=AlfredDataset.collate_fn, pin_memory=pin_memory)

        # initialize summary writer for tensorboardX
        self.summary_writer = SummaryWriter(log_dir=args.dout)

        # dump config
        fconfig = os.path.join(args.dout, 'config.json')
        with open(fconfig, 'wt') as f:
            json.dump(vars(args), f, indent=2)

        # optimizer
        if vars(args).get('lang_model', '') == 'bert':
            self.emb_word.requires_grad = False
            self.lang_projection.requires_grad = False

            regular_optim_params = [p for p in self.parameters() if p.requires_grad]

            # Set up optimizer for bert. 
            self.emb_word.requires_grad = True
            self.lang_projection.requires_grad = True
            bert_params = [p for p in self.emb_word.parameters()] + [p for p in self.lang_projection.parameters()]

            bert_optim = AdamW(bert_params, lr=5e-5, weight_decay=0.01)

            # Scheduler for warming up bert fine-tuning. 
            warmup_steps = int((args.epoch * (len(train) / args.batch)) * 0.1)
            bert_scheduler = WarmupConstantSchedule(bert_optim, warmup_steps=warmup_steps)
        
        else: 
            regular_optim_params = self.parameters()

        optimizer = optimizer or torch.optim.Adam(regular_optim_params, lr=args.lr)

        # display dout
        print("Saving to: %s" % self.args.dout)
        best_loss = {'train': 1e10, 'valid_seen': 1e10, 'valid_unseen': 1e10}
        train_iter, valid_seen_iter, valid_unseen_iter = 0, 0, 0
        for epoch in trange(0, args.epoch, desc='epoch'):
            m_train = collections.defaultdict(list)
            self.train()
            self.adjust_lr(optimizer, args.lr, epoch, decay_epoch=args.decay_epoch)
            p_train = {}
            total_train_loss = list()

            with tqdm.tqdm(train_loader, unit='batch', total=len(train_loader), ncols=80) as batch_iterator:
                for i_batch, (batch, feat) in enumerate(batch_iterator):
                    # s_time = time.time()

                    out = self.forward(feat)
                    preds = self.extract_preds(out, batch, feat)
                    if 'action_low_mask' in preds:
                        # these are expensive to store, and we only currently use them in eval when we're interacting with the simulator
                        del preds['action_low_mask']
                    p_train.update(preds)
                    loss = self.compute_loss(out, batch, feat)
                    for k, v in loss.items():
                        ln = 'loss_' + k
                        m_train[ln].append(v.item())
                        self.summary_writer.add_scalar('train/' + ln, v.item(), train_iter)

                    # optimizer backward pass, also optimize bert params if needed. 
                    optimizer.zero_grad()                    
                    if vars(args).get('lang_model', '') == 'bert':
                        bert_optim.zero_grad()
                    
                    sum_loss = sum(loss.values())
                    sum_loss.backward()

                    optimizer.step()
                    if vars(args).get('lang_model', '') == 'bert':
                        bert_optim.step()
                        bert_scheduler.step()

                    self.summary_writer.add_scalar('train/loss', sum_loss, train_iter)
                    sum_loss = sum_loss.detach().cpu()
                    total_train_loss.append(float(sum_loss))
                    train_iter += self.args.batch

                    # e_time = time.time()
                    # print('Batch time in seconds: {}'.format(e_time - s_time))

            # Scheduler step for bert warmup if needed. 
            if vars(args).get('lang_model', '') == 'bert':
                pass#bert_scheduler.step()

            print('\ntrain subset metrics\n')
            # compute metrics for train
            m_train = {k: sum(v) / len(v) for k, v in m_train.items()}
            m_train.update(self.compute_metric(p_train, train_subset))
            m_train['total_loss'] = sum(total_train_loss) / len(total_train_loss)
            self.summary_writer.add_scalar('train/total_loss', m_train['total_loss'], train_iter)

            print('\nvalid seen\n')
            # compute metrics for valid_seen
            p_valid_seen, valid_seen_iter, total_valid_seen_loss, m_valid_seen = self.run_pred(valid_seen_loader, args=args, name='valid_seen', iter=valid_seen_iter)
            m_valid_seen.update(self.compute_metric(p_valid_seen, valid_seen))
            m_valid_seen['total_loss'] = float(total_valid_seen_loss)
            self.summary_writer.add_scalar('valid_seen/total_loss', m_valid_seen['total_loss'], valid_seen_iter)

            # compute metrics for valid_unseen
            print('\nvalid unseen\n')
            p_valid_unseen, valid_unseen_iter, total_valid_unseen_loss, m_valid_unseen = self.run_pred(valid_unseen_loader, args=args, name='valid_unseen', iter=valid_unseen_iter)
            m_valid_unseen.update(self.compute_metric(p_valid_unseen, valid_unseen))
            m_valid_unseen['total_loss'] = float(total_valid_unseen_loss)
            self.summary_writer.add_scalar('valid_unseen/total_loss', m_valid_unseen['total_loss'], valid_unseen_iter)

            stats = {'epoch': epoch, 'train': m_train, 'valid_seen': m_valid_seen, 'valid_unseen': m_valid_unseen}

            # new best valid_seen loss
            if total_valid_seen_loss < best_loss['valid_seen']:
                print('\nFound new best valid_seen!! Saving...')
                fsave = os.path.join(args.dout, 'best_seen.pth')
                torch.save({
                    'metric': stats,
                    'model': self.state_dict(),
                    'optim': optimizer.state_dict(),
                    'args': self.args,
                    'vocab': self.vocab,
                }, fsave)
                fbest = os.path.join(args.dout, 'best_seen.json')
                with open(fbest, 'wt') as f:
                    json.dump(stats, f, indent=2)

                if not args.no_make_debug:
                    fpred = os.path.join(args.dout, 'valid_seen.epoch-{}.debug.preds.json'.format(epoch))
                    with open(fpred, 'wt') as f:
                        json.dump(self.make_debug(p_valid_seen, valid_seen), f, indent=2)
                best_loss['valid_seen'] = total_valid_seen_loss

            # new best valid_unseen loss
            if total_valid_unseen_loss < best_loss['valid_unseen']:
                print('Found new best valid_unseen!! Saving...')
                fsave = os.path.join(args.dout, 'best_unseen.pth')
                torch.save({
                    'metric': stats,
                    'model': self.state_dict(),
                    'optim': optimizer.state_dict(),
                    'args': self.args,
                    'vocab': self.vocab,
                }, fsave)
                fbest = os.path.join(args.dout, 'best_unseen.json')
                with open(fbest, 'wt') as f:
                    json.dump(stats, f, indent=2)

                if not args.no_make_debug:
                    fpred = os.path.join(args.dout, 'valid_unseen.epoch-{}.debug.preds.json'.format(epoch))
                    with open(fpred, 'wt') as f:
                        json.dump(self.make_debug(p_valid_unseen, valid_unseen), f, indent=2)

                best_loss['valid_unseen'] = total_valid_unseen_loss

            # save the latest checkpoint
            if args.save_every_epoch:
                fsave = os.path.join(args.dout, 'net_epoch_%d.pth' % epoch)
            else:
                fsave = os.path.join(args.dout, 'latest.pth')
            torch.save({
                'metric': stats,
                'model': self.state_dict(),
                'optim': optimizer.state_dict(),
                'args': self.args,
                'vocab': self.vocab,
            }, fsave)

            # debug action output json
            if not args.no_make_debug:
                fpred = os.path.join(args.dout, 'train_subset.epoch-{}.debug.preds.json'.format(epoch))
                with open(fpred, 'wt') as f:
                    json.dump(self.make_debug(p_train, train_subset), f, indent=2)

            iters_by_split = {
                'train': train_iter,
                'valid_seen': valid_seen_iter,
                'valid_unseen': valid_unseen_iter,
            }

            # write stats
            for split in stats.keys():
                if isinstance(stats[split], dict):
                    for k, v in stats[split].items():
                        self.summary_writer.add_scalar(split + '/' + k, v, iters_by_split[split])
            pprint.pprint(stats)

    def run_pred(self, dev_loader, args=None, name='dev', iter=0):
        '''
        validation loop
        '''
        args = args or self.args
        m_dev = collections.defaultdict(list)
        p_dev = {}
        self.eval()
        total_loss = list()
        dev_iter = iter

        with tqdm.tqdm(dev_loader, unit='batch', total=len(dev_loader), ncols=80) as batch_iterator:
            for i_batch, (batch, feat) in enumerate(batch_iterator):

                out = self.forward(feat)
                preds = self.extract_preds(out, batch, feat)
                if 'action_low_mask' in preds:
                    # these are expensive to store, and we only currently use them in eval when we're interacting with the simulator
                    del preds['action_low_mask']
                p_dev.update(preds)
                #pdb.set_trace()
                loss = self.compute_loss(out, batch, feat)
                for k, v in loss.items():
                    ln = 'loss_' + k
                    m_dev[ln].append(v.item())
                    self.summary_writer.add_scalar("%s/%s" % (name, ln), v.item(), dev_iter)
                sum_loss = sum(loss.values())
                self.summary_writer.add_scalar("%s/loss" % (name), sum_loss, dev_iter)
                total_loss.append(float(sum_loss.detach().cpu()))
                dev_iter += len(batch)

        m_dev = {k: sum(v) / len(v) for k, v in m_dev.items()}
        total_loss = sum(total_loss) / len(total_loss)
        return p_dev, dev_iter, total_loss, m_dev
