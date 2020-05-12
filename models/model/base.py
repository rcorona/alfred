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
    def __init__(self, args, data, model_class, test_mode, featurize=True):
        self.data = data
        self.model_class = model_class
        self.args = args
        self.test_mode = test_mode
        self.featurize = featurize

    def __getitem__(self, idx):

        # Load task from dataset.
        task = self.model_class.load_task_json(self.args, self.data[idx], None)[0]

        # Create dict of features from dict.
        if self.featurize:
            feat = self.model_class.featurize(task, self.args, self.test_mode)
        else:
            feat = None

        return (task, feat)

    def __len__(self):
        return len(self.data)

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
            elif k in {'instr_chunk_labels'}:
                # from Chunker
                seqs = [torch.tensor(vv, dtype=torch.long) for vv in v]
                pad_seq = pad_sequence(pad_sequence(seqs, batch_first=True, padding_value=0))
                seq_lengths = np.array(list(map(len, v)))
                feat[k] = (pad_seq, seq_lengths)
            elif k in {'lang_instr_len'}:
                feat[k] = torch.tensor(v, dtype=torch.long)
            else:
                # default: tensorize and pad sequence
                seqs = [tensorize(vv, k) for vv in v]
                pad_seq = pad_sequence(seqs, batch_first=True, padding_value=pad)
                feat[k] = pad_seq

        return (batch, feat)

class BaseModule(nn.Module):
    """inheritable by seq2seq and instruction chunker"""
    @classmethod
    def serialize_lang_action(cls, feat, test_mode):
        '''
        append segmented instr language and low-level actions into single sequences
        '''
        is_serialized = not isinstance(feat['num']['lang_instr'][0], list)
        if not is_serialized:
            feat['num']['sub_instr_lengths'] = [len(desc) for desc in feat['num']['lang_instr']]
            feat['num']['lang_instr'] = [word for desc in feat['num']['lang_instr'] for word in desc]
        if not test_mode:
            feat['num']['action_low'] = [a for a_group in feat['num']['action_low'] for a in a_group]

    @classmethod
    def filter_subgoal(self, data, subgoal_name):
        '''
        Filter a loaded json to only include examples from a specific type of subgoal.
        Returns a list of individual examples for subgoal type followed by the NoOp action.
        '''
        # First get idx of segments where specified subgoal takes place.
        valid_idx = set()

        for subgoal in data['plan']['high_pddl']:

            curr_subgoal = subgoal['discrete_action']['action']

            # Keep the subgoal we're looking for.
            if curr_subgoal == subgoal_name:
                valid_idx.add(subgoal['high_idx'])

            # As well as the final NoOp operation, we will copy this to the end of every subgoal example.
            elif curr_subgoal == 'NoOp':
                noop_idx = subgoal['high_idx']

        # No examples will be added from this file to dataset.
        if len(valid_idx) == 0:
            return []

        # Use for counting dataset examples.
        #self.subgoal_count += len(valid_idx) - 1
        #self.trajectories_count += 1
        #for subgoal in data['plan']['high_pddl']: self.subgoal_set.add(subgoal['discrete_action']['action'])
        #return

        # Create an example from each instance of the subgoal.
        examples = []

        for idx in valid_idx:

            # Copy data to make individual example.
            data_cp = deepcopy(data)

            # Filter language instructions.
            data_cp['num']['lang_instr'] = [data_cp['num']['lang_instr'][idx]]

            # Filter low level actions.
            data_cp['plan']['low_actions'] = [a for a in data_cp['plan']['low_actions'] if a['high_idx'] == idx]

            # Filter images.
            data_cp['images'] = [img for img in data_cp['images'] if img['high_idx'] == idx]

            # Fix image idx.
            low_idxs = sorted(list(set([img['low_idx'] for img in data_cp['images']])))

            for img in data_cp['images']:
                img['low_idx'] = low_idxs.index(img['low_idx'])

            # Filter action-low.
            for i in range(len(data_cp['num']['action_low'])):
                data_cp['num']['action_low'][i] = [a for a in data_cp['num']['action_low'][i] if a['high_idx'] == idx]

            data_cp['num']['action_low'] = [a for a in data_cp['num']['action_low'] if len(a) > 0]

            # Extend low-level actions with stop action.
            if not len(data_cp['num']['action_low']) == 1:
                continue

            assert(len(data_cp['num']['action_low']) == 1)

            data_cp['num']['action_low'][0].append(data['num']['action_low'][-1][0])

            examples.append(data_cp)

        return examples


    @classmethod
    def load_task_json(cls, args, task, subgoal=None):
        '''
        load preprocessed json from disk
        '''
        json_path = os.path.join(args.data, task['task'], '%s' % args.pp_folder, 'ann_%d.json' % task['repeat_idx'])
        with open(json_path) as f:
            data = json.load(f)

        data['repeat_idx'] = task['repeat_idx']

        if subgoal is not None:

            # Will return list of subgoal datapoints.
            return cls.filter_subgoal(data, subgoal)

        # Otherwise return list with singleton item.
        return [data]

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
        model = cls(save['args'], save['vocab'])
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


        # Put dataset splits into wrapper class for parallelizing data-loading.
        train = AlfredDataset(args, train, self.__class__, False)
        valid_seen = AlfredDataset(args, valid_seen, self.__class__, False)
        valid_unseen = AlfredDataset(args, valid_unseen, self.__class__, False)

        train_subset = AlfredDataset(args, train_subset, self.__class__, False)

        # this didn't seem to give a speedup
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
        if args.lang_model == 'bert':
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
                    p_train.update(preds)
                    loss = self.compute_loss(out, batch, feat)
                    for k, v in loss.items():
                        ln = 'loss_' + k
                        m_train[ln].append(v.item())
                        self.summary_writer.add_scalar('train/' + ln, v.item(), train_iter)

                    # optimizer backward pass, also optimize bert params if needed. 
                    optimizer.zero_grad()                    
                    if args.lang_model == 'bert': 
                        bert_optim.zero_grad()
                    
                    sum_loss = sum(loss.values())
                    sum_loss.backward()

                    optimizer.step()
                    if args.lang_model == 'bert': 
                        bert_optim.step()
                        bert_scheduler.step()
    
                    self.summary_writer.add_scalar('train/loss', sum_loss, train_iter)
                    sum_loss = sum_loss.detach().cpu()
                    total_train_loss.append(float(sum_loss))
                    train_iter += self.args.batch

                    # e_time = time.time()
                    # print('Batch time in seconds: {}'.format(e_time - s_time))

            # Scheduler step for bert warmup if needed. 
            if args.lang_model == 'bert':
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

                fpred = os.path.join(args.dout, 'valid_seen.debug.preds.json')
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

                fpred = os.path.join(args.dout, 'valid_unseen.debug.preds.json')
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
            fpred = os.path.join(args.dout, 'train.debug.preds.json')
            with open(fpred, 'wt') as f:
                json.dump(self.make_debug(p_train, train), f, indent=2)

            # write stats
            for split in stats.keys():
                if isinstance(stats[split], dict):
                    for k, v in stats[split].items():
                        self.summary_writer.add_scalar(split + '/' + k, v, train_iter)
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
