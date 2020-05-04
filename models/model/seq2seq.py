import os
import random
import json
import torch
import pprint
import collections
import numpy as np
from torch import nn
from tensorboardX import SummaryWriter
import tqdm
from tqdm import trange
from torch.utils.data import Dataset, DataLoader
import pdb
from copy import deepcopy
import pickle
import time
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

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
                seq_lengths = np.array(list(map(len, v)))
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

        return (batch, feat)  

class Module(nn.Module):

    # static sentinel tokens
    pad = 0
    seg = 1

    # Static variables.
    feat_pt = 'feat_conv.pt'

    def __init__(self, args, vocab):
        '''
        Base Seq2Seq agent with common train and val loops
        '''
        super().__init__()

        # args and vocab
        self.args = args
        self.vocab = vocab

        # emb modules
        self.emb_word = nn.Embedding(len(vocab['word']), args.demb)
        self.emb_action_low = nn.Embedding(len(vocab['action_low']), args.demb)

        # end tokens
        self.stop_token = self.vocab['action_low'].word2index("<<stop>>", train=False)
        self.seg_token = self.vocab['action_low'].word2index("<<seg>>", train=False)

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

        # Put dataset splits into wrapper class for parallelizing data-loading. 
        train = AlfredDataset(args, train, self.__class__, False)
        valid_seen = AlfredDataset(args, valid_seen, self.__class__, False)
        valid_unseen = AlfredDataset(args, valid_unseen, self.__class__, False)

        # DataLoaders
        train_loader = DataLoader(train, batch_size=args.batch, shuffle=True, num_workers=0, collate_fn=AlfredDataset.collate_fn)
        valid_seen_loader = DataLoader(valid_seen, batch_size=args.batch, shuffle=False, num_workers=8, collate_fn=AlfredDataset.collate_fn)
        valid_unseen_loader = DataLoader(valid_unseen, batch_size=args.batch, shuffle=False, num_workers=8, collate_fn=AlfredDataset.collate_fn)

        # initialize summary writer for tensorboardX
        self.summary_writer = SummaryWriter(log_dir=args.dout)

        # dump config
        fconfig = os.path.join(args.dout, 'config.json')
        with open(fconfig, 'wt') as f:
            json.dump(vars(args), f, indent=2)

        # optimizer
        optimizer = optimizer or torch.optim.Adam(self.parameters(), lr=args.lr)

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
                
            with tqdm.tqdm(train_loader, unit='batch', total=len(train_loader)) as batch_iterator:
                for i_batch, (batch, feat) in enumerate(batch_iterator):
                    s_time = time.time()

                    out = self.forward(feat)
                    preds = self.extract_preds(out, batch, feat)
                    p_train.update(preds)
                    loss = self.compute_loss(out, batch, feat)
                    for k, v in loss.items():
                        ln = 'loss_' + k
                        m_train[ln].append(v.item())
                        self.summary_writer.add_scalar('train/' + ln, v.item(), train_iter)

                    # optimizer backward pass
                    optimizer.zero_grad()
                    sum_loss = sum(loss.values())
                    sum_loss.backward()
                    optimizer.step()

                    self.summary_writer.add_scalar('train/loss', sum_loss, train_iter)
                    sum_loss = sum_loss.detach().cpu()
                    total_train_loss.append(float(sum_loss))
                    train_iter += self.args.batch

                    e_time = time.time()
                    print('Batch time in seconds: {}'.format(e_time - s_time))

            
            print('\ntrain metrics\n')
            # compute metrics for train
            m_train = {k: sum(v) / len(v) for k, v in m_train.items()}
            m_train.update(self.compute_metric(p_train, train))
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

            # debug action output josn
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

        with tqdm.tqdm(dev_loader, unit='batch', total=len(dev_loader)) as batch_iterator:
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

    @classmethod
    def serialize_lang_action(cls, feat, test_mode):
        '''
        append segmented instr language and low-level actions into single sequences
        '''
        is_serialized = not isinstance(feat['num']['lang_instr'][0], list)
        if not is_serialized:
            feat['num']['lang_instr'] = [word for desc in feat['num']['lang_instr'] for word in desc]
            if not test_mode:
                feat['num']['action_low'] = [a for a_group in feat['num']['action_low'] for a in a_group]

    @classmethod
    def featurize(cls, ex, args, test_mode, load_mask=True, load_frames=True):
        '''
        tensorize and pad batch input
        '''
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
            # low-level action
            feat['action_low'] = [a['action'] for a in ex['num']['action_low']]

            # low-level valid interact
            feat['action_low_valid_interact'] = np.array([a['valid_interact'] for a in ex['num']['action_low']])

        return feat

    def forward(self, feat, max_decode=100):
        raise NotImplementedError()

    def extract_preds(self, out, batch, feat):
        raise NotImplementedError()

    def compute_loss(self, out, batch, feat):
        raise NotImplementedError()

    def compute_metric(self, preds, data):
        raise NotImplementedError()

    def make_debug(self, preds, data):
        '''
        readable output generator for debugging
        '''
        debug = {}
        for ex, feat in data:
            if 'repeat_idx' in ex: ex = self.load_task_json(ex, None)[0]
            i = ex['task_id']
            debug[i] = {
                'lang_goal': ex['turk_annotations']['anns'][ex['ann']['repeat_idx']]['task_desc'],
                'action_low': [a['discrete_action']['action'] for a in ex['plan']['low_actions']],
                'p_action_low': preds[i]['action_low'].split(),
            }
        return debug

    @classmethod
    def load_task_json(cls, args, task, subgoal=None):
        '''
        load preprocessed json from disk
        '''
        json_path = os.path.join(args.data, task['task'], '%s' % args.pp_folder, 'ann_%d.json' % task['repeat_idx'])
        with open(json_path) as f:
            data = json.load(f)

        if subgoal is not None:

            # Will return list of subgoal datapoints. 
            return cls.filter_subgoal(data, subgoal)

        # Otherwise return list with singleton item. 
        return [data]

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
    def get_task_root(cls, ex, args):
        '''
        returns the folder path of a trajectory
        '''
        return os.path.join(args.data, ex['split'], *(ex['root'].split('/')[-2:]))

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

    def iterate(self, data, batch_size):
        '''
         breaks dataset into batch_size chunks for training
        '''
        for i in trange(0, len(data), batch_size, desc='batch'):
            
            batch = data[i:i+batch_size]

            # Load json files if needed. 
            if 'repeat_idx' in batch[0]: 
                batch = [self.load_task_json(task, None)[0] for task in batch]

            feat = self.featurize(batch)
            yield batch, feat

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
    def has_interaction(cls, action):
        '''
        check if low-level action is interactive
        '''
        non_interact_actions = ['MoveAhead', 'Rotate', 'Look', '<<stop>>', '<<pad>>', '<<seg>>']
        if any(a in action for a in non_interact_actions):
            return False
        else:
            return True
