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
            random.shuffle(train) # shuffle every epoch
            for batch, feat in self.iterate(train, args.batch, args.subgoal):
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
                if train_iter > 2000:
                    break

            
            print('\ntrain metrics\n')
            # compute metrics for train
            m_train = {k: sum(v) / len(v) for k, v in m_train.items()}
            #import pdb;pdb.set_trace()
            #m_train.update(self.compute_metric(p_train, train))
            m_train['total_loss'] = sum(total_train_loss) / len(total_train_loss)
            self.summary_writer.add_scalar('train/total_loss', m_train['total_loss'], train_iter)

            print('\nvalid seen\n')
            # compute metrics for valid_seen
            p_valid_seen, valid_seen_iter, total_valid_seen_loss, m_valid_seen = self.run_pred(valid_seen, args=args, name='valid_seen', iter=valid_seen_iter)
            m_valid_seen.update(self.compute_metric(p_valid_seen, valid_seen))
            m_valid_seen['total_loss'] = float(total_valid_seen_loss)
            self.summary_writer.add_scalar('valid_seen/total_loss', m_valid_seen['total_loss'], valid_seen_iter)

            # compute metrics for valid_unseen
            print('\nvalid unseen\n')
            p_valid_unseen, valid_unseen_iter, total_valid_unseen_loss, m_valid_unseen = self.run_pred(valid_unseen, args=args, name='valid_unseen', iter=valid_unseen_iter)
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

#             # debug action output josn
#             fpred = os.path.join(args.dout, 'train.debug.preds.json')
#             with open(fpred, 'wt') as f:
#                 json.dump(self.make_debug(p_train, train), f, indent=2)

            # write stats
            for split in stats.keys():
                if isinstance(stats[split], dict):
                    for k, v in stats[split].items():
                        self.summary_writer.add_scalar(split + '/' + k, v, train_iter)
            pprint.pprint(stats)

    def run_pred(self, dev, args=None, name='dev', iter=0):
        '''
        validation loop
        '''
        args = args or self.args
        m_dev = collections.defaultdict(list)
        p_dev = {}
        self.eval()
        total_loss = list()
        dev_iter = iter
        for batch, feat in self.iterate(dev, args.batch, args.subgoal):
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

    def featurize(self, batch):
        raise NotImplementedError()

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
