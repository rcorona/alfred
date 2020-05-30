import os
import torch
import numpy as np
import nn.vnn as vnn
import collections
import os
import random
import json
import torch
import pprint
import collections
import numpy as np
from torch import nn
from tensorboardX import SummaryWriter
from tqdm import trange
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from model.seq2seq import Module as Base
from models.utils.metric import compute_f1, compute_exact
from gen.utils.image_util import decompress_mask
from torch.utils.data import Dataset, DataLoader



class AlfredBaselineDataset(Dataset):

    def __init__(self, args, data):
        self.pad = 0
        self.seg = 1
        self.args = args
        self.data = data
        self.device = torch.device('cuda') if self.args.gpu else torch.device('cpu')

    def get_task_root(self, ex):
        '''
        returns the folder path of a trajectory
        '''
        return os.path.join(self.args.data, ex['split'], *(ex['root'].split('/')[-2:]))


    def featurize(self, ex):
        '''
        tensorize and pad batch input
        '''

        root = self.get_task_root(ex)
        im = torch.load(os.path.join(root, "feat_conv.pt")) # sizes of images are 512 x 7 x 7
        keep = [None] * len(ex['plan']['low_actions'])
        for idx, img_info in enumerate(ex['images']): # only add frames linked with low-level actions (i.e. skip filler frames like smooth rotations and dish washing)
            if keep[img_info['low_idx']] is None:
                keep[img_info['low_idx']] = (im[idx], img_info['high_idx'])
        keep.append((keep[-1][0], keep[-1][1]+1))  # stop frame

        high_level = torch.stack([keep[j][0] for j in range(len(keep))], dim=0).type(torch.long) # seqlen x 512 x 7 x 7
        low_level_context = [torch.stack([keep[j][0] for j in range(len(keep)) if keep[j][1] == i], dim=0).type(torch.long) for i in range(len(ex['ann']['instr']))] # seqlen x 512 x 7 x 7

        high_level = high_level.to(self.device)
        low_level_context = [ll.to(self.device) for ll in low_level_context]
        target_idx = random.randrange(len(low_level_context))
        low_level_target = low_level_context[target_idx]
        del low_level_context[target_idx]
        padded_context = torch.cat([high_level] + [torch.tensor(self.seg, dtype = torch.long).repeat(1, 512, 7, 7).to(self.device)] + low_level_context, dim=0)

        return (padded_context, low_level_target)

    def __getitem__(self, idx):

        # Load task from dataset.
        task = self.data[idx]

        json_path = os.path.join(self.args.data, task['task'], '%s' % self.args.pp_folder, 'ann_%d.json' % task['repeat_idx'])
        with open(json_path) as f:
            data = json.load(f)

        # Create dict of features from dict.
        feat = self.featurize(data)

        return feat

    def __len__(self):
        return len(self.data)

    def collate(self):
        def collate_fn(batch):
            batch_size = len(batch)

            contexts = []
            targets = []
            labels = []
            for idx in range(batch_size):
                contexts.append(batch[idx][0])
                targets.append(batch[idx][1])
                labels.append(torch.tensor(idx).unsqueeze(0).to(self.device))

            padded_contexts = pad_sequence(contexts)
            padded_targets = pad_sequence(targets)
            padded_labels = torch.cat(labels, dim=0)

            return (padded_contexts, padded_targets, padded_labels)

        return collate_fn



class Module(Base):
    def __init__(self, args, vocab):
        super().__init__(args, vocab)
        self.device = torch.device('cuda') if self.args.gpu else torch.device('cpu')

        self.pad = 0
        self.seg = 1

        # args and vocab
        self.args = args
        self.vocab = vocab

        # encoder and self-attention
        self.embed = nn.Embedding(512, args.demb)
        self.enc = nn.LSTM(args.demb, args.dhid, bidirectional=True, batch_first=True)
        self.to(self.device)


    def encoder(self, batch):
        '''
        Input: stacked tensor of [high_level, seq, low_level_context]
        '''

        embedded = self.embed(torch.transpose(batch, 0, 1))
        # Input h,c are randomized/zero or should be prev
        h_0 = torch.zeros(2, embedded.shape[0], self.args.dhid).to(self.device)
        c_0 = torch.zeros(2, embedded.shape[0], self.args.dhid).to(self.device)
        out, (h, c) = self.enc(embedded, (h_0, c_0))
        torch.sum(h, dim=0)

        # Add small feed forward/projection layer + ReLU?
        return torch.sum(h, dim=0)

    def forward(self, contexts, targets):
        '''
        Takes in contexts and targets and returns the dot product of each enc(context) with each enc(target)
        '''

        enc_contexts = self.encoder(contexts) # N x Emb
        enc_targets = self.encoder(targets) # C x Emb
        sim_m = torch.matmul(enc_contexts, torch.transpose(enc_targets, 0, 1)) # N x C
        logits = F.log_softmax(sim_m, dim = 1)

        return logits

    def run_train(self, splits, optimizer, args=None):

        args = args or self.args
        self.writer = SummaryWriter('runs/img_baseline')
        fsave = os.path.join(args.dout, 'best.pth')

        # Loading Data
        # splits = self.load_data_into_ram()
        # valid_idx = np.arange(start=0, stop=len(splits), step=10)
        # eval_idx = np.arange(start=1, stop=len(splits), step=10)
        # train_idx = [i for i in range(len(splits)) if i not in valid_idx and i not in eval_idx]

        train_data = splits['train']
        valid_data = splits['valid_seen'] + splits['valid_unseen']

        valid_dataset = AlfredBaselineDataset(self.args, valid_data)
        train_dataset = AlfredBaselineDataset(self.args, train_data)

        valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=True, collate_fn=valid_dataset.collate())
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=train_dataset.collate())

        # Optimizer
        optimizer = optimizer or torch.optim.Adam(self.parameters(), lr=args.lr)

        # Training loop
        best_loss = 1e10
        for epoch in range(args.epoch):
            print('Epoch', epoch)
            self.train()
            total_train_loss = torch.tensor(0, dtype=torch.float)
            total_train_acc = torch.tensor(0, dtype=torch.float)
            total_train_size = torch.tensor(0, dtype=torch.float)
            for batch in train_loader:
                optimizer.zero_grad()
                contexts, targets, labels = batch
                logits = self.forward(contexts, targets)
                loss = F.nll_loss(logits, labels)
                total_train_loss += loss
                total_train_size += labels.shape[0]
                most_likely = torch.argmax(logits, dim=1)
                acc = torch.eq(most_likely, labels)
                total_train_acc += torch.sum(acc)
                loss.backward()
                optimizer.step()
            self.writer.add_scalar('Accuracy/train', (total_train_acc/total_train_size).item(), epoch)
            self.writer.add_scalar('Loss/train', total_train_loss.item(), epoch)
            print("Train Accuracy: " + str((total_train_acc/total_train_size).item()))
            print("Train Loss: " + str(total_train_loss.item()))

            self.eval()
            total_valid_loss = torch.tensor(0, dtype=torch.float)
            total_valid_acc = torch.tensor(0, dtype=torch.float)
            total_valid_size = torch.tensor(0, dtype=torch.float)
            with torch.no_grad():
                for batch in valid_loader:
                    contexts, targets, labels = batch
                    logits = self.forward(contexts, targets)
                    loss = F.nll_loss(logits, labels)
                    total_valid_loss += loss
                    total_valid_size += labels.shape[0]
                    most_likely = torch.argmax(logits, dim=1)
                    acc = torch.eq(most_likely, labels)
                    total_valid_acc += torch.sum(acc)
                self.writer.add_scalar('Accuracy/validation', (total_valid_acc/total_valid_size).item(), epoch)
                self.writer.add_scalar('Loss/validation', total_valid_loss.item(), epoch)
                print("Validation Accuracy: " + str((total_valid_acc/total_valid_size).item()))
                print("Validation Loss: " + str(total_valid_loss.item()))

            if total_valid_loss < best_loss:
                print( "Obtained a new best validation loss of {:.2f}, saving model checkpoint to {}...".format(total_valid_loss, fsave))
                torch.save({
                    'model': self.state_dict(),
                    'optim': optimizer.state_dict(),
                    'args': self.args,
                    'vocab': self.vocab
                }, fsave)
                best_loss = total_valid_loss



            # Loss is going down but accuracy remains at zero

    def load_data_into_ram(self):
        '''
        Loads all data into RAM
        '''
        # Load data from all_data.json (language only dataset)
        json_path = os.path.join(self.args.data, "all_data.json")
        with open(json_path) as f:
            raw_data = json.load(f)

        # Turn data into array of [high_level, low_level_context, low_level_target, target_idx]
        split_data = []
        for key in raw_data.keys():
            for r_idx in range(len(raw_data[key]["high_level"])):
                high_level = raw_data[key]["high_level"][r_idx]
                low_levels = raw_data[key]["low_level"][r_idx]
                split_data += [{"high_level": high_level, "low_level_context": low_levels}]

        return split_data


# Loss is going down but accuracy remains at zero
# Switched from concatenation to sum of h for encoding - problem?
