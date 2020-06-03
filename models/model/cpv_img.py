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
        self.vis_encoder = ResnetVisualEncoder(dframe=256)

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
        high_level = self.vis_encoder(torch.stack([keep[j][0] for j in range(len(keep))], dim=0)).type(torch.float) # seqlen x 512 x 7 x 7
        low_level_context = [self.vis_encoder(torch.stack([keep[j][0] for j in range(len(keep)) if keep[j][1] == i], dim=0)).type(torch.float) for i in range(len(ex['ann']['instr']))] # seqlen x 512 x 7 x 7

        high_level = high_level.to(self.device)
        low_level_context = [ll.to(self.device) for ll in low_level_context]
        target_idx = random.randrange(len(low_level_context))
        low_level_target = low_level_context[target_idx]
        del low_level_context[target_idx]

        return (high_level, low_level_context, low_level_target)

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

            highs = []
            contexts = []
            targets = []
            labels = []
            lengths = torch.tensor([len(batch[idx][1]) for idx in range(batch_size)], dtype=torch.long).to(self.device)
            all_sorted_lengths = []
            longest_seq = 0
            for idx in range(batch_size):
                longest_seq = max(longest_seq, max([batch[idx][1][seq].shape[0] for seq in range(len(batch[idx][1]))]))
            for idx in range(batch_size):
                highs.append(batch[idx][0])
                padded_contexts = torch.zeros((len(batch[idx][1]), longest_seq, batch[idx][1][0].shape[1]), dtype=torch.float).to(self.device)
                context_lengths = torch.tensor([batch[idx][1][seq].shape[0] for seq in range(len(batch[idx][1]))], dtype=torch.float).to(self.device)
                for seq_idx in range(len(batch[idx][1])):
                    padded_contexts[seq_idx, :batch[idx][1][seq_idx].shape[0], :] = batch[idx][1][seq_idx]
                sorted_lengths, sort_idx = torch.sort(context_lengths, 0, descending=True)
                sorted_contexts = padded_contexts[sort_idx]

                # packed_padded_contexts = pack_padded_sequence(sorted_contexts, sorted_lengths, batch_first=True)
                # contexts.append(packed_padded_contexts.data)
                all_sorted_lengths.append(sorted_lengths)
                contexts.append(sorted_contexts)
                targets.append(batch[idx][2])
                labels.append(torch.tensor(idx).unsqueeze(0).to(self.device))

            padded_lengths = pad_sequence(all_sorted_lengths, batch_first=True, padding_value=longest_seq)
            packed_contexts = pad_sequence(contexts, batch_first=True)
            padded_highs = pad_sequence(highs, batch_first=True)
            padded_targets = pad_sequence(targets, batch_first=True)
            padded_labels = torch.cat(labels, dim=0)

            # padded_contexts = torch.zeros((batch_size, lengths.max(), longest_seq), dtype=torch.long)
            mask = torch.zeros((batch_size, lengths.max(), self.args.dhid), dtype=torch.float).to(self.device)
            for idx in range(batch_size):
                # padded_contexts[idx, :lengths[idx], :longest_seq] = contexts[idx]
                mask[idx, :lengths[idx], :self.args.dhid] = torch.ones((lengths[idx], self.args.dhid)).to(self.device)

            return (padded_highs, packed_contexts, padded_targets, padded_labels, padded_lengths, mask)
        return collate_fn

class ResnetVisualEncoder(nn.Module):
    '''
    visual encoder
    '''

    def __init__(self, dframe):
        super(ResnetVisualEncoder, self).__init__()
        self.dframe = dframe
        self.flattened_size = 64*7*7

        self.conv1 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)
        self.fc = nn.Linear(self.flattened_size, self.dframe)
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(64)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))

        x = self.conv2(x)
        x = F.relu(self.bn2(x))

        x = x.view(-1, self.flattened_size)
        x = self.fc(x)

        return x

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
        self.embed = nn.Embedding(len(self.vocab['word']), args.demb)
        self.enc = nn.LSTM(256, args.dhid, bidirectional=True, batch_first=True)
        self.to(self.device)


    def encoder(self, batch, batch_size):
        '''
        Input: stacked tensor of [high_level, seq, low_level_context]
        '''

        # embedded = self.embed(torch.transpose(batch, 0, 1))
        # Input h,c are randomized/zero or should be prev
        h_0 = torch.zeros(2, batch_size, self.args.dhid).to(self.device)
        c_0 = torch.zeros(2, batch_size, self.args.dhid).to(self.device)
        out, (h, c) = self.enc(batch, (h_0, c_0))
        torch.sum(h, dim=0)

        # Add small feed forward/projection layer + ReLU?
        return torch.sum(h, dim=0)

    def forward(self, highs, contexts, targets, lengths, mask):
        '''
        Takes in contexts and targets and returns the dot product of each enc(high) - enc(context) with each enc(target)
        '''
        batch_size = contexts.shape[1]
        num_low_levels = contexts.shape[0]

        # embed_highs = self.embed(highs)
        enc_highs = self.encoder(highs, highs.shape[0]) # B x Emb

        # embed_contexts = self.embed(contexts)

        flat_contexts = torch.reshape(contexts, (contexts.shape[0]*contexts.shape[1], contexts.shape[2], contexts.shape[3]))
        flat_lengths = torch.reshape(lengths, (lengths.shape[0]*lengths.shape[1],))

        packed_contexts = pack_padded_sequence(flat_contexts, flat_lengths, batch_first=True, enforce_sorted=False)
        enc_contexts = self.encoder(packed_contexts, contexts.shape[0]*contexts.shape[1]) # B * N x Emb
        full_enc_contexts = enc_contexts.reshape(num_low_levels, batch_size, -1) # N x B x Emb
        masked_enc_contexts = torch.einsum('ijk, ijk -> ijk', full_enc_contexts, mask) # B x N x Emb
        summed_contexts = torch.sum(masked_enc_contexts, dim=1)  # B x Emb
        comb_contexts = enc_highs - summed_contexts

        # embed_targets = self.embed(targets)
        enc_targets = self.encoder(targets, targets.shape[0]) # C x Emb
        sim_m = torch.matmul(comb_contexts, torch.transpose(enc_targets, 0, 1)) # N x C
        logits = F.log_softmax(sim_m, dim = 1)

        return logits

    def run_train(self, splits, optimizer, args=None):

        args = args or self.args
        self.writer = SummaryWriter('runs/img_cpv')
        fsave = os.path.join(args.dout, 'best.pth')


        train_data = splits['train']
        valid_data = splits['valid_seen'] + splits['valid_unseen']

        valid_dataset = AlfredBaselineDataset(self.args, valid_data)
        train_dataset = AlfredBaselineDataset(self.args, train_data)

        valid_loader = DataLoader(valid_dataset, batch_size=self.args.batch, shuffle=True, collate_fn=valid_dataset.collate())
        train_loader = DataLoader(train_dataset, batch_size=self.args.batch, shuffle=True, collate_fn=train_dataset.collate())

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
            i=0
            for batch in train_loader:
                print(i)
                i += 1
                optimizer.zero_grad()
                highs, contexts, targets, labels, lengths, mask = batch
                # highs = highs.to(self.device)
                # contexts = contexts.to(self.device)
                # targets = targets.to(self.device)
                # labels = labels.to(self.device)
                # lengths = lengths.to(self.device)
                # mask = mask.to(self.device)
                logits = self.forward(highs, contexts, targets, lengths, mask)
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
                    highs, contexts, targets, labels, lengths, mask = batch
                    # highs = highs.to(self.device)
                    # contexts = contexts.to(self.device)
                    # targets = targets.to(self.device)
                    # labels = labels.to(self.device)
                    # lengths = lengths.to(self.device)
                    # mask = mask.to(self.device)
                    logits = self.forward(highs, contexts, targets, lengths, mask)
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
# Switched from concatenation to sum of h for encoding - problem?
