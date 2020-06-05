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

        ### IMG TRAJ ###
        # Get ResNet Features -> Size = 512 x 7 x 7
        root = self.get_task_root(ex)
        im = torch.load(os.path.join(root, "feat_conv.pt"))

        # Prune out excess images (only one per time step)
        keep = [None] * len(ex['plan']['low_actions'])
        for idx, img_info in enumerate(ex['images']): # only add frames linked with low-level actions (i.e. skip filler frames like smooth rotations and dish washing)
            if keep[img_info['low_idx']] is None:
                keep[img_info['low_idx']] = (im[idx], img_info['high_idx'])
        keep.append((keep[-1][0], keep[-1][1]+1))  # stop frame

        low_level_context = [torch.stack([keep[j][0] for j in range(len(keep)) if keep[j][1] == i], dim=0).type(torch.float) for i in range(len(ex['ann']['instr']))] # -> N x 512 x 7 x 7

        ### LANG VEC ###
        high_level = torch.tensor(ex['num']['lang_goal']).type(torch.long) # -> M
        target_idx = random.randrange(len(low_level_context))
        low_level_target = torch.tensor(ex['num']['lang_instr'][target_idx]) # -> T

        # Remove target from low levels
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
            resnet_size = (512, 7, 7)

            # Initialize arrays for each type of tensor
            highs = []
            contexts = []
            targets = []
            labels = []
            all_sorted_lengths = []

            # Array of number of low level instructions for each high level
            lengths = torch.tensor([len(batch[idx][1]) for idx in range(batch_size)], dtype=torch.long) # -> B

            # Calculate longest seqence (L) - this is the number we will be padding the low level instructions to
            longest_seq = 0
            for idx in range(batch_size):
                longest_seq = max(longest_seq, max([batch[idx][1][seq].shape[0] for seq in range(len(batch[idx][1]))]))

            # Pad the low level tensors to be the correct length - seq len x 512 x 7 x 7 -> L x 512 x 7 x 7
            for idx in range(batch_size):
                # Create new zeros tensor that will hold the low level tensors for this high level instruction
                padded_contexts = torch.zeros((len(batch[idx][1]), longest_seq, resnet_size[0], resnet_size[1], resnet_size[2]), dtype=torch.float) # -> num_ll x L x 512 x 7 x 7

                # Keep track of original lengths of each tensor for masking
                context_lengths = torch.tensor([batch[idx][1][seq].shape[0] for seq in range(len(batch[idx][1]))], dtype=torch.float) # -> num_ll

                # Copy entire low level tensor into padded tensor
                for seq_idx in range(len(batch[idx][1])):
                    padded_contexts[seq_idx, :batch[idx][1][seq_idx].shape[0], :, :, :] = batch[idx][1][seq_idx] # -> num_ll x L x 512 x 7 x 7

                # Sort by length for packing
                sorted_lengths, sort_idx = torch.sort(context_lengths, 0, descending=True) # -> num_ll
                sorted_contexts = padded_contexts[sort_idx]

                # Append all data to arrays for padding
                highs.append(batch[idx][0]) # -> seq_len
                contexts.append(sorted_contexts) # -> num_ll x L x 512 x 7 x 7
                targets.append(batch[idx][2]) # -> seq_len
                labels.append(torch.tensor(idx).unsqueeze(0)) # -> 1
                all_sorted_lengths.append(sorted_lengths) # -> num_ll

            # Pad all sequences to make big tensors
            padded_lengths = pad_sequence(all_sorted_lengths, batch_first=True, padding_value=longest_seq) # -> B x NL
            packed_contexts = pad_sequence(contexts, batch_first=True) # -> B x NL x L x 512 x 7 x 7 (NL = Largest # of Low Levels)
            padded_highs = pad_sequence(highs, batch_first=True) # -> B x M (M = Longest High Level)
            padded_targets = pad_sequence(targets, batch_first=True) # -> B x T (T = Longest Target)
            padded_labels = torch.cat(labels, dim=0) # -> B

            # Mask for the second level of padding on low levels - pad on # of low levels
            mask = torch.zeros((batch_size, lengths.max()), dtype=torch.float) # -> B x NL
            for idx in range(batch_size):
                mask[idx, :lengths[idx]] = torch.ones((lengths[idx],))

            return (padded_highs, packed_contexts, padded_targets, padded_labels, padded_lengths, mask)
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
        self.embed = self.embed = nn.Embedding(len(self.vocab['word']), args.demb)
        self.linear = nn.Linear(args.demb, args.dframe)
        self.enc = nn.LSTM(args.dframe, args.dhid, bidirectional=True, batch_first=True)
        self.vis_encoder = vnn.ResnetVisualEncoder(dframe=args.dframe)
        self.to(self.device)


    def encoder(self, batch, batch_size):
        '''
        Input: stacked tensor of [high_level, seq, low_level_context]
        '''
        ### INPUTS ###
        # Batch -> B x H x E

        ### LSTM ###
        h_0 = torch.zeros(2, batch_size, self.args.dhid).type(torch.float).to(self.device) # -> L * 2 x B x H
        c_0 = torch.zeros(2, batch_size, self.args.dhid).type(torch.float).to(self.device) # -> L * 2 x B x H
        out, (h, c) = self.enc(batch, (h_0, c_0)) # -> L * 2 x B x H

        ## COMB ##
        hid_sum = torch.sum(h, dim=0) # -> B x H

        return hid_sum

    def forward(self, highs, contexts, targets, lengths, mask):
        '''
        Takes in contexts and targets and returns the dot product of each enc(high) - enc(context) with each enc(target)
        '''
        ### INPUT ###
        # Highs ->  B x M (M = Highs Seq Len)
        # Contexts -> B x NL x L x 512 x 7 x 7 (NL = Largest # of Low Levels, L = Lows Seq Len)
        # Targets -> B x T (T = Targets Seq Len)
        # Lengths -> B x NL
        # Mask -> B x NL

        batch_size = contexts.shape[0]
        highs_len = highs.shape[1]
        contexts_num = contexts.shape[1]
        contexts_len = contexts.shape[2]
        targets_len = targets.shape[1]
        resnet_size = (512, 7, 7)

        ### HIGHS ###
        # Embedding:
        embed_highs = self.embed(highs) # -> B x M x D (D = embedding size)
        # Projection:
        tall_highs = self.linear(embed_highs) # -> B x M x E
        # Encoding:
        enc_highs = self.encoder(tall_highs, batch_size) # -> B x H

        ### CONTEXTS ###
        # Reshaping:
        flat_contexts = contexts.reshape(batch_size * contexts_num * contexts_len, resnet_size[0], resnet_size[1], resnet_size[2]) # -> B * NL * L x 512 x 7 x 7
        # Dimension Reduction:
        emb_contexts = self.vis_encoder(flat_contexts) # -> B * NL * L x E
        # Reshaping:
        tall_contexts = emb_contexts.reshape(batch_size * contexts_num, contexts_len, self.args.dframe) # -> B * NL x L x 256
        tall_lengths = lengths.reshape(batch_size * contexts_num) # -> B * NL
        # Packing:
        packed_contexts = pack_padded_sequence(tall_contexts, tall_lengths, batch_first=True, enforce_sorted=False)
        # Encoding:
        enc_contexts = self.encoder(packed_contexts, batch_size * contexts_num) # -> B * NL x H
        # Reshaping:
        full_enc_contexts = enc_contexts.reshape(batch_size, contexts_num, -1) # -> B x NL x H
        # Masking:
        masked_enc_contexts = torch.einsum('ijk, ij -> ijk', full_enc_contexts, mask) # -> B x NL x H
        # Sum all low levels that correspond to the same high level:
        summed_enc_contexts = torch.sum(masked_enc_contexts, dim=1)  # B x H

        ### TARGETS ###
        # Embedding:
        embed_targets = self.embed(targets) # -> B x T x D
        # Projection:
        tall_targets = self.linear(embed_targets) # -> B x T x E
        # Encoding:
        enc_targets = self.encoder(tall_targets, batch_size) # B x H

        ### COMB ###
        # Combining high levels and low levels:
        comb_contexts = enc_highs - summed_enc_contexts # -> B x H
        # Dot product:
        sim_m = torch.matmul(comb_contexts, torch.transpose(enc_targets, 0, 1)) # -> B x B
        logits = F.log_softmax(sim_m, dim = 1)

        return logits

    def run_train(self, splits, optimizer, args=None):

        ### SETUP ###
        args = args or self.args
        self.writer = SummaryWriter('runs/multi_cpv')
        fsave = os.path.join(args.dout, 'best.pth')

        # Get splits
        train_data = splits['train']
        valid_data = splits['valid_seen'] + splits['valid_unseen']

        # Initialize Datasets
        valid_dataset = AlfredBaselineDataset(self.args, valid_data)
        train_dataset = AlfredBaselineDataset(self.args, train_data)

        # Initialize Dataloaders
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch, shuffle=True, num_workers=8, collate_fn=valid_dataset.collate())
        train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=8, collate_fn=train_dataset.collate())

        # Optimizer
        optimizer = optimizer or torch.optim.Adam(list(self.parameters()) + list(self.vis_encoder.parameters()), lr=args.lr)

        ### TRAINING LOOP ###
        best_loss = 1e10
        for epoch in range(args.epoch):
            print('Epoch', epoch)
            self.train()
            total_train_loss = torch.tensor(0, dtype=torch.float)
            total_train_acc = torch.tensor(0, dtype=torch.float)
            total_train_size = torch.tensor(0, dtype=torch.float)
            for batch in train_loader:
                optimizer.zero_grad()
                highs, contexts, targets, labels, lengths, mask = batch

                # Transfer to GPU
                highs = highs.to(self.device)
                contexts = contexts.to(self.device)
                targets = targets.to(self.device)
                labels = labels.to(self.device)
                lengths = lengths.to(self.device)
                mask = mask.to(self.device)

                # Forward
                logits = self.forward(highs, contexts, targets, lengths, mask)

                # Calculate Loss and Accuracy
                loss = F.nll_loss(logits, labels)
                total_train_loss += loss
                total_train_size += labels.shape[0]
                most_likely = torch.argmax(logits, dim=1)
                acc = torch.eq(most_likely, labels)
                total_train_acc += torch.sum(acc)

                # Backpropogate
                loss.backward()
                optimizer.step()

            # Write to TensorBoardX
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

                    # Transfer to GPU
                    highs = highs.to(self.device)
                    contexts = contexts.to(self.device)
                    targets = targets.to(self.device)
                    labels = labels.to(self.device)
                    lengths = lengths.to(self.device)
                    mask = mask.to(self.device)

                    # Forward
                    logits = self.forward(highs, contexts, targets, lengths, mask)

                    # Calculate Loss and Accuracy
                    loss = F.nll_loss(logits, labels)
                    total_valid_loss += loss
                    total_valid_size += labels.shape[0]
                    most_likely = torch.argmax(logits, dim=1)
                    acc = torch.eq(most_likely, labels)
                    total_valid_acc += torch.sum(acc)

                # Write to TensorBoardX
                self.writer.add_scalar('Accuracy/validation', (total_valid_acc/total_valid_size).item(), epoch)
                self.writer.add_scalar('Loss/validation', total_valid_loss.item(), epoch)
                print("Validation Accuracy: " + str((total_valid_acc/total_valid_size).item()))
                print("Validation Loss: " + str(total_valid_loss.item()))

            # Save Model iff validation loss is improved
            if total_valid_loss < best_loss:
                print( "Obtained a new best validation loss of {:.2f}, saving model checkpoint to {}...".format(total_valid_loss, fsave))
                torch.save({
                    'model': self.state_dict(),
                    'optim': optimizer.state_dict(),
                    'args': self.args,
                    'vocab': self.vocab
                }, fsave)
                best_loss = total_valid_loss
