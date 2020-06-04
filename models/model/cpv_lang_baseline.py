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

    def featurize(self, ex):
        '''
        tensorize and pad batch input
        '''
        # Collect instructions from dictionary
        high_level = torch.tensor(ex["high_level"]) # -> seq_len
        low_level_context = [torch.tensor(ll) for ll in ex["low_level_context"]] # -> seq_len

        # Remove target instruction
        target_idx = random.randrange(len(low_level_context))
        low_level_target = low_level_context[target_idx] # -> T
        del low_level_context[target_idx]

        # Stack instructions
        padded_context = torch.cat([high_level] + [torch.tensor(self.seg).unsqueeze(0)] + low_level_context, dim=0) # -> N

        return (padded_context, low_level_target)

    def __getitem__(self, idx):

        # Load task from dataset.
        task = self.data[idx]

        # Create dict of features from dict.
        feat = self.featurize(task)

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
                labels.append(torch.tensor(idx).unsqueeze(0))

            padded_contexts = pad_sequence(contexts, batch_first=True) # -> B x N
            padded_targets = pad_sequence(targets, batch_first=True) # -> B x T
            padded_labels = torch.cat(labels, dim=0) # -> B

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
        self.embed = nn.Embedding(len(self.vocab['word']), args.demb)
        self.enc = nn.LSTM(args.demb, args.dhid, bidirectional=True, batch_first=True)
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

    def forward(self, contexts, targets):
        '''
        Takes in contexts and targets and returns the dot product of each enc(context) with each enc(target)
        '''
        ### INPUT ###
        # Contexts -> B x N
        # Targets -> B x T
        batch_size = contexts.shape[0]

        ### CONTEXTS ###
        # Embedding:
        emb_contexts = self.embed(contexts) # -> B x N x E
        # Encoding:
        enc_contexts = self.encoder(emb_contexts, batch_size) # -> B x H

        ### TARGETS ###
        # Embedding:
        emb_targets = self.embed(targets) # -> B x T x E
        # Encoding:
        enc_targets = self.encoder(emb_targets, batch_size) # -> B x E

        ### COMB ###
        # Dot product:
        sim_m = torch.matmul(enc_contexts, torch.transpose(enc_targets, 0, 1)) # B x B
        logits = F.log_softmax(sim_m, dim = 1)

        return logits

    def run_train(self, splits, optimizer, args=None):

        ### SETUP ###
        args = args or self.args
        self.writer = SummaryWriter('runs/lang_baseline')
        fsave = os.path.join(args.dout, 'best.pth')

        # Getting splits
        splits = self.load_data_into_ram()
        valid_idx = np.arange(start=0, stop=len(splits), step=10)
        eval_idx = np.arange(start=1, stop=len(splits), step=10)
        train_idx = [i for i in range(len(splits)) if i not in valid_idx and i not in eval_idx]

        # Initialize Datasets
        valid_dataset = AlfredBaselineDataset(args, [splits[i] for i in range(len(splits)) if i in valid_idx])
        train_dataset = AlfredBaselineDataset(args, [splits[i] for i in range(len(splits)) if i in train_idx])

        # Initalize Dataloaders
        valid_loader = DataLoader(valid_dataset, batch_size=self.args.batch, shuffle=True, num_workers=8, collate_fn=valid_dataset.collate())
        train_loader = DataLoader(train_dataset, batch_size=self.args.batch, shuffle=True, num_workers=8, collate_fn=train_dataset.collate())

        # Optimizer
        optimizer = optimizer or torch.optim.Adam(self.parameters(), lr=args.lr)

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
                contexts, targets, labels = batch

                # Transfer to GPU
                contexts = contexts.to(self.device)
                targets = targets.to(self.device)
                labels = labels.to(self.device)

                # Forward
                logits = self.forward(contexts, targets)

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
                    contexts, targets, labels = batch

                    # Transfer to GPU
                    contexts = contexts.to(self.device)
                    targets = targets.to(self.device)
                    labels = labels.to(self.device)

                    # Forward
                    logits = self.forward(contexts, targets)

                    # Calcualte Loss and Accuracy
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
