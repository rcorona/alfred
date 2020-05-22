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

        high_level = torch.tensor(ex["high_level"])
        low_level_context = [torch.tensor(ll) for ll in ex["low_level_context"]]
        low_level_target = torch.tensor(ex["low_level_target"]).to(self.device)
        padded_context = torch.cat([high_level] + [torch.tensor(self.seg).unsqueeze(0)] + low_level_context, dim=0).to(self.device)
        label = torch.tensor(ex["target_id"]).to(self.device)

        return (padded_context, low_level_target, label)

    def __getitem__(self, idx):

        # Load task from dataset.
        task = self.data[idx]

        # Create dict of features from dict.
        feat = self.featurize(task)

        return feat

    def __len__(self):
        return len(self.data)

def collate_fn(batch):
    batch_size = len(batch)

    contexts = []
    targets = []
    labels = []
    for idx in range(batch_size):
        contexts.append(batch[idx][0])
        targets.append(batch[idx][1])
        labels.append(torch.tensor(idx).unsqueeze(0))

    padded_contexts = pad_sequence(contexts)
    padded_targets = pad_sequence(targets)
    padded_labels = torch.cat(labels, dim=0)

    return (padded_contexts, padded_targets, padded_labels)



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
        # Twice the size it should be
        logits = F.log_softmax(sim_m, dim = 1)

        return logits

    def run_train(self, splits, optimizer, args=None):

        args = args or self.args
        self.writer = SummaryWriter('runs/baseline')

        # Loading Data
        splits = self.load_data_into_ram()
        valid_idx = np.arange(start=0, stop=len(splits), step=10)
        eval_idx = np.arange(start=1, stop=len(splits), step=10)
        train_idx = [i for i in range(len(splits)) if i not in valid_idx and i not in eval_idx]

        valid_dataset = AlfredBaselineDataset(args, [splits[i] for i in range(len(splits)) if i in valid_idx])
        eval_dataset = AlfredBaselineDataset(args, [splits[i] for i in range(len(splits)) if i in eval_idx])
        train_dataset = AlfredBaselineDataset(args, [splits[i] for i in range(len(splits)) if i in train_idx])

        valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)

        # Optimizer
        optimizer = optimizer or torch.optim.Adam(self.parameters(), lr=args.lr)

        # Training loop
        best_loss = 1e10
        for epoch in range(args.epoch):
            print('Epoch', epoch)
            self.train()
            total_train_loss = 0
            total_train_acc = 0
            total_train_size = 0
            for batch in train_loader:
                optimizer.zero_grad()
                contexts, targets, labels = batch
                labels = labels.to(self.device)
                logits = self.forward(contexts, targets)
                loss = F.nll_loss(logits, labels)
                total_train_loss += loss
                total_train_size += labels.shape[0]
                most_likely = torch.argmax(logits, dim=1)
                acc = torch.eq(most_likely, labels)
                total_train_acc += torch.sum(acc)
                loss.backward()
                optimizer.step()
            self.writer.add_scalar('Accuracy/train', total_train_acc/total_train_size, epoch)
            self.writer.add_scalar('Loss/train', total_train_loss, epoch)
            print("Train Accuracy: " + str(total_train_acc/total_train_size))
            print("Train Loss: " + str(total_train_loss))

            self.eval()
            total_valid_loss = 0
            total_valid_acc = 0
            total_valid_size = 0
            with torch.no_grad():
                for batch in valid_loader:
                    contexts, targets, labels = batch
                    labels = labels.to(self.device)
                    logits = self.forward(contexts, targets)
                    loss = F.nll_loss(logits, labels)
                    total_valid_loss += loss
                    total_valid_size += labels.shape[0]
                    most_likely = torch.argmax(logits, dim=1)
                    acc = torch.eq(most_likely, labels)
                    total_valid_acc += torch.sum(acc)
                self.writer.add_scalar('Accuracy/validation', total_valid_acc/total_valid_size, epoch)
                self.writer.add_scalar('Loss/validation', total_valid_loss, epoch)
                print("Validation Accuracy: " + str(total_valid_acc/total_valid_size))
                print("Validation Loss: " + str(total_valid_loss))

            if total_valid_loss < best_loss:
                print( "Obtained a new best validation loss of {:.2f}, saving model checkpoint to {}...".format(total_valid_loss, args.dout))
                torch.save(model.state_dict(), args.dout)
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
                target_idx = random.randrange(len(low_levels))
                target = low_levels[target_idx]
                del low_levels[target_idx]
                split_data += [{"high_level": high_level, "low_level_context": low_levels, "low_level_target": target, "target_id": target_idx}]

        return split_data


# Input h,c are randomized/zero or should be prev (ZERO)
# They have a vocab argument which I assume gives the vocab
# Should I linearize hidden layer before output in encoder? (Assume don't have to cause hidden size is constant)

# Do the featurizing in the training loop rather than in the dataset
