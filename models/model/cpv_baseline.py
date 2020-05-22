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
        self.args = args
        self.test_mode = test_mode
        self.data = data


    def featurize(self, ex, load_mask=True, load_frames=True):
        '''
        tensorize and pad batch input
        '''
        device = torch.device('cuda') if self.args.gpu else torch.device('cpu')

        high_level = torch.tensor(ex["high_level"])
        low_level_context = [torch.tensor(ll) for ll in ex["low_level_context"]]
        low_level_target = torch.tensor(ex["low_level_target"])
        padded_context = torch.cat([high_level] + [self.sep] + low_level_context, dim=0)
        label = torch.tensor(ex["target_id"])

        return (padded_context, low_level_target, label)

    def __getitem__(self, idx):

        # Load task from dataset.
        task = self.data[idx]

        # Create dict of features from dict.
        feat = self.featurize(task, self.args, self.test_mode)

        return feat

    def __len__(self):
        return len(self.data)

    def collate_fn(batch):
        batch_size = len(batch)

        contexts = []
        targets = []
        labels = []
        for idx in range(batch_size):
            contexts.append(batch[0][idx])
            contexts.append(batch[1][idx])
            contexts.append(batch[2][idx])

        padded_contexts = pad_sequence(contexts)
        padded_targets = pad_sequence(targets)
        padded_labels = torch.cat(labels, dim=0)

        return (padded_contexts, padded_targets, padded_labels)



class BaselineModule(Base):
    def __init__(self, args, vocab):
        '''
        Seq2Seq agent
        '''
        super().__init__(args, vocab)

        self.pad = 0
        self.seg = 1

        # args and vocab
        self.args = args
        self.vocab = vocab

        # encoder and self-attention
        self.embed = nn.Embedding(len(self.vocab), args.demb)
        self.enc = nn.LSTM(args.demb, args.dhid, bidirectional=True, batch_first=True)
        self.enc_att = vnn.SelfAttn(args.dhid*2)


        # internal states
        self.state_t = None
        self.e_t = None
        self.test_mode = False

        # bce reconstruction loss
        self.bce_with_logits = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.mse_loss = torch.nn.MSELoss(reduction='none')

        # paths
        self.root_path = os.getcwd()


        # reset model
        self.reset()

    def encoder(self, batch):
        '''
        Input: stacked tensor of [high_level, seq, low_level_context]
        '''
        embedded = self.embed(batch)
        # Input h,c are randomized/zero or should be prev
        h_0 = torch.zeros(2, embedded.shape[0], self.args.dhid)
        c_0 = torch.zeros(2, embedded.shape[0], self.args.dhid)
        out, h, c = self.enc(embedded, h_0, c_0)

        # Add small feed forward/projection layer + ReLU?
        return h

    def forward(self, contexts, targets):
        '''
        Takes in contexts and targets and returns the dot product of each enc(context) with each enc(target)
        '''

        enc_contexts = self.encoder(contexts) # N x Emb
        enc_targets = self.encoder(targets) # C x Emb

        sim_m = torch.matmul(enc_contexts, torch.transpose(enc_targets, 0, 1)) # N x C
        logits = F.log_softmax(sim_m, dim = 1)

        return logits

    def run_train(self, args=None, model_file):
        args = args or self.args

        # Loading Data
        splits = load_data_into_ram()
        valid_idx = np.arange(start=0, stop=len(splits), step=10)
        eval_idx = np.arange(start=1, stop=len(split), step=10)
        train_idx = [i for i in range(len(splits)) if i not in valid_idx and i not in eval_idx]

        valid_dataset = AlfredBaselineDataset(args, [splits[i] for i in range(len(splits)) if i in valid_idx])
        eval_dataset = AlfredBaselineDataset(args, [splits[i] for i in range(len(splits)) if i in eval_idx])
        train_dataset = AlfredBaselineDataset(args, [splits[i] for i in range(len(splits)) if i in train_idx])

        valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=True)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

        # Optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=args.lr)

        # Training loop
        best_loss = 1e10
        for epoch in range(10):
            print('Epoch', epoch)
            model.train()
            total_train_loss = 0
            total_train_acc = 0
            total_train_size = 0
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

            model.eval()
            total_valid_loss = 0
            total_valid_acc = 0
            total_valid_size = 0
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

            if total_valid_loss > best_loss:
                print( "Obtained a new best validation loss of {:.2f}, saving model checkpoint to {}...".format(total_valid_loss, model_file))
                torch.save(model.state_dict(), model_file)
                best_loss = total_valid_loss

            # TODO: Add tensorboardX logging

    def load_data_into_ram():
        '''
        Loads all data into RAM
        '''
        # Load data from all_data.json (language only dataset)
        json_path = os.path.join(self.args.data, "json_feat_2.1.0", "all_data.json")
        with open(json_path) as f:
            raw_data = json.load(f)

        # Turn data into array of [high_level, low_level_context, low_level_target, target_idx]
        split_data = []
        for key in raw_data.keys():
            for r_idx in range(len(raw_data[key]["high_level"])):
                high_level = raw_data[key]["high_level"][r_idx]
                low_levels = raw_data[key]["high_level"][r_idx]
                target_idx = random.randrange(len(low_levels))
                target = low_levels[target_idx]
                del low_levels[target_idx]
                split_data += [{"high_level": high_level, "low_level_context": low_levels, "low_level_target": target, "target_id": target_idx}]

        return split_data


# Input h,c are randomized/zero or should be prev (ZERO)
# They have a vocab argument which I assume gives the vocab
# Should I linearize hidden layer before output in encoder? (Assume don't have to cause hidden size is constant)

# Do the featurizing in the training loop rather than in the dataset
