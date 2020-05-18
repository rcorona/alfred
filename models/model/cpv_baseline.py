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



class AlfredDataset(Dataset):
    'Loads all data into RAM'
    def __init__(self, args, data, model_class, test_mode):
        self.model_class = model_class
        self.args = args
        self.test_mode = test_mode
        self.data = load_data_into_ram()

    def load_data_into_ram():
        # Load data from all_data.json (language only dataset)
        json_path = os.path.join(self.args.data, "json_feat_2.1.0", "all_data.json")
        with open(json_path) as f:
            raw_data = json.load(f)

        # Turn data into array of [high_level, low_level_context, low_level_target]
        split_data = []
        for key in raw_data.keys():
            for r_idx in range(len(raw_data[key]["high_level"])):
                high_level = raw_data[key]["high_level"][r_idx]
                low_levels = raw_data[key]["high_level"][r_idx]
                target_idx = random.randrange(len(low_levels))
                target = low_levels[target_idx]
                del low_levels[target_idx]
                split_data += [{"high_level": high_level, "low_level_context": low_levels, "low_level_target": target}]

        return split_data

    def __getitem__(self, idx):

        # Load task from dataset.
        task = self.data[idx]

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
        for k in range(len(feat[0])):
            feat[k] = pad_sequence([element[k] for element in feats])

        return (batch, feats)


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

    def featurize(self, ex, load_mask=True, load_frames=True):
        '''
        tensorize and pad batch individual input
        '''
        device = torch.device('cuda') if self.args.gpu else torch.device('cpu')

        high_level = torch.tensor(ex["high_level"])
        low_level_context = [torch.tensor(ll) for ll in ex["low_level_context"]]
        low_level_target = torch.tensor(ex["low_level_target"])
        padded_context = torch.cat([high_level] + [self.seg] + low_level_context, dim=0)

        return (padded_context, low_level_target)

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


# Input h,c are randomized/zero or should be prev (ZERO)
# They have a vocab argument which I assume gives the vocab
# Should I linearize hidden layer before output in encoder? (Assume don't have to cause hidden size is constant)
