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
import sklearn.metrics
from vocab import Vocab
from torch import nn
from tensorboardX import SummaryWriter
from tqdm import trange
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from model.seq2seq import Module as Base
from models.utils.metric import compute_f1, compute_exact
from gen.utils.image_util import decompress_mask
from models.utils.helper_utils import plot_confusion_matrix
from torch.utils.data import Dataset, DataLoader


class AlfredBaselineDataset(Dataset):

    def __init__(self, args, data):
        self.pad = 0
        self.seg = 1
        self.args = args
        self.data = data
        self.device = torch.device('cuda') if self.args.gpu else torch.device('cpu')
        self.goals = Vocab(['GotoLocation', 'PickupObject', 'PutObject', 'CoolObject', 'HeatObject', 'CleanObject', 'SliceObject', 'ToggleObject', 'End'])


    def featurize(self, ex):
        '''
        tensorize and pad batch input
        '''

        # # Collect instructions from dictionary
        # high_level = torch.tensor(ex["high_level"])
        # low_level_context = [torch.tensor(ll) for ll in ex["low_level_context"]]
        #
        # # Remove target from low levels
        # target_idx = random.randrange(len(low_level_context))
        # low_level_target = low_level_context[target_idx]
        # del low_level_context[target_idx]
        # return (high_level, low_level_context, low_level_target)

        high_level = torch.tensor(ex['num']['lang_goal']).type(torch.long) # -> M
        low_level_context = [torch.tensor(ex['num']['lang_instr'][idx]) for idx in range(len(ex['ann']['instr']))]
        target_idx = random.randrange(len(low_level_context))
        low_level_target = torch.tensor(ex['num']['lang_instr'][target_idx]) # -> T
        del low_level_context[target_idx]

        padded_context = torch.cat([high_level] + [torch.tensor(self.seg).unsqueeze(0)] + low_level_context, dim=0) # -> N x 512 x 7 x 7

        # Categorize the correct target for error analysis
        category = self.goals.word2index(ex['plan']['high_pddl'][target_idx]['planner_action']['action'])

        return (high_level, low_level_context, low_level_target, category)



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

            # Initalize arrays for each type of tensor
            highs = []
            contexts = []
            targets = []
            labels = []

            highs_lens =  torch.tensor([batch[idx][0].shape[0] for idx in range(batch_size)], dtype=torch.long) # -> B
            contexts_lens = []
            targets_lens = torch.tensor([batch[idx][2].shape[0] for idx in range(batch_size)], dtype=torch.long) # -> B

            # Array of number of low level instructions for each high level
            contexts_nums = torch.tensor([len(batch[idx][1]) for idx in range(batch_size)], dtype=torch.long) # -> B

            # Calculate longest seqence (L) - this is the number we will be padding the low level instructions to
            longest_seq = 0
            for idx in range(batch_size):
                longest_seq = max(longest_seq, max([batch[idx][1][seq].shape[0] for seq in range(len(batch[idx][1]))]))

            # Pad the low level tensors to be the correct length - seq len -> L
            for idx in range(batch_size):
                # Create new zeros tensor that will hold the low level tensors for this high level instruction
                padded_contexts = torch.zeros((len(batch[idx][1]), longest_seq), dtype=torch.long) # -> num_ll x L

                # Keep track of original lengths of each tensor for masking
                curr_context_lens = torch.tensor([batch[idx][1][seq].shape[0] for seq in range(len(batch[idx][1]))], dtype=torch.long) # -> num_ll

                # Copy entire low level tensor into padded tensor
                for seq_idx in range(len(batch[idx][1])):
                    padded_contexts[seq_idx, :batch[idx][1][seq_idx].shape[0]] = batch[idx][1][seq_idx] # -> num_ll x L

                # Sory by length for packing
                sorted_lengths, sort_idx = torch.sort(curr_context_lens , 0, descending=True) # -> num_ll
                sorted_contexts = padded_contexts[sort_idx]

                # Append all data to arrays for padding
                highs.append(batch[idx][0]) # -> seq_len
                contexts.append(sorted_contexts) # -> num_ll x L
                targets.append(batch[idx][2]) # -> seq_len
                labels.append(torch.tensor(idx).unsqueeze(0)) # -> 1
                contexts_lens.append(sorted_lengths) # -> num_ll

            # Pad all sequences to make big tensors
            contexts_lens = pad_sequence(contexts_lens, batch_first=True, padding_value=longest_seq) # -> B x NL
            packed_contexts = pad_sequence(contexts, batch_first=True) # -> B x NL x L (NL = Largest # of Low Levels)
            padded_highs = pad_sequence(highs, batch_first=True) # -> B x M (M = Longest High Level)
            padded_targets = pad_sequence(targets, batch_first=True) # -> B x T (T = Longest Target)
            padded_labels = torch.cat(labels, dim=0) # -> B

            # Mask for the second level of padding on low levels - pad on # of low levels
            mask = torch.zeros((batch_size, contexts_nums.max()), dtype=torch.float) # -> B x NL
            for idx in range(batch_size):
                mask[idx, :contexts_nums[idx]] = torch.ones((contexts_nums[idx],))

            categories = [batch[idx][3] for idx in range(batch_size)]

            return (padded_highs, packed_contexts, padded_targets, padded_labels, highs_lens, contexts_lens, targets_lens, mask, categories)
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
        self.classes = ['GotoLocation', 'PickupObject', 'PutObject', 'CoolObject', 'HeatObject', 'CleanObject', 'SliceObject', 'ToggleObject', 'End']

        # encoder and self-attention
        self.embed = nn.Embedding(len(self.vocab['word']), args.demb)
        self.enc = nn.LSTM(args.demb, args.dhid, bidirectional=True, batch_first=True)
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

    def forward(self, highs, contexts, targets, highs_lens, contexts_lens, targets_lens, mask):
        '''
        Takes in contexts and targets and returns the dot product of each enc(high) - enc(context) with each enc(target)
        '''
        ### INPUTS ###
        # Highs ->  B x M (M = Highs Seq Len)
        # Contexts -> B x NL x L (NL = Largest # of Low Levels, L = Lows Seq Len)
        # Targets -> B x T (T = Targets Seq Len)
        # Highs_Lens -> B
        # Contexts_Lens -> B x NL
        # Targets_Lens -> B
        # Mask -> B x NL

        batch_size = contexts.shape[0]
        contexts_num = contexts.shape[1]
        contexts_len = contexts.shape[2]

        ### HIGHS ###
        # Embedding:
        embed_highs = self.embed(highs) # -> B x M x E (E = embedding size)
        # Packing:
        packed_highs = pack_padded_sequence(embed_highs, highs_lens, batch_first=True, enforce_sorted=False)
        # Encoding:
        enc_highs = self.encoder(packed_highs, batch_size) # -> B x H

        ### CONTEXTS ###
        # Reshaping:
        flat_contexts = contexts.reshape(batch_size * contexts_num, contexts_len) # -> B * NL x L
        flat_lengths = contexts_lens.reshape(batch_size * contexts_num) # -> B * NL
        # Embedding:
        embed_contexts = self.embed(flat_contexts) # -> B * NL x L x E
        # Packing:
        packed_contexts = pack_padded_sequence(embed_contexts, flat_lengths, batch_first=True, enforce_sorted=False)
        # Encoding:
        enc_contexts = self.encoder(packed_contexts, batch_size * contexts_num) # -> B * NL x H
        # Reshaping:
        full_enc_contexts = enc_contexts.reshape(batch_size, contexts_num, self.args.dhid) # -> B x NL x H
        # Masking:
        masked_enc_contexts = torch.einsum('ijk, ij -> ijk', full_enc_contexts, mask) # -> B x NL x H
        # Sum all low levels that correspond to the same high level:
        summed_contexts = torch.sum(masked_enc_contexts, dim=1)  # -> B x H

        ### TARGETS ###
        # Embedding:
        embed_targets = self.embed(targets) # -> B x T x E
        # Packing:
        packed_targets = pack_padded_sequence(embed_targets, targets_lens, batch_first=True, enforce_sorted=False)
        # Encoding:
        enc_targets = self.encoder(packed_targets, batch_size) # B x H

        ### COMB ###
        # Combining high levels and low levels:
        comb_contexts = enc_highs - summed_contexts # -> B x H
        # Dot product:
        sim_m = torch.matmul(comb_contexts, torch.transpose(enc_targets, 0, 1)) # -> B x B
        logits = F.log_softmax(sim_m, dim = 1)

        return logits

    def run_train(self, splits, optimizer, args=None):

        ### SETUP ###
        args = args or self.args
        self.writer = SummaryWriter('runs/lang_cpv')
        fsave = os.path.join(args.dout, 'best.pth')

        # # Get splits
        # splits = self.load_data_into_ram()
        # valid_idx = np.arange(start=0, stop=len(splits), step=10)
        # eval_idx = np.arange(start=1, stop=len(splits), step=10)
        # train_idx = [i for i in range(len(splits)) if i not in valid_idx and i not in eval_idx]
        #
        # # Initialize Datasets
        # valid_dataset = AlfredBaselineDataset(args, [splits[i] for i in range(len(splits)) if i in valid_idx])
        # train_dataset = AlfredBaselineDataset(args, [splits[i] for i in range(len(splits)) if i in train_idx])

        train_data = splits['train']
        valid_data = splits['valid_seen'] + splits['valid_unseen']

        # Initialize Datasets
        valid_dataset = AlfredBaselineDataset(self.args, valid_data)
        train_dataset = AlfredBaselineDataset(self.args, train_data)

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
                highs, contexts, targets, labels, highs_lens, contexts_lens, targets_lens, mask, categories = batch

                # Transfer to GPU
                highs = highs.to(self.device)
                contexts = contexts.to(self.device)
                targets = targets.to(self.device)
                labels = labels.to(self.device)
                highs_lens = highs_lens.to(self.device)
                contexts_lens = contexts_lens.to(self.device)
                targets_lens = targets_lens.to(self.device)
                mask = mask.to(self.device)

                # Forward
                logits = self.forward(highs, contexts, targets, highs_lens, contexts_lens, targets_lens, mask)

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

            # Will contain pairs of (predicted category, actual category) for analysis
            predicted = []
            expected = []

            with torch.no_grad():
                for batch in valid_loader:
                    highs, contexts, targets, labels, highs_lens, contexts_lens, targets_lens, mask, categories = batch

                    # Transfer to GPU
                    highs = highs.to(self.device)
                    contexts = contexts.to(self.device)
                    targets = targets.to(self.device)
                    labels = labels.to(self.device)
                    highs_lens = highs_lens.to(self.device)
                    contexts_lens = contexts_lens.to(self.device)
                    targets_lens = targets_lens.to(self.device)
                    mask = mask.to(self.device)

                    # Forward
                    logits = self.forward(highs, contexts, targets, highs_lens, contexts_lens, targets_lens, mask)

                    # Calculate Loss and Accuracy
                    loss = F.nll_loss(logits, labels)
                    total_valid_loss += loss
                    total_valid_size += labels.shape[0]
                    most_likely = torch.argmax(logits, dim=1)
                    acc = torch.eq(most_likely, labels)
                    total_valid_acc += torch.sum(acc)

                    # Enter results
                    predicted.extend([categories[torch.argmax(logits, dim=1)[i]] for i in range(logits.shape[0])])
                    expected.extend(categories)

                # Write to TensorBoardX
                self.writer.add_scalar('Accuracy/validation', (total_valid_acc/total_valid_size).item(), epoch)
                self.writer.add_scalar('Loss/validation', total_valid_loss.item(), epoch)
                print("Validation Accuracy: " + str((total_valid_acc/total_valid_size).item()))
                print("Validation Loss: " + str(total_valid_loss.item()))

                cm = sklearn.metrics.confusion_matrix(expected, predicted)
                figure = plot_confusion_matrix(cm, class_names=self.classes)

                self.writer.add_figure("Confusion Matrix", figure, global_step=epoch)

            self.writer.flush()
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

        self.writer.close()

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
