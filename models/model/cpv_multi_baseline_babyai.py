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
import tqdm
import sklearn.metrics
import matplotlib.pyplot as plt
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
import time
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


class AlfredBaselineDataset(Dataset):

    def __init__(self, args, data):
        self.pad = 0
        self.seg = 1
        self.args = args
        self.data = data
        self.device = torch.device('cuda') if self.args.gpu else torch.device('cpu')
        # self.goals = Vocab(['GotoLocation', 'PickupObject', 'PutObject', 'CoolObject', 'HeatObject', 'CleanObject', 'SliceObject', 'ToggleObject', 'End'])


    def featurize(self, ex):
        '''
        tensorize and pad batch input
        '''
        task_folder = ex['folder']
        task_file = ex['file']
        task_num = ex['ann']

        lang_root = os.path.join(self.args.data, task_folder, task_file + ".json")
        img_root = os.path.join(self.args.data, task_folder, "imgs" + task_file[4:] + ".npz")

        with open(lang_root) as file:
            data = json.load(file)[task_num]
            img_file = np.load(img_root) # sizes of images are 7 x 7 x 3
            imgs = img_file["arr_" + str(task_num)]
            imgs = np.split(imgs, len(imgs) // 7)
            img_file.close()

        # Low levels and targets are image trajectories
        final_shape = len(imgs[0]) * len(imgs[0][0]) * len(imgs[0][0][0])
        low_levels = [torch.tensor(img, dtype=torch.float).reshape(final_shape) for img in imgs]
        target_idx = random.randrange(len(low_levels))
        target_length = torch.tensor(len(low_levels) - target_idx)
        low_level_target = low_levels[target_idx:] # -> T x 147
        low_level_context = low_levels[:target_idx] # -> N x 147

        if len(low_level_context) == 0:
            padded_context = torch.tensor([[self.pad for x in range(final_shape)]], dtype=torch.float)
        else:
            padded_context = torch.stack(low_level_context, dim=0) # -> N x 147

        if len(low_level_target) == 0:
            padded_target = torch.tensor([[self.pad for x in range(final_shape)]], dtype=torch.float)
        else:
            padded_target = torch.stack(low_level_target, dim=0) # -> N x 147

        # High level and target are language instructions
        high_level = torch.tensor(data['num_instr'])

        return (high_level, padded_context, padded_target, target_length)


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

            highs = []
            contexts = []
            targets = []
            labels = []

            highs_lens = []
            contexts_lens = []
            targets_lens = []

            target_length = []

            for idx in range(batch_size):
                highs.append(batch[idx][0])
                highs_lens.append(batch[idx][0].shape[0])
                contexts.append(batch[idx][1])
                contexts_lens.append(batch[idx][1].shape[0])
                targets.append(batch[idx][2])
                targets_lens.append(batch[idx][2].shape[0])
                labels.append(torch.tensor(idx).unsqueeze(0))
                target_length.append(batch[idx][3])


            padded_highs = pad_sequence(highs, batch_first=True) # -> B x M (M = longest high seq)
            padded_contexts = pad_sequence(contexts, batch_first=True) # -> B x N x 147
            padded_targets = pad_sequence(targets,  batch_first=True) # -> B x T x 147 (T = longest target seq)
            padded_labels = torch.cat(labels, dim=0)

            highs_lens = torch.tensor(highs_lens)
            contexts_lens = torch.tensor(contexts_lens)
            targets_lens = torch.tensor(targets_lens)

            target_length = torch.stack(target_length)

            return (padded_highs, padded_contexts, padded_targets, padded_labels, highs_lens, contexts_lens, targets_lens, target_length)

        return collate_fn


class Module(nn.Module):
    def __init__(self, args, vocab):
        # super().__init__(args, vocab)
        super().__init__()
        self.pad = 0
        self.seg = 1

        # args and vocab
        self.args = args
        self.vocab = vocab
        self.pseudo = args.pseudo

        self.device = torch.device('cuda') if self.args.gpu else torch.device('cpu')

        self.img_shape =  147

        # self.classes = ['GotoLocation', 'PickupObject', 'PutObject', 'CoolObject', 'HeatObject', 'CleanObject', 'SliceObject', 'ToggleObject', 'End']

        # encoder and self-attention
        self.embed = nn.Embedding(len(self.vocab), args.demb)
        self.linear = nn.Linear(args.demb, self.img_shape)
        self.enc = nn.LSTM(self.img_shape, args.dhid, bidirectional=True, batch_first=True)
        self.to(self.device)


    def encoder(self, batch, batch_size, h_0, c_0):
        '''
        Input: stacked tensor of [high_level, seq, low_level_context]
        '''
        ### INPUTS ###
        # Batch -> B x H x E

        ### LSTM ###
        if h_0 is None or c_0 is None:
            h_0 = torch.zeros(2, batch_size, self.args.dhid).type(torch.float).to(self.device) # -> L * 2 x B x H
            c_0 = torch.zeros(2, batch_size, self.args.dhid).type(torch.float).to(self.device) # -> L * 2 x B x H
        out, (h, c) = self.enc(batch, (h_0, c_0)) # -> L * 2 x B x H

        ## COMB ##
        hid_sum = torch.sum(h, dim=0) # -> B x H

        return hid_sum, h, c

    def forward(self, highs, contexts, targets, highs_lens, contexts_lens, targets_lens):
        '''
        Takes in contexts and targets and returns the dot product of each enc(context) with each enc(target)
        '''
        ### INPUT ###
        # Highs -> B x M (M = Highs Seq Len)
        # Contexts ->  B x N x 147 (N = Contexts Seq Len)
        # Targets -> B x 147 (T = Targets Seq Len)

        batch_size = contexts.shape[0]

        ### HIGHS & CONTEXTS ###
        # Embedding:
        emb_highs = self.embed(highs) # -> B x M x D
        # Projection from lang embedding dimension to image embedding dimension:
        tall_highs = self.linear(emb_highs) # -> B x M x E
        # Packing:
        packed_highs = pack_padded_sequence(tall_highs, highs_lens, batch_first=True, enforce_sorted=False)
        # Encoding:
        enc_highs, h_t, c_t = self.encoder(packed_highs, batch_size, None, None) # -> L * 2 x B x H
        # Packing:
        packed_contexts = pack_padded_sequence(contexts, contexts_lens, batch_first=True, enforce_sorted=False)
        # Encoding:
        enc_contexts, _, _ = self.encoder(packed_contexts, batch_size, h_t, c_t) # -> B x H

        ### TARGETS ###
        # Packing:
        packed_targets = pack_padded_sequence(targets, targets_lens, batch_first=True, enforce_sorted=False)
        # Encoding:
        enc_targets, _, _ = self.encoder(targets, batch_size, None, None) # -> B x H

        ### FULL TRAJ ###
        # Packing:
        # Packing:
        packed_contexts = pack_padded_sequence(contexts, contexts_lens, batch_first=True, enforce_sorted=False)
        # Encoding:
        enc_traj, h_t, c_t = self.encoder(packed_contexts, batch_size, None, None) # -> B x H
        # Packing:
        packed_targets = pack_padded_sequence(targets, targets_lens, batch_first=True, enforce_sorted=False)
        # Encoding:
        enc_traj, _, _ = self.encoder(targets, batch_size, h_t, c_t) # -> B x H


        ### COMB ###
        # Dot Product:
        sim_m = torch.matmul(enc_contexts, torch.transpose(enc_targets, 0, 1)) # -> B x B
        logits = F.log_softmax(sim_m, dim = 1)

        highdottraj = torch.bmmm(enc_highs.reshape(batch_size, 1, -1), enc_traj(batch_size, -1, 1))
        hnorm = torch.bmm(enc_highs.reshape(batch_size, 1, -1), enc_highs.reshape(batch_size, -1, 1)).squeeze()


        return logits, highdottraj, hnorm

    def run_train(self, splits, optimizer, args=None):

        args = args or self.args
        self.writer = SummaryWriter('runs/babyai_new_baseline')
        fsave = os.path.join(args.dout, 'best.pth')

        # Get splits
        with open(splits['train'], 'r') as file:
            train_data = json.load(file)
        with open(splits['valid'], 'r') as file:
            valid_data = json.load(file)

        # Initialize Datasets
        valid_dataset = AlfredBaselineDataset(self.args, valid_data)
        train_dataset = AlfredBaselineDataset(self.args, train_data)

        # Initialize Dataloaders
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch, shuffle=True, num_workers=self.args.workers, collate_fn=valid_dataset.collate())
        train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=self.args.workers, collate_fn=train_dataset.collate())

        # Optimizer
        optimizer = optimizer or torch.optim.Adam(list(self.parameters()), lr=args.lr)

        # Training loop
        best_loss = 1e10
        if self.pseudo:
            print("Starting...")
            pseudo_epoch = 0
            pseudo_epoch_batches = len(train_dataset)//(args.pseudo_epoch * args.batch)
            print(pseudo_epoch_batches)

        for epoch in range(args.epoch):
            print('Epoch', epoch)
            desc_train = "Epoch " + str(epoch) + ", train"
            desc_valid = "Epoch " + str(epoch) + ", valid"

            total_train_loss = torch.tensor(0, dtype=torch.float)
            total_train_acc = torch.tensor(0, dtype=torch.float)
            total_train_size = torch.tensor(0, dtype=torch.float)

            if self.pseudo:
                pseudo_train_size = torch.tensor(0, dtype=torch.float)
                pseudo_train_acc = torch.tensor(0, dtype=torch.float)
                pseudo_train_loss = torch.tensor(0, dtype=torch.float)
                batch_idx = 0
                accuracy_by_length = {}
                num_by_length = {}

            self.train()
            for batch in tqdm.tqdm(train_loader, desc=desc_train):
                optimizer.zero_grad()
                highs, contexts, targets, labels, highs_lens, contexts_lens, targets_lens, target_length = batch

                # Transfer to GPU
                highs = highs.to(self.device)
                contexts = contexts.to(self.device)
                targets = targets.to(self.device)
                labels = labels.to(self.device)
                highs_lens = highs_lens.to(self.device)
                contexts_lens = contexts_lens.to(self.device)
                targets_lens = targets_lens.to(self.device)
                # Forward
                logits, highdottraj, hnorm = self.forward(highs, contexts, targets, highs_lens, contexts_lens, targets_lens)

                # Calculate Loss and Accuracy
                loss = F.nll_loss(logits, labels) + F.mse(highdottraj, hnorm**2)
                total_train_loss += loss
                pseudo_train_loss += loss
                total_train_size += labels.shape[0]
                pseudo_train_size += labels.shape[0]
                most_likely = torch.argmax(logits, dim=1)
                acc = torch.eq(most_likely, labels)
                total_train_acc += torch.sum(acc)
                pseudo_train_acc += torch.sum(acc)

                if self.pseudo:
                    for idx in range(len(batch)):
                        if not target_length[idx].item() in accuracy_by_length.keys():
                            accuracy_by_length[target_length[idx].item()] = int(acc[idx])
                            num_by_length[target_length[idx].item()] = 1
                        else:
                            accuracy_by_length[target_length[idx].item()] += int(acc[idx])
                            num_by_length[target_length[idx].item()] += 1

                # Backpropogate
                loss.backward()
                optimizer.step()

                if self.pseudo and batch_idx == pseudo_epoch_batches:
                    # Write to TensorBoardX
                    self.writer.add_scalar('PseudoAccuracy/train', (pseudo_train_acc/pseudo_train_size).item(), pseudo_epoch)
                    self.writer.add_scalar('PseudoLoss/train', (pseudo_train_loss/pseudo_train_size).item(), pseudo_epoch)

                    self.run_valid(valid_loader, pseudo_epoch)

                    pseudo_epoch += 1
                    batch_idx = -1
                    pseudo_train_loss = torch.tensor(0, dtype=torch.float)
                    pseudo_train_acc = torch.tensor(0, dtype=torch.float)
                    pseudo_train_size = torch.tensor(0, dtype=torch.float)

                    a = []
                    b = []
                    for idx in sorted(accuracy_by_length.keys()):
                        a.append(idx)

                        b.append(accuracy_by_length[idx]/num_by_length[idx])
                    fig = plt.figure()
                    plt.plot(a, b)
                    self.writer.add_figure('Accuracy By Length', fig, global_step=pseudo_epoch)
                    self.train()

                    accuracy_by_length = {}
                    num_by_length = {}
                batch_idx += 1

            # Write to TensorBoardX
            self.writer.add_scalar('Accuracy/train', (total_train_acc/total_train_size).item(), epoch)
            self.writer.add_scalar('Loss/train', (total_train_loss/total_train_size).item(), epoch)
            print("Train Accuracy: " + str((total_train_acc/total_train_size).item()))
            print("Train Loss: " + str((total_train_loss/total_train_size).item()))

            total_valid_loss = self.run_valid(valid_loader, epoch, pseudo=False, desc_valid=desc_valid)

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

    def run_valid(self, valid_loader, epoch, pseudo=True, desc_valid=None):

        self.eval()
        total_valid_loss = torch.tensor(0, dtype=torch.float)
        total_valid_acc = torch.tensor(0, dtype=torch.float)
        total_valid_size = torch.tensor(0, dtype=torch.float)

        if pseudo:
            loader = valid_loader

        else:
            loader = tqdm.tqdm(valid_loader, desc = desc_valid)

        with torch.no_grad():
            for batch in loader:
                highs, contexts, targets, labels, highs_lens, contexts_lens, targets_lens, target_length = batch

                # Transfer to GPU
                highs = highs.to(self.device)
                contexts = contexts.to(self.device)
                targets = targets.to(self.device)
                labels = labels.to(self.device)
                highs_lens = highs_lens.to(self.device)
                contexts_lens = contexts_lens.to(self.device)
                targets_lens = targets_lens.to(self.device)

                # Forward
                logits, highdottraj, hnorm = self.forward(highs, contexts, targets, highs_lens, contexts_lens, targets_lens)

                # Calculate Loss and Accuracy
                loss = F.nll_loss(logits, labels) + F.mse(highdottraj, hnorm**2)
                total_valid_loss += loss
                total_valid_size += labels.shape[0]
                most_likely = torch.argmax(logits, dim=1)
                acc = torch.eq(most_likely, labels)
                total_valid_acc += torch.sum(acc)



        if not pseudo:
            # Write to TensorBoardX
            self.writer.add_scalar('Accuracy/validation', (total_valid_acc/total_valid_size).item(), epoch)
            self.writer.add_scalar('Loss/validation', (total_valid_loss/total_valid_size).item(), epoch)
            print("Validation Accuracy: " + str((total_valid_acc/total_valid_size).item()))
            print("Validation Loss: " + str((total_valid_loss/total_valid_size).item()))
        else:
            self.writer.add_scalar('PseudoAccuracy/validation', (total_valid_acc/total_valid_size).item(), epoch)
            self.writer.add_scalar('PseudoLoss/validation', (total_valid_loss/total_valid_size).item(), epoch)


        return total_valid_loss
