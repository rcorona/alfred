import os
import random
import collections
import json
import numpy as np
import tqdm
import sklearn.metrics
import matplotlib.pyplot as plt
from vocab import Vocab
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

class BaselineDataset(Dataset):
    def __init__(self, args, data):
        self.pad = 0
        self.seg = 1
        self.args = args
        self.data = data
        self.device = torch.device('cuda') if self.args.gpu else torch.device('cpu')
        self.object_default = np.array([np.eye(11) for _ in range(49)])
        self.color_default = np.array([np.eye(6) for _ in range(49)])
        self.state_default = np.array([np.eye(3) for _ in range(49)])

    def featurize(self, ex):
        '''
        Takes in a single data point (defined by the dictionary ex) and featurizes it.
        '''

        task_folder = ex["folder"] # The folder is also the task type
        task_file = ex["file"]
        task_num = ex["ann"]

        lang_root = os.path.join(self.args.data, task_folder, task_file + ".json") # Contains all the information about the data
        img_root = os.path.join(self.args.data, task_folder, "imgs" + task_file[4:] + ".npz") # Contains the trajectory itself

        with open(lang_root) as file:
            data = json.load(file)[task_num]
            img_file = np.load(img_root)
            imgs = img_file["arr_" + str(task_num)]
            imgs = np.split(imgs, len(imgs) // 7)
            img_file.close()

        final_shape = 7 * 7 * (11 + 6 + 3)


        imgs = [np.reshape(img, (49, -1)) for img in imgs]

        low_level_object = [torch.tensor(self.object_default[list(range(49)), img[:, 0], :], dtype=torch.float) for img in imgs]
        low_level_color = [torch.tensor(self.color_default[list(range(49)), img[:, 1], :], dtype=torch.float) for img in imgs]
        low_level_state = [torch.tensor(self.state_default[list(range(49)), img[:, 2], :], dtype=torch.float) for img in imgs]

        low_levels = [torch.cat([low_level_object[i], low_level_color[i], low_level_state[i]], dim=1).reshape(final_shape) for i in range(len(imgs))]
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

        high_level = torch.tensor(data['num_instr'])

        return {"high" : high_level, "context": padded_context, "target": padded_target}

    def __getitem__(self, idx):
        '''
        Returns the featurized data point at index idx.
        '''

        task = self.data[idx]
        feat = self.featurize(task)
        return feat

    def __len__(self):
        '''
        Returns the number of data points total.
        '''

        return len(self.data)

    def collate_gen(self):
        '''
        Returns a function that collates the dataset. It is inside this function because it needs access to some of the
        args passed in and it cannot take a self argument.
        '''
        def collate(batch):
            '''
            Collates a batch of datapoints to these specifications:
                high - stacked and padded high level instructions -> [B x M]
                context - stacked and padded low level instructions [:target_idx] -> [B x N x 147]
                target - stacked and padded low level instructions [target_idx:] -> [B x T x 147]
                labels - a matrix with elements {1 ... B} -> [B]
                high_lens - array of length of instruction per batch, used for packing -> [B]
                context_lens - array of length of context per batch, used for packing -> [B]
                target_lens - array of length of target per batch, used for packing -> [B]
            '''

            batch_size = len(batch)
            high = []
            high_lens = []
            context = []
            context_lens = []
            target = []
            target_lens = []

            for idx in range(batch_size):
                high.append(batch[idx]["high"])
                high_lens.append(batch[idx]["high"].shape[0])
                context.append(batch[idx]["context"])
                context_lens.append(batch[idx]["context"].shape[0])
                target.append(batch[idx]["target"])
                target_lens.append(batch[idx]["target"].shape[0])

            high = pad_sequence(high, batch_first=True) # -> B x M
            high_lens = torch.tensor(high_lens) # -> B
            context = pad_sequence(context, batch_first=True) # B x N x 147
            context_lens = torch.tensor(context_lens) # -> B
            target = pad_sequence(target, batch_first=True) # B x T x 147
            target_lens = torch.tensor(target_lens) # -> B
            labels = torch.tensor([*range(batch_size)]) # -> B

            return {"high": high, "high_lens": high_lens, "context": context, "context_lens": context_lens, "target": target, "target_lens": target_lens, "labels": labels}
        return collate


class Module(nn.Module):

    def __init__(self, args, vocab):
        super().__init__()

        self.args = args
        self.vocab = vocab

        self.pseudo = args.pseudo
        self.img_shape = 7 * 7 * (11 + 6 + 3) # This is based off of the Babyai img size

        self.embed = nn.Embedding(len(self.vocab), args.demb)
        self.linear = nn.Linear(self.img_shape, args.demb)
        self.lang_enc = nn.LSTM(args.demb, args.dhid, num_layers=2, batch_first=True)
        self.img_enc = nn.LSTM(args.demb, args.dhid, num_layers=2, batch_first=True)
        self.lin_1 = nn.Linear(args.dhid * 4, args.dhid)
        self.lin_2 = nn.Linear(args.dhid, 1)

        self.device = torch.device('cuda') if self.args.gpu else torch.device('cpu')
        self.to(self.device)

    def language_encoder(self, batch, batch_size, h_0=None, c_0=None):
        '''
        Encodes a stacked tensor.
        '''

        if h_0 is None or c_0 is None:
            h_0 = torch.zeros(2, batch_size, self.args.dhid).type(torch.float).to(self.device) # -> 2 x B x H
            c_0 = torch.zeros(2, batch_size, self.args.dhid).type(torch.float).to(self.device) # -> 2 x B x H
        out, (h, c) = self.lang_enc(batch, (h_0, c_0)) # -> M x B x H

        hid_sum = torch.transpose(h, 0, 1).reshape(batch_size, -1)

        return hid_sum, h, c

    def image_encoder(self, batch, batch_size, h_0=None, c_0=None):
        '''
        Encodes a stacked tensor.
        '''

        if h_0 is None or c_0 is None:
            h_0 = torch.zeros(2, batch_size, self.args.dhid).type(torch.float).to(self.device) # -> 2 x B x H
            c_0 = torch.zeros(2, batch_size, self.args.dhid).type(torch.float).to(self.device) # -> 2 x B x H
        out, (h, c) = self.img_enc(batch, (h_0, c_0)) # -> M x B x H

        hid_sum = torch.transpose(h, 0, 1).reshape(batch_size, -1)

        return hid_sum, h, c

    def forward(self, high, context, high_lens, context_lens):
        '''

        '''

        B = context.shape[0]

        ### High ###
        high = self.embed(high) # -> B x M x D
        high = pack_padded_sequence(high, high_lens, batch_first=True, enforce_sorted=False)
        high, _, _ = self.language_encoder(high, B) # -> B x H

        ### Context ###
        context = self.linear(context)
        context = pack_padded_sequence(context, context_lens, batch_first=True, enforce_sorted=False)
        context, _, _ = self.image_encoder(context, B) # -> B x H

        ### Combination ###
        combination = torch.stack([torch.cat((high[i].unsqueeze(0).repeat(B, 1), context), dim=1) for i in range(B)]) # -> B x B x 2H
        combination = self.lin_1(combination) # -> B x B x 10H
        combination = F.relu(combination) # -> B x B x 10H
        combination = self.lin_2(combination).squeeze() # -> B x B
        combination = F.tanh(combination) # -> B x B

        output = {}
        output["prediction"] = combination

        return output

    def run_train(self, splits, optimizer, args=None):
        '''
        '''

        ### SETUP ###
        args = args or self.args
        self.writer = SummaryWriter('runs/babyai_baseline_simple_subset')
        fsave = os.path.join(args.dout, 'best.pth')
        psave = os.path.join(args.dout, 'pseudo_best.pth')

        with open(splits['train'], 'r') as file:
            train_data = json.load(file)
        with open(splits['valid'], 'r') as file:
            valid_data = json.load(file)

        valid_dataset = BaselineDataset(self.args, valid_data)
        train_dataset = BaselineDataset(self.args, train_data)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch, shuffle=True, num_workers=args.workers, collate_fn=valid_dataset.collate_gen())
        train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=args.workers, collate_fn=train_dataset.collate_gen())

        optimizer = optimizer or torch.optim.Adam(list(self.parameters()), lr=args.lr)

        best_loss = 1e10
        if self.pseudo:
            pseudo_epoch = 0
            pseudo_epoch_batch_size = len(train_dataset)//(args.pseudo_epoch * args.batch)

        for epoch in range(args.epoch):
            print('Epoch', epoch)
            desc_train = "Epoch " + str(epoch) + ", train"
            desc_valid = "Epoch " + str(epoch) + ", valid"

            loss = {
                "total": torch.tensor(0, dtype=torch.float)
            }
            size = torch.tensor(0, dtype=torch.float)

            if self.pseudo:
                pseudo_loss = {
                    "total": torch.tensor(0, dtype=torch.float)
                }
                batch_idx = 0
                pseudo_size = torch.tensor(0, dtype=torch.float)

            self.train()

            for batch in tqdm.tqdm(train_loader, desc=desc_train):
                optimizer.zero_grad()

                batch_size = batch["high"].shape[0]
                size += batch_size
                pseudo_size += batch_size
                high = batch["high"].to(self.device)
                context = batch["context"].to(self.device)
                target = batch["target"].to(self.device)
                high_lens = batch["high_lens"].to(self.device)
                context_lens = batch["context_lens"].to(self.device)
                target_lens = batch["target_lens"].to(self.device)

                output = self.forward(high, context, high_lens, context_lens)

                total_loss = 0
                for b in range(batch_size):
                    correctness_mask = torch.ones((batch_size,)).to(self.device) * -1
                    correctness_mask[b] = 1
                    progress = context_lens.float()/(context_lens.float() + target_lens.float())
                    c_loss = F.mse_loss(output["prediction"][b], progress * correctness_mask, reduction='none')
                    weight_mask = torch.ones((batch_size,)).to(self.device)
                    weight_mask[b] = batch_size - 1
                    contrast_loss += torch.dot(c_loss, weight_mask)

                loss["total"] += total_loss
                pseudo_loss["total"] += total_loss

                total_loss.backward()
                optimizer.step()
                torch.cuda.empty_cache()

                if self.pseudo and batch_idx == pseudo_epoch_batch_size:
                    self.write(pseudo_loss, pseudo_size, pseudo_epoch, pseudo=True)
                    self.run_valid(valid_loader, pseudo_epoch, pseudo=True)

                    pseudo_epoch += 1
                    batch_idx = -1
                    pseudo_size = torch.tensor(0, dtype=torch.float)
                    pseudo_loss["total"] = torch.tensor(0, dtype=torch.float)
                    torch.save({
                        'model': self.state_dict(),
                        'optim': optimizer.state_dict(),
                        'args': self.args,
                        'vocab': self.vocab
                    }, psave)

                self.train()
                batch_idx += 1

            self.write(loss, size, epoch, pseudo=False)
            valid_loss = self.run_valid(valid_loader, epoch, desc_valid=desc_valid)

            print("Train Loss: " + str((loss["total"]/size).item()))
            print("Validation Loss: " + str((valid_loss["total"]/size).item()))

            self.writer.flush()

            if valid_loss["total"] < best_loss:
                print( "Obtained a new best validation loss of {:.2f}, saving model checkpoint to {}...".format(valid_loss["total"], fsave))
                torch.save({
                    'model': self.state_dict(),
                    'optim': optimizer.state_dict(),
                    'args': self.args,
                    'vocab': self.vocab
                }, fsave)
                best_loss = valid_loss["total"]

        self.writer.close()

    def run_valid(self, valid_loader, epoch, pseudo=False, desc_valid=None):
        self.eval()
        loss = {
            "total": torch.tensor(0, dtype=torch.float)
        }

        size = torch.tensor(0, dtype=torch.float)

        loader = valid_loader

        if not pseudo:
            loader = tqdm.tqdm(loader, desc = desc_valid)

        with torch.no_grad():
            for batch in loader:
                batch_size = batch["high"].shape[0]
                size += batch_size
                high = batch["high"].to(self.device)
                context = batch["context"].to(self.device)
                target = batch["target"].to(self.device)
                high_lens = batch["high_lens"].to(self.device)
                context_lens = batch["context_lens"].to(self.device)
                target_lens = batch["target_lens"].to(self.device)

                output = self.forward(high, context, high_lens, context_lens)

                total_loss = 0
                for b in range(batch_size):
                    correctness_mask = torch.ones((batch_size,)).to(self.device) * -1
                    correctness_mask[b] = 1
                    progress = context_lens.float()/(context_lens.float() + target_lens.float())
                    c_loss = F.mse_loss(output["prediction"][b], progress * correctness_mask, reduction='none')
                    weight_mask = torch.ones((batch_size,)).to(self.device)
                    weight_mask[b] = batch_size - 1
                    contrast_loss += torch.dot(c_loss, weight_mask)

                loss["total"] += total_loss

        self.write(loss, size, epoch, train=False, pseudo=pseudo)
        return loss


    def write(self, loss, size, epoch, train=True, pseudo=False):
        if train:
            type = "train"
        else:
            type = "valid"
        if self.pseudo:
            self.writer.add_scalar('PseudoLoss/' + type, (loss["total"]/size).item(), epoch)
        else:
            self.writer.add_scalar('Loss/' + type, (loss["total"]/size).item(), epoch)
