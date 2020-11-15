import os
import random
import collections
import json
import revtok
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
import pdb

torch.multiprocessing.set_sharing_strategy('file_system')

class CPVDataset(Dataset):
    def __init__(self, args, vocab, data):
        self.pad = 0
        self.seg = 1
        self.args = args
        self.data = data
        self.device = torch.device('cuda') if self.args.gpu else torch.device('cpu')

        self.object_default = np.array([np.eye(11) for _ in range(49)])
        self.color_default = np.array([np.eye(6) for _ in range(49)])
        self.state_default = np.array([np.eye(3) for _ in range(49)])

        then_merge = vocab.word2index([",", "then"], train=False)
        and_merge = vocab.word2index(["and"], train=False)
        after_merge = vocab.word2index(["after", "you"], train=False)

        self.merges = [then_merge, and_merge, after_merge]


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
        target_idx = random.randrange(0, len(low_levels))
        target_length = torch.tensor(len(low_levels) - target_idx)
        low_level_target = low_levels[target_idx:] # -> T x 147
        low_level_context = low_levels[:target_idx] # -> N x 147





        if len(low_level_context) == 0:
            action = torch.tensor(6)
            padded_context = torch.tensor([[self.pad for x in range(final_shape)]], dtype=torch.float)
        else:
            action = torch.tensor(data['act_idx'][target_idx - 1])
            padded_context = torch.stack(low_level_context, dim=0) # -> N x 147

        if len(low_level_target) == 0:
            padded_target = torch.tensor([[self.pad for x in range(final_shape)]], dtype=torch.float)
        else:
            padded_target = torch.stack(low_level_target, dim=0) # -> N x 147

        high_level = torch.tensor(data['num_instr'])


        return {"high" : high_level, "context": padded_context, "target": padded_target, "action": action}

    def merge_highs(self, high_1, high_2):
        merge = torch.tensor(self.merges[random.randrange(3)])
        return torch.cat([high_1, merge, high_2], dim=0)

    def merge_lows(self, low_1, low_2):
        return torch.cat([low_1, low_2], dim=0)


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
            merged = []
            merged_lens = []
            merged_img = []
            merged_img_lens = []
            actions = []

            l1 = self.merge_lows(batch[0]["context"], batch[0]["target"])

            for idx in range(batch_size):
                high.append(batch[idx]["high"])
                high_lens.append(batch[idx]["high"].shape[0])
                context.append(batch[idx]["context"])
                context_lens.append(batch[idx]["context"].shape[0])
                target.append(batch[idx]["target"])
                target_lens.append(batch[idx]["target"].shape[0])


                if self.args.imitation_loss:
                    actions.append(batch[idx]["action"])

                if self.args.generalizing_loss:
                    h1 = batch[idx]["high"]
                    l2 = self.merge_lows(batch[idx]["context"], batch[idx]["target"])
                    m_img = self.merge_lows(l1, l2)
                    merged_img.append(m_img)
                    merged_img_lens.append(m_img.shape[0])

                    for jdx in range(batch_size):
                        h2 = batch[jdx]["high"]
                        l2 = self.merge_lows(batch[jdx]["context"], batch[jdx]["target"])
                        m = self.merge_highs(h1, h2)
                        merged.append(m)
                        merged_lens.append(m.shape[0])


            high = pad_sequence(high, batch_first=True) # -> B x M
            high_lens = torch.tensor(high_lens) # -> B
            context = pad_sequence(context, batch_first=True) # B x N x 147
            context_lens = torch.tensor(context_lens) # -> B
            target = pad_sequence(target, batch_first=True) # B x T x 147
            target_lens = torch.tensor(target_lens) # -> B
            labels = torch.tensor([*range(batch_size)]) # -> B
            if self.args.generalizing_loss:
                merged = pad_sequence(merged, batch_first=True) # -> B x 2M
                merged_lens = torch.tensor(merged_lens) # -> B
                merged_img = pad_sequence(merged_img, batch_first=True) # -> B x 2M
                merged_img_lens = torch.tensor(merged_img_lens) # -> B
            if self.args.imitation_loss:
                actions = torch.tensor(actions)


            return {
                "high": high,
                "high_lens": high_lens,
                "context": context,
                "context_lens": context_lens,
                "target": target,
                "target_lens": target_lens,
                "merged": merged,
                "merged_lens": merged_lens,
                "merged_img": merged_img,
                "merged_img_lens": merged_img_lens,
                "actions": actions,
                "labels": labels
            }
        return collate



class Module(nn.Module):

    def __init__(self, args, vocab):
        super().__init__()

        self.args = args
        self.vocab = vocab

        self.pseudo = args.pseudo
        self.img_shape = 7 * 7 * (11 + 6 + 3) # This is based off of the Babyai img size
        self.num_actions = 7

        self.embed = nn.Embedding(len(self.vocab), args.demb)
        self.linear = nn.Linear(self.img_shape, args.demb)
        self.lang_enc = nn.LSTM(args.demb, args.dhid, bidirectional=False, num_layers=2, batch_first=True)
        self.img_enc = nn.LSTM(args.demb, args.dhid, bidirectional=False, num_layers=2, batch_first=True)
        self.obs_enc = nn.LSTM(args.demb, args.dhid, bidirectional=False, num_layers=2, batch_first=True)

        self.im_linear_1 =  nn.Linear(args.dhid * 2, args.demb)
        self.im_linear_2 =  nn.Linear(args.demb, self.num_actions)

        self.device = torch.device('cuda') if self.args.gpu else torch.device('cpu')
        self.to(self.device)

    def language_encoder(self, batch, batch_size, h_0=None, c_0=None):
        '''
        Encodes a stacked tensor.
        '''

        if h_0 is None or c_0 is None:
            h_0 = torch.zeros(2, batch_size, self.args.dhid).type(torch.float).to(self.device) # -> 2 x B x H
            c_0 = torch.zeros(2, batch_size, self.args.dhid).type(torch.float).to(self.device) # -> 2 x B x H
        out, (h, c) = self.lang_enc(batch, (h_0, c_0)) # -> 2 x B x H

        hid_sum = h[-1].squeeze() # -> B x H

        return hid_sum, h, c

    def image_encoder(self, batch, batch_size, h_0=None, c_0=None):
        '''
        Encodes a stacked tensor.
        '''

        if h_0 is None or c_0 is None:
            h_0 = torch.zeros(2, batch_size, self.args.dhid).type(torch.float).to(self.device) # -> 2 x B x H
            c_0 = torch.zeros(2, batch_size, self.args.dhid).type(torch.float).to(self.device) # -> 2 x B x H
        out, (h, c) = self.img_enc(batch, (h_0, c_0)) # -> 2 x B x H

        hid_sum = h[-1].squeeze() # -> B x H

        return hid_sum, h, c

    def observation_encoder(self, batch, batch_size, h_0=None, c_0=None):
        '''
        Encodes a stacked tensor.
        '''

        if h_0 is None or c_0 is None:
            h_0 = torch.zeros(2, batch_size, self.args.dhid).type(torch.float).to(self.device) # -> 2 x B x H
            c_0 = torch.zeros(2, batch_size, self.args.dhid).type(torch.float).to(self.device) # -> 2 x B x H
        out, (h, c) = self.obs_enc(batch, (h_0, c_0)) # -> 2 x B x H

        hid_sum = h[-1].squeeze() # -> B x H

        return hid_sum, h, c

    def forward(self, high, context, target, merged, merged_img, high_lens, context_lens, target_lens, merged_lens, merged_img_lens):
        '''

        '''

        B = context.shape[0]

        ### High ###
        high = self.embed(high) # -> B x M x D
        high = pack_padded_sequence(high, high_lens, batch_first=True, enforce_sorted=False)
        high, _, _ = self.language_encoder(high, B) # -> B x H

        ### Context ###
        context = self.linear(context)
        packed_context = pack_padded_sequence(context, context_lens, batch_first=True, enforce_sorted=False)
        context, h, c = self.image_encoder(packed_context, B)

        ### Target ###
        target = self.linear(target)
        packed_target = pack_padded_sequence(target, target_lens, batch_first=True, enforce_sorted=False)
        target, _, _ = self.image_encoder(packed_target, B)

        ### Full Trajectory ###
        trajectory, _, _ = self.image_encoder(packed_target, B, h, c)

        if self.args.generalizing_loss:
            ### Merged Highs ###
            merged = self.embed(merged)
            merged = pack_padded_sequence(merged, merged_lens, batch_first=True, enforce_sorted=False)
            merged, _, _ = self.language_encoder(merged, B * B)

            summed = torch.einsum('ik, jk -> ijk', high, high)
            summed = torch.reshape(summed, (B * B, -1))

            ### Merged Lows ###
            merged_img = self.linear(merged_img)
            merged_img = pack_padded_sequence(merged_img, merged_img_lens, batch_first=True, enforce_sorted=False)
            merged_img, _, _ = self.image_encoder(merged_img, B)

            summed_img = trajectory[0].unsqueeze(0).repeat(B, 1) + trajectory

        if self.args.imitation_loss:
            ## plan -> high - context
            plan = high - context
            ## current -> Put context trajectory through a different lstm
            current, _, _ = self.observation_encoder(packed_context, B)
            ## state -> concat plan and current
            state = torch.cat([plan, current], dim=1)
            ## put state through ff
            state = self.im_linear_1(state)
            state = F.relu(state)
            state = self.im_linear_2(state)

        ### Combinations ###
        output = {}
        output["H * C"] = torch.matmul(high, torch.transpose(context, 0, 1)) # -> B x B
        output["<H, C>"] = torch.bmm(high.reshape(B, 1, -1), context.reshape(B, -1, 1)).squeeze() # -> B
        output["<H, T>"] = torch.bmm(high.reshape(B, 1, -1), target.reshape(B, -1, 1)).squeeze() # -> B
        output["<H, N>"] = torch.bmm(high.reshape(B, 1, -1), trajectory.reshape(B, -1, 1)).squeeze() # -> B
        output["<H, C + T>"] = torch.bmm(high.reshape(B, 1, -1), (context + target).reshape(B, -1, 1)).squeeze() # -> B
        output["norm(H)"] = torch.norm(high, dim=1) # -> B
        output["norm(C)"] = torch.norm(context, dim=1) # -> B
        output["norm(T)"] = torch.norm(target, dim=1) # -> B
        output["norm(N)"] = torch.norm(trajectory, dim=1) # -> B
        output["cos(H, N)"] = F.cosine_similarity(high, trajectory) # -> B
        if self.args.generalizing_loss:
            output["H12"] = merged
            output["H1 + H2"] = summed
            output["T12"] = merged_img
            output["T1 + T2"] = summed_img
        if self.args.imitation_loss:
            output["At+1"] = state

        return output

    def compute_losses(self, batch, loss, pseudo_loss=None):
        """
        Compute losses given a batch during training/eval.

        batch - The batch to process.
        loss - Dict to update with losses.
        pseudo_loss - Dict to update with losses (multiple times per epoch).

        """
        batch_size = batch["high"].shape[0]
        high = batch["high"].to(self.device)
        context = batch["context"].to(self.device)
        target = batch["target"].to(self.device)
        high_lens = batch["high_lens"].to(self.device)
        context_lens = batch["context_lens"].to(self.device)
        target_lens = batch["target_lens"].to(self.device)

        if self.args.imitation_loss:
            actions = batch["actions"].to(self.device)

        if self.args.generalizing_loss:
            merged = batch["merged"].to(self.device)
            merged_img = batch["merged_img"].to(self.device)
            merged_lens = batch["merged_lens"].to(self.device)
            merged_img_lens = batch["merged_img_lens"].to(self.device)
            output = self.forward(high, context, target, merged, merged_img, high_lens, context_lens, target_lens, merged_lens, merged_img_lens)
        else:
            output = self.forward(high, context, target, None, None, high_lens, context_lens, target_lens, None, None)

        contrast_loss = 0 # Pixl2r loss
        for b in range(batch_size):
            if self.args.negative_contrast:
                correctness_mask = torch.ones((batch_size,)).to(self.device) * -1
            else:
                correctness_mask = torch.zeros((batch_size,)).to(self.device)
            correctness_mask[b] = 1
            progress = context_lens.float() / (context_lens.float() + target_lens.float())
            # c_loss = F.mse_loss(output["H * C"][b], output["norm(H)"][b]**2 * progress * correctness_mask, reduction='none')
            c_loss = F.mse_loss(output["H * C"][b], progress * correctness_mask, reduction='none')
            weight_mask = torch.ones((batch_size,)).to(self.device)
            weight_mask[b] = batch_size - 1
            contrast_loss += torch.dot(c_loss, weight_mask)

        sum_loss = F.mse_loss(output["<H, C + T>"], output["norm(H)"]**2)
        equal_loss = F.mse_loss(output["norm(H)"]**2, output["<H, N>"])

        total_loss = sum_loss + equal_loss + contrast_loss

        if self.args.generalizing_loss:
            generalizing_loss = F.mse_loss(output["H1 + H2"], output["H12"]) + F.mse_loss(output["T1 + T2"], output["T12"])
            total_loss += generalizing_loss
            loss["generalizing"] += generalizing_loss
            if pseudo_loss:
                pseudo_loss["generalizing"] += generalizing_loss
        if self.args.imitation_loss:
            imitation_loss = F.cross_entropy(output["At+1"], actions)
            total_loss += imitation_loss
            loss["imitation"] += imitation_loss
            if pseudo_loss:
                pseudo_loss["imitation"] += imitation_loss

        # hnorm_loss = sum([output["norm(H)"][i] if output["norm(H)"][i].item() > torch.tensor(1.) else 0 for i in range(batch_size)]) * self.args.lbda
        # cnorm_loss = sum([output["norm(C)"][i] if output["norm(C)"][i].item() > torch.tensor(1.) else 0 for i in range(batch_size)]) * self.args.lbda
        # tnorm_loss = sum([output["norm(T)"][i] if output["norm(T)"][i].item() > torch.tensor(1.) else 0 for i in range(batch_size)]) * self.args.lbda
        # cosine_loss = -output["cos(H, N)"].sum()

        loss["total"] += total_loss
        loss["contrast"] += contrast_loss
        loss["sum"] += sum_loss
        loss["equal"] += equal_loss

        # loss["hnorm"] += hnorm_loss
        # loss["cnorm"] += cnorm_loss
        # loss["tnorm"] += tnorm_loss
        # loss["cosine"] += cosine_loss

        if pseudo_loss:
            pseudo_loss["total"] += total_loss
            pseudo_loss["contrast"] += contrast_loss
            pseudo_loss["sum"] += sum_loss
            pseudo_loss["equal"] += equal_loss

            # pseudo_loss["hnorm"] += hnorm_loss
            # pseudo_loss["cnorm"] += cnorm_loss
            # pseudo_loss["tnorm"] += tnorm_loss
            # pseudo_loss["cosine"] += cosine_loss

        return total_loss

    def run_train(self, splits, optimizer, args=None):
        '''
        '''

        ### SETUP ###
        args = args or self.args
        self.writer = SummaryWriter('runs_2/cpv_babyai_nov_12_traj_generalize_one')
        fsave = os.path.join(args.dout, 'best.pth')
        psave = os.path.join(args.dout, 'pseudo_best.pth')

        with open(splits['train'], 'r') as file:
            train_data = json.load(file)
        with open(splits['valid'], 'r') as file:
            valid_data = json.load(file)

        valid_dataset = CPVDataset(self.args, self.vocab, valid_data)
        train_dataset = CPVDataset(self.args, self.vocab, train_data)
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
                "total": torch.tensor(0, dtype=torch.float),
                "contrast": torch.tensor(0, dtype=torch.float),
                "sum": torch.tensor(0, dtype=torch.float),
                "equal": torch.tensor(0, dtype=torch.float),
                "hnorm": torch.tensor(0, dtype=torch.float),
                "cnorm": torch.tensor(0, dtype=torch.float),
                "tnorm": torch.tensor(0, dtype=torch.float),
                "cosine": torch.tensor(0, dtype=torch.float),
                "generalizing": torch.tensor(0, dtype=torch.float),
                "imitation": torch.tensor(0, dtype=torch.float)
            }
            size = torch.tensor(0, dtype=torch.float)

            if self.pseudo:
                pseudo_loss = {
                    "total": torch.tensor(0, dtype=torch.float),
                    "contrast": torch.tensor(0, dtype=torch.float),
                    "sum": torch.tensor(0, dtype=torch.float),
                    "equal": torch.tensor(0, dtype=torch.float),
                    "hnorm": torch.tensor(0, dtype=torch.float),
                    "cnorm": torch.tensor(0, dtype=torch.float),
                    "tnorm": torch.tensor(0, dtype=torch.float),
                    "cosine": torch.tensor(0, dtype=torch.float),
                    "generalizing": torch.tensor(0, dtype=torch.float),
                    "imitation": torch.tensor(0, dtype=torch.float)
                }
                batch_idx = 0
                pseudo_size = torch.tensor(0, dtype=torch.float)
            else:
                pseudo_loss = None

            self.train()

            for batch in tqdm.tqdm(train_loader, desc=desc_train):
                optimizer.zero_grad()

                # Process batch for losses and update dicts.
                size += batch['high'].shape[0]

                if self.pseudo:
                    pseudo_size += batch['high'].shape[0]

                total_loss = self.compute_losses(batch, loss, pseudo_loss)

                total_loss.backward()
                optimizer.step()
                torch.cuda.empty_cache()

                if self.pseudo and batch_idx == pseudo_epoch_batch_size:
                    self.write(pseudo_loss, pseudo_size, pseudo_epoch, pseudo=True)
                    self.run_valid(valid_loader, pseudo_epoch, pseudo=True)

                    torch.save({
                        'model': self.state_dict(),
                        'optim': optimizer.state_dict(),
                        'args': self.args,
                        'vocab': self.vocab
                    }, psave)

                    pseudo_epoch += 1
                    batch_idx = -1
                    pseudo_size = torch.tensor(0, dtype=torch.float)
                    pseudo_loss["total"] = torch.tensor(0, dtype=torch.float)
                    pseudo_loss["contrast"] = torch.tensor(0, dtype=torch.float)
                    pseudo_loss["sum"] = torch.tensor(0, dtype=torch.float)
                    pseudo_loss["equal"] = torch.tensor(0, dtype=torch.float)
                    pseudo_loss["hnorm"] = torch.tensor(0, dtype=torch.float)
                    pseudo_loss["cnorm"] = torch.tensor(0, dtype=torch.float)
                    pseudo_loss["tnorm"] = torch.tensor(0, dtype=torch.float)
                    pseudo_loss["cosine"] = torch.tensor(0, dtype=torch.float)
                    pseudo_loss["generalizing"] = torch.tensor(0, dtype=torch.float)
                    pseudo_loss["imitation"] = torch.tensor(0, dtype=torch.float)

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
            "total": torch.tensor(0, dtype=torch.float),
            "contrast": torch.tensor(0, dtype=torch.float),
            "sum": torch.tensor(0, dtype=torch.float),
            "equal": torch.tensor(0, dtype=torch.float),
            "hnorm": torch.tensor(0, dtype=torch.float),
            "cnorm": torch.tensor(0, dtype=torch.float),
            "tnorm": torch.tensor(0, dtype=torch.float),
            "cosine": torch.tensor(0, dtype=torch.float),
            "generalizing": torch.tensor(0, dtype=torch.float)
        }

        size = torch.tensor(0, dtype=torch.float)

        loader = valid_loader

        if not pseudo:
            loader = tqdm.tqdm(loader, desc = desc_valid)

        with torch.no_grad():
            for batch in loader:

                # Update loss dict.
                size += batch['high'].shape[0]
                _ = self.compute_losses(batch, loss)

        self.write(loss, size, epoch, train=False, pseudo=pseudo)
        return loss


    def write(self, loss, size, epoch, train=True, pseudo=False):
        if train:
            type = "train"
        else:
            type = "valid"
        if pseudo:
            self.writer.add_scalar('PseudoLoss/' + type, (loss["total"]/size).item(), epoch)
            self.writer.add_scalar('PseudoContrastLoss/' + type, (loss["contrast"]/size).item(), epoch)
            self.writer.add_scalar('PseudoSumLoss/' + type, (loss["sum"]/size).item(), epoch)
            self.writer.add_scalar('PseudoEqualLoss/' + type, (loss["equal"]/size).item(), epoch)
            self.writer.add_scalar('PseudoHNormLoss/' + type, (loss["hnorm"]/size).item(), epoch)
            self.writer.add_scalar('PseudoCNormLoss/' + type, (loss["cnorm"]/size).item(), epoch)
            self.writer.add_scalar('PseudoTNormLoss/' + type, (loss["tnorm"]/size).item(), epoch)
            self.writer.add_scalar('PseudoCosineLoss/' + type, (loss["cosine"]/size).item(), epoch)
            self.writer.add_scalar('PseudoGeneralizingLoss/' + type, (loss["generalizing"]/size).item(), epoch)
            self.writer.add_scalar('PseudoImitationLoss/' + type, (loss["imitation"]/size).item(), epoch)
        else:
            self.writer.add_scalar('Loss/' + type, (loss["total"]/size).item(), epoch)
            self.writer.add_scalar('ContrastLoss/' + type, (loss["contrast"]/size).item(), epoch)
            self.writer.add_scalar('SumLoss/' + type, (loss["sum"]/size).item(), epoch)
            self.writer.add_scalar('EqualLoss/' + type, (loss["equal"]/size).item(), epoch)
            self.writer.add_scalar('HNormLoss/' + type, (loss["hnorm"]/size).item(), epoch)
            self.writer.add_scalar('CNormLoss/' + type, (loss["cnorm"]/size).item(), epoch)
            self.writer.add_scalar('TNormLoss/' + type, (loss["tnorm"]/size).item(), epoch)
            self.writer.add_scalar('CosineLoss/' + type, (loss["cosine"]/size).item(), epoch)
            self.writer.add_scalar('GeneralizingLoss/' + type, (loss["generalizing"]/size).item(), epoch)
            self.writer.add_scalar('ImitationLoss/' + type, (loss["imitation"]/size).item(), epoch)
