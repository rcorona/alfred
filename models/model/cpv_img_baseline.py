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
        high_level = torch.stack([keep[j][0] for j in range(len(keep))], dim=0).type(torch.float) # N x 512 x 7 x 7
        low_level_context = [torch.stack([keep[j][0] for j in range(len(keep)) if keep[j][1] == i], dim=0).type(torch.float) for i in range(len(ex['ann']['instr']))] # N x 512 x 7 x 7

        high_level = high_level
        low_level_context = [ll for ll in low_level_context]
        target_idx = random.randrange(len(low_level_context))
        low_level_target = low_level_context[target_idx]
        del low_level_context[target_idx]

        # Categorize the correct target for error analysis
        category = self.goals.word2index(ex['plan']['high_pddl'][target_idx]['planner_action']['action'])

        padded_context = torch.cat([high_level] + [torch.tensor(self.seg, dtype = torch.float).repeat(1, 512, 7, 7)] + low_level_context, dim=0)
        return (padded_context, low_level_target, category)

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

            contexts = []
            targets = []
            labels = []
            categories = []

            contexts_lens = []
            targets_lens = []

            for idx in range(batch_size):
                contexts.append(batch[idx][0])
                contexts_lens.append(batch[idx][0].shape[0])
                targets.append(batch[idx][1])
                targets_lens.append(batch[idx][1].shape[0])
                labels.append(torch.tensor(idx).unsqueeze(0))
                categories.append(batch[idx][2])

            padded_contexts = pad_sequence(contexts, batch_first=True)
            padded_targets = pad_sequence(targets,  batch_first=True)
            padded_labels = torch.cat(labels, dim=0)

            contexts_lens = torch.tensor(contexts_lens, dtype=torch.long) # -> B
            targets_lens = torch.tensor(targets_lens, dtype=torch.long) # -> B

            return (padded_contexts, padded_targets, padded_labels, contexts_lens, targets_lens, categories)

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
        self.vis_encoder = vnn.ResnetVisualEncoder(dframe=self.args.dframe)
        self.enc = nn.LSTM(self.args.dframe, args.dhid, bidirectional=True, batch_first=True)
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

    def forward(self, contexts, targets, contexts_lens, targets_lens):
        '''
        Takes in contexts and targets and returns the dot product of each enc(context) with each enc(target)
        '''
        ### INPUT ###
        # Contexts ->  B x N x 512 x 7 x 7 (N = Contexts Seq Len)
        # Targets -> B x T x 512 x 7 x 7 (T =  Targets Seq Len)
        # Contexts_Lens -> B
        # Targets_Lens -> B

        batch_size = contexts.shape[0]
        contexts_seq_len = contexts.shape[1]
        targets_seq_len = targets.shape[1]
        resnet_size = (contexts.shape[2], contexts.shape[3], contexts.shape[4])

        ### CONTEXTS ###
        # Reshaping:
        flat_contexts = contexts.reshape(batch_size * contexts_seq_len, resnet_size[0], resnet_size[1], resnet_size[2]) # -> B * N x 512 x 7 x 7
        # Dimension Reduction:
        emb_contexts = self.vis_encoder(flat_contexts) # -> B * N x E
        # Reshaping:
        tall_contexts = emb_contexts.reshape(batch_size, contexts_seq_len, self.args.dframe) # -> B x N x E
        # Packing:
        packed_contexts = pack_padded_sequence(tall_contexts, contexts_lens, batch_first=True, enforce_sorted=False)
        # Encoding:
        enc_contexts = self.encoder(packed_contexts, batch_size) # -> B x H (H = hidden dim of LSTM)

        ### TARGETS ###
        # Reshaping
        flat_targets = targets.reshape(batch_size * targets_seq_len,  resnet_size[0], resnet_size[1], resnet_size[2]) # -> B * T x 512 x 7 x 7
        # Dimension Reduction:
        emb_targets = self.vis_encoder(flat_targets) # -> B * T x E
        # Reshaping:
        tall_targets = emb_targets.reshape(batch_size, targets_seq_len, self.args.dframe) # -> B x T x E
        # Packing:
        packed_targets = pack_padded_sequence(tall_targets, targets_lens, batch_first=True, enforce_sorted=False)
        #Encoding:
        enc_targets = self.encoder(packed_targets, batch_size) # -> B x H

        ### COMB ###
        # Dot Product:
        sim_m = torch.matmul(enc_contexts, torch.transpose(enc_targets, 0, 1)) # -> B x B
        logits = F.log_softmax(sim_m, dim = 1)

        return logits

    def run_train(self, splits, optimizer, args=None):

        args = args or self.args
        self.writer = SummaryWriter('runs/img_baseline')
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

        # Training loop
        best_loss = 1e10
        for epoch in range(args.epoch):
            print('Epoch', epoch)
            self.train()
            total_train_loss = torch.tensor(0, dtype=torch.float)
            total_train_acc = torch.tensor(0, dtype=torch.float)
            total_train_size = torch.tensor(0, dtype=torch.float)
            print(len(train_loader))
            for batch in train_loader:
                optimizer.zero_grad()
                contexts, targets, labels, contexts_lens, targets_lens, categories = batch

                # Transfer to GPU
                contexts = contexts.to(self.device)
                targets = targets.to(self.device)
                labels = labels.to(self.device)
                contexts_lens = contexts_lens.to(self.device)
                targets_lens = targets_lens.to(self.device)

                # Forward
                logits = self.forward(contexts, targets, contexts_lens, targets_lens)

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
                    contexts, targets, labels, contexts_lens, targets_lens, categories = batch

                    # Transfer to GPU
                    contexts = contexts.to(self.device)
                    targets = targets.to(self.device)
                    labels = labels.to(self.device)
                    contexts_lens = contexts_lens.to(self.device)
                    targets_lens = targets_lens.to(self.device)

                    # Forward
                    logits = self.forward(contexts, targets, contexts_lens, targets_lens)

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


            # self.run_eval(eval_dataset, epoch)
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

            self.writer.flush()

        self.writer.close()


    # def run_eval(self, eval_dataset, epoch):
    #     # Initalize Dataloaders
    #     eval_loader = DataLoader(eval_dataset, batch_size=self.args.batch, shuffle=True, num_workers=8, collate_fn=eval_dataset.collate())
    #
    #     # Will contain pairs of (predicted category, actual category) for analysis
    #     predicted = []
    #     expected = []
    #
    #     self.eval()
    #     total_eval_loss = torch.tensor(0, dtype=torch.float)
    #     total_eval_acc = torch.tensor(0, dtype=torch.float)
    #     total_eval_size = torch.tensor(0, dtype=torch.float)
    #
    #     with torch.no_grad():
    #         for batch in eval_loader:
    #             contexts, targets, labels, contexts_lens, targets_lens, categories = batch
    #
    #             # Transfer to GPU
    #             contexts = contexts.to(self.device)
    #             targets = targets.to(self.device)
    #             labels = labels.to(self.device)
    #             contexts_lens = contexts_lens.to(self.device)
    #             targets_lens = targets_lens.to(self.device)
    #
    #             # Forward
    #             logits = self.forward(contexts, targets, contexts_lens, targets_lens)
    #
    #             # Calculate Loss and Accuracy
    #             loss = F.nll_loss(logits, labels)
    #             total_eval_loss += loss
    #             total_eval_size += labels.shape[0]
    #             most_likely = torch.argmax(logits, dim=1)
    #             acc = torch.eq(most_likely, labels)
    #             total_eval_acc += torch.sum(acc)
    #
    #             # Enter results
    #             predicted.extend([categories[torch.argmax(logits, dim=1)[i]] for i in range(logits.shape[0])])
    #             expected.extend(categories)
    #
    #
    #     # Write to TensorBoardX
    #     self.writer.add_scalar('Accuracy/evaluation', (total_eval_acc/total_eval_size).item(), epoch)
    #     self.writer.add_scalar('Loss/evaluation', total_eval_loss.item(), epoch)
    #     print("Evaluation Accuracy: " + str((total_eval_acc/total_eval_size).item()))
    #     print("Evaluation Loss: " + str(total_eval_loss.item()))
    #
    #     cm = sklearn.metrics.confusion_matrix(expected, predicted)
    #     figure = plot_confusion_matrix(cm, class_names=self.classes)
    #
    #     self.writer.add_figure("Confusion Matrix", figure, global_step=epoch)
