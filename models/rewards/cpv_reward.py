import torch
import revtok
import numpy as np
from torch import nn
from vocab import Vocab

class CPV(nn.Module):
    def __init__(self, primed_model):
        super().__init__()

        self.pad = 0
        self.seg = 1

        self.device = torch.device('cuda')
        self.args = primed_model['args']
        self.vocab = primed_model['vocab']
        self.encoder_type ='lstm'

        self.img_shape = 7 * 7 * 20

        self.embed = nn.Embedding(len(self.vocab), self.args.demb)
        self.linear = nn.Linear(self.img_shape, self.args.demb)

        # Use either RNNs or Transformers.
        if self.encoder_type == 'transformer':
            self.lang_enc_layer = nn.TransformerEncoderLayer(d_model=self.args.demb, nhead=5)
            self.lang_enc = nn.TransformerEncoder(self.lang_enc_layer, num_layers=6)
            self.lang_out_linear = nn.Sequential(nn.Linear(self.args.demb, self.args.dhid), nn.ReLU())

            self.img_enc_layer = nn.TransformerEncoderLayer(d_model=self.args.demb, nhead=5)
            self.img_enc = nn.TransformerEncoder(self.img_enc_layer, num_layers=6)
            self.img_out_linear = nn.Sequential(nn.Linear(self.args.demb, self.args.dhid), nn.ReLU())

        elif self.encoder_type == 'lstm':
            self.lang_enc = nn.LSTM(self.args.demb, self.args.dhid, bidirectional=False, num_layers=2, batch_first=True)
            self.img_enc = nn.LSTM(self.args.demb, self.args.dhid, bidirectional=False, num_layers=2, batch_first=True)

        self.to(self.device)

        # self.load_state_dict(primed_model['model'], strict=False)


    def language_encoder(self, batch, batch_size, h_0=None, c_0=None):
        '''
        Encodes a stacked tensor.
        '''

        if h_0 is None or c_0 is None:
            h_0 = torch.zeros(2, batch_size, self.args.dhid).type(torch.float).to(self.device) # -> 2 x B x H
            c_0 = torch.zeros(2, batch_size, self.args.dhid).type(torch.float).to(self.device) # -> 2 x B x H
        out, (h, c) = self.lang_enc(batch, (h_0, c_0)) # -> 2 x B x H

        hid_sum = torch.sum(h, dim=0) # -> B x H

        return hid_sum, h, c

    def image_encoder(self, batch, batch_size, h_0=None, c_0=None):
        '''
        Encodes a stacked tensor.
        '''

        if h_0 is None or c_0 is None:
            h_0 = torch.zeros(2, batch_size, self.args.dhid).type(torch.float).to(self.device) # -> 2 x B x H
            c_0 = torch.zeros(2, batch_size, self.args.dhid).type(torch.float).to(self.device) # -> 2 x B x H
        out, (h, c) = self.img_enc(batch, (h_0, c_0)) # -> 2 x B x H

        hid_sum = torch.sum(h, dim=0) # -> B x H

        return hid_sum, h, c

    def forward(self, high, context, target):
        '''
        Takes in language mission (string), past observations (list of imgs), and
        the next action observation (img) and returns the dot product of each
        enc(high) - enc(context) with each enc(target)
        '''

        B = 1

        # Full trajectory. TODO move into transformer loop since this isn't needed by LSTM.
        # traj = torch.cat([context, target], dim=1)

        # Embed all inputs.
        high = self.embed(high) # -> B x M x D
        context = self.linear(context)
        target = self.linear(target)
        # traj = self.linear(traj)

        if self.encoder_type == 'transformer':

            high = self.lang_enc(high)
            high = high[:,0,:] + high[:,-1,:]
            high = self.lang_out_linear(high)

            context = self.img_enc(context)
            context = context[:,0,:] + context[:,-1,:]
            context = self.img_out_linear(context)

            target = self.img_enc(target)
            target = target[:,0,:] + target[:,-1,:]
            target = self.img_out_linear(target)

            traj = self.img_enc(traj)
            traj = traj[:,0,:] + traj[:,-1,:]
            trajectory = self.img_out_linear(traj)

        else:
            ### High ###

            # high = pack_padded_sequence(high, high_lens, batch_first=True, enforce_sorted=False)
            high, _, _ = self.language_encoder(high, B) # -> B x H
            high = high.squeeze()

            ### Context ###c]
            context = context.reshape(1, -1, self.args.demb)
            # context = pack_padded_sequence(context, context_lens, batch_first=True, enforce_sorted=False)
            context, h, c = self.image_encoder(context, B)
            context = context.squeeze()

            ### Target ###
            packed_target = target.reshape(1, -1, self.args.demb)
            # packed_target = pack_padded_sequence(target, target_lens, batch_first=True, enforce_sorted=False)
            target, _, _ = self.image_encoder(packed_target, B)
            target = target.squeeze()

            ### Full Trajectory ###
            trajectory, _, _ = self.image_encoder(packed_target, B, h, c)
            trajectory = trajectory.squeeze()


        ### COMB ###
        reward = torch.dot(high/high.norm(), trajectory/high.norm()) - torch.dot(high/high.norm(), context/high.norm())

        done = torch.dot(high, context)

        return reward, done, context.norm()

    def remove_spaces(self, s):
        cs = ' '.join(s.split())
        return cs

    def remove_spaces_and_lower(self, s):
        cs = self.remove_spaces(s)
        cs = cs.lower()
        return cs

    def calculate_reward(self, high, contexts, target):

        # high = torch.tensor(high).unsqueeze(0).to(self.device)
        #
        # final_shape = 7 * 7 * (11 + 6 + 3)
        # object_default = np.array([np.eye(11) for _ in range(49)])
        # color_default = np.array([np.eye(6) for _ in range(49)])
        # state_default = np.array([np.eye(3) for _ in range(49)])
        #
        # contexts = [np.reshape(img, (49, -1)) for img in contexts]
        # contexts_object = [torch.tensor(object_default[list(range(49)), img[:, 0], :], dtype=torch.float) for img in contexts]
        # contexts_color = [torch.tensor(color_default[list(range(49)), img[:, 1], :], dtype=torch.float) for img in contexts]
        # contexts_state = [torch.tensor(state_default[list(range(49)), img[:, 2], :], dtype=torch.float) for img in contexts]
        # contexts = [torch.cat([contexts_object[i], contexts_color[i], contexts_state[i]], dim=1).reshape(final_shape) for i in range(len(contexts))]
        #
        # if len(contexts) == 0:
        #     contexts = torch.tensor([[self.pad for x in range(final_shape)]], dtype=torch.float).to(self.device)
        # else:
        #     contexts = torch.stack(contexts, dim=0).to(self.device)
        #
        # target = np.reshape(target, (49, -1))
        # target_object = torch.tensor(object_default[list(range(49)), target[:, 0], :], dtype=torch.float)
        # target_color = torch.tensor(color_default[list(range(49)), target[:, 1], :], dtype=torch.float)
        # target_state = torch.tensor(state_default[list(range(49)), target[:, 2], :], dtype=torch.float)
        # target = torch.cat([target_object, target_color, target_state], dim=1).reshape(final_shape).to(self.device)
        #
        # self.eval()
        # reward, done, cnorm = self.forward(high, contexts, target)
        # reward = reward.to(torch.device('cpu')).detach().item()
        # done = done.to(torch.device('cpu')).detach().item()
        # cnorm = cnorm.to(torch.device('cpu')).detach().item()
        # return reward, done, cnorm

        high = revtok.tokenize(self.remove_spaces_and_lower(high))
        high = self.vocab.word2index([w.strip().lower() for w in high])

        return high

    def works(self, high):
        high = revtok.tokenize(self.remove_spaces_and_lower(high)) # -> M
        for w in high:
            if w.strip().lower() not in self.vocab.to_dict():
                return False;
        return True;
