import torch
import revtok
import numpy as np
from torch import nn
from torch.nn import functional as F
from vocab import Vocab

class Baseline(nn.Module):
    def __init__(self, primed_model):
        super().__init__()

        self.pad = 0
        self.seg = 1

        self.device = torch.device('cuda')
        self.args = primed_model['args']
        self.vocab = primed_model['vocab']

        self.img_shape = 7 * 7 * 20

        self.embed = nn.Embedding(len(self.vocab), self.args.demb)
        self.linear = nn.Linear(self.img_shape, self.args.demb)
        self.lang_enc = nn.LSTM(self.args.demb, self.args.dhid, num_layers=2, batch_first=True)
        self.img_enc = nn.LSTM(self.args.demb, self.args.dhid, num_layers=2, batch_first=True)
        self.lin_1 = nn.Linear(self.args.dhid * 4, self.args.dhid)
        self.lin_2 = nn.Linear(self.args.dhid, 1)

        self.to(self.device)

        self.load_state_dict(primed_model['model'], strict=False)

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

    def forward(self, high, context, target):
        '''
        Takes in language mission (string), past observations (list of imgs), and
        the next action observation (img) and returns the dot product of each
        enc(high) - enc(context) with each enc(target)
        '''
        ### High ###
        high = self.embed(high) # -> B x M x D
        high, _, _ = self.language_encoder(high, 1) # -> 1 x H
        high = high.squeeze()

        ### Context ###
        context = self.linear(context)
        context, _, _ = self.image_encoder(context.reshape(1, -1, self.args.demb), 1) # -> 1 x H
        context = context.squeeze()

        ### Combination ###
        combination = torch.cat((high, context), dim=0)# -> 2H
        combination = self.lin_1(combination) # -> 10H
        combination = F.relu(combination) # -> 10H
        combination = self.lin_2(combination).squeeze() # -> 1
        combination = F.tanh(combination) # -> 1


        return combination

    def remove_spaces(self, s):
        cs = ' '.join(s.split())
        return cs

    def remove_spaces_and_lower(self, s):
        cs = self.remove_spaces(s)
        cs = cs.lower()
        return cs

    def calculate_reward(self, high, contexts, target):

        high = revtok.tokenize(self.remove_spaces_and_lower(high)) # -> M
        high = self.vocab.word2index([w.strip().lower() if w.strip().lower() in self.vocab.to_dict() else '<<goal>>' for w in high]) # -> M
        high = torch.tensor(high, dtype=torch.long) # -> M
        high = high.reshape(1, -1).to(self.device) # -> 1 x M

        final_shape = 7 * 7 * (11 + 6 + 3)
        object_default = np.array([np.eye(11) for _ in range(49)])
        color_default = np.array([np.eye(6) for _ in range(49)])
        state_default = np.array([np.eye(3) for _ in range(49)])

        contexts = [np.reshape(img, (49, -1)) for img in contexts]
        contexts_object = [torch.tensor(object_default[list(range(49)), img[:, 0], :], dtype=torch.float) for img in contexts]
        contexts_color = [torch.tensor(color_default[list(range(49)), img[:, 1], :], dtype=torch.float) for img in contexts]
        contexts_state = [torch.tensor(state_default[list(range(49)), img[:, 2], :], dtype=torch.float) for img in contexts]
        contexts = [torch.cat([contexts_object[i], contexts_color[i], contexts_state[i]], dim=1).reshape(final_shape) for i in range(len(contexts))]

        if len(contexts) == 0:
            contexts = torch.tensor([[self.pad for x in range(final_shape)]], dtype=torch.float).to(self.device)
        else:
            contexts = torch.stack(contexts, dim=0).to(self.device)

        target = np.reshape(target, (49, -1))
        target_object = torch.tensor(object_default[list(range(49)), target[:, 0], :], dtype=torch.float)
        target_color = torch.tensor(color_default[list(range(49)), target[:, 1], :], dtype=torch.float)
        target_state = torch.tensor(state_default[list(range(49)), target[:, 2], :], dtype=torch.float)
        target = torch.cat([target_object, target_color, target_state], dim=1).reshape(final_shape).to(self.device)

        self.eval()
        reward = self.forward(high, contexts, target).to(torch.device('cpu')).detach().numpy()

        return reward, 0, 0
