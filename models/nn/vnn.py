import torch
from torch import nn
from torch.nn import functional as F
import pdb

BIG_NEG = -1e9

class SelfAttn(nn.Module):
    '''
    self-attention with learnable parameters
    '''

    def __init__(self, dhid):
        super().__init__()
        self.scorer = nn.Linear(dhid, 1)

    def forward(self, inp):
        scores = F.softmax(self.scorer(inp), dim=1)
        cont = scores.transpose(1, 2).bmm(inp).squeeze(1)
        return cont

class MLPAdapter(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.l1 = nn.Linear(dim, dim)
        self.l2 = nn.Linear(dim, dim)

    def forward(self, inp):
        return self.l2(torch.relu(self.l1(inp)))

class DotAttn(nn.Module):
    '''
    dot-attention (or soft-attention)
    '''
    def __init__(self): 
        super(DotAttn, self).__init__()
        self.raw_score = None

    def forward(self, inp, h, mask=None):
        # mask: bool tensor, with False for locations that should *not* be attended to
        score = self.softmax(inp, h, mask)
        return score.expand_as(inp).mul(inp).sum(1), score

    def softmax(self, inp, h, mask):
        raw_score = inp.bmm(h.unsqueeze(2))
        if mask is not None:
            mask = mask.unsqueeze(-1)
            assert mask.size() == raw_score.size(), (mask.size(), raw_score.size())
            # ~ not implemented on bool tensors in 1.1
            raw_score.masked_fill_(~(mask.byte()), BIG_NEG)
        self.raw_score = raw_score
        score = F.softmax(raw_score, dim=1)
        return score


class ResnetVisualEncoder(nn.Module):
    '''
    visual encoder
    '''

    def __init__(self, dframe):
        super(ResnetVisualEncoder, self).__init__()
        self.dframe = dframe
        self.flattened_size = 64*7*7

        self.conv1 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)
        self.fc = nn.Linear(self.flattened_size, self.dframe)
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(64)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))

        x = self.conv2(x)
        x = F.relu(self.bn2(x))

        x = x.view(-1, self.flattened_size)
        x = self.fc(x)

        return x


class MaskDecoder(nn.Module):
    '''
    mask decoder
    '''

    def __init__(self, dhid, pframe=300, hshape=(64,7,7)):
        super(MaskDecoder, self).__init__()
        self.dhid = dhid
        self.hshape = hshape
        self.pframe = pframe

        self.d1 = nn.Linear(self.dhid, hshape[0]*hshape[1]*hshape[2])
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn1 = nn.BatchNorm2d(16)
        self.dconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.dconv2 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.dconv1 = nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.d1(x))
        x = x.view(-1, *self.hshape)

        x = self.upsample(x)
        x = self.dconv3(x)
        x = F.relu(self.bn2(x))

        x = self.upsample(x)
        x = self.dconv2(x)
        x = F.relu(self.bn1(x))

        x = self.dconv1(x)
        x = F.interpolate(x, size=(self.pframe, self.pframe), mode='bilinear')

        return x

class ConvFrameMaskDecoder(nn.Module):
    '''
    action decoder
    '''

    def __init__(self, emb, dframe, dhid, pframe=300,
                 attn_dropout=0., hstate_dropout=0., actor_dropout=0., input_dropout=0.,
                 teacher_forcing=False):
        super().__init__()
        demb = emb.weight.size(1)

        self.emb = emb
        self.pframe = pframe
        self.dhid = dhid
        self.vis_encoder = ResnetVisualEncoder(dframe=dframe)
        self.cell = nn.LSTMCell(dhid+dframe+demb, dhid)
        self.attn = DotAttn()
        self.input_dropout = nn.Dropout(input_dropout)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.hstate_dropout = nn.Dropout(hstate_dropout)
        self.actor_dropout = nn.Dropout(actor_dropout)
        self.go = nn.Parameter(torch.Tensor(demb))
        self.actor = nn.Linear(dhid+dhid+dframe+demb, demb)
        self.mask_dec = MaskDecoder(dhid=dhid+dhid+dframe+demb, pframe=self.pframe)
        self.teacher_forcing = teacher_forcing
        self.h_tm1_fc = nn.Linear(dhid, dhid)

        nn.init.uniform_(self.go, -0.1, 0.1)

    def step(self, enc, frame, e_t, state_tm1, encoder_mask=None, hstate_dropout_mask=None):
        # previous decoder hidden state
        h_tm1 = state_tm1[0]

        # encode vision and lang feat
        vis_feat_t = self.vis_encoder(frame)
        lang_feat_t = enc # language is encoded once at the start

        # attend over language
        weighted_lang_t, lang_attn_t = self.attn(
            self.attn_dropout(lang_feat_t), self.h_tm1_fc(h_tm1), mask=encoder_mask,
        )

        # concat visual feats, weight lang, and previous action embedding
        inp_t = torch.cat([vis_feat_t, weighted_lang_t, e_t], dim=1)
        inp_t = self.input_dropout(inp_t)

        # update hidden state
        state_t = self.cell(inp_t, state_tm1)
        state_t = [self.hstate_dropout(x) for x in state_t]
        if hstate_dropout_mask is not None:
            state_t[0] = hstate_dropout_mask * state_t[0]
        h_t = state_t[0]

        # decode action and mask
        cont_t = torch.cat([h_t, inp_t], dim=1)
        action_emb_t = self.actor(self.actor_dropout(cont_t))
        action_t = action_emb_t.mm(self.emb.weight.t())
        mask_t = self.mask_dec(cont_t)

        return action_t, mask_t, state_t, lang_attn_t

    def forward(self, enc, frames, gold=None, max_decode=150, state_0=None, encoder_mask=None, hstate_dropout_mask=None):
        max_t = gold.size(1) if self.training else min(max_decode, frames.shape[1])
        batch = enc.size(0)
        e_t = self.go.repeat(batch, 1)
        state_t = state_0

        actions = []
        masks = []
        attn_scores = []
        for t in range(max_t):
            try: 
                action_t, mask_t, state_t, attn_score_t = self.step(
                    enc, frames[:, t], e_t, state_t, encoder_mask=encoder_mask,
                    hstate_dropout_mask=hstate_dropout_mask
                )
            except: 
                #pdb.set_trace()
                action_t, mask_t, state_t, attn_score_t = self.step(
                    enc, frames[:, -1], e_t, state_t, encoder_mask=encoder_mask,
                    hstate_dropout_mask=hstate_dropout_mask
                )

            masks.append(mask_t)
            actions.append(action_t)
            attn_scores.append(attn_score_t)
            if self.teacher_forcing and self.training:
                w_t = gold[:, t]
            else:
                w_t = action_t.max(1)[1]
            e_t = self.emb(w_t)

        results = {
            'out_action_low': torch.stack(actions, dim=1),
            'out_action_low_mask': torch.stack(masks, dim=1),
            'out_attn_scores': torch.stack(attn_scores, dim=1),
            'state_t': state_t
        }

        return results

def clone_module(module):
    import pickle
    return pickle.loads(pickle.dumps(module))

class ConvFrameMaskDecoderModular(nn.Module):
    '''
    action decoder
    '''

    def __init__(self, emb, dframe, dhid, pframe=300,
                 attn_dropout=0., hstate_dropout=0., actor_dropout=0., input_dropout=0.,
                 teacher_forcing=False, n_modules=8, controller_type='attention',
                 cloned_module_initialization=False, 
                 init_model_path=None,
                 modularize_actor_mask=False,
                 h_translation=None, c_translation=None
                 ):
        super().__init__()
        demb = emb.weight.size(1)
        self.controller_type = controller_type
        self.cloned_module_initialization = cloned_module_initialization
        self.modularize_actor_mask=modularize_actor_mask

        self.n_modules = n_modules

        self.emb = emb
        self.pframe = pframe
        self.dhid = dhid
        self.vis_encoder = ResnetVisualEncoder(dframe=dframe)

#         if self.use_fc_nodes:
#             self.fc_nodes = nn.ModuleList([nn.Linear(dhid, dhid) for i in range(n_modules)])
        # High level controller.
        if self.controller_type == 'attention':
            self.controller = nn.LSTMCell(dhid+dhid+dframe+demb, dhid)
            self.controller_attn = DotAttn()
            self.controller_h_tm1_fc = nn.Linear(dhid, dhid)

            # Attention over modules.
            self.module_attn = DotAttn()
        else:
            self.controller = None
            self.controller_attn = None
            self.controller_h_tm1_fc = None
            self.stop_embedding = None
            self.module_attn = None

        self._h_translation = h_translation
        self._c_translation = c_translation

        # STOP module for high level controller.
        self.stop_embedding = torch.nn.init.uniform_(nn.Parameter(torch.zeros((self.dhid,))))

        self.input_dropout = nn.Dropout(input_dropout)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.hstate_dropout = nn.Dropout(hstate_dropout)
        self.actor_dropout = nn.Dropout(actor_dropout)
        self.go = nn.Parameter(torch.Tensor(demb))
        self.actor = nn.Linear(dhid+dhid+dframe+demb, demb)
        self.mask_dec = MaskDecoder(dhid=dhid+dhid+dframe+demb, pframe=self.pframe)
        self.teacher_forcing = teacher_forcing

        nn.init.uniform_(self.go, -0.1, 0.1)

        if self.h_translation == 'linear':
            self.h_translators_in = nn.ModuleList([nn.Linear(dhid, dhid) for _ in range(n_modules)])
            self.h_translators_out = nn.ModuleList([nn.Linear(dhid, dhid) for _ in range(n_modules)])
        elif self.h_translation == 'mlp':
            self.h_translators_in = nn.ModuleList([MLPAdapter(dhid) for _ in range(n_modules)])
            self.h_translators_out = nn.ModuleList([MLPAdapter(dhid) for _ in range(n_modules)])
        elif self.h_translation == None:
            self.h_translators_in = [lambda x: x for _ in range(n_modules)]
            self.h_translators_out = [lambda x: x for _ in range(n_modules)]
        else:
            raise NotImplementedError("--h_translation {}".format(self.h_translation))

        if self.c_translation == 'linear':
            self.c_translators_in = nn.ModuleList([nn.Linear(dhid, dhid) for _ in range(n_modules)])
            self.c_translators_out = nn.ModuleList([nn.Linear(dhid, dhid) for _ in range(n_modules)])
        elif self.h_translation == 'mlp':
            self.c_translators_in = nn.ModuleList([MLPAdapter(dhid) for _ in range(n_modules)])
            self.c_translators_out = nn.ModuleList([MLPAdapter(dhid) for _ in range(n_modules)])
        elif self.c_translation == None:
            self.c_translators_in = [lambda x: x for _ in range(n_modules)]
            self.c_translators_out = [lambda x: x for _ in range(n_modules)]
        else:
            raise NotImplementedError("--c_translation {}".format(self.c_translation))

        if cloned_module_initialization:
 
            self.cell = nn.LSTMCell(dhid+dframe+demb, dhid)
            self.attn = DotAttn()
            self.h_tm1_fc = nn.Linear(dhid, dhid)

            # Use pre-trained parameters if desired. 
            if init_model_path: 
            
                # Load model parameters first. 
                params = torch.load(init_model_path, map_location='cpu')['model']

                # Filter out only the paramters we need. 
                params = {k.replace('dec.', ''): params[k] for k in params}
                params = {k.replace('mask_', 'mask_dec.'): params[k] for k in params}
                params = {k: params[k] for k in params if not 'enc.' in k}
                params = {k: params[k] for k in params if not 'emb_' in k}
                params = {k: params[k] for k in params if not 'enc_att' in k}

                # Load parameters. 
                model_dict = self.state_dict()
                model_dict.update(params)
                self.load_state_dict(model_dict)

            # Clone parameters across modules. 
            self.cell = nn.ModuleList([clone_module(self.cell) for i in range(n_modules)])
            self.attn = nn.ModuleList([clone_module(self.attn) for i in range(n_modules)])
            self.h_tm1_fc = nn.ModuleList([clone_module(self.h_tm1_fc) for i in range(n_modules)])

            # TODO: make this an argument
            # Freeze GotoLocation module. 
            # for p in self.cell[0].parameters():
            #     p.requires_grad = False
            #
            # for p in self.attn[0].parameters():
            #     p.requires_grad = False
            #
            # for p in self.h_tm1_fc[0].parameters():
            #     p.requires_grad = False

            if modularize_actor_mask: 
                self.actor = nn.ModuleList([clone_module(self.actor) for i in range(n_modules)])
                self.mask_dec = nn.ModuleList([clone_module(self.mask_dec) for i in range(n_modules)])

            if self.h_translation == 'linear':
                for layer in self.h_translators_in:
                    layer.weight.data = torch.eye(layer.weight.size(0))
                    layer.bias.data.fill_(0)
                for layer in self.h_translators_out:
                    layer.weight.data = torch.eye(layer.weight.size(0))
                    layer.bias.data.fill_(0)

            if self.c_translation == 'linear':
                for layer in self.c_translators_in:
                    layer.weight.data = torch.eye(layer.weight.size(0))
                    layer.bias.data.fill_(0)
                for layer in self.c_translators_out:
                    layer.weight.data = torch.eye(layer.weight.size(0))
                    layer.bias.data.fill_(0)

        else:
            self.cell = nn.ModuleList([nn.LSTMCell(dhid+dframe+demb, dhid) for i in range(n_modules)])
            self.attn = nn.ModuleList([DotAttn() for i in range(n_modules)])
            self.h_tm1_fc = nn.ModuleList([nn.Linear(dhid, dhid) for i in range(n_modules)])

    # for backward compatibility with serialized models
    @property
    def h_translation(self):
        try:
            return self._h_translation
        except:
            return None

    @property
    def c_translation(self):
        try:
            return self._c_translation
        except:
            return None

    def step(self, enc, frame, e_t, state_tm1, state_transitioned_tm1, controller_state_tm1, controller_mask=None,
             transition_mask=None, next_transition_mask=None,
             hstate_dropout_mask=None,
             ):
        # transition_mask and next_transition_mask are not
        # previous decoder hidden state for modules.

        if not self.modularize_actor_mask: 
            return self.single_actor_mask_step(
                enc, frame, e_t, state_tm1, state_transitioned_tm1, controller_state_tm1,
                controller_mask, transition_mask, next_transition_mask,
                hstate_dropout_mask
            )
        else:
            return self.multiple_actor_mask_step(
                enc, frame, e_t, state_tm1, state_transitioned_tm1, controller_state_tm1,
                controller_mask, transition_mask, next_transition_mask,
                hstate_dropout_mask
            )

    def multiple_actor_mask_step(self, enc, frame, e_t, state_tm1, state_transitioned_tm1,
                                 controller_state_tm1, controller_mask=None,
                                 transition_mask=None, next_transition_mask=None,
                                 hstate_dropout_mask=None):
        # transition_mask and next_transition_mask are not
        # previous decoder hidden state for modules.
        raise NotImplementedError("implement state_transitioned_tm1 in multiple_actor_mask_step")
        h_tm1 = state_tm1[0]

        batch_sz = frame.size(0)

        # encode vision and lang feat
        vis_feat_t = self.vis_encoder(frame)
        lang_feat_t = enc # language is encoded once at the start

        lang_attn_t = []
        h_t = []
        c_t = []
        action_t = []
        mask_t = []

        # Iterate over each module. 
        for i in range  (self.n_modules): 

            # attend over language
            weighted_lang_ti, lang_attn_ti = self.attn[i](self.attn_dropout(lang_feat_t), self.h_tm1_fc[i](h_tm1))
            lang_attn_t.append(lang_attn_ti)

            # concat visual feats, weight lang, and previous action embedding
            inp_ti = torch.cat([vis_feat_t, weighted_lang_ti, e_t], dim=1)
            inp_ti = self.input_dropout(inp_ti)

            # update hidden state
            state_t = self.cell[i](inp_ti, (h_tm1, state_tm1[1]))
            h, c = [self.hstate_dropout(x) for x in state_t]
            if hstate_dropout_mask is not None:
                h = hstate_dropout_mask * h
            state_t = h, c

            # batch x dhid
            h_t.append(state_t[0])
            c_t.append(state_t[1])

            # decode action and mask
            cont_t = torch.cat([state_t[0], inp_ti], dim=-1)
            action_emb_t = self.actor[i](self.actor_dropout(cont_t))
            action_t.append(action_emb_t.mm(self.emb.weight.t()))
            mask_t.append(self.mask_dec[i](cont_t))

        # Attend over each modules output.
        h_t_in = self.attn_dropout(torch.cat([h.unsqueeze(1) for h in h_t], dim=1)) # batch x n_modules x dhid
        c_t = torch.cat([c.unsqueeze(1) for c in c_t], dim=1)
        action_t = torch.cat([a.unsqueeze(1) for a in action_t], dim=1)
        mask_t = torch.cat([m.unsqueeze(1) for m in mask_t], dim=1)
        lang_attn_t = torch.cat(lang_attn_t, dim=-1)

        # Add stop embedding. 

        h_t_in = torch.cat([h_t_in, self.stop_embedding.view(1,1,-1).expand(h_t_in.size(0), 1, self.dhid)], dim=1)

        # Attend over submodules hidden states. 
        # TODO Use ground truth attention here if provided, but keep generated attention since we need it to train attention mechanism.
        if self.controller_type == 'attention':
            _, module_scores = self.controller_attn(h_t_in, self.controller_h_tm1_fc(controller_state_tm1[0]))
            module_attn_logits = self.controller_attn.raw_score
        else:
            assert controller_mask is not None, "when not using an attention-based controller, must pass controller_mask to step()"

        # If no supervised/forced attention is provided, then used inferred values.
        if controller_mask is None:
            module_attn = module_scores

            # Only pay attention to a single module.
            max_subgoal = module_attn.max(1)[1].squeeze()

            module_attn = torch.zeros_like(module_attn)
            module_attn[:,max_subgoal,:] = 1.0

        # Otherwise use ground truth attention.
        else:
            module_attn = controller_mask.unsqueeze(-1).float()

        h_t_in = module_attn.expand_as(h_t_in).mul(h_t_in).sum(1)
        c_t = module_attn[:,:-1,:].expand_as(c_t).mul(c_t).sum(1)
        lang_attn_t = module_attn[:,:-1,:].view(batch_sz,1,8).expand_as(lang_attn_t).mul(lang_attn_t).sum(-1)
        action_t = module_attn[:,:-1,:].expand_as(action_t).mul(action_t).sum(1)
        mask_t = module_attn[:,:-1,:].unsqueeze(-1).unsqueeze(-1).expand_as(mask_t).mul(mask_t).sum(1)

        # update controller hidden state
        if self.controller_type == 'attention':
            controller_state_t = self.controller(cont_t, controller_state_tm1)
            controller_state_t = [self.hstate_dropout(x) for x in controller_state_t]
            module_attn_logits_ret = module_attn_logits.view(batch_sz,1,9)
        else:
            controller_state_t = None
            module_attn_logits_ret = None

        # Package weighted state for output. 
        state_t = (h_t_in, c_t)

        return action_t, mask_t, state_t, controller_state_t, lang_attn_t, module_attn_logits_ret, module_attn

    def translate(self, var, transforms, controller_mask):
        # TODO: this might be faster with unconditional computation and a gather
        # controller_mask: bsz x (num_modules+1)
        transformed = []
        for batch_index in range(controller_mask.size(0)):
            modules_active = controller_mask[batch_index].nonzero().flatten()
            assert modules_active.size(0) <= 1
            if modules_active.size() == 1 and modules_active.item() < len(transforms):
                transform = transforms[modules_active.item()]
                transformed.append(transform(var[batch_index]))
            else:
                transformed.append(var[batch_index])
        return torch.stack(transformed, 0)

    def transform_h_in(self, h, controller_mask):
        if self.h_translation is None:
            return h
        return self.translate(h, self.h_translators_in, controller_mask)

    def transform_h_out(self, h, controller_mask):
        if self.h_translation is None:
            return h
        return self.translate(h, self.h_translators_out, controller_mask)

    def transform_c_in(self, c, controller_mask):
        if self.c_translation is None:
            return c
        return self.translate(c, self.c_translators_in, controller_mask)

    def transform_c_out(self, c, controller_mask):
        if self.c_translation is None:
            return c
        return self.translate(c, self.c_translators_out, controller_mask)


    def single_actor_mask_step(self, enc, frame, e_t, state_tm1, state_transitioned_tm1,
                               controller_state_tm1, controller_mask=None,
                               transition_mask=None, next_transition_mask=None, hstate_dropout_mask=None):
        # transition_mask and next_transition_mask are not
        # previous decoder hidden state for modules.

        # h_tm1 = state_tm1[0]
        # c_tm1 = state_tm1[1]

        if self.controller_type == 'attention':
            raise NotImplementedError()

        if transition_mask.sum() > 0:
            h_tm1, c_tm1 = state_tm1
            h_tm1 = h_tm1.clone()
            c_tm1 = c_tm1.clone()
            # need the byte case here because otherwise it will treat transition_mask as indices
            byte_tm = transition_mask.byte()
            h_tm1[byte_tm] = self.transform_h_in(
                state_transitioned_tm1[0][byte_tm], controller_mask[byte_tm]
            )
            c_tm1[transition_mask.byte()] = self.transform_c_in(
                state_transitioned_tm1[1][byte_tm], controller_mask[byte_tm]
            )
        else:
            h_tm1, c_tm1 = state_tm1

        batch_sz = frame.size(0)

        # encode vision and lang feat
        vis_feat_t = self.vis_encoder(frame)
        lang_feat_t = enc # language is encoded once at the start

        lang_attn_t = []
        h_t = []
        c_t = []
        inp_t = []

        # Iterate over each module. 
        for i in range  (self.n_modules): 

            # attend over language
            weighted_lang_ti, lang_attn_ti = self.attn[i](self.attn_dropout(lang_feat_t), self.h_tm1_fc[i](h_tm1))
            lang_attn_t.append(lang_attn_ti)

            # concat visual feats, weight lang, and previous action embedding
            inp_ti = torch.cat([vis_feat_t, weighted_lang_ti, e_t], dim=1)
            inp_ti = self.input_dropout(inp_ti)
            inp_t.append(inp_ti.unsqueeze(1))

            # update hidden state
            state_t = self.cell[i](inp_ti, (h_tm1, c_tm1))
            h, c = [self.hstate_dropout(x) for x in state_t]
            if hstate_dropout_mask is not None:
                h = hstate_dropout_mask * h
            state_t = h, c

            # batch x dhid
            h_t.append(state_t[0])
            c_t.append(state_t[1])

        # Attend over each modules output.
        h_t = torch.cat([h.unsqueeze(1) for h in h_t], dim=1) # batch x n_modules x dhid
        c_t = torch.cat([c.unsqueeze(1) for c in c_t], dim=1)

        h_t_in = self.attn_dropout(h_t)
        # Add stop embedding.
        h_t_in = torch.cat([h_t_in, self.stop_embedding.view(1,1,-1).expand(h_t_in.size(0), 1, self.dhid)], dim=1)

        inp_t = torch.cat(inp_t, dim=1)
        lang_attn_t = torch.cat(lang_attn_t, dim=-1)


        # Attend over submodules hidden states. 
        # TODO Use ground truth attention here if provided, but keep generated attention since we need it to train attention mechanism.
        if self.controller_type == 'attention':
            _, module_scores = self.controller_attn(h_t_in, self.controller_h_tm1_fc(controller_state_tm1[0]))
            module_attn_logits = self.controller_attn.raw_score
        else:
            assert controller_mask is not None, "when not using an attention-based controller, must pass controller_mask to step()"

        # If no supervised/forced attention is provided, then used inferred values.
        if controller_mask is None:
            module_attn = module_scores

            # Only pay attention to a single module.
            max_subgoal = module_attn.max(1)[1].squeeze()

            module_attn = torch.zeros_like(module_attn)
            module_attn[:,max_subgoal,:] = 1.0

        # Otherwise use ground truth attention.
        else:
            module_attn = controller_mask.unsqueeze(-1).float()

        h_t = module_attn[:,:-1,:].expand_as(h_t).mul(h_t).sum(1)
        h_t_in = module_attn.expand_as(h_t_in).mul(h_t_in).sum(1)
        c_t = module_attn[:,:-1,:].expand_as(c_t).mul(c_t).sum(1)
        inp_t = module_attn[:,:-1,:].expand_as(inp_t).mul(inp_t).sum(1)
        lang_attn_t = module_attn[:,:-1,:].view(batch_sz,1,8).expand_as(lang_attn_t).mul(lang_attn_t).sum(-1)

        if controller_mask is None:
            raise NotImplementedError()
        h_t_translated = self.transform_h_out(h_t, controller_mask)
        c_t_translated = self.transform_c_out(h_t, controller_mask)

        # decode action and mask
        cont_t = torch.cat([h_t_in, inp_t], dim=1) # TODO Add controller_lang_emb
        action_emb_t = self.actor(self.actor_dropout(cont_t))
        action_t = action_emb_t.mm(self.emb.weight.t())
        mask_t = self.mask_dec(cont_t)

        # update controller hidden state
        if self.controller_type == 'attention':
            controller_state_t = self.controller(cont_t, controller_state_tm1)
            controller_state_t = [self.hstate_dropout(x) for x in controller_state_t]
            module_attn_logits_ret = module_attn_logits.view(batch_sz,1,9)
        else:
            controller_state_t = None
            module_attn_logits_ret = None

        # Package weighted state for output. 
        state_t = (h_t_in, c_t)
        state_transitioned_t = (h_t_translated, c_t_translated)

        return action_t, mask_t, state_t, state_transitioned_t, controller_state_t, lang_attn_t, module_attn_logits_ret, module_attn

    def forward(self, enc, frames, gold=None, max_decode=150, state_0=None, state_0_transitioned=None,
                controller_state_0=None, controller_mask=None, transition_mask=None, hstate_dropout_mask=None):
        # transition_mask is not used
        max_t = gold.size(1) if self.training else min(max_decode, frames.shape[1])
        batch = enc.size(0)
        e_t = self.go.repeat(batch, 1)
        state_t = state_0
        state_t_transitioned = state_0_transitioned
        controller_state_t = controller_state_0

        actions = []
        masks = []
        attn_scores = []
        modules_used = []
        if self.controller_type == 'attention':
            module_attn_scores = []
        else:
            assert controller_mask is not None, "when not using an attention-based controller, must pass controller_mask to step()"

        for t in range(max_t):

            # Mask input for high level controller. 
            if controller_mask is None: 
                controller_mask_in = None
            else: 
                controller_mask_in = controller_mask[:,t]

            action_t, mask_t, state_t, state_t_transitioned, controller_state_t, attn_score_t, module_attn_score_t, module_attn_t = self.step(
                enc, frames[:, t], e_t, state_t, state_t_transitioned, controller_state_t, controller_mask_in,
                transition_mask=transition_mask[:,t],
                next_transition_mask=transition_mask[:,t+1] if t+1 < transition_mask.size(1) else None,
                hstate_dropout_mask=hstate_dropout_mask
            )
            masks.append(mask_t)
            actions.append(action_t)
            attn_scores.append(attn_score_t)
            if self.controller_type == 'attention':
                module_attn_scores.append(module_attn_score_t)
            # after squeezing, will be bsz x n_modules
            modules_used.append(module_attn_t.squeeze(-1))
            if self.teacher_forcing and self.training:
                w_t = gold[:, t]
            else:
                w_t = action_t.max(1)[1]
            e_t = self.emb(w_t)

        results = {
            'out_action_low': torch.stack(actions, dim=1),
            'out_action_low_mask': torch.stack(masks, dim=1),
            'out_attn_scores': torch.stack(attn_scores, dim=1),
            'state_t': state_t,
            'state_t_transitioned': state_t_transitioned,
            'controller_state': controller_state_t,
            'modules_used': torch.stack(modules_used, dim=1)
        }
        if self.controller_type == 'attention':
            results['out_module_attn_scores'] = torch.cat(module_attn_scores, dim=1)

        return results


class ConvFrameMaskDecoderProgressMonitor(nn.Module):
    '''
    action decoder with subgoal and progress monitoring
    '''

    def __init__(self, emb, dframe, dhid, pframe=300,
                 attn_dropout=0., hstate_dropout=0., actor_dropout=0., input_dropout=0.,
                 teacher_forcing=False):
        super().__init__()
        demb = emb.weight.size(1)

        self.emb = emb
        self.pframe = pframe
        self.dhid = dhid
        self.vis_encoder = ResnetVisualEncoder(dframe=dframe)
        self.cell = nn.LSTMCell(dhid+dframe+demb, dhid)
        self.attn = DotAttn()
        self.input_dropout = nn.Dropout(input_dropout)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.hstate_dropout = nn.Dropout(hstate_dropout)
        self.actor_dropout = nn.Dropout(actor_dropout)
        self.go = nn.Parameter(torch.Tensor(demb))
        self.actor = nn.Linear(dhid+dhid+dframe+demb, demb)
        self.mask_dec = MaskDecoder(dhid=dhid+dhid+dframe+demb, pframe=self.pframe)
        self.teacher_forcing = teacher_forcing
        self.h_tm1_fc = nn.Linear(dhid, dhid)

        self.subgoal = nn.Linear(dhid+dhid+dframe+demb, 1)
        self.progress = nn.Linear(dhid+dhid+dframe+demb, 1)

        nn.init.uniform_(self.go, -0.1, 0.1)

    def step(self, enc, frame, e_t, state_tm1, encoder_mask=None,
             hstate_dropout_mask=None):
        # previous decoder hidden state
        h_tm1 = state_tm1[0]

        # encode vision and lang feat
        vis_feat_t = self.vis_encoder(frame)
        lang_feat_t = enc # language is encoded once at the start

        # attend over language
        weighted_lang_t, lang_attn_t = self.attn(
            self.attn_dropout(lang_feat_t), self.h_tm1_fc(h_tm1), mask=encoder_mask
        )

        # concat visual feats, weight lang, and previous action embedding
        inp_t = torch.cat([vis_feat_t, weighted_lang_t, e_t], dim=1)
        inp_t = self.input_dropout(inp_t)

        # update hidden state
        state_t = self.cell(inp_t, state_tm1)
        state_t = [self.hstate_dropout(x) for x in state_t]
        if hstate_dropout_mask is not None:
            state_t[0] = hstate_dropout_mask * state_t[0]
        h_t, c_t = state_t[0], state_t[1]

        # decode action and mask
        cont_t = torch.cat([h_t, inp_t], dim=1)
        action_emb_t = self.actor(self.actor_dropout(cont_t))
        action_t = action_emb_t.mm(self.emb.weight.t())
        mask_t = self.mask_dec(cont_t)

        # predict subgoals completed and task progress
        subgoal_t = F.sigmoid(self.subgoal(cont_t))
        progress_t = F.sigmoid(self.progress(cont_t))

        return action_t, mask_t, state_t, lang_attn_t, subgoal_t, progress_t

    def forward(self, enc, frames, gold=None, max_decode=150, state_0=None, encoder_mask=None, hstate_dropout_mask=None):
        max_t = gold.size(1) if self.training else min(max_decode, frames.shape[1])
        batch = enc.size(0)
        e_t = self.go.repeat(batch, 1)
        state_t = state_0

        actions = []
        masks = []
        attn_scores = []
        subgoals = []
        progresses = []
        for t in range(max_t):
            action_t, mask_t, state_t, attn_score_t, subgoal_t, progress_t = self.step(
                enc, frames[:, t], e_t, state_t, encoder_mask=encoder_mask,
                hstate_dropout_mask=hstate_dropout_mask
            )
            masks.append(mask_t)
            actions.append(action_t)
            attn_scores.append(attn_score_t)
            subgoals.append(subgoal_t)
            progresses.append(progress_t)

            # find next emb
            if self.teacher_forcing and self.training:
                w_t = gold[:, t]
            else:
                w_t = action_t.max(1)[1]
            e_t = self.emb(w_t)

        results = {
            'out_action_low': torch.stack(actions, dim=1),
            'out_action_low_mask': torch.stack(masks, dim=1),
            'out_attn_scores': torch.stack(attn_scores, dim=1),
            'out_subgoal': torch.stack(subgoals, dim=1),
            'out_progress': torch.stack(progresses, dim=1),
            'state_t': state_t,
        }
        return results

class ConvFrameDecoder(nn.Module):
    '''
    action decoder
    '''

    def __init__(self, emb, dframe, dhid, pframe=300,
                 attn_dropout=0., hstate_dropout=0., actor_dropout=0., input_dropout=0.,
                 teacher_forcing=False):
        super().__init__()
        demb = emb.weight.size(1)

        self.emb = emb
        self.pframe = pframe
        self.dhid = dhid
        self.vis_encoder = ResnetVisualEncoder(dframe=dframe)
        self.cell = nn.LSTMCell(dhid+dframe+demb, dhid)
        self.attn = DotAttn()
        self.input_dropout = nn.Dropout(input_dropout)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.hstate_dropout = nn.Dropout(hstate_dropout)
        self.actor_dropout = nn.Dropout(actor_dropout)
        self.go = nn.Parameter(torch.Tensor(demb))
        self.actor = nn.Linear(dhid+dhid+dframe+demb, demb)
        self.teacher_forcing = teacher_forcing
        self.h_tm1_fc = nn.Linear(dhid, dhid)

        nn.init.uniform_(self.go, -0.1, 0.1)

    def step(self, enc, frame, e_t, state_tm1, hstate_dropout_mask=None):
        
        # previous decoder hidden state
        h_tm1 = state_tm1[0]

        # encode vision and lang feat
        vis_feat_t = self.vis_encoder(frame)
        lang_feat_t = enc # language is encoded once at the start

        # attend over language
        weighted_lang_t, lang_attn_t = self.attn(self.attn_dropout(lang_feat_t), self.h_tm1_fc(h_tm1))

        # concat visual feats, weight lang, and previous action embedding
        inp_t = torch.cat([vis_feat_t, weighted_lang_t, e_t], dim=1)
        inp_t = self.input_dropout(inp_t)

        # update hidden state
        state_t = self.cell(inp_t, state_tm1)
        state_t = [self.hstate_dropout(x) for x in state_t]
        if hstate_dropout_mask is not None:
            state_t[0] = state_t[0] * hstate_dropout_mask
        h_t = state_t[0]

        # decode action and mask
        cont_t = torch.cat([h_t, inp_t], dim=1)
        action_emb_t = self.actor(self.actor_dropout(cont_t))
        action_t = action_emb_t.mm(self.emb.weight.t())

        return action_t, state_t, lang_attn_t

    def forward(self, enc, frames, gold=None, max_decode=25, state_0=None, hstate_dropout_mask=None):
        max_t = gold.size(1) if self.training else min(max_decode, frames.shape[1])
        batch = enc.size(0)
        e_t = self.go.repeat(batch, 1)
        state_t = state_0

        actions = []
        attn_scores = []
        for t in range(max_t):
            action_t, state_t, attn_score_t = self.step(enc, frames[:, t], e_t, state_t, hstate_dropout_mask=hstate_dropout_mask)

            actions.append(action_t)
            attn_scores.append(attn_score_t)
            if self.teacher_forcing and self.training:
                w_t = gold[:, t]
            else:
                w_t = action_t.max(1)[1]
            e_t = self.emb(w_t)

        results = {
            'out_action_high': torch.stack(actions, dim=1),
            'out_attn_scores': torch.stack(attn_scores, dim=1),
            'state_t': state_t
        }

        return results

    
class SubtaskModule(nn.Module):
    def __init__(self,dhid, din, index, use_fc_nodes=True):
        super().__init__()
        self.cell = nn.LSTMCell(din, dhid)
        self.use_fc_nodes = use_fc_nodes
        if self.use_fc_nodes:
            self.fc_h_in = nn.Linear(dhid, dhid)
            self.fc_h_out = nn.Linear(dhid, dhid)
        self.index = index
        
    # def forward(self, x, controller_mask):
    #     hx, cx  = self.cell(x, (self.hx, self.cx))
    #
    #     # Only update state if selected
    #     # float casts for compatibility with pytorch 1.1
    #     self.cx = cx*controller_mask[:, self.index].unsqueeze(1).float() + self.cx*(1-controller_mask[:, self.index].unsqueeze(1)).float()
    #     self.hx = hx*controller_mask[:, self.index].unsqueeze(1).float() + self.hx*(1-controller_mask[:, self.index].unsqueeze(1)).float()
    #
    #     return self.hx, self.cx
    
    def forward(self, x, controller_mask, outer_h, transition_mask):
        # x: bsz x _
        # controller_mask: n_modules x bsz
        # outer_h: bsz x _
        # transition_mask: bsz
        # self.hx: bsz x _
        t_mask = transition_mask.unsqueeze(1) #[:, self.index].unsqueeze(1)
        t_mask = t_mask.float()
        # If not transitioning, keep current mask, if transitioning, then replace h with translation of the input h.
        #import pdb; pdb.set_trace()
        if self.use_fc_nodes:
            self.hx = self.hx*(1-t_mask) + self.fc_h_in(outer_h)*t_mask
        else:
            self.hx = self.hx*(1-t_mask) + outer_h*t_mask

        # If not transitioning, keep current cell, if transitioning, then replace c with initial state.
        self.cx = self.cx*(1-t_mask) + self.state_0[1]*t_mask

        hx, cx  = self.cell(x, (self.hx, self.cx))

        # Only update state if selected
        c_mask = controller_mask[:, self.index].unsqueeze(1)
        c_mask = c_mask.float()
        self.cx = cx*c_mask + self.cx*(1-c_mask)
        self.hx = hx*c_mask + self.hx*(1-c_mask)


        return self.hx, self.cx

    def reset(self, state_0):
        self.state_0 = state_0
        self.hx = state_0[0]
        self.cx = state_0[1]
    
    def translate_out(self, h, translation_mask, controller_mask):
        # if transitioning out and this module was active, replace h with translation of h.
        c_mask = controller_mask[:, self.index].unsqueeze(1)
        if self.use_fc_nodes:
                next_t_mask = translation_mask.unsqueeze(1)
                next_t_mask = next_t_mask.float()
                hx = self.fc_h_out(self.hx)*next_t_mask+ self.hx*(1-next_t_mask)
                self.hx = hx*c_mask + self.hx*(1-c_mask)
        return self.hx * c_mask 
        
class ConvFrameMaskDecoderModularIndependent(nn.Module):
    '''
    action decoder
    '''

    def __init__(self, emb, dframe, dhid, pframe=300,
                 attn_dropout=0., hstate_dropout=0., actor_dropout=0., input_dropout=0.,
                 teacher_forcing=False, n_modules=8, use_fc_nodes=True, controller_type='attention',
                 cloned_module_initialization=False, init_model_path=None, modularize_actor_mask=False):
        super().__init__()
        assert init_model_path is None
        assert not modularize_actor_mask
        demb = emb.weight.size(1)
        self.controller_type = controller_type
        self.cloned_module_initialization = cloned_module_initialization

        self.n_modules = n_modules

        self.emb = emb
        self.pframe = pframe
        self.dhid = dhid
        self.vis_encoder = ResnetVisualEncoder(dframe=dframe)
        self.cell = nn.ModuleList([SubtaskModule(din=dhid+dframe+demb, dhid=dhid, index=i, use_fc_nodes=use_fc_nodes) for i in range(n_modules)])
        self.attn = nn.ModuleList([DotAttn() for i in range(n_modules)])
        self.h_tm1_fc = nn.ModuleList([nn.Linear(dhid, dhid) for i in range(n_modules)])
        if cloned_module_initialization:
            raise NotImplementedError()
        # High level controller.
        if self.controller_type == 'attention':
            self.controller = nn.LSTMCell(dhid+dhid+dframe+demb, dhid)
            self.controller_attn = DotAttn()
            self.controller_h_tm1_fc = nn.Linear(dhid, dhid)

            # Attention over modules.
            self.module_attn = DotAttn()
        else:
            self.controller = None
            self.controller_attn = None
            self.controller_h_tm1_fc = None
            self.module_attn = None

        # STOP module for high level controller.
        self.stop_embedding = torch.nn.init.uniform_(nn.Parameter(torch.zeros((self.dhid,))))

        self.input_dropout = nn.Dropout(input_dropout)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.hstate_dropout = nn.Dropout(hstate_dropout)
        self.actor_dropout = nn.Dropout(actor_dropout)
        self.go = nn.Parameter(torch.Tensor(demb))
        self.actor = nn.Linear(dhid+dhid+dframe+demb, demb)
        self.mask_dec = MaskDecoder(dhid=dhid+dhid+dframe+demb, pframe=self.pframe)
        self.teacher_forcing = teacher_forcing

        nn.init.uniform_(self.go, -0.1, 0.1)

    def step(self, enc, frame, e_t, state_tm1, controller_state_tm1, controller_mask, transition_mask, prev_controller_mask):
        # previous decoder hidden state for modules. 
        h_tm1 = state_tm1[0]
        module_ids = controller_mask.max(1)[1].squeeze()
        batch_sz = frame.size(0)

        # encode vision and lang feat
        vis_feat_t = self.vis_encoder(frame)
        lang_feat_t = enc # language is encoded once at the start

        lang_attn_t = []
        h_t = []
        c_t = []
        inp_t = []
        
        transition_mask = transition_mask.float()
        controller_mask = controller_mask.float()
        # Do outer translation:
        if transition_mask.sum() > 0:
            new_h = state_tm1[0]*(1-transition_mask.unsqueeze(1))
            for i in range  (self.n_modules): 
                new_h += self.cell[i].translate_out(state_tm1[0], transition_mask, controller_mask)*transition_mask.unsqueeze(1)
            state_tm1 = (new_h, state_tm1[1])

        # Iterate over each module. 
        for i in range  (self.n_modules): 

            # attend over language
            weighted_lang_ti, lang_attn_ti = self.attn[i](self.attn_dropout(lang_feat_t), self.h_tm1_fc[i](h_tm1))
            lang_attn_t.append(lang_attn_ti)

            # concat visual feats, weight lang, and previous action embedding
            inp_ti = torch.cat([vis_feat_t, weighted_lang_ti, e_t], dim=1)
            inp_ti = self.input_dropout(inp_ti)
            inp_t.append(inp_ti.unsqueeze(1))

            # update hidden state
            state_t = self.cell[i](inp_ti, controller_mask, state_tm1[0], transition_mask)
            state_t = [self.hstate_dropout(x) for x in state_t]
            h_t.append(state_t[0])
            c_t.append(state_t[1])

        # Attend over each modules output.
        h_t_in = self.attn_dropout(torch.cat([h.unsqueeze(1) for h in h_t], dim=1))
        c_t = torch.cat([c.unsqueeze(1) for c in c_t], dim=1)
        inp_t = torch.cat(inp_t, dim=1)
        lang_attn_t = torch.cat(lang_attn_t, dim=-1)

        # Add stop embedding. 
        h_t_in = torch.cat([h_t_in, self.stop_embedding.view(1,1,-1).expand(h_t_in.size(0), 1, self.dhid)], dim=1)

        # Attend over submodules hidden states. 
        # TODO Use ground truth attention here if provided, but keep generated attention since we need it to train attention mechanism.
        if self.controller_type == 'attention':
            _, module_scores = self.controller_attn(h_t_in, self.controller_h_tm1_fc(controller_state_tm1[0]))
            module_attn_logits = self.controller_attn.raw_score
        else:
            assert controller_mask is not None, "when not using an attention-based controller, must pass controller_mask to step()"

        # If no supervised/forced attention is provided, then used inferred values.  
        if controller_mask is None: 
            module_attn = module_scores

            # Only pay attention to a single module. 
            max_subgoal = module_attn.max(1)[1].squeeze()
            
            module_attn = torch.zeros_like(module_attn)
            module_attn[:,max_subgoal,:] = 1.0

        # Otherwise use ground truth attention. 
        else:
            module_attn = controller_mask.unsqueeze(-1).float()

        h_t_in = module_attn.expand_as(h_t_in).mul(h_t_in).sum(1)
        c_t = module_attn[:,:-1,:].expand_as(c_t).mul(c_t).sum(1)
        inp_t = module_attn[:,:-1,:].expand_as(inp_t).mul(inp_t).sum(1)
        lang_attn_t = module_attn[:,:-1,:].view(batch_sz,1,8).expand_as(lang_attn_t).mul(lang_attn_t).sum(-1)

        # decode action and mask
        cont_t = torch.cat([h_t_in, inp_t], dim=1) # TODO Add controller_lang_emb
        action_emb_t = self.actor(self.actor_dropout(cont_t))
        action_t = action_emb_t.mm(self.emb.weight.t())
        mask_t = self.mask_dec(cont_t)

        # update controller hidden state
        if self.controller_type == 'attention':
            controller_state_t = self.controller(cont_t, controller_state_tm1)
            controller_state_t = [self.hstate_dropout(x) for x in controller_state_t]
            module_attn_logits_ret = module_attn_logits.view(batch_sz,1,9)
        else:
            controller_state_t = None
            module_attn_logits_ret = None

        # Package weighted state for output. 
        state_t = (h_t_in, c_t)

        return action_t, mask_t, state_t, controller_state_t, lang_attn_t, module_attn_logits_ret, module_attn

    def forward(self, enc, frames, gold=None, max_decode=150, state_0=None, controller_state_0=None, controller_mask=None, transition_mask=None, hstate_dropout_mask=None):
        # TODO: use hstate_dropout_mask
        max_t = gold.size(1) if self.training else min(max_decode, frames.shape[1])
        batch = enc.size(0)
        e_t = self.go.repeat(batch, 1)
        state_t = state_0
        for n in range(self.n_modules):
            self.cell[n].reset(state_0)
        controller_state_t = controller_state_0

        actions = []
        masks = []
        attn_scores = []
        modules_used = []
        if self.controller_type == 'attention':
            module_attn_scores = []
        else:
            assert controller_mask is not None, "when not using an attention-based controller, must pass controller_mask to step()"
        for t in range(max_t):

            # Mask input for high level controller. 
            if controller_mask is None: 
                controller_mask_in = None
            else: 
                controller_mask_in = controller_mask[:,t]

            transition_mask_in = transition_mask[:,t]
            if t+1 < max_t:
                next_transition_mask_in = transition_mask[:,t+1]
            else:
                next_transition_mask_in = None
            #print("t", t, "mask", transition_mask_in)
            action_t, mask_t, state_t, controller_state_t, attn_score_t, module_attn_score_t, module_attn_t = self.step(enc, frames[:, t], e_t,
                                                                                                                        state_t, controller_state_t,
                                                                                                                        controller_mask_in, transition_mask_in,
                                                                                                                        next_transition_mask_in
                                                                                                                       )
            
            if transition_mask is None:
                # Manually get transition mask by checking for action=2 (Stop)
                _, idx =  action_t.max(dim=1)
                transition_mask_in = idx==2
            elif t+1 < max_t:
                transition_mask_in = transition_mask[:,t+1]
            
            masks.append(mask_t)
            actions.append(action_t)
            attn_scores.append(attn_score_t)
            if self.controller_type == 'attention':
                module_attn_scores.append(module_attn_score_t)
            # after squeezing, will be bsz x n_modules
            modules_used.append(module_attn_t.squeeze(-1))
            if self.teacher_forcing and self.training:
                w_t = gold[:, t]
            else:
                w_t = action_t.max(1)[1]
            e_t = self.emb(w_t)

        results = {
            'out_action_low': torch.stack(actions, dim=1),
            'out_action_low_mask': torch.stack(masks, dim=1),
            'out_attn_scores': torch.stack(attn_scores, dim=1),
            'state_t': state_t,
            'controller_state': controller_state_t,
            'modules_used': torch.stack(modules_used, dim=1)
        }
        if self.controller_type == 'attention':
            results['out_module_attn_scores'] = torch.cat(module_attn_scores, dim=1)

        return results
