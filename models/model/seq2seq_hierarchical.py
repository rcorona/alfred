import os
import torch
import numpy as np
import nn.vnn as vnn
import collections
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from model.seq2seq import Module as Base
from model.seq2seq_im_mask import Module as Seq2SeqIM
from models.utils.metric import compute_f1, compute_exact
from gen.utils.image_util import decompress_mask
import pdb
from vocab import Vocab

class Module(Base):

    def __init__(self, args, vocab):
        '''
        Modular Seq2Seq agent
        '''
        super().__init__(args, vocab)

        # subgoal monitoring
        self.subgoal_monitoring = (self.args.pm_aux_loss_wt > 0 or self.args.subgoal_aux_loss_wt > 0)

        # Individual network for each of the 8 submodules. 
        self.submodules = nn.ModuleList([Seq2SeqIM(args, vocab) for i in range(8)])

        # Dictionary from submodule names to idx. 
        self.submodule_names = [
            'PAD', 
            'GotoLocation', 
            'PickupObject', 
            'PutObject', 
            'CoolObject', 
            'HeatObject', 
            'CleanObject', 
            'SliceObject', 
            'ToggleObject',
            'NoOp'
        ]

        self.high_vocab = Vocab(self.submodule_names) 
    
        # Embeddings for high-level actions. 
        self.emb_action_high = nn.Embedding(len(self.high_vocab), args.demb)

        # end tokens
        self.stop_token = self.high_vocab.word2index("NoOp", train=False)

        # internal states
        self.state_t = None
        self.e_t = None
        self.test_mode = False

        # encoder and self-attention
        self.enc =
        self.enc_att = vnn.SelfAttn(args.dhid*2)

        # encoder and self-attention
        self.enc = nn.LSTM(args.demb, args.dhid, bidirectional=True, batch_first=True)
        self.enc_att = vnn.SelfAttn(args.dhid*2)

        # frame decoder (no masks)
        decoder = vnn.ConvFrameDecoderProgressMonitor if self.subgoal_monitoring else vnn.ConvFrameDecoder

        # Have one decoder per module.
        self.decoders = nn.ModuleList([
            self.dec = decoder(self.emb_action_high, args.dframe, 2*args.dhid,
                               pframe=args.pframe,
                               attn_dropout=args.attn_dropout,
                               hstate_dropout=args.hstate_dropout,
                               actor_dropout=args.actor_dropout,
                               input_dropout=args.input_dropout,
                               teacher_forcing=args.dec_teacher_forcing)
        for i in range(8)])

        # dropouts
        self.vis_dropout = nn.Dropout(args.vis_dropout)
        self.lang_dropout = nn.Dropout(args.lang_dropout, inplace=True)
        self.input_dropout = nn.Dropout(args.input_dropout)

        # bce reconstruction loss
        self.bce_with_logits = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.mse_loss = torch.nn.MSELoss(reduction='none')

        # paths
        self.root_path = os.getcwd()
        self.feat_pt = 'feat_conv.pt'

        # params
        self.max_subgoals = 25

        # reset model
        self.reset()

    def featurize(self, batch, load_mask=True, load_frames=True):
        '''
        tensorize and pad batch input
        '''
        device = torch.device('cuda') if self.args.gpu else torch.device('cpu')
        feat = collections.defaultdict(list)

        for ex in batch:

            #########
            # inputs
            #########

            # Combine language instructions into single sequence.            
            is_serialized = not isinstance(ex['num']['lang_instr'][0], list)
            if not is_serialized:
                ex['num']['lang_instr'] = [word for desc in ex['num']['lang_instr'] for word in desc]
 
            # goal and instr language
            lang_goal, lang_instr = ex['num']['lang_goal'], ex['num']['lang_instr']

            # append goal + instr
            feat['lang_goal_instr'].append(lang_goal + lang_instr)

            # Get id of first time step from each subgoal segment. 
            seg_lens = [len(ex['num']['action_low'][i]) for i in range(ex['plan']['low_actions'][-1]['high_idx'] + 1)]
            first_idx = 0
            img_action_ids = []

            for i in range(len(seg_lens)): 
                img_action_ids.append(first_idx)
                first_idx += seg_lens[i]

            # load Resnet features from disk
            if load_frames and not self.test_mode:
                root = self.get_task_root(ex)
                im = torch.load(os.path.join(root, self.feat_pt))
                keep = [None] * (len(seg_lens) + 1) # One more for NoOp
                for i, d in enumerate(ex['images']):
                    
                    # only add frames linked with low-level actions (i.e. skip filler frames like smooth rotations and dish washing)
                    if d['low_idx'] in img_action_ids and keep[img_action_ids.index(d['low_idx'])] is None: 
                        keep[img_action_ids.index(d['low_idx'])] = im[i]

                # Add last image in trajectory for NoOp action. 
                keep[-1] = im[-1]
            
                # Some high level actions are repeated and have no associated image, delete these. 
                while None in keep: 
                    repeat_idx = keep.index(None)

                    # NoOp should not be repeated. 
                    assert(ex['plan']['high_pddl'][repeat_idx]['discrete_action']['action'] != 'NoOp')

                    # Remove. 
                    del keep[repeat_idx]
                    del ex['plan']['high_pddl'][repeat_idx]

                try: 
                    feat['frames'].append(torch.stack(keep, dim=0))
                except: 
                    pdb.set_trace() # TODO not all time steps have an associated image. 

            # Ground trutch high-idx for modular model 
            a_high = [self.high_vocab.word2index(a['discrete_action']['action']) for a in ex['plan']['high_pddl']]

            # Some trajectories don't have low-level actions associated with the last high level actions. 
            a_high = a_high[:ex['plan']['low_actions'][-1]['high_idx'] + 1]

            # Add NoOp
            if not self.high_vocab.word2index('NoOp', train=False) in a_high: 
                a_high.append(self.high_vocab.word2index('NoOp', train=False))

            feat['action_high'].append(a_high)

            try: 
                assert(len(a_high) == len(feat['frames'][-1]))
            except:
                pdb.set_trace()

        # tensorization and padding
        for k, v in feat.items():
            if k in {'lang_goal_instr'}:
                # language embedding and padding
                seqs = [torch.tensor(vv, device=device) for vv in v]
                pad_seq = pad_sequence(seqs, batch_first=True, padding_value=self.pad)
                seq_lengths = np.array(list(map(len, v)))
                embed_seq = self.emb_word(pad_seq)
                packed_input = pack_padded_sequence(embed_seq, seq_lengths, batch_first=True, enforce_sorted=False)
                feat[k] = packed_input
            else:
                # default: tensorize and pad sequence
                seqs = [torch.tensor(vv, device=device, dtype=torch.float if ('frames' in k) else torch.long) for vv in v]
                pad_seq = pad_sequence(seqs, batch_first=True, padding_value=self.pad)
                feat[k] = pad_seq

        return feat

    def forward(self, feat, max_decode=25):
        
        cont_lang, enc_lang = self.encode_lang(feat)
        state_0 = cont_lang, torch.zeros_like(cont_lang)
        frames = self.vis_dropout(feat['frames'])
        res = self.dec(enc_lang, frames, max_decode=max_decode, gold=feat['action_high'], state_0=state_0)
        feat.update(res)
        return feat

    def encode_lang(self, feat):
        '''
        encode goal+instr language
        '''
        emb_lang_goal_instr = feat['lang_goal_instr']
        self.lang_dropout(emb_lang_goal_instr.data)
        enc_lang_goal_instr, _ = self.enc(emb_lang_goal_instr)
        enc_lang_goal_instr, _ = pad_packed_sequence(enc_lang_goal_instr, batch_first=True)
        self.lang_dropout(enc_lang_goal_instr)
        cont_lang_goal_instr = self.enc_att(enc_lang_goal_instr)

        return cont_lang_goal_instr, enc_lang_goal_instr

    def reset(self):
        '''
        reset internal states (used for real-time execution during eval)
        '''
        self.r_state = {
            'state_t': None,
            'e_t': None,
            'cont_lang': None,
            'enc_lang': None
        }

    def step(self, feat, prev_action=None):
        '''
        forward the model for a single time-step (used for real-time execution during eval)
        '''

        # encode language features
        if self.r_state['cont_lang'] is None and self.r_state['enc_lang'] is None:
            self.r_state['cont_lang'], self.r_state['enc_lang'] = self.encode_lang(feat)

        # initialize embedding and hidden states
        if self.r_state['e_t'] is None and self.r_state['state_t'] is None:
            self.r_state['e_t'] = self.dec.go.repeat(self.r_state['enc_lang'].size(0), 1)
            self.r_state['state_t'] = self.r_state['cont_lang'], torch.zeros_like(self.r_state['cont_lang'])

        # previous action embedding
        e_t = self.embed_action(prev_action) if prev_action is not None else self.r_state['e_t']

        # decode and save embedding and hidden states
        out_action_high, state_t, *_ = self.dec.step(self.r_state['enc_lang'], feat['frames'][:, 0], e_t=e_t, state_tm1=self.r_state['state_t'])

        # save states
        self.r_state['state_t'] = state_t
        self.r_state['e_t'] = self.dec.emb(out_action_high.max(1)[1])

        # output formatting
        feat['out_action_high'] = out_action_high.unsqueeze(0)
        feat['out_action_low_mask'] = out_action_low_mask.unsqueeze(0)
        return feat

    def make_debug(self, preds, data):
        '''
        readable output generator for debugging
        '''
        debug = {}
        for ex in data:
            if 'repeat_idx' in ex: ex = self.load_task_json(ex, None)[0]
            i = ex['task_id']
            debug[i] = {
                'lang_goal': ex['turk_annotations']['anns'][ex['ann']['repeat_idx']]['task_desc'],
                'action_low': [a['discrete_action']['action'] for a in ex['plan']['low_actions']],
                'p_action_high': preds[i]['action_high'].split(),
            }
        return debug

    def extract_preds(self, out, batch, feat, clean_special_tokens=True):
        '''
        output processing
        '''
        pred = {}
        for ex, ahigh in zip(batch, feat['out_action_high'].max(2)[1].tolist()):
            # remove padding tokens
            if self.pad in ahigh:
                pad_start_idx = ahigh.index(self.pad)
                ahigh = ahigh[:pad_start_idx]

            # index to API actions
            words = self.high_vocab.index2word(ahigh)

            pred[ex['task_id']] = {
                'action_high': ' '.join(words),
            }

        return pred

    def embed_action(self, action):
        '''
        embed high-level action
        '''
        device = torch.device('cuda') if self.args.gpu else torch.device('cpu')
        action_num = torch.tensor(self.high_vocab.word2index(action), device=device)
        action_emb = self.dec.emb(action_num).unsqueeze(0)
        return action_emb

    def compute_loss(self, out, batch, feat):
        '''
        loss function for Seq2Seq agent
        '''
        losses = dict()

        # GT and predictions
        p_ahigh = out['out_action_high'].view(-1, len(self.high_vocab))
        l_ahigh = feat['action_high'].view(-1)

        ahigh_loss = F.cross_entropy(p_ahigh, l_ahigh, reduction='mean', ignore_index=self.pad)
        losses['action_high'] = ahigh_loss * self.args.action_loss_wt

        return losses

    def compute_metric(self, preds, data):
        '''
        compute f1 and extract match scores for output
        '''
        m = collections.defaultdict(list)
        for ex in data:
            
            # Load task.
            if 'repeat_idx'in ex: ex = self.load_task_json(ex, None)[0]
            i = ex['task_id']

            # Compute the metrics.             
            label = ' '.join([a['discrete_action']['action'] for a in ex['plan']['high_pddl']])
            m['action_high_f1'].append(compute_f1(label.lower(), preds[i]['action_high'].lower()))
            m['action_high_em'].append(compute_exact(label.lower(), preds[i]['action_high'].lower()))
        return {k: sum(v)/len(v) for k, v in m.items()}
