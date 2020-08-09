import os
import sys
sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'models'))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'gen'))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'data'))

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="2"

import os
import torch
import pprint
import json
from data.preprocess import Dataset
from importlib import import_module
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from models.utils.helper_utils import optimizer_to, print_git_info


def add_data_args(parser):
    parser.add_argument('--data', help='dataset folder', default='data/json_feat_2.1.0')
    parser.add_argument('--splits', help='json file containing train/dev/test splits', default='splits/oct21.json')
    parser.add_argument('--preprocess', help='store preprocessed data to json files', action='store_true')
    parser.add_argument('--pp_folder', help='folder name for preprocessed data', default='pp')
    parser.add_argument('--preloaded_dataset', help='Path to preloaded json dataset, set to save time from diskread overhead.', default=None)

    parser.add_argument('--train_on_subtrajectories', action='store_true', help='chop up full trajectories and instructions into subtrajectories')
    parser.add_argument('--train_on_subtrajectories_full_instructions', action='store_true', help='chop up full trajectories into subtrajectories, but keep the full instructions')
    parser.add_argument('--add_stop_in_subtrajectories', action='store_true', help='Add STOP action at end of each subtrajectory.')

    # debugging
    parser.add_argument('--fast_epoch', help='fast epoch during debugging', action='store_true')
    parser.add_argument('--dataset_fraction', help='use fraction of the dataset for debugging (0 indicates full size)', default=0, type=int)

def make_parser():
    # parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    add_data_args(parser)

    # settings
    parser.add_argument('--seed', help='random seed', default=123, type=int)
    parser.add_argument('--save_every_epoch', help='save model after every epoch (warning: consumes a lot of space)', action='store_true')
    parser.add_argument('--model', help='model to use', default='seq2seq_im')
    parser.add_argument('--gpu', help='use gpu', action='store_true')
    parser.add_argument('--dout', help='where to save model', default='exp/model:{model}')
    parser.add_argument('--resume', help='load a checkpoint')
    parser.add_argument('--num_workers', type=int, default=8, help='number of threads to use in DataLoaders')

    # hyper parameters
    parser.add_argument('--batch', help='batch size', default=8, type=int)
    parser.add_argument('--epoch', help='number of epochs', default=20, type=int)
    parser.add_argument('--lr', help='optimizer learning rate', default=1e-4, type=float)
    parser.add_argument('--decay_epoch', help='num epoch to adjust learning rate', default=10, type=int)
    parser.add_argument('--dhid', help='hidden layer size', default=512, type=int)
    parser.add_argument('--dframe', help='image feature vec size', default=2500, type=int)
    parser.add_argument('--demb', help='language embedding size', default=100, type=int)
    parser.add_argument('--pframe', help='image pixel size (assuming square shape eg: 300x300)', default=300, type=int)
    parser.add_argument('--mask_loss_wt', help='weight of mask loss', default=1., type=float)
    parser.add_argument('--action_loss_wt', help='weight of action loss', default=1., type=float)
    parser.add_argument('--subgoal_aux_loss_wt', help='weight of subgoal completion predictor', default=0., type=float)
    parser.add_argument('--pm_aux_loss_wt', help='weight of progress monitor', default=0., type=float)

    parser.add_argument('--lang_model', help='Type of language  modeling to use.', default='default', type=str, choices=['default', 'bert'])
    parser.add_argument('--indep-modules', help='uses independent submodules that keep their own hidden state', action='store_true')
    parser.add_argument('--hierarchical_controller', choices=['attention', 'chunker'], default='attention')
    parser.add_argument('--cloned_module_initialization', help='initialize module parameters to the same values (but allow divergence in training)', action='store_true')
    parser.add_argument('--init_model_path', help='Path to monolithic model to initialize parameters with.', type=str, default=None)
    parser.add_argument('--modularize_actor_mask', help='Give each submodule its own action and mask decoder.', action='store_true')

    # dropouts
    parser.add_argument('--zero_goal', help='zero out goal language', action='store_true')
    parser.add_argument('--zero_instr', help='zero out step-by-step instr language', action='store_true')
    parser.add_argument('--lang_dropout', help='dropout rate for language (goal + instr)', default=0., type=float)
    parser.add_argument('--input_dropout', help='dropout rate for concatted input feats', default=0., type=float)
    parser.add_argument('--vis_dropout', help='dropout rate for Resnet feats', default=0.3, type=float)
    parser.add_argument('--hstate_dropout', help='dropout rate for LSTM hidden states during unrolling', default=0.3, type=float)
    parser.add_argument('--attn_dropout', help='dropout rate for attention', default=0., type=float)
    parser.add_argument('--actor_dropout', help='dropout rate for actor fc', default=0., type=float)

    # other settings
    parser.add_argument('--dec_teacher_forcing', help='use gpu', action='store_true')
    parser.add_argument('--temp_no_history', help='use gpu', action='store_true')

    # Custom parameters.
    parser.add_argument('--subgoal', help='Train only a single subgoal.', default=None, type=str)
    parser.add_argument('--subgoal_pairs', help='Train on contiguous subgoal pairs.', action='store_true')
    parser.add_argument('--subgoal_pairs_and_singles', help='Train on contiguous subgoal pairs and single subgoals.', action='store_true')
    parser.add_argument('--subgoal_pairs_validate_full', help='but use full datasets for validation', action='store_true')

    parser.add_argument('--print_git', action='store_true')
    parser.add_argument('--no_make_debug', action='store_true', help="don't write the predictions to a json file")

    return parser

if __name__ == '__main__':
    parser = make_parser()

    # args and init
    args = parser.parse_args()
    args.dout = args.dout.format(**vars(args))
    torch.manual_seed(args.seed)

    print(' '.join(sys.argv[1:]))

    if args.print_git:
        print_git_info()

    # check if dataset has been preprocessed
    if not os.path.exists(os.path.join(args.data, "%s.vocab" % args.pp_folder)) and not args.preprocess:
        raise Exception("Dataset not processed; run with --preprocess")

    # make output dir
    pprint.pprint(args)
    if not os.path.isdir(args.dout):
        os.makedirs(args.dout)

    # load train/valid/tests splits
    with open(args.splits) as f:
        splits = json.load(f)
        pprint.pprint({k: len(v) for k, v in splits.items()})

    # preprocess and save
    if args.preprocess:
        print("\nPreprocessing dataset and saving to %s folders ... This will take a while. Do this once as required." % args.pp_folder)
        dataset = Dataset(args, None)
        dataset.preprocess_splits(splits)
        vocab = torch.load(os.path.join(args.dout, "%s.vocab" % args.pp_folder))
    else:
        vocab = torch.load(os.path.join(args.data, "%s.vocab" % args.pp_folder))

    # load model
    M = import_module('model.{}'.format(args.model))
    if args.resume:
        print("Loading: " + args.resume)
        model, optimizer = M.Module.load(args.resume)
    else:
        model = M.Module(args, vocab)
        optimizer = None

    # to gpu
    if args.gpu:
        model = model.to(torch.device('cuda'))
        if not optimizer is None:
            optimizer_to(optimizer, torch.device('cuda'))

    # start train loop
    model.run_train(splits, optimizer=optimizer)
