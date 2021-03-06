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
from models.utils.helper_utils import optimizer_to

from models.train.train_seq2seq import  add_data_args

from models.model.instruction_chunker import Chunker

def make_parser():
    # parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    add_data_args(parser)
    Chunker.add_arguments(parser)

    # other settings
    parser.add_argument('--seed', help='random seed', default=123, type=int)
    parser.add_argument('--save_every_epoch', help='save model after every epoch (warning: consumes a lot of space)', action='store_true')
    parser.add_argument('--gpu', help='use gpu', action='store_true')
    parser.add_argument('--dout', help='where to save model', default='exp/chunker')
    parser.add_argument('--resume', help='load a checkpoint')
    parser.add_argument('--num_workers', type=int, default=8, help='number of threads to use in DataLoaders')
    parser.add_argument('--batch', help='batch size', default=8, type=int)
    parser.add_argument('--epoch', help='number of epochs', default=20, type=int)
    parser.add_argument('--lr', help='optimizer learning rate', default=1e-4, type=float)
    parser.add_argument('--decay_epoch', help='num epoch to adjust learning rate', default=10, type=int)

    return parser

def main():
    parser = make_parser()

    # args and init
    args = parser.parse_args()
    args.dout = args.dout.format(**vars(args))
    torch.manual_seed(args.seed)

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
    if args.resume:
        print("Loading: " + args.resume)
        model, optimizer = Chunker.load(args.resume)
    else:
        model = Chunker(args, vocab)
        optimizer = None

    # to gpu
    if args.gpu:
        model = model.to(torch.device('cuda'))
        if not optimizer is None:
            optimizer_to(optimizer, torch.device('cuda'))

    # start train loop
    model.run_train(splits, optimizer=optimizer)

if __name__ == "__main__":
    main()
