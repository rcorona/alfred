import os
import sys
sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'gen'))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'models'))

import argparse
import torch.multiprocessing as mp
from eval_task import EvalTask
from eval_subgoals import EvalSubgoals
from eval_benchmark import EvalBenchmark


if __name__ == '__main__':
    # multiprocessing settings
    mp.set_start_method('spawn')
    manager = mp.Manager()

    # parser
    parser = argparse.ArgumentParser()

    # settings
    parser.add_argument('--splits', type=str, default="data/splits/oct21.json")
    parser.add_argument('--data', type=str, default="data/json_2.1.0")
    parser.add_argument('--reward_config', default='models/config/rewards.json')
    parser.add_argument('--eval_split', type=str, default='valid_seen', choices=['train', 'valid_seen', 'valid_unseen'])
    parser.add_argument('--model_path', type=str, default="model.pth")
    parser.add_argument('--model', type=str, default='models.model.seq2seq_im_mask')
    parser.add_argument('--preprocess', dest='preprocess', action='store_true')
    parser.add_argument('--shuffle', dest='shuffle', action='store_true')
    parser.add_argument('--gpu', dest='gpu', action='store_true')
    parser.add_argument('--num_threads', type=int, default=1)

    # eval params
    parser.add_argument('--max_steps', type=int, default=500, help='max steps before episode termination')
    parser.add_argument('--max_fails', type=int, default=10, help='max API execution failures before episode termination')

    # eval settings
    parser.add_argument('--subgoals', type=str, help="subgoals to evaluate independently, eg:all or GotoLocation,PickupObject...", default="")
    parser.add_argument('--smooth_nav', dest='smooth_nav', action='store_true', help='smooth nav actions (might be required based on training data)')
    parser.add_argument('--skip_model_unroll_with_expert', action='store_true', help='forward model with expert actions')
    parser.add_argument('--no_teacher_force_unroll_with_expert', action='store_true', help='no teacher forcing with expert')

    parser.add_argument('--modular_subgoals', action='store_true', help='this model was trained with the --subgoal argument to train_seq2seq; should also likely run with --skip_model_unroll_with_expert')

    # debug
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--fast_epoch', dest='fast_epoch', action='store_true')

    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--benchmark_batch_size', type=int, default=1)

    # parse arguments
    args = parser.parse_args()

    if args.benchmark:
        eval = EvalBenchmark(args, manager)
    else:
        # eval mode
        if args.subgoals:
            eval = EvalSubgoals(args, manager)
        else:
            eval = EvalTask(args, manager)

    # start threads
    eval.spawn_threads()
