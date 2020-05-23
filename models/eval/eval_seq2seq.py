import os
import sys

sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'gen'))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'models'))

import pprint
import argparse
import torch.multiprocessing as mp
from eval_task import EvalTask
from eval_subgoals import EvalSubgoals
from eval_hierarchical import EvalHierarchical

from models.utils.helper_utils import print_git_info

if __name__ == '__main__':
    # multiprocessing settings
    print(' '.join(sys.argv))
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

    parser.add_argument('--indep-modules', help='uses independent submodules that keep their own hidden state', action='store_true')
    parser.add_argument('--hierarchical_controller', choices=['attention', 'chunker'], default='attention')
    parser.add_argument('--hierarchical_controller_chunker_model_path', help='path to the chunker model (for use with --hierarchical_controller=chunker)')


    parser.add_argument('--instance_limit', type=int)

    # eval params
    parser.add_argument('--max_steps', type=int, default=500, help='max steps before episode termination')
    parser.add_argument('--max_fails', type=int, default=10, help='max API execution failures before episode termination')

    # eval settings
    parser.add_argument('--subgoals', type=str, help="subgoals to evaluate independently, eg:all or GotoLocation,PickupObject...", default="")
    parser.add_argument('--eval_type', type=str, help="Which type of model to evaluate", choices=['subgoals', 'hierarchical'], default='')
    parser.add_argument('--smooth_nav', dest='smooth_nav', action='store_true', help='smooth nav actions (might be required based on training data)')
    parser.add_argument('--skip_model_unroll_with_expert', action='store_true', help='forward model with expert actions')
    parser.add_argument('--no_teacher_force_unroll_with_expert', action='store_true', help='no teacher forcing with expert')

    parser.add_argument('--modular_subgoals', action='store_true', help='this model was trained with the --subgoal argument to train_seq2seq; should also likely run with --skip_model_unroll_with_expert')
    parser.add_argument('--oracle', action='store_true', help='Use oracle for high-level controller.')

    # TODO: just read these from the model arguments, possibly setting --skip_model_unroll_with_expert
    parser.add_argument('--trained_on_subtrajectories', action='store_true', help='this model was trained with the --train_on_subtrajectories argument to train_seq2seq; should also likely run with --skip_model_unroll_with_expert')
    parser.add_argument('--trained_on_subtrajectories_full_instructions', action='store_true', help='this model was trained with the --train_on_subtrajectories_full_instructions argument to train_seq2seq; should also likely run with --skip_model_unroll_with_expert')

    parser.add_argument('--subgoals_length_constrained', action='store_true', help='force the model to decode for exactly the length of the true segment')

    # debug
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--fast_epoch', dest='fast_epoch', action='store_true')

    parser.add_argument('--print_git', action='store_true')

    # parse arguments
    args = parser.parse_args()
    pprint.pprint(vars(args))

    if args.print_git:
        print_git_info()

    if args.trained_on_subtrajectories or args.trained_on_subtrajectories_full_instructions:
        if args.eval_type != 'subgoals':
            raise NotImplementedError("subtrajectory training and non-subgoal evaluation is not implemented")

    # eval mode
    if args.eval_type == 'subgoals':
        eval = EvalSubgoals(args, manager)

    elif args.eval_type == 'hierarchical':

        eval = EvalHierarchical(args, manager)

    else:
        eval = EvalTask(args, manager)

    # start threads
    eval.spawn_threads()
