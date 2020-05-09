import os
import sys
sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'models'))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'gen'))
# sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'data'))

import tqdm
import argparse
import json
import pprint

from collections import defaultdict

from models.model.base import AlfredDataset, AlfredSubtrajectoryDataset
from models.model.seq2seq_im_mask import Module

def iterate_low_actions(task):
    for actions in task['num']['action_low']:
        for action in actions:
            yield action['action']

def action_count_no_stop(task):
    return sum(1 if action_ix != 2 else 0 for action_ix in iterate_low_actions(task))

def count_stop(task):
    return sum(1 if action_ix == 2 else 0 for action_ix in iterate_low_actions(task))

# todo: figure out how many tasks don't have a NoOp at the end (found at least one)

def instructions(task, include_stop=False):
    for instr in task['num']['lang_instr']:
        if include_stop or instr != [34]:
            yield instr

def total_instruction_length(task, include_stop):
    return sum(len(instr_tokens) for instr_tokens in instructions(task, include_stop))

def num_instructions(task, include_stop):
    return sum(1 for instr in instructions(task, include_stop))

def collect_stats(args, tasks, split_into_subtrajectories):
    cls = AlfredSubtrajectoryDataset if split_into_subtrajectories else AlfredDataset
    dataset = cls(
        args, tasks, Module, test_mode=False, featurize=False
    )
    stats = defaultdict(float)
    for i in tqdm.trange(len(dataset)):
        task, feat = dataset[i]
        if task is None:
            continue
        stats['num_instances'] += 1
        stats['action_count_no_stop'] += action_count_no_stop(task)
        stats['stop_count'] += count_stop(task)
        stats['num_instructions'] += num_instructions(task, True)
        stats['total_instruction_length'] += total_instruction_length(task, True)
        stats['num_instructions_no_stop'] += num_instructions(task, False)
        stats['total_instruction_length_no_stop'] += total_instruction_length(task, False)
    return stats

def check_tasks(args, tasks):
    split_stats = collect_stats(args, tasks, True)
    unsplit_stats = collect_stats(args, tasks, False)
    print(unsplit_stats)
    print(split_stats)
    for key in ['action_count_no_stop', 'num_instructions_no_stop', 'total_instructions_length_no_stop']:
        assert split_stats[key] == unsplit_stats[key], key
    print("tests pass")
    # assert split_stats['total_instruction_length'] == unsplit_stats['total_instruction_length'] - unsplit_stats['num_instances']

def main(args):
    with open(args.splits) as f:
        splits = json.load(f)
        pprint.pprint({k: len(v) for k, v in splits.items()})

    print("train")
    check_tasks(args, splits['train'])
    print("valid_seen")
    check_tasks(args, splits['valid_seen'])
    print("valid_unseen")
    check_tasks(args, splits['valid_unseen'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='dataset folder', default='data/json_feat_2.1.0')
    parser.add_argument('--splits', help='json file containing train/dev/test splits', default='splits/oct21.json')
    parser.add_argument('--pp_folder', help='folder name for preprocessed data', default='pp')

    args = parser.parse_args()

    main(args)
