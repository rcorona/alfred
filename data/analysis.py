import pdb
import json
import os
import numpy as np
import tqdm
import torch
import pickle
import matplotlib
matplotlib.use("Agg") # Needed for headless server.
import matplotlib.pyplot as plt

# Global variables. 
SPLITS = '/home/rcorona/alfred/data/splits/oct21.json'
PP_FOLDER = 'pp'
DATA = '/home/rcorona/alfred/data/json_feat_2.1.0'
SUBGOALS = ['GotoLocation', 'PickupObject', 'PutObject', 'CoolObject', 'HeatObject', 'CleanObject', 'SliceObject', 'ToggleObject']

ACTION2TYPE = {
    'LookDown_15': 'Look', 'LookUp_15': 'Look', 
    'RotateLeft_90': 'Rotate', 'RotateRight_90': 'Rotate',
    'MoveAhead_25': 'Move', 
    'PickupObject': 'Put_Pickup', 'PutObject': 'Put_Pickup',
    'OpenObject': 'Open_Close',  'CloseObject': 'Open_Close',
    'ToggleObjectOn': 'Toggle', 'ToggleObjectOff': 'Toggle', 
    'SliceObject': 'Slice'
}

ACTION_SPACE = set([
    'Look', 'Rotate', 'Move', 'Put_Pickup', 
    'Open_Close', 'Toggle', 'Slice'
])


def load_task_json(task):
    '''
    load preprocessed json from disk
    '''
    json_path = os.path.join(DATA, task['task'], '%s' % PP_FOLDER, 'ann_%d.json' % task['repeat_idx'])
    with open(json_path) as f:
        data = json.load(f)

    return data

def collect_subgoal_stats(example): 

    # Subgoals to collect statistics over. 
    stats = {s: [] for s in SUBGOALS}

    # Get indexes of each high-level subgoal. 
    subgoals = [s['discrete_action']['action'] for s in example['plan']['high_pddl']]
    subgoals.remove('NoOp')

    # Gather low-level actions pertaining to each subgoal in trajectory. 
    actions = [[] for i in range(len(subgoals))]
    
    for a in example['plan']['low_actions']:
        actions[a['high_idx']].append(a)

    # Now iterate over each subgoal and collect its statistics. 
    for i in range(len(subgoals)): 

        # Length.
        length = len(actions[i])

        # Count action types per subgoal. 
        action_types = {a_type: 0 for a_type in ACTION_SPACE}

        for action in actions[i]: 
            action_types[ACTION2TYPE[action['discrete_action']['action']]] += 1

        # Package statistics. 
        subgoal_stats = {
            'length': length, 
            'action_types': action_types
        }

        # Add to specific subgoal stats. 
        stats[subgoals[i]].append(subgoal_stats)

    return stats

def collect_dataset_subgoal_stats(dataset): 
    """
    Collect statistics across subgoals over entire dataset. 
    """
    # Subgoals to collect statistics over. 
    stats = {s: [] for s in SUBGOALS}

    for task in tqdm.tqdm(dataset): 
        
        # Load example and collect statistics. 
        data = load_task_json(task)
        example_stats = collect_subgoal_stats(data)

        # Add to the dataset stats. 
        for k in stats: 
            stats[k].extend(example_stats[k])

    # Average stats.
    averages = {s: {} for s in SUBGOALS}

    for subgoal in SUBGOALS: 
        for stat in stats[subgoal][0]:
            
            if stat in ('length',): 
                averages[subgoal][stat] = np.mean([ex[stat] for ex in stats[subgoal]])
            
            elif stat in ('action_types',): 
                
                # Compute proportions for each action type for each subgoal. 
                sums = {t: sum([ex[stat][t] for ex in stats[subgoal]]) for t in ACTION_SPACE}
                total = sum([sums[t] for t in sums])
                proportions = {t: sums[t] / total for t in sums}

                averages[subgoal][stat] = proportions

    return averages

def plot_subgoal_stats(stats): 

    # Plot length. 
    lengths = {s: stats[s]['length'] for s in stats}
    
    x = np.arange(len(SUBGOALS))
    y = [lengths[s] for s in SUBGOALS]
    width = 0.35
    xticks = [s.replace('Object', '').replace('Location', '') for s in SUBGOALS]

    plt.title('Average Subgoal Segment Lengths')
    plt.xlabel('Subgoal')
    plt.ylabel('Avg. Length')
    plt.xticks(x, xticks)
    plt.bar(x, y, width, label=SUBGOALS)
    plt.savefig('lengths.png')
    plt.close()

    # Plot action type distributions. 
    action_types = {s: stats[s]['action_types'] for s in stats}
    type_names = list(ACTION_SPACE)
    proportions = [[action_types[s][t] for s in SUBGOALS] for t in type_names]

    fix, ax = plt.subplots()

    for i in range(len(type_names)): 
    
        if i == 0:
            bottom = 0
        else: 
            bottom = np.sum(proportions[:i], axis=0)

        ax.bar(xticks, proportions[i], width, bottom=bottom, label=type_names[i])

    ax.set_ylabel('Action Distribution')
    ax.set_title('Action Type Distributions per Subgoal')
    ax.legend(ncol=3)
    ax.set_ylim([0.0, 1.3])

    plt.savefig('action_distributions.png')
    
if __name__ == '__main__':

    # Load statistics if they exist. 
    if os.path.isfile('stats.pkl'): 
        stats = pickle.load(open('stats.pkl', 'rb'))

    # Otherwise collect them. 
    else: 
        # load train/valid/tests splits
        with open(SPLITS) as f:
            splits = json.load(f)

        # Unpack splits. 
        train = splits['train']
        valid_seen = splits['valid_seen']
        valid_unseen = splits['valid_unseen']

        # Collect statistics for each dataset. 
        train_stats = collect_dataset_subgoal_stats(train)
        seen_stats = collect_dataset_subgoal_stats(valid_seen)
        unseen_stats = collect_dataset_subgoal_stats(valid_unseen)

        print('Train: {}\n'.format(train_stats))
        print('Seen: {}\n'.format(seen_stats))
        print('Unseen: {}\n'.format(unseen_stats))

        # Pickle the statistics for faster loading. 
        stats = {
            'train': train_stats, 
            'seen': seen_stats, 
            'unseen': unseen_stats
        }

        pickle.dump(stats, open('stats.pkl', 'wb'))

    # Make plots for each dataset. 
    plot_subgoal_stats(stats['train'])
