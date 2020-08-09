import json
import pprint
import random
import time
from datetime import timedelta
import torch
import torch.multiprocessing as mp

from models.model.base import AlfredDataset, move_dict_to_cuda
from models.nn.resnet import Resnet
from data.preprocess import Dataset
from importlib import import_module

class Eval(object):

    # tokens
    STOP_TOKEN = "<<stop>>"
    SEQ_TOKEN = "<<seg>>"
    TERMINAL_TOKENS = [STOP_TOKEN, SEQ_TOKEN]

    def __init__(self, args, manager):
        # args and manager
        self.args = args
        self.manager = manager

        # load splits
        with open(self.args.splits) as f:
            self.splits = json.load(f)
            pprint.pprint({k: len(v) for k, v in self.splits.items()})

        # load model
        print("Loading: ", self.args.model_path)
        M = import_module(self.args.model)
        self.model, optimizer = M.Module.load(self.args.model_path)
        self.model.share_memory()
        self.model.eval()

        # updated args
        self.model.args.dout = self.args.model_path.replace(self.args.model_path.split('/')[-1], '')
        self.model.args.data = self.args.data if self.args.data else self.model.args.data

        if self.args.hierarchical_controller_chunker_model_path:
            from models.model.instruction_chunker_subgoal import SubgoalChunker, SubgoalChunkerSelfTransitions, SubgoalChunkerNoTransitions
            ChunkerModule = {
                'instruction_chunker_subgoal': SubgoalChunker,
                'instruction_chunker_subgoal_self_transitions': SubgoalChunkerSelfTransitions,
                'instruction_chunker_subgoal_no_transitions': SubgoalChunkerNoTransitions,
            }[self.args.hierarchical_controller_chunker_model_type]
            self.chunker_model, optimizer = ChunkerModule.load(self.args.hierarchical_controller_chunker_model_path)
            self.chunker_model.share_memory()
            self.chunker_model.eval()

            # updated args
            self.chunker_model.args.dout = self.args.hierarchical_controller_chunker_model_path.replace(
                self.args.hierarchical_controller_chunker_model_path.split('/')[-1], ''
            )
            self.chunker_model.args.data = self.args.data if self.args.data else self.chunker_model.args.data
        else:
            self.chunker_model = None

        # preprocess and save
        if args.preprocess:
            print("\nPreprocessing dataset and saving to %s folders ... This is will take a while. Do this once as required:" % self.model.args.pp_folder)
            self.model.args.fast_epoch = self.args.fast_epoch
            dataset = Dataset(self.model.args, self.model.vocab)
            dataset.preprocess_splits(self.splits)

        # load resnet
        args.visual_model = 'resnet18'
        self.resnet = Resnet(args, eval=True, share_memory=True, use_conv_feat=True)

        # gpu
        if self.args.gpu:
            self.model = self.model.to(torch.device('cuda'))

            if self.chunker_model is not None:
                self.chunker_model = self.chunker_model.to(torch.device('cuda'))

        # success and failure lists
        self.create_stats()

        self.args.start_time = time.time()

        # set random seed for shuffling
        random.seed(int(self.args.start_time))

    def queue_tasks(self):
        '''
        create queue of trajectories to be evaluated
        '''
        task_queue = self.manager.Queue()
        files = self.splits[self.args.eval_split]

        # debugging: fast epoch
        if self.args.fast_epoch:
            files = files[:16]

        if self.args.instance_limit is not None:
            files = files[:self.args.instance_limit]

        if self.args.shuffle:
            random.shuffle(files)
        for traj in files:
            task_queue.put(traj)
        return task_queue

    def spawn_threads(self):
        '''
        spawn multiple threads to run eval in parallel
        '''
        task_queue = self.queue_tasks()

        # start threads
        threads = []
        lock = self.manager.Lock()
        if self.args.num_threads > 1:
            for n in range(self.args.num_threads):
                thread = mp.Process(target=self.run, args=(self.model, self.resnet, self.chunker_model, task_queue, self.args, lock,
                                                           self.successes, self.failures, self.results))
                thread.start()
                threads.append(thread)

            for t in threads:
                t.join()
        else:
            self.run(self.model, self.resnet, self.chunker_model, task_queue, self.args, lock, self.successes, self.failures, self.results)

        # save
        self.save_results()

    @classmethod
    def setup_scene(cls, env, traj_data, r_idx, args, reward_type='dense'):
        '''
        intialize the scene and agent from the task info
        '''
        # scene setup
        scene_num = traj_data['scene']['scene_num']
        object_poses = traj_data['scene']['object_poses']
        dirty_and_empty = traj_data['scene']['dirty_and_empty']
        object_toggles = traj_data['scene']['object_toggles']

        scene_name = 'FloorPlan%d' % scene_num
        env.reset(scene_name)
        env.restore_scene(object_poses, object_toggles, dirty_and_empty)

        # initialize to start position
        env.step(dict(traj_data['scene']['init_action']))

        # print goal instr
        print("Task: %s" % (traj_data['turk_annotations']['anns'][r_idx]['task_desc']))
        current_time = time.time()
        print("total time elapsed: {}".format(timedelta(seconds=current_time-args.start_time)))

        # setup task for reward
        env.set_task(traj_data, args, reward_type=reward_type)

    @classmethod
    def run(cls, model, resnet, chunker_model, task_queue, args, lock, successes, failures, results):
        raise NotImplementedError()

    @classmethod
    def evaluate(cls, env, model, r_idx, resnet, chunker_model, traj_data, args, lock, successes, failures, results):
        raise NotImplementedError()

    @classmethod
    def predict_submodules_from_chunker(cls, chunker_model, traj_data, args):
        feat = chunker_model.featurize(traj_data, chunker_model.args, test_mode=True)
        feat = AlfredDataset.collate_fn([(None, feat)])[1]
        if args.gpu:
            move_dict_to_cuda(feat)

        out = chunker_model.forward(feat)
        preds = chunker_model.extract_preds(out, [traj_data], feat)
        assert len(preds) == 1
        single_preds = next(iter(preds.values()))
        pred_submodules = chunker_model.subtask_sequence(single_preds, ensure_noop_at_end=True)
        return pred_submodules

    def save_results(self):
        raise NotImplementedError()

    def create_stats(self):
        raise NotImplementedError()
