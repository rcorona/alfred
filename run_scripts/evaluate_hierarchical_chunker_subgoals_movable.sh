#!/bin/bash

export DISPLAY=":0"
export ALFRED_ROOT="`pwd`"

source activate alfred

model_dir=$1

chunker_model_dir=$2

subgoals=all

date=`date -Iminutes`

for split in valid_seen valid_unseen
#for split in valid_seen
#for split in valid_unseen valid_seen
do
  python -u models/eval/eval_seq2seq.py \
    --model_path ${model_dir}/best_seen.pth \
    --eval_split $split \
    --data data/json_feat_2.1.0 \
    --splits data/splits/movable.json \
    --model models.model.seq2seq_hierarchical \
    --gpu \
    --num_threads 3 \
    --subgoals $subgoals \
    --eval_type subgoals \
    --hierarchical_controller_chunker_model_path ${chunker_model_dir}/best_seen.pth \
    --print_git \
    | tee ${model_dir}/eval_chunker_subgoals_${subgoals}_${split}_movable_${date}.out
done
