#!/bin/bash

export DISPLAY=":0"
export ALFRED_ROOT="`pwd`"

source activate alfred

model_dir=$1

subgoals=$2

#for split in valid_seen valid_unseen
for split in valid_seen
#for split in valid_unseen valid_seen
do
  python -u models/eval/eval_seq2seq.py \
    --model_path ${model_dir}/best_seen.pth \
    --eval_split $split \
    --data data/json_feat_2.1.0 \
    --model models.model.seq2seq_hierarchical \
    --splits data/splits/object_receptacle_partitioned.json \
    --gpu \
    --num_threads 3 \
    --subgoals $subgoals \
    --eval_type subgoals \
    --oracle \
    --print_git \
    | tee ${model_dir}/eval_oracle_subgoals_${subgoals}_${split}_generalization.out
done
