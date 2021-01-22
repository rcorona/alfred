#!/bin/bash

export DISPLAY=":0"
export ALFRED_ROOT="`pwd`"

source activate alfred

model_dir=$1

date=`date -Iminutes`
for split in valid_seen valid_unseen
do
  python -u models/eval/eval_seq2seq.py \
    --model_path ${model_dir}/best_seen.pth \
    --eval_split $split \
    --data data/json_feat_2.1.0 \
    --splits data/splits/pick_2.json \
    --model models.model.seq2seq_im_mask \
    --gpu \
    --num_threads 3 \
    --subgoals all \
    --eval_type subgoals \
    --print_git \
    | tee ${model_dir}/eval_subgoals_all_${split}_pick-2_${date}.out
done
