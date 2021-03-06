#!/bin/bash

export DISPLAY=":0"
export ALFRED_ROOT="`pwd`"

source activate alfred

model_dir=$1

for split in valid_seen valid_unseen
do
  python -u models/eval/eval_seq2seq.py \
    --model_path ${model_dir}/best_seen.pth \
    --eval_split $split \
    --data data/json_feat_2.1.0 \
    --model models.model.seq2seq_im_mask \
    --gpu \
    --num_threads 3 \
    | tee ${model_dir}/eval_${split}.out
done
