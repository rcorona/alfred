#!/bin/bash

export DISPLAY=":0"
export ALFRED_ROOT="`pwd`"

source activate alfred

model_dir=$1

subgoals=$2
if [[ -z $subgoals ]]
then
  subgoals=all
fi

for split in valid_seen valid_unseen
#for split in valid_seen
do
  python -u models/eval/eval_seq2seq.py \
    --model_path ${model_dir}/best_seen.pth \
    --eval_split $split \
    --data data/json_feat_2.1.0 \
    --model models.model.seq2seq_im_mask \
    --gpu \
    --num_threads 3 \
    --subgoals $subgoals \
    --trained_on_subtrajectories \
    --trained_on_subtrajectories_full_instructions \
    --skip_model_unroll_with_expert \
    --eval_type subgoals \
    | tee ${model_dir}/eval_subtraj-trained-full-instructions_subgoals_${subgoals}_${split}.out
done
