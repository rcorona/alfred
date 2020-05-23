#!/bin/bash

export DISPLAY=":0"
export ALFRED_ROOT="`pwd`"

source activate alfred

model_dir=$1
chunker_model_dir=$2

if [[ -z $chunker_model_dir ]]
then
  echo "pass a chunker model"
  exit 1;
fi

for split in valid_seen valid_unseen
do
  python -u models/eval/eval_seq2seq.py \
    --model_path ${model_dir}/best_seen.pth \
    --eval_split $split \
    --data data/json_feat_2.1.0 \
    --model models.model.seq2seq_hierarchical \
    --gpu \
    --num_threads 3 \
    --hierarchical_controller chunker \
    --hierarchical_controller_chunker_model_path ${chunker_model_dir}/best_seen.pth \
    --print_git \
    | tee ${model_dir}/eval_${split}.out
done
