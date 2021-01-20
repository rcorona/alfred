#!/bin/bash

dhid=$1

if [[ -z $dhid ]]
then
  dhid=128
fi

name=chunker_subgoal_self_dhid:${dhid}

out_dir="exp_chunker/${name}"

mkdir $out_dir 2> /dev/null

python -u models/train/train_chunker.py \
  --data data/json_feat_2.1.0 \
  --model instruction_chunker_subgoal_self_transitions \
  --dout $out_dir \
  --splits data/splits/oct21.json \
  --batch 8 \
  --gpu \
  --num_workers 8 \
  --print_git \
  --epoch 15 \
  | tee ${out_dir}/stdout.log
