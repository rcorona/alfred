#!/bin/bash

dhid=$1

if [[ -z $dhid ]]
then
  dhid=512
fi

name=monolithic_dhid:${dhid}

out_dir="exp_pairs/${name}"

mkdir $out_dir 2> /dev/null

python -u models/train/train_seq2seq.py \
  --data data/json_feat_2.1.0 \
  --model seq2seq_im_mask \
  --dout $out_dir \
  --splits data/splits/oct21.json \
  --batch 8 \
  --pm_aux_loss_wt 0.0 \
  --subgoal_aux_loss_wt 0.0 \
  --zero_goal \
  --gpu \
  --dhid $dhid \
  --num_workers 8 \
  --print_git \
  --train_on_subtrajectories \
  --subgoal_pairs \
  | tee ${out_dir}/stdout.log
