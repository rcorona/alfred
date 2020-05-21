#!/bin/bash

subgoal=$1
dhid=$2
batch=$3
hstate_dropout=$4
actor_dropout=$5

if [[ -z $dhid ]]
then
  dhid=512
fi

if [[ -z $batch ]]
then
  batch=8
fi

name=segmented_modular_subgoal:${subgoal}_dhid:${dhid}_batch:${batch}

if [[ -z $hstate_dropout ]]
then
  hstate_dropout=0.3
else
  name=${name}_hstate_dropout:${hstate_dropout}
fi

if [[ -z $actor_dropout ]]
then
  actor_dropout=0
else
  name=${name}_actor_dropout:${actor_dropout}
fi

out_dir="exp/${name}"

mkdir $out_dir 2> /dev/null


python -u models/train/train_seq2seq.py \
  --data data/json_feat_2.1.0 \
  --model seq2seq_im_mask \
  --dout $out_dir \
  --splits data/splits/oct21.json \
  --batch $batch \
  --pm_aux_loss_wt 0.0 \
  --subgoal_aux_loss_wt 0.0 \
  --zero_goal \
  --gpu \
  --dhid $dhid \
  --num_workers 8 \
  --train_on_subtrajectories \
  --train_on_subtrajectories_full_instructions \
  --print_git \
  --hstate_dropout $hstate_dropout \
  --actor_dropout $actor_dropout \
  --subgoal $subgoal \
  --epoch 12 \
  | tee ${out_dir}/stdout.log
