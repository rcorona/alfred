#!/bin/bash

pretrain_path=$1
dhid=$2
hstate_dropout=$3
actor_dropout=$4

if [[ -z $dhid ]]
then
  dhid=512
fi

name=modular_finetune_cloned-actor_dhid:${dhid}_2

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

out_dir="exp_pairs/${name}"

mkdir -p $out_dir 2> /dev/null

python -u models/train/train_seq2seq.py \
  --data data/json_feat_2.1.0 \
  --model seq2seq_hierarchical \
  --dout $out_dir \
  --splits data/splits/oct21.json \
  --batch 6 \
  --pm_aux_loss_wt 0.0 \
  --subgoal_aux_loss_wt 0.0 \
  --zero_goal \
  --gpu \
  --dhid $dhid \
  --num_workers 8 \
  --print_git \
  --hstate_dropout $hstate_dropout \
  --actor_dropout $actor_dropout \
  --hierarchical_controller chunker \
  --train_on_subtrajectories \
  --subgoal_pairs \
  --cloned_module_initialization \
  --init_model_path $pretrain_path \
  --modularize_actor_mask \
  | tee ${out_dir}/stdout.log