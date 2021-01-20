#!/bin/bash

pretrain_path=$1
seed=$2
dhid=$3
hstate_dropout=$4
actor_dropout=$5

if [[ -z $seed ]]
then 
  seed=123
fi

if [[ -z $dhid ]]
then
  dhid=512
fi

name=modular_finetune_dhid:${dhid}:${seed}

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
  --model seq2seq_hierarchical \
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
  --hstate_dropout $hstate_dropout \
  --actor_dropout $actor_dropout \
  --hierarchical_controller chunker \
  --cloned_module_initialization \
  --init_model_path $pretrain_path \
  --seed $seed \
  | tee ${out_dir}/stdout.log
