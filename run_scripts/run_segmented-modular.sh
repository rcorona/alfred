#!/bin/bash
# takes one argument, <dhid>
for subgoal in GotoLocation CleanObject PickupObject PutObject CoolObject HeatObject SliceObject ToggleObject
do
  run_scripts/run_segmented-modular_subgoal.sh $subgoal $@
done
