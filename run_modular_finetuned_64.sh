#!/bin/bash
./run_scripts/evaluate_hierarchical_chunker_subgoals.sh ../models_nlp1/exp/modular_finetune_dhid:64/modular_finetune_dhid:64/ ../models_nlp1/exp_chunker/chunker_subgoal_dhid:128/
./run_scripts/evaluate_hierarchical_chunker.sh ../models_nlp1/exp/modular_finetune_dhid:64/modular_finetune_dhid:64/ ../models_nlp1/exp_chunker/chunker_subgoal_dhid:128/

./run_scripts/evaluate_hierarchical_chunker_subgoals.sh ../models_nlp1/exp_pairs/modular_finetune_dhid:64/modular_finetune_dhid:64/ ../models_nlp1/exp_chunker_pairs/chunker_subgoal_dhid:128/
./run_scripts/evaluate_hierarchical_chunker.sh ../models_nlp1/exp_pairs/modular_finetune_dhid:64/modular_finetune_dhid:64/ ../models_nlp1/exp_chunker_pairs/chunker_subgoal_dhid:128/

./run_scripts/evaluate_hierarchical_chunker_subgoals_movable.sh ../models_nlp1/exp_movable/modular_finetune_dhid:64/modular_finetune_dhid:64/ ../models_nlp1/exp_chunker_movable/chunker_subgoal_dhid:128/
./run_scripts/evaluate_hierarchical_chunker_movable.sh ../models_nlp1/exp_movable/modular_finetune_dhid:64/modular_finetune_dhid:64/ ../models_nlp1/exp_chunker_movable/chunker_subgoal_dhid:128/

./run_scripts/evaluate_hierarchical_chunker_subgoals_pick-2.sh ../models_nlp1/exp_pick-2/modular_finetune_dhid:64/modular_finetune_dhid:64/ ../models_nlp1/exp_chunker_pick-2/chunker_subgoal_dhid:128/
./run_scripts/evaluate_hierarchical_chunker_pick-2.sh ../models_nlp1/exp_pick-2/modular_finetune_dhid:64/modular_finetune_dhid:64/ ../models_nlp1/exp_chunker_pick-2/chunker_subgoal_dhid:128/
