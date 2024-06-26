#!/bin/bash

echo "Remaking all plots for RL Assignment 2..."

if [ ! -d "figures" ]; then
  mkdir figures
fi
if [ ! -d "results" ]; then
  mkdir results
fi

echo "Training DQN"
python3 run_experiment.py --filename='classic_dqn'

echo "Training DQN-ER"
python3 run_experiment.py --no_experience_replay --filename='ablate_er'

echo "Training DQN-TN"
python3 run_experiment.py --no_target_network --filename='ablate_tn'

echo "Training DQN-ER-TN"
python3 run_experiment.py --no_experience_replay --no_target_network --filename='ablate_both'

echo "Performing best exploration comparisons"
python3 run_experiment.py --filename='best_egreedy' --policy='egreedy'

echo "Making exploration comparison plot"
python3 best_explos.py

echo "Compiling results"
python3 ablation_study.py

echo "Training on LunarLander-v2"
python3 run_experiment.py --env="LunarLander-v2" --num_repetitions=5 --filename="lunarlander"

echo "DONE!"
