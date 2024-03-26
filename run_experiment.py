import numpy as np
import gymnasium as gym

from babymode_dqn import CartPoleDQN
import matplotlib.pyplot as plt
from policies import Policy

from test import plot_cartpole_learning


if __name__ == "__main__":
    num_epochs = 400
    num_repetitions = 5

    model_params = {
            'lr': 1e-3,  
            'exp_param': .9,
            'policy': Policy.SOFTMAX,
            'batch_size': 512,
            'gamma': 0.99,
            'target_network_update_time': 100,
            'do_target_network': True,
            'do_experience_replay': True,
            'burnin_time': 2048,   # == buffer_capacity
            'eval_interval': 20,
            'n_eval_episodes': 10,
            'anneal_exp_param': True,
            'anneal_timescale': 50,
    }
    
    plot_cartpole_learning(num_epochs=num_epochs, 
                           num_repetitions=num_repetitions, 
                           model_params=model_params,
                           render_final_dqn=True,
                           )