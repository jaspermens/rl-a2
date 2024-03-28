from policies import Policy
from plot_learning import plot_cartpole_learning    


if __name__ == "__main__":
    num_epochs = 500
    num_repetitions = 3
    model_params = {
            'lr': 5e-4,  
            'exp_param': .9,
            'policy': Policy.EGREEDY,
            'batch_size': 256,
            'gamma': 0.995,
            'target_network_update_time': 100,
            'do_target_network': True,
            'do_experience_replay': True,
            'buffer_capacity': 10000,   # == buffer_capacity
            'eval_interval': 10,
            'n_eval_episodes': 10,
            'anneal_exp_param': True,
            'anneal_timescale': 100,
            'early_stopping_reward': 500,
    }
    
    plot_cartpole_learning(num_epochs = num_epochs, 
                           num_repetitions = num_repetitions, 
                           model_params = model_params,
                           filename = "test_results.png"
                        )