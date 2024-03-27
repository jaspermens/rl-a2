from policies import Policy
from test import plot_cartpole_learning    


if __name__ == "__main__":
    num_epochs = 300
    num_repetitions = 1
    model_params = {
            'lr': 1e-3,  
            'exp_param': 0.2,
            'policy': Policy.EGREEDY,
            'batch_size': 128,
            'gamma': 0.99,
            'target_network_update_time': 100,
            'do_target_network': True,
            'do_experience_replay': True,
            'burnin_time': 10000,   # == buffer_capacity
            'eval_interval': 10,
            'n_eval_episodes': 10,
            'anneal_exp_param': False,
            'anneal_timescale': num_epochs,
            'early_stopping_reward': 500,
    }
    
    plot_cartpole_learning(num_epochs=num_epochs, 
                           num_repetitions=num_repetitions, 
                            model_params=model_params,
                            )