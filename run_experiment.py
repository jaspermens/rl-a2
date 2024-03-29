from policies import Policy
from plot_learning import plot_cartpole_learning    
from argparse import ArgumentParser 


if __name__ == "__main__":
    num_epochs = 500
    num_repetitions = 3
    parser = ArgumentParser(description="""DQN agent training.""")
    parser.add_argument("-no_target_network", dest='target_network', action='store_false', help="""Use a target network for more stable updates.""")
    parser.add_argument("-no_experience_replay", dest='experience_replay', action='store_false', help="""Use experience replay to break correlations.""")
    cmdargs = parser.parse_args()
    
    model_params = {
            'lr': 5e-4,  
            'exp_param': .9,
            'policy': Policy.EGREEDY,
            'batch_size': 256,
            'gamma': 0.995,
            'target_network_update_time': 100,
            'do_target_network': cmdargs.target_network,
            'do_experience_replay': cmdargs.experience_replay,
            'buffer_capacity': 10000, 
            'eval_interval': 10,
            'n_eval_episodes': 10,
            'anneal_exp_param': True,
            'anneal_timescale': 100,
            'early_stopping_reward': 480,
    }
    plot_cartpole_learning(num_epochs = num_epochs, 
                           num_repetitions = num_repetitions, 
                           model_params = model_params,
                           filename = "test_results.png",
                           environment_name="CartPole-v1",
                        )