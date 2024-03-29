from policies import Policy
from plot_learning import plot_cartpole_learning    
from argparse import ArgumentParser 


if __name__ == "__main__":
    parser = ArgumentParser(description="""DQN agent training.""")

    parser.add_argument("--no_target_network", 
                        dest='target_network', 
                        action='store_false', 
                        help="""Train without a target network.""")
    parser.add_argument("--no_experience_replay", 
                        dest='experience_replay', 
                        action='store_false', 
                        help="""Train without experience replay.""")
    parser.add_argument("--plot_filename",
                        dest='plot_filename',
                        type=str,
                        default='test_results.png',
                        help="Filename for the learning progression figure.")
    parser.add_argument("--num_epochs",
                        type=int,
                        default=500,
                        help="Max number of training episodes.")
    parser.add_argument("--num_repetitions",
                        type=int,
                        default=3,
                        help="Number of experiments to average results over.")
    parser.add_argument("--env",
                        type=str,
                        choices=["CartPole-v1", "CartPole-v0", "LunarLander-v2"],
                        default="CartPole-v1",
                        help="Name of the Gym environment where the DQN will try to learn."
                        )
    parser.add_argument("--policy",
                        type=str,
                        choices=["egreedy", "softmax"],
                        default="egreedy",
                        help="Exploration strategy. Either epsilon-greedy or softmax/boltzmann",
                        )
    cmdargs = parser.parse_args()
    
    policy_param_annealtime = {
        "egreedy": (Policy.EGREEDY, 0.9, 100),
        "softmax": (Policy.SOFTMAX, 100, 50),
    }

    policy, exp_param, anneal_timescale = policy_param_annealtime[cmdargs.policy] 

    model_params = {
            'lr': 5e-4,  
            'exp_param': exp_param,
            'policy': policy,
            'batch_size': 256,
            'gamma': 0.995,
            'target_network_update_time': 100,
            'do_target_network': cmdargs.target_network,
            'do_experience_replay': cmdargs.experience_replay,
            'buffer_capacity': 10000, 
            'eval_interval': 10,
            'n_eval_episodes': 10,
            'anneal_exp_param': True,
            'anneal_timescale': anneal_timescale,
            'early_stopping_reward': 480,
    }
    plot_cartpole_learning(num_epochs = cmdargs.num_epochs, 
                           num_repetitions = cmdargs.num_repetitions, 
                           model_params = model_params,
                           filename = cmdargs.plot_filename,
                           environment_name = cmdargs.env,
                        )