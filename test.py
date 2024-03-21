import numpy as np
import gymnasium as gym

from babymode_dqn import CartPoleDQN
import matplotlib.pyplot as plt
from policies import Policy
    
def plot_cartpole_learning(num_repetitions: int, num_epochs: int, model_params):
    epsilons = np.zeros([num_repetitions, num_epochs])
    rewards = np.zeros([num_repetitions, num_epochs])
    for i in range(num_repetitions):
        env = gym.make("CartPole-v1")#, render_mode="human") 

        dqn = CartPoleDQN(env=env,
                          **model_params)
        
        dqn.train_model_with_buffer_and_target_network(num_epochs=num_epochs)    

        epsilons[i] = dqn.epoch_epsilons
        rewards[i] = dqn.ep_rewards

    fig, ax = plt.subplots(1,1)
    ax.plot(np.median(rewards, axis=0), c='black')
    ax.fill_between(np.arange(num_epochs), *np.quantile(rewards, q=[.33, .67], axis=0), interpolate=True, alpha=.5, zorder=0, color='teal')
    ax.fill_between(np.arange(num_epochs), np.min(rewards, axis=0), np.max(rewards, axis=0), interpolate=True, alpha=.3, zorder=-1, color='teal')
    plt.show()


if __name__ == "__main__":
    model_params = {
            'lr': 1e-3,  
            'exp_param': 0.5,
            'policy': Policy.EGREEDY,
            'batch_size': 128,
            'gamma': 0.9,
            'target_network_update_time': 200,
            'anneal_timescale': 1000,
            'burnin_time': 1000,
    }

    num_repetitions = 3
    num_epochs = 10
    plot_cartpole_learning(num_epochs=num_epochs, 
                           num_repetitions=num_repetitions,
                            model_params=model_params)