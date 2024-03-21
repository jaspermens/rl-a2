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

    fig, axs = plt.subplots(1,2,figsize=(14,7),gridspec_kw={'width_ratios': [9, 1]})
    axs[0].plot(np.median(rewards, axis=0), c='black',label="Median")
    axs[0].fill_between(np.arange(num_epochs), *np.quantile(rewards, q=[.33, .67], axis=0), interpolate=True, alpha=.5, zorder=0, color='teal',label="1-$\sigma$? quantile")
    axs[0].fill_between(np.arange(num_epochs), np.min(rewards, axis=0), np.max(rewards, axis=0), interpolate=True, alpha=.3, zorder=-1, color='teal',label="Total range")
    #aesthetics
    axs[0].set_title(f"Results of {str(model_params['policy'])[14:-23]} policy for {num_repetitions} repetitions of {num_epochs} epochs",fontsize=20)
    axs[0].set_ylabel("Reward attained",fontsize=16)
    axs[0].set_xlabel("Epoch",fontsize=16)
    axs[0].legend(loc="lower right")
    axs[0].grid(alpha=0.5)
    target_string,anneal="target","anneal"
    hyper_param_txt = f"$\\alpha$ = {dqn.lr}\n$\epsilon_0$ = {dqn.epsilon_0}\n$\gamma$ = {dqn.gamma}\n$\omega_{{{target_string}}}$ = {dqn.target_network_update_time}\
    \n$\epsilon_{{{anneal}}}$ = {dqn.anneal_timescale}\nburn-in = {dqn.burnin_time}"
    axs[1].set_title(f"Hyperparameters",x=0.6)
    axs[1].text(0.0,0.7,hyper_param_txt,fontsize=12)
    axs[1].set_axis_off()
    
    plt.tight_layout()
    plt.savefig("test_results.png")
    plt.show()


if __name__ == "__main__":
    model_params = {
            'lr': 1e-3,  
            'exp_param': 0.5,
            'policy': Policy.SOFTMAX,
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