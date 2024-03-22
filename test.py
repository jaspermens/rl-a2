import numpy as np
import gymnasium as gym

from babymode_dqn import CartPoleDQN
import matplotlib.pyplot as plt
from policies import Policy
    

def plot_cartpole_learning(num_repetitions: int, num_epochs: int, model_params):
    losses = np.zeros([num_repetitions, num_epochs])
    epsilons = np.zeros([num_repetitions, num_epochs])
    rewards = np.zeros([num_repetitions, num_epochs])

    eval_times = np.arange(0, num_epochs, model_params['eval_interval'])
    eval_rewards = np.zeros([num_repetitions, len(eval_times)])

    for repetition_i in range(num_repetitions):
        env = gym.make("CartPole-v1")#, render_mode="human") 

        dqn = CartPoleDQN(env=env,
                          **model_params)
        
        dqn.train_model(num_epochs=num_epochs)    

        eval_rewards[repetition_i] = dqn.eval_rewards
        losses[repetition_i] = dqn.episode_losses
        epsilons[repetition_i] = dqn.epoch_epsilons
        rewards[repetition_i] = dqn.ep_rewards

    fig, (ax, axlegend) = plt.subplots(1, 2, figsize=(14, 7), width_ratios=[9,1])

    # median_reward = np.median(rewards, axis=0)
    # ax.plot(median_reward, c='black',label="Median")

    mean_eval_rewards = np.mean(eval_rewards, axis=0)
    eval_reward_sigma = np.std(eval_rewards, axis=0)
    ax.errorbar(eval_times, mean_eval_rewards, yerr= eval_reward_sigma, c='black', label='eval rewards')
    
    # mean_reward = np.mean(rewards, axis=0)
    # ax.plot(mean_reward, c='black', label='Mean', ls='--')

    ax.fill_between(np.arange(num_epochs), *np.quantile(rewards, q=[.33, .67], axis=0), interpolate=True, alpha=.5, zorder=0, color='teal',label="1-$\sigma$? quantile")
    ax.fill_between(np.arange(num_epochs), np.min(rewards, axis=0), np.max(rewards, axis=0), interpolate=True, alpha=.3, zorder=-1, color='teal',label="Total range")
    
    # plot losses over time (?)
    # ax1 = ax.twinx()
    # ax1.plot(np.mean(losses, axis=0), c='red', ls='dotted')
    ax1 = ax.twinx()
    ax1.plot(epsilons[0],label="Exploration parameter")

    # aesthetics
    ax.set_title(f"Results of {str(model_params['policy'])[14:-23]} policy for {num_repetitions} repetitions of {num_epochs} epochs", fontsize=20)
    ax.set_ylabel("Reward attained", fontsize=16)
    ax.set_xlabel("Epoch", fontsize=16)

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax1.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="lower right")
    ax.grid(alpha=0.5)

    target_string, anneal = "target", "anneal"
    
    hyper_param_txt = f"$\\alpha$ = {dqn.lr}\n$\epsilon_0$ = {dqn.init_exp_param}\n$\gamma$ = {dqn.gamma}\n$\omega_{{{target_string}}}$ = {dqn.target_network_update_time}\
    \n$\epsilon_{{{anneal}}}$ = {dqn.anneal_timescale}\nburn-in = {dqn.burnin_time}"
    
    axlegend.set_title(f"Hyperparameters", x=0.6)
    axlegend.text(0.0, 0.7, hyper_param_txt, fontsize=12)
    axlegend.set_axis_off()
    
    plt.tight_layout()
    plt.savefig("test_results.png")
    plt.show()


if __name__ == "__main__":
    model_params = {
            'lr': 5e-4,  
            'exp_param': 1.,
            'policy': Policy.EGREEDY,
            'batch_size': 256,
            'gamma': 0.99,
            'target_network_update_time': 50,
            'do_target_network': True,
            'do_experience_replay': True,
            'anneal_timescale': 500,
            'burnin_time': 10000,
            'eval_interval': 10,
            'n_eval_episodes': 10,
    }

    num_repetitions = 5
    num_epochs = 500
    plot_cartpole_learning(num_epochs=num_epochs, 
                           num_repetitions=num_repetitions,
                            model_params=model_params)