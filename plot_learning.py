import numpy as np
import gymnasium as gym

from babymode_dqn import CartPoleDQN
import matplotlib.pyplot as plt
    

def plot_cartpole_learning(num_repetitions: int, num_epochs: int, 
                           model_params: dict, render_final_dqn=True, 
                           filename: str = "test_results.png",
                           environment_name: str = "CartPole-v1"):
    """ Plots the (averaged) learning progression for the given parameter combination """

    eval_times = np.arange(0, num_epochs, model_params['eval_interval'])
    eval_rewards = np.zeros([num_repetitions, len(eval_times)])

    def pad_early_stopped(arr1: list[int], arr2):
        """pads arr1 to match the length of arr2"""
        npad = len(arr2) - len(arr1)
        padded = np.pad(np.array(arr1), pad_width=(0, npad), mode='constant', constant_values=arr1[-1])
        return padded
    
    early_stop_epochs = [0]*(num_repetitions)                 # list to keep track of early stopping epochs (used in plotting)
    for repetition_i in range(num_repetitions):
        env = gym.make(environment_name)
        env.reset(seed=np.random.randint(0,999999999))

        dqn = CartPoleDQN(env=env,
                          **model_params)
        
        early_stop_epochs[repetition_i] = dqn.train_model(num_epochs=num_epochs)    

        eval_rewards[repetition_i] = pad_early_stopped(dqn.eval_rewards, eval_rewards[0]) 

    epsilons = [dqn.exp_param_at_epoch(epoch=ep) for ep in range(num_epochs)]

    fig, (ax, axlegend) = plt.subplots(1, 2, figsize=(14, 7), width_ratios=[9,1])

    mean_eval_rewards = np.mean(eval_rewards, axis=0)
    # eval_reward_sigma = np.std(eval_rewards, axis=0)

    #ax.plot(eval_times, mean_eval_rewards, c='black', label='mean reward')
    # ax.errorbar(eval_times, mean_eval_rewards, yerr=eval_reward_sigma, c='black', label='mean reward')
    
    #ax.fill_between(eval_times, *np.quantile(eval_rewards, q=[.33, .67], axis=0), interpolate=True, alpha=.5, zorder=0, color='teal',label="1-$\sigma$ quantile")
    #ax.fill_between(eval_times, np.min(eval_rewards, axis=0), np.max(eval_rewards, axis=0), interpolate=True, alpha=.3, zorder=-1, color='teal',label="Total range")
    print(early_stop_epochs,eval_rewards.shape)
    colors = ["red","orange","green","cyan","purple","brown","gray"]
    for i,epoch in enumerate(early_stop_epochs):
        ax.axvline(x=epoch,ls="--",alpha=0.8,color=colors[i]) #early stop epochs
        ax.plot(np.arange(len(eval_rewards[0]))*10,eval_rewards[i],label=f"Run {i}",alpha=0.8,color=colors[i])

    ax1 = ax.twinx()
    ax1.plot(epsilons, label="Exploration parameter")

    # aesthetics
    ax.set_title(f"Results of {str(model_params['policy'])[14:-23]} policy for {num_repetitions} repetitions of {num_epochs} epochs", fontsize=20)
    ax.set_ylabel("Reward attained", fontsize=16)
    ax.set_xlabel("Epoch", fontsize=16)

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax1.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="lower right")
    ax.grid(alpha=0.5)

    target_string, anneal = "target", "anneal"
    
    if model_params['anneal_exp_param']:
        hyper_param_txt = f"$\\alpha$ = {model_params['lr']}\n$\epsilon_0$ = {model_params['exp_param']}\n$\gamma$ = {model_params['gamma']}\n$\omega_{{{target_string}}}$ = {model_params['target_network_update_time']}\
        \n$\epsilon_{{{anneal}}}$ = {model_params['anneal_exp_param']}\nbuffer size = {model_params['buffer_capacity']}"
    else:
        hyper_param_txt = f"$\\alpha$ = {model_params['lr']}\n$\epsilon_0$ = {model_params['exp_param']}\n$\gamma$ = {model_params['gamma']}\n$\omega_{{{target_string}}}$ = {model_params['target_network_update_time']}\
        \n$\epsilon_{{{anneal}}}$ = $\infty$\nbuffer size = {model_params['buffer_capacity']}"
    
    axlegend.set_title(f"Hyperparameters", x=0.6)
    axlegend.text(0.0, 0.7, hyper_param_txt, fontsize=12)
    axlegend.set_axis_off()
    
    plt.tight_layout()
    plt.savefig(f"figures/{filename}")
    plt.show()
    
    if render_final_dqn:
        env = gym.make(environment_name, render_mode="human") 

        dqn.dqn_render_run(env=env)

