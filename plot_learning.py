import numpy as np
import gymnasium as gym
import os

from babymode_dqn import CartPoleDQN
import matplotlib.pyplot as plt
    

def train_runs(num_repetitions: int, 
               num_epochs: int, 
               model_params: dict,
               environment_name: str = "CartPole-v1"):
    
    def pad_early_stopped(arr1: list[int], arr2):
        """pads arr1 to match the length of arr2"""
        npad = len(arr2) - len(arr1)
        padded = np.pad(np.array(arr1), pad_width=(0, npad), mode='constant', constant_values=arr1[-1])
        return padded
    
    eval_times = np.arange(0, num_epochs, model_params['eval_interval'])
    eval_rewards = np.zeros([num_repetitions, len(eval_times)])
    final_eval_rewards = np.zeros([num_repetitions, model_params['n_eval_episodes']*10])

    for repetition_i in range(num_repetitions):
        env = gym.make(environment_name)
        env.reset(seed=np.random.randint(0,999999999))

        dqn = CartPoleDQN(env=env,
                          **model_params)
        
        _ = dqn.train_model(num_epochs=num_epochs)
        
        eval_rewards[repetition_i] = pad_early_stopped(dqn.eval_rewards, eval_rewards[0]) 
        final_eval_rewards[repetition_i] = dqn.final_eval_rewards
    
    return eval_rewards, final_eval_rewards

def get_eval_rewards(num_repetitions: int, num_epochs: int, 
                           model_params: dict, render_final_dqn=True, 
                           filename: str = "test_results.png",
                           environment_name: str = "CartPole-v1"):
    """ Plots the (averaged) learning progression for the given parameter combination """


    training_reward_fn = f'results/{filename}_training.npy'
    final_reward_fn = f'results/{filename}_final.npy'
    if os.path.exists(training_reward_fn) and os.path.exists(final_reward_fn):
        print("USING SAVED DATA!!!")
        eval_rewards = np.load(training_reward_fn)
        final_eval_rewards = np.load(final_reward_fn)
    
    else:
        eval_rewards, final_eval_rewards = train_runs(num_repetitions=num_repetitions, 
                                  num_epochs=num_epochs, 
                                  model_params=model_params, 
                                  environment_name=environment_name)

        np.save(file=training_reward_fn, arr=eval_rewards)
        np.save(file=final_reward_fn, arr=final_eval_rewards)

    return eval_rewards, final_eval_rewards


def plot_learning_progression(eval_times, eval_rewards, filename: str, show_plot: bool):
    fig, ax = plt.subplots(1, 1, figsize=[6,4], dpi=150, layout='constrained')

    ax.plot(eval_times, np.mean(eval_rewards, axis=0), label='Mean reward', color='darkslategray')
    ax.fill_between(eval_times, np.min(eval_rewards, axis=0), np.max(eval_rewards, axis=0), 
                    interpolate=True, alpha=.3, zorder=-1, color='teal',label="Total range")
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Reward')
    ax.legend()
    
    plt.savefig(f'figures/{filename}_training.png',dpi=500)
    plt.savefig(f'figures/{filename}_training.pdf')
    if show_plot:
        plt.show()
    plt.close()


def make_learning_plots(num_repetitions: int, num_epochs: int, 
                           model_params: dict, 
                           show_plot: bool,
                           render_final_dqn=True, 
                           filename: str = "test_results.png",
                           environment_name: str = "CartPole-v1",
                           ):
    
    eval_times = np.arange(0, num_epochs, model_params['eval_interval'])
    eval_times_fn = f'results/{filename}_eval_times.npy'
    np.save(eval_times_fn, arr=eval_times)

    eval_rewards, final_rewards = get_eval_rewards(num_repetitions=num_repetitions, 
                                    num_epochs=num_epochs, 
                                    model_params=model_params, 
                                    filename=filename,
                                    environment_name=environment_name)
    
    plot_learning_progression(eval_times, eval_rewards, filename, show_plot)


def plot_cartpole_learning(num_repetitions: int, num_epochs: int, 
                           model_params: dict, 
                           do_plot: bool,
                           render_final_dqn=True, 
                           filename: str = "test_results.png",
                           environment_name: str = "CartPole-v1",
                           ):
    """ Plots the (averaged) learning progression for the given parameter combination """

    eval_times = np.arange(0, num_epochs, model_params['eval_interval'])

    eval_rewards, final_rewards = get_eval_rewards(num_repetitions=num_repetitions, 
                                    num_epochs=num_epochs, 
                                    model_params=model_params, 
                                    filename=filename,
                                    environment_name=environment_name,)

    # plotting code
    fig, (ax, axlegend) = plt.subplots(1, 2, figsize=(14, 7), width_ratios=[9,1])

    mean_eval_rewards = np.mean(eval_rewards, axis=0)
    # eval_reward_sigma = np.std(eval_rewards, axis=0)

    ax.plot(eval_times, mean_eval_rewards, c='black', label='mean reward')
    # ax.errorbar(eval_times, mean_eval_rewards, yerr=eval_reward_sigma, c='black', label='mean reward')
    #np.save(file=f"Exploration_testing/mean_reward_{str(model_params['policy'])[14:-23]}{model_params['exp_param']}",arr=mean_eval_rewards)
    #ax.fill_between(eval_times, *np.quantile(eval_rewards, q=[.33, .67], axis=0), interpolate=True, alpha=.5, zorder=0, color='teal',label="1-$\sigma$ quantile")
    #ax.fill_between(eval_times, np.min(eval_rewards, axis=0), np.max(eval_rewards, axis=0), interpolate=True, alpha=.3, zorder=-1, color='teal',label="Total range")
    print(early_stop_epochs,eval_rewards.shape)
    colors = ["red","orange","green","cyan","purple","brown","gray"]
    for i,epoch in enumerate(early_stop_epochs):
         #early stop epochs
        if len(early_stop_epochs) <= len(colors):
            ax.plot(eval_times,eval_rewards[i],label=f"Run {i}",alpha=0.8,color=colors[i])
            ax.axvline(x=epoch,ls="--",alpha=0.8,color=colors[i])
        else:
            ax.plot(eval_times,eval_rewards[i],label=f"Run {i}",alpha=0.8)
            ax.axvline(x=epoch,ls="--",alpha=0.8)
    #np.save(file=f"Exploration_testing/training_rewards_{str(model_params['policy'])[14:-23]}{model_params['exp_param']}",arr=eval_rewards)
    ax1 = ax.twinx()
    ax1.plot(epsilons, label="Exploration parameter")

    # aesthetics
    ax.set_title(f"Results of {str(model_params['policy'])[14:-23]} policy for {num_repetitions} repetitions of {num_epochs} epochs", fontsize=20)
    ax.set_ylabel("Reward attained", fontsize=16)
    ax.set_xlabel("Epoch", fontsize=16)

    ax.legend(loc="lower right")
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
    plt.savefig(f"figures/{filename}.png",dpi=500)
    plt.savefig(f"figures/{filename}.pdf")
    plt.show()
    plt.close()
    

