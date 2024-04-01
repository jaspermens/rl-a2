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
        print(f"Training model {repetition_i+1} of {num_repetitions}")
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
    ax.grid(alpha=0.5)
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
