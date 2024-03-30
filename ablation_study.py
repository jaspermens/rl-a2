import numpy as np 
import os
import matplotlib.pyplot as plt

ABLATION_HANDLES = ['classic_dqn', 'ablate_er', 'ablate_tn', 'ablate_both']

def read_ablation_performance():
    final_fns = [f'results/{fn}_final.npy' for fn in ABLATION_HANDLES]
    training_fns = [f'results/{fn}_training.npy' for fn in ABLATION_HANDLES]
    
    if not all([os.path.exists(fn) for fn in final_fns]):
        raise FileNotFoundError

    final_evals = np.array([np.load(final) for final in final_fns])
    training_evals = np.array([np.load(training) for training in training_fns])

    return final_evals, training_evals

def end_time_for_run(training_evals: np.ndarray):
    return np.argwhere(training_evals < 500)[-1]


def get_experiment_end_times(training_evals):
    end_times = np.zeros((4, len(training_evals[0])))

    for i, exp_t in enumerate(training_evals):
        end_times[i] = np.array([end_time_for_run(training_evals=run) for run in exp_t]).reshape(-1)

    print(end_times)
    print(np.mean(end_times, axis=1))
    return end_times


def print_table():
    final, training = read_ablation_performance()
    end_times = get_experiment_end_times(training_evals=training)
    mean_tend = np.mean(end_times, axis=1)
    spread_tend = np.std(end_times, axis=1)
    average_scores = np.mean(final, axis=(1,2))
    score_spread = np.std(final, axis=(1,2))

    for experiment in range(4):
        print(f"Experiment {experiment}:")
        print(f"Average score: {average_scores[experiment]} \pm {score_spread[experiment]}")
        print(f"Training time: {mean_tend[experiment]} \pm {spread_tend[experiment]}")
        print()


def plot_ablation_results():
    # eval times are (should be) identical across all four experiments
    eval_times_fn = f'results/{ABLATION_HANDLES[0]}_eval_times.npy'
    titles = ['Best model', 'No TN', 'No ER', 'Neither']
    eval_times = np.load(eval_times_fn)
    _, training_rewards = read_ablation_performance()
    

    fig, axes = plt.subplots(2, 2, figsize=[8,6], dpi=150, layout='constrained', sharex=True)

    for i, ax in enumerate(axes.flatten()):
        eval_rewards = training_rewards[i]
        ax.plot(eval_times, np.mean(eval_rewards, axis=0), label='Mean reward', color='darkslategray')
        ax.fill_between(eval_times, np.min(eval_rewards, axis=0), np.max(eval_rewards, axis=0), 
                        interpolate=True, alpha=.3, zorder=-1, color='teal',label="Total range")
        
        ax.set_title(titles[i])
        ax.set_ylabel('Reward')

    axes[0,0].legend()
    axes[1,0].set_xlabel('Epoch')
    axes[1,1].set_xlabel('Epoch')
    
    plt.savefig(f'figures/ablation_2by2_training.png',dpi=500)
    plt.savefig(f'figures/ablation_2by2_training.pdf')
    plt.close()



if __name__ == '__main__':
    print_table()
    plot_ablation_results()