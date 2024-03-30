import numpy as np 
import os

def read_ablation_performance():
    ablation_handles = ['classic_dqn', 'ablate_er', 'ablate_tn', 'ablate_both']
    final_fns = [f'results/{fn}_final.npy' for fn in ablation_handles]
    training_fns = [f'results/{fn}_training.npy' for fn in ablation_handles]
    
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

if __name__ == '__main__':
    print_table()