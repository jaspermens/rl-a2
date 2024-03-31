import numpy as np
import matplotlib.pyplot as plt

best_softmaxs = np.load("results/classic_dqn_training.npy")
best_egreedys= np.load("results/best_egreedy_training.npy")

egreedy_mean = np.mean(best_egreedys,axis=0)
softmax_mean = np.mean(best_softmaxs,axis=0)

eval_times = np.arange(0,len(egreedy_mean),1)

plt.figure(figsize=(8,8))
plt.plot(eval_times,egreedy_mean,color="blue",label="$\epsilon$=0.1")
plt.plot(eval_times,softmax_mean,color="red",label="$\\tau$=0.5")
plt.legend()
plt.title("Comparison of best exploration parameters",fontsize=20)
plt.xlabel("Epoch",fontsize=16)
plt.ylabel("Total reward attained",fontsize=16)
plt.tight_layout()
plt.grid(alpha=0.5)
plt.savefig("figures/Exploration_testing_bestparams_2.png",dpi=500)
