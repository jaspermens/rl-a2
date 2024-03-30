import numpy as np
import matplotlib.pyplot as plt


colors = [
    (0.2, 0.6, 0.8),   # Blue
    (0.8, 0.4, 0.2),   # Orange
    (0.4, 0.6, 0.2),   # Green
    (0.8, 0.2, 0.6),   # Pink
    (1.0, 0.0, 0.0),   # Purple
]

eval_times = np.arange(0, 500, 10)

#epsilon
eps_1 = np.load("Exploration_testing/mean_reward_egreedy0.1.npy")
eps_2 = np.load("Exploration_testing/mean_reward_egreedy0.3.npy")
eps_3 = np.load("Exploration_testing/mean_reward_egreedy0.5.npy")
eps_4 = np.load("Exploration_testing/mean_reward_egreedy0.7.npy")
eps_5 = np.load("Exploration_testing/mean_reward_egreedy0.9.npy")

#boltzmann
tau_0 = np.load("Exploration_testing/mean_reward_softmax0.05.npy")
tau_1 = np.load("Exploration_testing/mean_reward_softmax0.5.npy")
tau_2 = np.load("Exploration_testing/mean_reward_softmax5.npy")
tau_3 = np.load("Exploration_testing/mean_reward_softmax10.npy")
tau_4 = np.load("Exploration_testing/mean_reward_softmax25.npy")
tau_5 = np.load("Exploration_testing/mean_reward_softmax50.npy")

fig,axs = plt.subplots(nrows= 1,ncols=2, sharey=True,figsize=(16,8))

axs[0].plot(eval_times,eps_1,color=colors[0],label="$\epsilon$=0.1")
axs[0].plot(eval_times,eps_2,color=colors[1],label="$\epsilon$=0.3")
axs[0].plot(eval_times,eps_3,color=colors[2],label="$\epsilon$=0.5")
axs[0].plot(eval_times,eps_4,color=colors[3],label="$\epsilon$=0.7")
axs[0].plot(eval_times,eps_5,color=colors[4],label="$\epsilon$=0.9")

#axs[1].plot(eval_times,tau_0,color=colors[0],label="$\\tau$=0.05")
axs[1].plot(eval_times,tau_1,color=colors[0],label="$\\tau$=0.5")
axs[1].plot(eval_times,tau_2,color=colors[1],label="$\\tau$=5")
axs[1].plot(eval_times,tau_3,color=colors[2],label="$\\tau$=10")
axs[1].plot(eval_times,tau_4,color=colors[3],label="$\\tau$=25")
axs[1].plot(eval_times,tau_5,color=colors[4],label="$\\tau$=50")

#aesthetics
fig.suptitle("Comparison of $\epsilon$-greedy and Boltzmann exploration",fontsize=20)

axs[0].grid(alpha=0.5),axs[1].grid(alpha=0.5)
axs[0].set_ylabel("Total reward attained",fontsize=16)
axs[0].set_xlabel("Epoch/Episode",fontsize=16),axs[1].set_xlabel("Epoch/Episode",fontsize=16)
axs[0].legend(),axs[1].legend()
plt.tight_layout()
#plt.savefig("Exploration_testing/Exploration_testing_plot_2.png",dpi=500)
plt.show()

plt.figure(figsize=(8,8))
plt.plot(eval_times,eps_1,color="blue",label="$\epsilon$=0.1")
plt.plot(eval_times,tau_1,color="red",label="$\\tau$=0.5")
plt.legend()
plt.title("Comparison of best exploration parameters",fontsize=20)
plt.xlabel("Epoch/Episode",fontsize=16)
plt.ylabel("Total reward attained",fontsize=16)
plt.tight_layout()
plt.grid(alpha=0.5)
#plt.savefig("Exploration_testing/Exploration_testing_bestparams_1.png",dpi=500)
plt.show()