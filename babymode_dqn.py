import numpy as np
from deep_q_agent import DeepQAgent
from policies import Policy
from dqn import DeepQModel
import gymnasium as gym
from torch.optim import Adam
from torch import nn
import torch
import matplotlib.pyplot as plt
from data_handling import Experience
from tqdm import tqdm

class CartPoleDQN:
    def __init__(self, 
                 env: gym.Env,
                 lr: float,  
                 exp_param: float,
                 policy: Policy,
                 batch_size: int,
                 gamma: float,
                 target_network_update_time: int,
                 do_target_network: bool,
                 do_experience_replay: bool,
                 anneal_timescale: int,
                 burnin_time: int,
                 ):
        
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size

        self.target_network_update_time = target_network_update_time
        self.do_target_network = do_target_network
        
        self.burnin_time = burnin_time
        self.init_exp_param = exp_param
        self.anneal_timescale = anneal_timescale

        if do_experience_replay:
            self.train_step_func = self._train_step_with_buffer
        else:
            self.train_step_func = self._train_step_without_buffer
        
        self.reset_counters()

        self.agent = DeepQAgent(env=env, 
                                policy=policy, 
                                exploration_parameter=exp_param, 
                                buffer_capacity=burnin_time)
        
        self.model = DeepQModel(n_inputs=4, n_actions=2)
        if self.do_target_network:
            self.target_network = DeepQModel(n_inputs=4, n_actions=2)
        else:
            self.target_network = self.model
        self.update_target_model()

        self.optimizer = Adam(self.model.parameters(), lr=lr)
        
    def reset_counters(self):
        self.episode_reward = 0 
        self.total_time = 0
        self.ep_rewards = []
        self.episode_losses = []
        self.epoch_epsilons = []

    def update_exp_param(self, time):
        new_exp_param = self.init_exp_param * 0.5**(time / self.anneal_timescale)
        
        self.exp_param = new_exp_param
        self.agent.exploration_parameter = self.exp_param

    def update_target_model(self):
        """ Updates the target network if applicable """
        if not self.do_target_network:
            return
        
        if self.total_time % self.target_network_update_time == 0:
            self.target_network.load_state_dict(self.model.state_dict())

    def _train_step_without_buffer(self):
        """ 
        train step using only empirical data. 
        note: not sure if this is correct actually
        """
        # take step
        state_before = self.agent.state

        reward, done = self.agent.take_step(self.model)
        self.episode_reward += reward

        new_state = self.agent.state
        

        # compute mse loss:
        q_values = self.model.forward(state_before) 
        expected_value = q_values.max(1)[0] # get the max q value (note: not the index)

        # next state value according to the (target) net
        new_state_q_value = self.target_network.forward(new_state).max(1)[0]
        new_state_value = self.gamma * new_state_q_value + reward

        loss = nn.MSELoss()(new_state_value, expected_value)

        
        return loss, done

    def _train_step_with_buffer(self):
        # take step
        reward, done = self.agent.take_step(self.model)
        self.episode_reward += reward

        # sample from the buffer
        experiences = self.agent.buffer.sample(batch_size=self.batch_size)
        # fucky tensor thing to reshape
        batch = Experience(*zip(*experiences))

        # if the run is terminated, then the value is 0, so let's mask those out
        # also, concatenate the batch data (just for nicer casting)

        non_final_mask = torch.tensor([s is not None for s in batch.new_state], 
                                        dtype=torch.bool)
        
        non_final_next_states = torch.cat([torch.Tensor(s) for s in batch.new_state
                                                    if s is not None])
        
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action).type(dtype=torch.int64).unsqueeze(-1)
        reward_batch = torch.cat(batch.reward)

        # q-values for the actions in the batch under current policy
        q_values_batch = self.model.forward(state_batch)
        state_action_values = q_values_batch.gather(1, action_batch)

        # best q-values for the states in the batch
        next_state_best_q_values = torch.zeros(self.batch_size)
        next_state_best_q_values[non_final_mask] = self.target_network.forward(non_final_next_states).max(1).values
        
        expected_state_action_values = reward_batch + (self.gamma * next_state_best_q_values)

        loss = nn.MSELoss()(state_action_values, expected_state_action_values.unsqueeze(1))
        
        return loss, done

    def train_model(self, num_epochs: int = 100):
        self.agent.burn_in(model=self.model)     

        for epoch_i in tqdm(range(num_epochs), total=num_epochs, desc=self.episode_reward):
            done = False
            while not done:
                loss, done = self.train_step_func()

                self.optimizer.zero_grad() # remove gradients from previous steps
                loss.backward()            # compute gradients
                torch.nn.utils.clip_grad_value_(self.model.parameters(), 100)

                self.optimizer.step()      # apply gradients

                self.total_time += 1
                self.update_exp_param(time=epoch_i)
                self.update_target_model()      # (only updates the model if applicable)

            self.episode_losses.append(loss.item())
            self.epoch_epsilons.append(self.exp_param)
            self.ep_rewards.append(self.episode_reward)
            self.episode_reward = 0

    def plot_ep_rewards(self):
        fig, ax = plt.subplots(1,1)
        ax.plot(self.ep_rewards)  
        ax1 = ax.twinx() 
        ax1.plot(self.epoch_epsilons, ls='--', c='red')
        plt.savefig('ep_rewards.png')
        from scipy.signal import medfilt
        ax.plot(medfilt(self.ep_rewards, kernel_size=5), c='black')
        plt.savefig('ep_rewards.png')

        plt.show()

