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
                 env = gym.Env,
                 lr: float = 1e-4,  
                 exp_param: float = 0.9,
                 policy: Policy = Policy.EGREEDY,
                 batch_size: int = 128,
                 gamma: float = 0.99,
                 target_network_update_time: int = 200,
                 anneal_timescale: int = 1000,
                 ):
        
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_network_update_time = target_network_update_time

        self.env = env
        self.agent = DeepQAgent(env=env, 
                                policy=policy, 
                                exploration_parameter=exp_param, 
                                buffer_capacity=self.batch_size*5)
        
        self.model = DeepQModel(n_inputs=4, n_actions=2)
        self.target_network = DeepQModel(n_inputs=4, n_actions=2)
        self.update_target_model

        self.optimizer = Adam(self.model.parameters(), lr=lr)
        
        self.episode_reward = 0 
        self.ep_rewards = []
        self.episode_losses = []
        self.epoch_epsilons = []

        self.epsilon_0 = exp_param
        self.anneal_timescale = anneal_timescale

    def get_epsilon(self, time):
        return self.epsilon_0 * 0.5**(time/self.anneal_timescale)

    def update_epsilon(self, time):
        self.epsilon = self.get_epsilon(time=time)

    def update_target_model(self):
        self.target_network.load_state_dict(self.model.state_dict())

    def train_model_babymode(self, num_epochs: int = 100):
        # train the model without ER or TN
        def train_step():
            # take step
            reward, done = self.agent.take_step(self.model)

            # compute mse loss:
            q_values = self.model.forward(self.agent.state)
            expected_value = q_values.max(1)[0] # get the max q value (note: not the index)

            next_state_value = torch.Tensor(np.array([reward]))

            loss = nn.MSELoss()(next_state_value, expected_value)

            self.episode_reward += reward
            
            return loss, done
    
        for _ in range(num_epochs):
            done = False
            while not done:
                loss, done = train_step()

                self.optimizer.zero_grad() # I think for stability
                loss.backward()            
                self.optimizer.step()    

            self.ep_rewards.append(self.episode_reward)
            self.episode_reward = 0



    def train_model_with_buffer(self, num_epochs: int = 100):
        # train the model without ER or TN
        def train_step():
            # take step
            reward, done = self.agent.take_step(self.model)
            self.episode_reward += reward

            experiences = self.agent.buffer.sample(batch_size=self.batch_size)
            # fucky tensor thing
            batch = Experience(*zip(*experiences))

            # Compute a mask of non-final states and concatenate the batch elements
            # (a final state would've been the one after which simulation ended)
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.new_state)), 
                                          dtype=torch.bool)
            non_final_next_states = torch.cat([torch.Tensor(s) for s in batch.new_state
                                                        if s is not None])
            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action).type(dtype=torch.int64).unsqueeze(-1)
            reward_batch = torch.cat(batch.reward)

            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
            # columns of actions taken. These are the actions which would've been taken
            # for each batch state according to policy_net
            q_values_batch = self.model.forward(state_batch)
            state_action_values = q_values_batch.gather(1, action_batch)

            q_values = torch.zeros(self.batch_size)
            q_values[non_final_mask] = self.model.forward(non_final_next_states).max(1).values
            
            expected_state_action_values = (q_values * self.gamma) + reward_batch

            loss = nn.MSELoss()(state_action_values, expected_state_action_values.unsqueeze(1))
            
            return loss, done
            
        self.agent.burn_in(model=self.model)        
        for _ in tqdm(range(num_epochs), total=num_epochs, desc=self.episode_reward):
            done = False
            while not done:
                loss, done = train_step()

                self.optimizer.zero_grad() # I think for stability
                loss.backward()            
                self.optimizer.step()    

            # print(self.episode_reward)
            self.ep_rewards.append(self.episode_reward)
            self.episode_reward = 0


    def train_model_with_buffer_and_target_network(self, num_epochs: int = 100):
        # train the model without ER or TN
        def train_step():
            # take step
            reward, done = self.agent.take_step(self.model)
            self.episode_reward += reward

            experiences = self.agent.buffer.sample(batch_size=self.batch_size)
            # fucky tensor thing
            batch = Experience(*zip(*experiences))

            # Compute a mask of non-final states and concatenate the batch elements
            # (a final state would've been the one after which simulation ended)
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.new_state)), 
                                          dtype=torch.bool)
            non_final_next_states = torch.cat([torch.Tensor(s) for s in batch.new_state
                                                        if s is not None])
            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action).type(dtype=torch.int64).unsqueeze(-1)
            reward_batch = torch.cat(batch.reward)

            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
            # columns of actions taken. These are the actions which would've been taken
            # for each batch state according to policy_net
            q_values_batch = self.model.forward(state_batch)
            state_action_values = q_values_batch.gather(1, action_batch)

            q_values = torch.zeros(self.batch_size)
            q_values[non_final_mask] = self.target_network.forward(non_final_next_states).max(1).values
            
            expected_state_action_values = (q_values * self.gamma) + reward_batch

            loss = nn.MSELoss()(state_action_values, expected_state_action_values.unsqueeze(1))
            
            return loss, done
            
        self.agent.burn_in(model=self.model)     
        n_steps = 0   
        self.anneal_timescale = num_epochs/2
        for _ in tqdm(range(num_epochs), total=num_epochs, desc=self.episode_reward):
            done = False
            while not done:
                self.update_epsilon(time=n_steps)

                loss, done = train_step()

                self.optimizer.zero_grad() # I think for stability
                loss.backward()            
                self.optimizer.step()    
                n_steps += 1

                if n_steps % self.target_network_update_time == 0:
                    self.update_target_model()

            self.epoch_epsilons.append(self.epsilon)
            # print(self.episode_reward)
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

    def test_model(self):
        # self.train_model_babymode(num_epochs=100)
        # self.train_model_with_buffer(num_epochs=500)
        self.train_model_with_buffer_and_target_network(num_epochs=1000)
        self.plot_ep_rewards()
        self.env.close()
    

def test_cartpole_learning():
    env=gym.make("CartPole-v1")#, render_mode="human") 
    dqn = CartPoleDQN(env=env)

    dqn.test_model()    

if __name__ == "__main__":
    test_cartpole_learning()