import numpy as np
from deep_q_agent import DeepQAgent
from policies import Policy
from dqn import DeepQModel
import gymnasium as gym
from torch.optim import Adam
from torch import nn
import torch
import matplotlib.pyplot as plt


class CartPoleDQN:
    def __init__(self, 
                 env = gym.Env,
                 lr: float = 1e-3,  
                 exp_param: float = 0.1,
                 policy: Policy = Policy.EGREEDY,
                 ):
        self.lr = lr
        
        self.env = env
        self.agent = DeepQAgent(env=env, 
                                policy=policy, 
                                exploration_parameter=exp_param, 
                                buffer_capacity=100)
        
        self.model = DeepQModel(n_inputs=4, n_actions=2)

        self.optimizer = Adam(self.model.parameters(), lr=lr)
        
        self.episode_reward = 0 
        self.ep_rewards = []


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


    def plot_ep_rewards(self):
        fig, ax = plt.subplots(1,1)
        ax.plot(self.ep_rewards)

        plt.show()

    def test_model(self):
        self.train_model_babymode(num_epochs=100)

    
def test_cartpole_learning():
    env=gym.make("CartPole-v1")#, render_mode="human") 
    dqn = CartPoleDQN(env=env)

    dqn.test_model()    
    dqn.plot_ep_rewards()
    dqn.env.close()

if __name__ == "__main__":
    test_cartpole_learning()