import numpy as np
import gymnasium as gym 

from policies import Policy
import torch
from torch import nn

from data_handling import Experience, ReplayBuffer
from dqn import DeepQModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DeepQAgent:
    def __init__(self, env: gym.Env, 
                 policy: Policy, 
                 exploration_parameter: float,
                 buffer_capacity: int,
                 ):
        self.env = env
        self.reset()

        self.policy = policy
        self.exploration_parameter = exploration_parameter
        self.burnin_length = buffer_capacity

        self.buffer = ReplayBuffer(capacity=buffer_capacity)
        
    def reset(self):
        state, _ = self.env.reset()
        self.state = torch.tensor(np.array([state]), device=DEVICE, dtype=torch.float32)#.unsqueeze(0)

    def select_action(self, model: DeepQModel):
        # basically self.model.evaluate(state)

        # get q values:
        q_values = model.forward(self.state)

        # greedy best action:
        action = self.policy(q_values, self.exploration_parameter)

        return action

    @torch.no_grad      # disable gradient calculation here. I think it saves memory
    def take_step(self, model: DeepQModel):
        action = self.select_action(model = model)

        new_state, reward, done, _, _ = self.env.step(action)
        
        if done:
            new_state = None
        #
        # this is where we'd do replay buffer stuff
        self.buffer.append(state=self.state, action=action, reward=reward, new_state=new_state, done=done)
        #
        
        if done:
            self.reset()
        else:
            self.state = torch.tensor(np.array([new_state]), dtype=torch.float32, device=DEVICE)#.unsqueeze(0)

        return reward, done
    
    def burn_in(self, model: DeepQModel):
        real_policy = self.policy

        # set the policy to fully random for this bit
        # the initial model might have a bias
        self.policy = Policy.RANDOM
        for _ in range(self.burnin_length):
            self.take_step(model=model)
        
        self.policy = real_policy