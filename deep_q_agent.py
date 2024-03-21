import numpy as np
import gymnasium as gym 

from policies import Policy
import torch
from torch import nn

from data_handling import Experience, ReplayBuffer
from dqn import DeepQModel

class DeepQAgent:
    def __init__(self, env: gym.Env, 
                 policy: Policy = Policy.EGREEDY, 
                 exploration_parameter: float = 0.1,
                 buffer_capacity: int = 100,
                 ):
        self.env = env
        self.reset()

        self.policy = policy
        self.exploration_parameter = exploration_parameter
        self.burnin_length = buffer_capacity

        self.buffer = ReplayBuffer(capacity=buffer_capacity)
        
    def reset(self):
        self.state, _ = self.env.reset()

    def select_action(self, model: DeepQModel):
        # basically self.model.evaluate(state)
        state = torch.tensor(np.array([self.state]))

        # get q values:
        q_values = model.forward(state)

        # greedy best action:
        action = self.policy(q_values, self.exploration_parameter)

        return action

    @torch.no_grad      # disable gradient calculation here. I think it saves memory
    def take_step(self, model: DeepQModel):
        action = self.select_action(model = model)

        new_state, reward, done, _, _ = self.env.step(action)
        
        #
        # this is where we'd do replay buffer stuff
        self.buffer.append(state=self.state, action=action, reward=reward, new_state=new_state, done=done)
        #
        
        self.state = new_state           

        if done:
            self.reset()

        return reward, done
    
    def burn_in(self, model: DeepQModel):
        for _ in range(self.burnin_length):
            self.take_step(model=model)