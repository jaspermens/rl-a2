import numpy as np
import gymnasium as gym 

from policies import Policy
import torch

from data_handling import ReplayBuffer
from dqn import DeepQModel


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
        
        if done:
            new_state = None
        #
        # this is where we'd do replay buffer stuff
        self.buffer.append(state=self.state, action=action, reward=reward, new_state=new_state, done=done)
        #
        
        self.state = new_state           
        
        if done:
            self.reset()

        return reward, done
    
    def burn_in(self, model: DeepQModel):
        real_policy = self.policy

        # set the policy to fully random for this bit
        # the initial model might have a bias
        self.policy = Policy.RANDOM
        for _ in range(self.burnin_length):
            self.take_step(model=model)
        
        self.policy = real_policy