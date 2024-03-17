import numpy as np
from deep_q_agent import DeepQAgent
from policies import Policy
from dqn import DeepQModel
from torch import Tensor
import gymnasium as gym

def test_cartpole_learning():
    agent = DeepQAgent(
                env=gym.make("CartPole-v1", render_mode="human") 
                )
    
    model = DeepQModel(n_inputs=4, n_actions=2)
    for _ in range(1000):
        reward, done = agent.take_step(model)

    agent.env.close()


if __name__ == "__main__":
    test_cartpole_learning()