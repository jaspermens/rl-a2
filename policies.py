from enum import Enum, auto
from dqn import DeepQModel
import torch
import numpy as np


def act_greedy(q_values: torch.Tensor, exp_param: None = None) -> int:
    _, maxind = torch.max(q_values, dim=1)
    return int(maxind.item())

def act_egreedy(q_values: torch.Tensor, epsilon: float = None) -> int:
    if epsilon is None:
        raise ValueError("attempted to use epsilon-greedy policy without epsilon")
    
    if np.random.random() < epsilon:
        action = np.random.choice(len(q_values))
    else:
        action = act_greedy(q_values=q_values)

    return action

def act_softmax(state):
    ...


class Policy(Enum):
    GREEDY = act_greedy
    EGREEDY = act_egreedy
    SOFTMAX = act_softmax
 