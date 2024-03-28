import torch
import numpy as np

from enum import Enum

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


def act_softmax(q_values: torch.Tensor, temperature: float = None) -> int:
    if temperature is None:
        raise ValueError("attempted to use softmax policy without temperature")
    x = q_values.numpy()[0]/temperature
    z = x - max(x)
    softmax = (np.exp(z))/((sum(np.exp(z))))
    action = np.random.choice([0,1], 1, p=softmax)[0]
    return action


def act_random(q_values: torch.Tensor | None = None, exp_param: None = None):
    return np.random.randint(0, 2)


class Policy(Enum):
    GREEDY = act_greedy
    EGREEDY = act_egreedy
    SOFTMAX = act_softmax
    RANDOM = act_random

