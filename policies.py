from enum import Enum, auto
from dqn import DeepQModel
import torch
import numpy as np
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

UNIFORM_PROBABILITIES = torch.tensor([1.,1.], dtype=torch.float32, device=DEVICE)

def act_greedy(q_values: torch.Tensor, exp_param: None = None) -> int:
    _, maxind = torch.max(q_values, dim=1)
    return int(maxind.item())

def act_egreedy(q_values: torch.Tensor, epsilon: float = None) -> int:
    if epsilon is None:
        raise ValueError("attempted to use epsilon-greedy policy without epsilon")
    
    if np.random.random() < epsilon:
        action = torch.multinomial(input=UNIFORM_PROBABILITIES, num_samples=1).item()
    else:
        action = act_greedy(q_values=q_values)

    return action

def act_softmax(q_values: torch.Tensor, temperature: float = None) -> int:
    if temperature is None:
        raise ValueError("attempted to use softmax policy without temperature")
    
    softmax = torch.nn.functional.softmax(input=q_values[0], dim=0)
    action = torch.multinomial(input=softmax, num_samples=1).item()
    return action


def act_random(q_values: torch.Tensor, exp_param: float):
    return np.random.randint(0, 2)

class Policy(Enum):
    GREEDY = act_greedy
    EGREEDY = act_egreedy
    SOFTMAX = act_softmax
    RANDOM = act_random

