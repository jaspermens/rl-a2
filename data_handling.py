from collections import deque, namedtuple
import numpy as np
import random
import torch

Experience = namedtuple(
    "Experience",
    field_names=["state", "action", "reward", "done", "new_state"],
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)

    def append(self, state, action, reward, done, new_state) -> None:
        def to_tensor(x):
            if isinstance(x, torch.Tensor):
                return x
            if x is None:
                return x
            return torch.tensor(np.array([x]), device=DEVICE, dtype=torch.float32)
        
        if new_state is not None and not isinstance(new_state, torch.Tensor):
            new_state = to_tensor(new_state)
        self.buffer.append(Experience(to_tensor(state), 
                                      torch.tensor(np.array([action]), device=DEVICE, dtype=torch.int64),
                                      to_tensor(reward),
                                      to_tensor(done),
                                      new_state,
        ))


    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)