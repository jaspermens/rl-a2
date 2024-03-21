from collections import deque, namedtuple
import numpy as np
import random
import torch

Experience = namedtuple(
    "Experience",
    field_names=["state", "action", "reward", "done", "new_state"],
)


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)

    def append(self, state, action, reward, done, new_state) -> None:
        to_tensor = lambda x: torch.Tensor(np.array([x]))
        action = int(action)
        self.buffer.append(Experience(to_tensor(state), 
                                      to_tensor(action),
                                      to_tensor(reward),
                                      to_tensor(done),
                                      to_tensor(new_state),
        ))

    # def append(self, experience: Experience) -> None:
    #     self.buffer.append(experience)

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)