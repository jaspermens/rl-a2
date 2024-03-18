from torch import nn
import torch.nn.functional as F
import numpy as np
import torch

class DeepQModel(nn.Module):
    def __init__(self, n_inputs, n_actions):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_actions = n_actions

        self.layer1 = nn.Linear(n_inputs, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)
        
    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.Tensor(np.array([x]))
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
