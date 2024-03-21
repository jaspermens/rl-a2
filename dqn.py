from torch import nn
import torch.nn.functional as F
import numpy as np
import torch

class DeepQModel(nn.Module):
    def __init__(self, n_inputs, n_actions):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_actions = n_actions

        n_nodes = 256
        self.layer1 = nn.Linear(n_inputs, n_nodes)
        self.layer2 = nn.Linear(n_nodes, n_nodes)
        self.layer3 = nn.Linear(n_nodes, n_actions)
        
    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.Tensor(np.array([x]))
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
