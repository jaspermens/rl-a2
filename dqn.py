from torch import nn
import torch.nn.functional as F
import numpy as np
import torch

class DeepQModel(nn.Module):
    def __init__(self, n_inputs, n_actions):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_actions = n_actions

        n_nodes = 16
        self.layer1 = nn.Linear(n_inputs, 4*n_nodes)
        # self.layer2 = nn.Linear(n_nodes, n_nodes)
        self.layer3 = nn.Linear(4*n_nodes, 2*n_nodes)
        self.layer4 = nn.Linear(2*n_nodes, n_actions)
        
    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.Tensor(np.array([x]))
        x = F.relu(self.layer1(x))
        # x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)
    
