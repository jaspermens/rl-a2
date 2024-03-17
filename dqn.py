from typing import Any
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import CSVLogger
from torch import nn

import gymnasium as gym


class DeepQModel(nn.Module):
    def __init__(self, n_inputs, n_actions):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_actions = n_actions
        
        self.net = nn.Sequential(
            nn.Linear(in_features=n_inputs, out_features=4),
            nn.ReLU(),
            nn.Linear(4, 4),
            nn.Linear(4, n_actions),
        )

    def forward(self, x):
        return self.net(x.float())
    
