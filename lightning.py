from typing import Any
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import CSVLogger
from torch import Tensor, nn
import gymnasium as gym

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=4, out_features=4),
            nn.ReLu(),
            nn.Linear(4, 4),
            nn.Linear(4, 2),
        )

    def forward(self, x):
        return self.net(x.float())


class Agent:
    def __init__(self, env) -> None:
        pass

    


class CartPoleDQN(LightningModule):
    def __init__(
        self,
        batch_size: int = 16,
        lr: float = 1e-2,
        env: str = "CartPole-v0",
        gamma: float = 0.99,
        sync_rate: int = 10,
        replay_size: int = 1000,
        warm_start_size: int = 1000,
        eps_last_frame: int = 1000,
        eps_start: float = 1.0,
        eps_end: float = 0.01,
        episode_length: int = 200,
        warm_start_steps: int = 1000,
    ) -> None:

        super().__init__() 
        self.save_hyperparameters()

        self.env = gym.make(self.hparams.env)
        self.agent = Agent(env=self.env)
        self.total_reward = 0
        self.episode_reward = 0
        

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
    
    def mse_loss(self, batch):
        states, actions, rewards, dones, next_states = batch

        