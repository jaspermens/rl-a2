import numpy as np
from dqn import DeepQModel
import gymnasium as gym 

from policies import Policy
import torch
from torch import Tensor, nn
from torch.optim import Adam
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader


class DeepQAgent:
    def __init__(self, env: gym.Env, 
                 policy: Policy = Policy.EGREEDY, 
                 exploration_parameter: float = 0.1,
                 ):
        self.env = env
        self.reset()

        self.policy = policy
        self.exploration_parameter = exploration_parameter
        
        
    def reset(self):
        self.state, _ = self.env.reset()

    def select_action(self, model: nn.Module):
        # basically self.model.evaluate(state)
        state = torch.tensor(np.array([self.state]))

        # get q values:
        q_values = model.forward(state)

        # greedy best action:
        action = self.policy(q_values, self.exploration_parameter)

        return action
    
    @torch.no_grad      # disable gradient calculation here. I think it saves memory
    def take_step(self, model: nn.Module):
        action = self.select_action(model = model)

        new_state, reward, terminated, _, _ = self.env.step(action)
        
        #
        # this is where we'd do replay buffer stuff
        #
        
        self.state = new_state           

        if terminated:
            self.reset()

        return reward, terminated
    

    
class CartPoleDQN(LightningModule):
    def __init__(
        self,
        lr: float = 1e-1,
        policy: Policy = Policy.EGREEDY,
        exp_param: float = 0.1,
        env_name: str = "CartPole-v1",
        gamma: float = 0.99,
        episode_length: int = 200,
    ) -> None:

        super().__init__()
        self.save_hyperparameters()

        self.env = gym.make(self.hparams.env_name)
        obs_size = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.n

        self.model = DeepQModel(obs_size, n_actions)

        self.agent = DeepQAgent(env=self.env, 
                                policy=policy, 
                                exploration_parameter=exp_param)
        
        self.total_reward = 0
        self.episode_reward = 0

    def forward(self, state: Tensor) -> Tensor:
        return self.model(state)


    def training_step(self):
        reward, done = self.take_step()

        self.episode_reward += reward

        # compute the mse loss:
        q_values = self.model.forward()
        expected_value = q_values.max(1)[0] # get the max q value (note: not the index)

        next_state_value = reward
        if done and reward != 0:
            raise Exception("eh?")

        loss = nn.MSELoss()(next_state_value, expected_value)

        if done:
            self.total_reward = self.episode_reward # ???
            self.episode_reward = 0

        return loss
    
    def configure_optimizers(self):
        opt = Adam(self.model.parameters(), lr=self.hparams.lr)
        return opt
    
    def train_dataloader(self):
        # moet voor de trainer - gebeurt van alles
        return DataLoader(dataset=None)
    
    
    
from pytorch_lightning import Trainer

model = CartPoleDQN()

trainer = Trainer(
    max_epochs=100,
)

trainer.fit(model)