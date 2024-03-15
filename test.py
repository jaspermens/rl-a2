import numpy as np
from enum import Enum, auto


class Policy(Enum):
    GREEDY = auto()
    EGREEDY = auto()
    SOFTMAX = auto()


class Model:
    # 4 in, 2 out (C)NN
    def __init__(self) -> None:
        pass

    def evaluate(self, state):
        pass

    def update(self):
        pass


class CartPoleAgent:
    def __init__(self, policy: Policy, model: Model):
        self.policy = policy
        self.model = model

    @staticmethod
    def rescale_state(state):
        # maybe clip/rescale the state values (bc dynamic range diff) 
        pass

    def update(self):
        # update q-table/model. I guess without e.r. (separate file/agent for that?)
        pass

    def select_action(self, state):
        # basically self.model.evaluate(state)
        return np.argmax(self.model.evaluate(state)) 

    def evaluate_performance(self, eval_env, n_episodes, max_episode_length):
        # run the environment a bunch of times and average the returns
        ...