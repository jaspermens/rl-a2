import numpy as np
from tqdm import tqdm

import gymnasium as gym

import torch
from torch import nn
from torch.optim import Adam

from data_handling import Experience
from deep_q_agent import DeepQAgent
from policies import Policy
from dqn import DeepQModel


class CartPoleDQN:
    """Main class handling the training of a DQN in a Gym environment"""
    def __init__(self, 
                 env: gym.Env,                      # gym environment we'll be training in
                 lr: float,                         # learning rate
                 policy: Policy,                    # (exploration) policy (e.g. e-greedy, softmax)
                 exp_param: float,                  # exploration parameter value (e.g. epsilon, temperature)
                 batch_size: int,                   # (only used with experience replay)
                 gamma: float,                  
                 do_target_network: bool,           # target network toggle
                 target_network_update_time: int,   # time interval between target network updates (in steps)
                 do_experience_replay: bool,        # experience replay toggle - if False no buffer is kept
                 buffer_capacity: int,              # replay buffer size
                 eval_interval: int,                # how many epochs between model evaluations
                 n_eval_episodes: int,              # how many epochs to average the eval reward over
                 anneal_exp_param: bool,            # exploration parameter annealing toggle
                 anneal_timescale: int,             # half-life time for the exploration parameter in epochs
                 early_stopping_reward: int | None = None, # critical reward value for early stopping
                 ):
        
        self.gamma = gamma
        self.batch_size = batch_size
        self.eval_interval = eval_interval
        self.n_eval_episodes = n_eval_episodes
        self.target_network_update_time = target_network_update_time
        self.do_target_network = do_target_network
        self.init_exp_param = exp_param
        self.anneal_exp_param = anneal_exp_param
        self.anneal_timescale = anneal_timescale
        self.env = env


        if early_stopping_reward is None:   # if not specified, then take from env
            self.early_stopping_reward = env.spec.reward_threshold
        else:
            self.early_stopping_reward = early_stopping_reward

        # set various counters, lists, etc
        self.reset_counters()

        if do_experience_replay:
            # regular batch-based training with buffer
            self.train_step_func = self._train_step_with_buffer
        else:
            # "1-step" training without buffer
            self.train_step_func = self._train_step_without_buffer
        
        # init the model and agent
        n_inputs = env.observation_space.shape[0]
        n_actions = env.action_space.n
        self.model = DeepQModel(n_inputs=n_inputs, n_actions=n_actions)

        self.agent = DeepQAgent(env=env, 
                                policy=policy, 
                                exploration_parameter=exp_param, 
                                buffer_capacity=buffer_capacity)

        if self.do_target_network:
            self.target_network = DeepQModel(n_inputs=n_inputs, n_actions=n_actions)
            self.target_network.load_state_dict(self.model.state_dict())
        else:
            self.target_network = self.model

        self.optimizer = Adam(self.model.parameters(), lr=lr)
        
    def reset_counters(self):
        self.exp_param = self.init_exp_param
        self.episode_reward = 0 
        self.total_time = 0
        self.eval_rewards = []
        self.final_eval_rewards = None

    def exp_param_at_epoch(self, epoch: int) -> float:
        """ Exploration parameter at the given time (epoch) according to exponential decay function """
        if not self.anneal_exp_param:
            return self.init_exp_param
        
        return self.init_exp_param * 0.5**(epoch / self.anneal_timescale)
    
    def update_exp_param(self, epoch: int) -> None:
        """ Apply the exploration parameter annealing """
        new_exp_param = self.exp_param_at_epoch(epoch)        
        
        self.exp_param = new_exp_param
        self.agent.exploration_parameter = new_exp_param   

    def update_target_network(self):
        """ Updates the target network if applicable """
        if not self.do_target_network:
            return
        
        if self.total_time % self.target_network_update_time == 0:
            self.target_network.load_state_dict(self.model.state_dict())

    def _train_step_without_buffer(self):
        """ 
        train step using only empirical data. 
        note: not sure if this is correct actually
        """
        # take step
        state_before = self.agent.state

        reward, done = self.agent.act_on_model(self.model)

        new_state = self.agent.state

        # compute mse loss:
        q_values = self.model.forward(state_before) 
        expected_value = q_values.max(1)[0] # get the max q value (note: not the index)

        # next state value according to the (target) net
        new_state_q_value = self.target_network.forward(new_state).max(1)[0]
        new_state_value = self.gamma * new_state_q_value + reward

        loss = nn.MSELoss()(expected_value, new_state_value)

        
        return loss, done, reward

    def _train_step_with_buffer(self):
        # take step
        reward, done = self.agent.act_on_model(self.model)

        # sample from the buffer
        experiences = self.agent.buffer.sample(batch_size=self.batch_size)
        # fucky tensor thing to reshape
        batch = Experience(*zip(*experiences))

        # if the run is terminated, then the value is 0, so let's mask those out
        # also, concatenate the batch data (just for nicer casting)

        non_final_mask = torch.tensor([s is not None for s in batch.new_state], 
                                        dtype=torch.bool)
        
        non_final_next_states = torch.cat([torch.Tensor(s) for s in batch.new_state
                                                    if s is not None])
        
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action).type(dtype=torch.int64).unsqueeze(-1)
        reward_batch = torch.cat(batch.reward)

        # q-values for the actions in the batch under current policy
        q_values_batch = self.model.forward(state_batch)
        state_action_values = q_values_batch.gather(1, action_batch)

        # best q-values for the states in the batch
        next_state_best_q_values = torch.zeros(self.batch_size)
        next_state_best_q_values[non_final_mask] = self.target_network.forward(non_final_next_states).max(1).values
        
        expected_state_action_values = reward_batch + (self.gamma * next_state_best_q_values)

        loss = nn.MSELoss()(state_action_values, expected_state_action_values.unsqueeze(1))
        
        return loss, done, reward

    def train_model(self, num_epochs: int):
        """ Train the model for num_epochs episodes. """
        for epoch_i in tqdm(range(num_epochs), total=num_epochs, desc=self.episode_reward):
            done = False
            episode_reward = 0
            while not done:
                loss, done, reward = self.train_step_func()

                self.optimizer.zero_grad() # remove gradients from previous steps
                loss.backward()            # compute gradients
                nn.utils.clip_grad_value_(self.model.parameters(), 100) # clip gradients
                self.optimizer.step()      # apply gradients

                self.total_time += 1
                episode_reward += reward

                if self.anneal_exp_param:
                    self.update_exp_param(epoch=epoch_i)

                self.update_target_network()      # (only updates the model if applicable)

            # evaluate the model if it's time
            if epoch_i % self.eval_interval == 0:
                stop_early = self.evaluate_model()
                if stop_early:
                    return epoch_i

        final_results = self.get_eval_rewards(10*self.n_eval_episodes)
        self.save_eval_rewards(eval_rewards=final_results)
        return num_epochs


    def evaluate_model(self) -> bool:                
        """ Evaluation routine. Returns boolean flag for early stopping """
        eval_rewards = self.get_eval_rewards(num_epochs=self.n_eval_episodes)
        mean_reward = np.mean(eval_rewards)

        self.eval_rewards.append(mean_reward)

        # are we done?
        if mean_reward < self.early_stopping_reward:
            return False
        
        # we might be done, so double check:
        double_check_eval_rewards = self.get_eval_rewards(num_epochs=3*self.n_eval_episodes)
        double_check_mean = np.mean(double_check_eval_rewards)
        if double_check_mean < self.early_stopping_reward: 
            # not quite there, but we're probably close...
            
            # decrease learning rate so we don't overshoot?
            # self.optimizer.param_groups[0]['lr'] /= 2

            print("almost stopped early...")
            return False       

        # passed both tests, so we're done!
        final_rewards = self.get_eval_rewards(num_epochs=6*self.n_eval_episodes)
        all_final_rewards = np.concatenate([eval_rewards, double_check_eval_rewards, final_rewards])
        self.save_eval_rewards(eval_rewards=all_final_rewards)
        
        print("stopping early!")
        return True
    
    def save_eval_rewards(self, eval_rewards: np.ndarray) -> np.ndarray:
        # filename = "run_rewards.npy"
        # if os.path.exists(filename):
            # eval_rewards = np.vstack([np.load(filename), eval_rewards])

        self.final_eval_rewards = eval_rewards
        # np.save(file=filename, arr=eval_rewards)

    def get_eval_rewards(self, num_epochs: int) -> np.ndarray:
        """ Evaluates the model across a few epochs/episodes """  
        rewards = np.zeros((num_epochs))

        for i in range(num_epochs):
            state, _ = self.env.reset()
            done = False
            ep_reward = 0
            
            while not done:
                state = torch.from_numpy(state).unsqueeze(0)
                with torch.no_grad():
                    q_values = self.model.forward(state)

                action = Policy.GREEDY(q_values)
                state, reward, terminated, truncated, _ = self.env.step(action=action)
                ep_reward += reward

                done = terminated or truncated

            rewards[i] = ep_reward

        self.env.reset()
    
        return rewards

    def dqn_render_run(self, env: gym.Env, n_episodes_to_plot: int = 10) -> None:
        """Runs a single evaluation episode while rendering the environment for visualization."""

        env.reset(seed=4309)
        for _ in range(n_episodes_to_plot):
            state, _ = env.reset()  # Uses the newly created environment with render=human
            done = False
            
            while not done:
                state = torch.from_numpy(state).unsqueeze(0)
                with torch.no_grad():
                    q_values = self.model.forward(state)
                action = Policy.GREEDY(q_values)
                state, _, terminated, truncated, _ = env.step(action=action)

                done = terminated or truncated

            self.env.reset()
        

