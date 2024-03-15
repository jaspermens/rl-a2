import numpy as np
from Functions import softmax, argmax

class BaseAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states,n_actions))

    def Q_network(self, s):
        # Returns the Q-values given a state s
        # NN
        return [Q_left, Q_right]
    
    def update(self, r_run):
        # Performs the back-propagation on the neural network
        # This is a batched update
        # The batch has the length of a single episode
        # Requires all the states of the run
        # 
        return
        
    def select_action(self, s, policy='egreedy', epsilon=None, temp=None):
        
        # s = [4 input params]
        # Q_network returns two Q-values: right and left

        if policy == 'greedy':
            # TO DO: Add own code --- DONE
            a = argmax(self.Q_network(s))
            
        elif policy == 'egreedy':
            if epsilon is None:
                raise KeyError("Provide an epsilon")
                
            # TO DO: Add own code --- DONE
            random_factor = np.random.random()
            if epsilon <= random_factor:
                a = argmax(self.Q_network(s))
            else:
                a = np.random.randint(4)

                 
        elif policy == 'softmax':
            if temp is None:
                raise KeyError("Provide a temperature")
                
            # TO DO: Add own code --- DONE
            probs = softmax(self.Q_network(s),temp)
            a = np.random.choice([0,1,2,3],1,p=probs)[0]
        return a
    
    # Loss function is als het ware de totale reward over een run - de run lengte
    # loss = r_tot - episode_length

    # 