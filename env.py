import numpy as np
import gym
from gym import spaces

# symmetric 2 player Markov game
# state: 2-dim array representing previous couple of choosen actions
class MarkovGame(gym.Env):
    def __init__(self, row_player, col_player, payoff_table, episode_len):
        super(MarkovGame, self).__init__()
        self.row_player = row_player
        self.col_player = col_player
        self.payoff_table = payoff_table
        self.episode_len = episode_len
        self.step_cntr = 0
        self.n_actions = payoff_table.shape[0]
        # All possible couple of actions i.e. states
        self.possible_states = [np.array([i, j], dtype=np.uint8) 
                                        for i in range(n_actions) 
                                        for j in range(n_actions)]
        self.n_states = len(self.possible_states)
        
        # Define observation space 
        self.observation_space = spaces.Box(low=0, high=n_actions-1,
                                            shape=(2,), dtype=np.uint8)
        # Define an action space
        self.action_space = spaces.Discrete(n_actions)
        
        self.index_to_state = {i : self.possible_states[i] 
                                   for i in range(self.n_states)}
        
    def reset(self):
        row_init_action = self.row_player.initial_action
        col_init_action = self.col_player.initial_action
        
        self.row_player_state = np.array([row_init_action, col_init_action], 
                                         dtype=np.uint8)
        self.col_player_state = np.array([col_init_action, row_init_action], 
                                         dtype=np.uint8)
        initial_state = self.row_player_state
        
        self.step_cntr = 0
        
        return initial_state
    
    def step(self, action):
        # column player action
        col_action = self.col_player.choose_action(self.col_player_state)
        
        # row player reward according to payoff table
        reward = self.payoff_table[action][col_action][0]
        
        # update states of the players
        self.row_player_state = np.array([action, col_action], dtype=np.uint8)
        self.col_player_state = np.array([col_action, action], dtype=np.uint8)
        state = self.row_player_state
        
        # check if terminal state is reached
        done = self.step_cntr == self.episode_len
        info = {}
        self.step_cntr += 1
        
        return state, reward, done, info
