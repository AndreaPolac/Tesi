from strategies import *

class StrategyLoader(object):
    def __init__(self, payoff_table, strategy_dict, GAMMA, ALPHA, n_states, n_actions):
        self.strategy_dict = strategy_dict
        self.n_actions = n_actions 
        self.payoff_table = payoff_table
        self.GAMMA = GAMMA
        self.ALPHA = ALPHA 
        self.n_states = n_states
        self.n_actions = n_actions
        
        self.strategies = {}

    def check_strategy(self, name):
        # check if strategies are available
        assert(name in self.strategy_dict)
        # check if the strategy can be applied to the current game
        assert(self.n_actions == self.payoff_table.shape[0])
        
    def init_strategy(self, name): 
        S = self.strategy_dict[name]
        strategy = S(self.GAMMA, self.ALPHA, self.n_states, self.n_actions)
        self.strategies[name] = strategy      
    
    def get_strategy(self, name):
        return self.strategies[name]
