from strategies import *

class StrategyLoader(object):
    def __init__(self, payoff_table, name_1, name_2, strategy_dict,
                         GAMMA, ALPHA, n_states, n_actions, input_dims):
        # available strategies = ['TIT_FOR_TAT', 'GROFMAN', 'CUNY']
        self.strategy_dict = strategy_dict
        self.n_actions = n_actions 
        self.payoff_table = payoff_table
       
        self.check_strategies(name_1, name_2)
        
        self.init_strategies(name_1, name_2, GAMMA, ALPHA, 
                                    n_states, n_actions, input_dims)
        
        #self.file_1 = name_1 + '.py'
        #self.file_2 = name_2 + '.py'
        #self.strategies_path = strategies_path
        #self.strategy_names = os.listdir(self.strategies_path)

    def check_strategies(self):
        # check if strategies are available
        assert(self.name_1 in self.strategy_dict)
        assert(self.name_2 in self.strategy_dict)
        
        # check if the strategy can be applied to the current game
        assert(self.n_actions == self.payoff_table.shape[0])
        
        # check if the strategy name is in the directory
        #assert(self.file_1 in self.strategy_names)
        #assert(self.file_2 in self.strategy_names)
        
    def init_strategies(self, name_1, name_2, GAMMA, ALPHA, 
                                n_states, n_actions, input_dims): 
        S_1 = self.strategy_dict[name_1]
        S_2 = self.strategy_dict[name_2]            
        
        self.strategy_1 = S_1(GAMMA, ALPHA, n_states, 
                                       n_actions, input_dims)
        self.strategy_2 = S_2(GAMMA, ALPHA, n_states, 
                                       n_actions, input_dims)        
    
    def get_strategies(self):
        return self.strategy_1, self.strategy_2
    
    def get_payoffs(self):
        return self.payoff_table
        
