class StrategyLoader(object):
    def __init__(self, payoff_table, name_1, name_2,
                 GAMMA, ALPHA, n_states, n_actions, input_dims):
        self.available_strategies = ['TIT_FOR_TAT', 'GROFMAN', 'CUNY']
        self.name_1 = name_1
        self.name_2 = name_2
        #self.file_1 = name_1 + '.py'
        #self.file_2 = name_2 + '.py'
        #self.strategies_path = strategies_path
        #self.strategy_names = os.listdir(self.strategies_path)
        self.n_actions = n_actions 
        self.payoff_table = payoff_table
       
        self.check_strategies()
        
        self.init_strategies()
        

    def check_strategies(self):
        # check if strategies are available
        assert(self.name_1 in self.available_strategies)
        assert(self.name_2 in self.available_strategies)
        
        # check if the strategy name is in the directory
        #assert(self.file_1 in self.strategy_names)
        #assert(self.file_2 in self.strategy_names)
        
        # check if the strategy can be applied to the current (symmetric) game
        assert(self.n_actions == self.payoff_table.shape[0])
        
        
    def init_strategies(self, GAMMA, ALPHA, n_states, n_actions, input_dims):             
        self.strategy_1 = Strategy(self.name_1, GAMMA, ALPHA, 
                                       n_states, n_actions, input_dims)
        self.strategy_2 = Strategy(self.name_2, GAMMA, ALPHA, 
                                       n_states, n_actions, input_dims)        
    
    def get_strategies(self):
        return self.strategy_1, self.strategy_2
    
    def get_payoffs(self):
        return self.payoff_table
