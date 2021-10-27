## METTERE ATTRIBUTO NOME IN MODO DA CARICARE CON IL LOADER
# (NAME, GAMMA.....)
class Strategy(object):
    def __init__(self, GAMMA, ALPHA, n_states, n_actions, input_dims):
        self.GAMMA = GAMMA
        self.ALPHA = ALPHA
         
        self.n_actions = n_actions
        self.n_states = n_states
        self.input_dims = input_dims
        
        self.start = True
        self.initial_action = 0
        self.interaction_history = []   # lis of tuples (C,D) (D,D) ....
        
        self.Q = {}
        self.init_Q()
        
    def init_Q(self):
        for state in range(self.n_states):
            self.Q[state] = {}
            for action in range(self.n_actions):
                self.Q[state][action] = 0.0
    
    def learn(self, state, action, reward, state_):
        a_max = max(self.Q[state], key=self.Q[state].get)
        self.Q[state][action] += self.ALPHA*(reward + 
                                         self.GAMMA*self.Q[state_][a_max] -
                                         self.Q[state][action])
        

class TIT_FOR_TAT(Strategy):
    def __init__(self, GAMMA, ALPHA, n_states, n_actions, input_dims):
        super().__init__(GAMMA, ALPHA, n_states, n_actions, input_dims)
        self.initial_action = 0
        
    def choose_action(self, state):
        if self.start:
            self.start = False
            return self.initial_action
        else:
            return state[1]
        
        
class GROFMAN(Strategy):
    def __init__(self, GAMMA, ALPHA, n_states, n_actions, input_dims):
        super().__init__(GAMMA, ALPHA, n_states, n_actions, input_dims)
        self.initial_action = 0
        
    def choose_action(self, state):
        if self.start:
            self.start = False
            action = self.initial_action
        else:
            if state[0] != state[1]:
                action = np.random.choice(2, p=[2/7, 1 - 2/7])
            else:
                action = 0
            
        return action
   
        
class CUNY(Strategy):
    def __init__(self, GAMMA, ALPHA, n_states, n_actions, input_dims):
        super().__init__(GAMMA, ALPHA, n_states, n_actions, input_dims)
        self.initial_action = 0
        self.cntr = 0
        self.defection = False
    
    def choose_action(self, state):
        if self.cntr < 10:
            action = 0
        else:
            if state[1] == 1:
                action = 1
                self.defection = True
            else:
                action = 0
                
            action = 1 if self.defection else 0
            
        return action 
    
Strategy_Dict = {'TIT_FOR_TAT' : TIT_FOR_TAT,
                'GROFMAN' : GROFMAN,
                'CUNY' : CUNY}


