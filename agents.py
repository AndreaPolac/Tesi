class TabularQLearningAgent(object):
    def __init__(self, GAMMA, ALPHA, epsilon, n_states,
                 n_actions, input_dims, eps_min=0.01, eps_dec=5e-7):
        self.GAMMA = GAMMA
        self.ALPHA = ALPHA
        self.epsilon = epsilon 
        self.eps_min = eps_min 
        self.eps_dec = eps_dec 
        self.n_actions = n_actions
        self.n_states = n_states
        self.input_dims = input_dims
        
        self.start = True
        self.initial_action = 0
        
        self.Q = {}
        self.init_Q()
    
    def init_Q(self):
        for state in range(self.n_states):
            self.Q[state] = {}
            for action in range(self.n_actions):
                self.Q[state][action] = 0.0
                
    def choose_action(self, state):
        if self.start:
            self.start = False
            return self.initial_action
        else:
            if np.random.random() > self.epsilon: # greedy action
                state_idx = toIndex(state)
                action = max(self.Q[state_idx], key=self.Q[state_idx].get)
            else:
                action = np.random.choice([i for i in range(self.n_actions)])
        return action
    
    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
                        if self.epsilon > self.eps_min else self.eps_min
               
    def learn(self, state, action, reward, state_):
        a_max = max(self.Q[state], key=self.Q[state].get)
        
        self.Q[state][action] += self.ALPHA*(reward + 
                                         self.GAMMA*self.Q[state_][a_max] -
                                         self.Q[state][action])
        self.decrement_epsilon()
        
class TIT_FOR_TAT(TabularQLearningAgent):
    def __init__(self, GAMMA, ALPHA, epsilon, n_states,
                 n_actions, input_dims, eps_min=0.01, eps_dec=5e-7):
        super().__init__(GAMMA, ALPHA, epsilon, n_states, n_actions, 
                                            input_dims, eps_min, eps_dec)
        self.initial_action = 0
        
    def choose_action(self, state):
        if self.start:
            self.start = False
            return self.initial_action
        else:
            return state[1]

    def learn(self, state, action, reward, state_):
        a_max = max(self.Q[state], key=self.Q[state].get)
        self.Q[state][action] += self.ALPHA*(reward + 
                                         self.GAMMA*self.Q[state_][a_max] -
                                         self.Q[state][action])
        
class P_COOPERATOR(TabularQLearningAgent):
    def __init__(self, GAMMA, ALPHA, epsilon, n_states,
                 n_actions, input_dims, eps_min=0.01, eps_dec=5e-7):
        super().__init__(GAMMA, ALPHA, epsilon, n_states, n_actions, 
                                            input_dims, eps_min, eps_dec)
        self.initial_action = 0 # cooperate
                        
    def choose_action(self, state):
        if self.start:
            self.start = False
            return self.initial_action
        else: # cooperatw with probability p=0.7
            return np.random.choice(2, p=[0.7, 0.3])
               
    def learn(self, state, action, reward, state_):
        a_max = max(self.Q[state], key=self.Q[state].get)
        
        self.Q[state][action] += self.ALPHA*(reward + 
                                         self.GAMMA*self.Q[state_][a_max] -
                                         self.Q[state][action])
        
class RandomAgent(TabularQLearningAgent):
    def __init__(self, GAMMA, ALPHA, epsilon, n_states,
                 n_actions, input_dims, eps_min=0.01, eps_dec=5e-7):
        super().__init__(GAMMA, ALPHA, epsilon, n_states, n_actions, 
                                            input_dims, eps_min, eps_dec)
        self.initial_action = 0  # Cooperate
                
    def choose_action(self, state):
        action = np.random.choice([i for i in range(self.n_actions)])
        return action
               
    def learn(self, state, action, reward, state_):
        a_max = max(self.Q[state], key=self.Q[state].get)
        
        self.Q[state][action] += self.ALPHA*(reward + 
                                         self.GAMMA*self.Q[state_][a_max] -
                                         self.Q[state][action])
