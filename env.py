class GameEnvironment(object):
    def __init__(self, column_player, T=5, R=3, P=1, S=0):
        self.action_space = [0, 1] # 0:cooperate 1:defect
        self.observation_space = {0 : np.array([0, 0], dtype=np.uint8), # CC
                                  1 : np.array([0, 1], dtype=np.uint8), # CD
                                  2 : np.array([1, 0], dtype=np.uint8), # DC
                                  3 : np.array([1, 1], dtype=np.uint8)} # DD
        
        self.payoff_table = np.array([[(R,R), (S,T)],
                                      [(T,S), (P,P)]])
        
        self.column_player = column_player
        
        
    def reset(self, row_player):
        self.state = np.array([row_player.initial_action, 
                               self.column_player.initial_action])
        self.column_state = np.array([self.state[1], self.state[0]])
        
        return self.state
    
    
    def step(self, action):
        column_action = self.column_player.choose_action(self.column_state)
        
        reward = self.payoff_table[action][column_action][0]
        
        self.state = np.array([action, column_action], dtype=np.uint8)
        self.column_state = np.array([self.state[1], self.state[0]])
        
        return reward, self.state
