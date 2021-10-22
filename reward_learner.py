class RewardLearner(object):
    def __init__(self, expert, precision, n_states, n_actions, input_dims, 
                 num_trajectories, episode_len, epsilon,
                 epsilon_min, eps_dec, ALPHA, GAMMA):
        self.num_trajectories = num_trajectories
        self.episode_len = episode_len
        self.precision = precision
        self.ALPHA = ALPHA 
        self.GAMMA = GAMMA
        self.n_states = n_states
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.eps_dec = eps_dec
        self.weights = []
        self.margins = []
        self.cntr = 0
        
        self.expert = expert
        self.expert_FE = self.featureExpectation(expert)
        
        self.policies = [] 
        self.FEs = []
        # for the projection step
        self.FEs__ = []

        self.lambdas = None
        self.lambdas_FE = None
                        
    # define the initial random policy 
    def init_policy(self):
        random_policy = RandomAgent(self.GAMMA, self.ALPHA, 
                                    self.epsilon, self.n_states, 
                                    self.n_actions, self.input_dims,
                                    self.epsilon_min, self.eps_dec)       
        
        FE_0 = self.featureExpectation(random_policy)
        self.policies.append(random_policy)
        self.FEs.append(FE_0)
        self.weights.append(np.zeros(2))    # dummy weight
        self.margins.append(1)              # dummy margin
        
    # feature expectation for a given policy/agent    
    def featureExpectation(self, policy):
        feature_sum = np.zeros(self.input_dims)  
        for i in range(self.num_trajectories):
            state = env.reset(row_player=policy)
            for t in range(self.episode_len):
                action = policy.choose_action(state)
                reward, state_ = env.step(action)      
                features = sigmoid(state)
                feature_sum += (GAMMA**t)*features
                state = state_
        feature_expectation = feature_sum/self.num_trajectories
        
        return feature_expectation
    
    # projection method
    def optimization_step(self):
        if self.cntr == 1:
            self.FEs__.append(self.FEs[0])
            w = self.expert_FE - self.FEs[0]
            margin = LA.norm(w, 2) 
            self.weights.append(w)
            self.margins.append(margin)  
        else:
            A = self.FEs__[-1]
            B = self.FEs[-1] - A
            C = self.expert_FE - A
            FE__ = A + (np.dot(B,C)/np.dot(B,B))*B
            w = self.expert_FE - FE__
            margin = LA.norm(w,2)
            
            self.FEs__.append(FE__)            
            self.weights.append(w)
            self.margins.append(margin)
        
        return w, margin
    
    def termination_condition(self):
        condition = self.margins[-1] < self.precision
        
        return condition
    
    # with a TabularQLearningAgent with epsilon greedy action choice
    def learn_new_policy(self, weight):
        agent = TabularQLearningAgent(self.GAMMA, self.ALPHA, self.epsilon, 
                                      self.n_states, self.n_actions, 
                                      self.input_dims, self.epsilon_min, 
                                      self.eps_dec)
        for i in range(self.num_trajectories):
            state = env.reset(row_player=agent)
            
            for j in range(self.episode_len):
                action = agent.choose_action(state)
                reward, state_ = env.step(action)
                # reward estimation
                features = sigmoid(state)
                reward = np.dot(weight, features)            
                # Q learning step 
                state_idx = toIndex(state)
                state_idx_ = toIndex(state_)
                agent.learn(state_idx, action, reward, state_idx_)
                state = state_
                
        return agent
        
    def learn(self):
        if self.cntr == 0:
            self.init_policy()
        else:
            w, margin = self.optimization_step()
            # policy for new reward
            new_policy = self.learn_new_policy(w)
            FE_new_policy = self.featureExpectation(new_policy)
            self.policies.append(new_policy)
            self.FEs.append(FE_new_policy)
        self.cntr += 1
    
            
    def mixed_policy(self):        
        # we consider the last d=(k+1) policies where k=2=input_dim
        d = self.input_dims + 1
        choosen_policies_FE = self.FEs[-d : ]
        
        M = np.zeros((d, d))
        c = np.zeros(d)
        for i in range(d):
            c[i] = np.dot(choosen_policies_FE[i], self.expert_FE)
            for j in range(d):
                M[i][j] = np.dot(choosen_policies_FE[i], choosen_policies_FE[j])
        
        P = cvxopt.matrix(2*M)
        q = cvxopt.matrix(-2*c)
        G = cvxopt.matrix(-np.eye(d))
        h = cvxopt.matrix(np.zeros(d))
        A = cvxopt.matrix(np.ones(d).astype('float').T)
        b = cvxopt.matrix(np.ones(1))
        
        # Compute the solution using the quadratic solver
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        
        self.lambdas = np.ravel(sol['x'])
        self.lambda_FE = np.sum([e[0]*e[1] for e in \
                                        zip(self.lambdas, choosen_policies_FE)])
