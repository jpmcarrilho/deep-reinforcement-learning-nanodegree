import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6,epsilon = 1,count = 0):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.epsilon = epsilon
        self.count = count
    def get_probs(self,state):
       """ obtains the action probabilities corresponding to e-greedy policy """
       policy_s = np.ones(self.nA) * self.epsilon/self.nA ##creates array with equal probability to all actions
       best_a = np.argmax(self.Q[state]) ## finds the action that maximex Q in the state used to call this function
       policy_s[best_a] = 1 - self.epsilon + (self.epsilon/self.nA) ## uses the equation to calculate the e-greedy probability
        ##in the the best action
       return policy_s
    
    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment
        - policy_p: vector with the probabilities of each possible action
        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        
        return np.random.choice(np.arange(self.nA), p= Agent.get_probs(self,state)) \
                                            if state in self.Q else np.random.choice(np.arange(self.nA))

    def step(self, state, action, reward, next_state, done, alpha =1, gamma =1):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """ 
        self.count += 1
        self.epsilon = 0.001
        
        old_Q = self.Q[state][action]
        
        if not done:
            self.Q[state][action] = old_Q + alpha*(reward + gamma*(np.dot( Agent.get_probs(self,state),self.Q[next_state])) - old_Q)
            state = next_state
        else: 
            self.Q[state][action] = old_Q + alpha*(reward - old_Q)
        