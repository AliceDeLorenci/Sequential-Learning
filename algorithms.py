import numpy as np

def simulate(bandit, algorithm, T):
        
        actions = []

        for t in range(T):
            k = algorithm.chooseArm()
            actions.append(k)
            reward = bandit.pullArm(k)
            algorithm.update(k, reward)

        return bandit.regret( actions )

class BernoulliBandit():
    def __init__(self, K, success_probs):
        self.K = K                                  # Number of arms
        self.success_probs = success_probs          # Success probabilities of each arm
        self.optimal = np.argmax(success_probs)     # Index of the optimal arm
        self.optimal_prob = np.max(success_probs)   # Success probability of the optimal arm
        self.suboptimality_gap = self.optimal_prob - self.success_probs # Suboptimality gap of each arm
    
    def pullArm(self, k):
        return np.random.rand() < self.success_probs[k] # Return 1 with probability success_probs[k]
    
    def regret(self, actions):
        """
        Calculate the cummulative pseudo-regret.

        Args:
            actions (list): List of actions chosen by the algorithm.
        """
        instant_regret = self.suboptimality_gap[actions]
        return np.cumsum(instant_regret)
    
class FollowTheLeader():
    def __init__(self, K):
        self.K = K                      # Number of arms
        self.n = np.zeros(K)            # Number of times each arm was chosen
        self.cumulator = np.zeros(K)    # Cumulative reward for each arm
        self.t = 0
    
    def update(self, k, reward):
        self.t += 1
        self.n[k] += 1
        self.cumulator[k] += reward
    
    def chooseArm(self):
        if self.t < self.K:
            return self.t
        else:
            return np.argmax(self.cumulator/self.n)
        
class UCB():
    def __init__(self, K, s2, xi=1.1):
        self.K = K                      # Number of arms
        self.s2 = s2                    # Rewards should be s2-subgaussian
        self.xi = xi                    
        self.n = np.zeros(K)            # Number of times each arm was chosen
        self.cumulator = np.zeros(K)    # Cumulative reward for each arm
        self.t = 0
    
    def update(self, k, reward):
        self.t += 1
        self.n[k] += 1
        self.cumulator[k] += reward
    
    def chooseArm(self):
        if self.t < self.K:
            return self.t
        else:
            # np.log(self.t+1) since t is starting from 0
            ucb = self.cumulator/self.n + np.sqrt(2*np.log(self.t+1)*self.xi*self.s2/self.n)
            return np.argmax(ucb)
        
class UCBV():
    def __init__(self, K, xi=1.1, c=1):
        self.K = K                      # Number of arms
        self.xi = xi     
        self.c = c               
        self.n = np.zeros(K)            # Number of times each arm was chosen
        self.cumulator = np.zeros(K)    # Cumulative reward for each arm
        self.variance = np.zeros(K)     # Empirical variance of each arm
        self.t = 0
    
    def update(self, k, reward):

        if self.t >= self.K:        
            mean_prev = self.cumulator[k]/self.n[k]
            mean_new = ( self.cumulator[k] + reward ) / ( self.n[k] + 1 )
            self.variance[k] = ( self.n[k]*self.variance[k] + ( reward - mean_prev )*( reward - mean_new ) ) / ( self.n[k]+1 )
        
        self.t += 1
        self.n[k] += 1
        self.cumulator[k] += reward
    
    def chooseArm(self):
        if self.t < self.K:
            return self.t
        else: 
            # np.log(self.t+1) since t is starting from 0
            ucbv = self.cumulator/self.n + np.sqrt( 2*np.log(self.t+1)*self.xi*self.variance/self.n ) + 3*self.c*self.xi/self.n
            return np.argmax(ucbv)
        
class ThompsonSampling():
    def __init__(self, K):
        self.K = K                      # Number of arms
        self.n = np.zeros(K)            # Number of times each arm was chosen
        self.cumulator = np.zeros(K)    # Cumulative reward for each arm
        self.t = 0
    
    def update(self, k, reward):
        self.t += 1
        self.n[k] += 1
        self.cumulator[k] += reward
    
    def chooseArm(self):
        if self.t == 0:
            theta = np.random.rand(self.K)
        else:
            alpha = self.cumulator + 1
            beta = self.n - self.cumulator + 1
            theta = np.random.beta(alpha, beta)
        return np.argmax(theta)
    
