import sys
import numpy as np
import matplotlib.pyplot as plt

class BernoulliBandit():
    def __init__(self, K, success_probs):
        self.K = K
        self.success_probs = success_probs
        self.optimal = np.argmax(success_probs)
        self.optimal_prob = np.max(success_probs)
    
    def pullArm(self, k):
        return np.random.rand() < self.success_probs[k]
    
    def regret(self, choices):
        T = np.sum(choices)
        return T * self.optimal_prob - np.sum( [n_k*self.success_probs[k] for k, n_k in enumerate(choices)] )
    

class UCB():
    def __init__(self, K, s2, xi=1.1):
        self.K = K                      # Number of arms
        self.s2 = s2                    # Rewards should be s2-subgaussian
        self.xi = xi                    
        self.choices = np.zeros(K)      # Number of times each arm was chosen
        self.cumulator = np.zeros(K)    # Cumulative reward for each arm
        self.t = 0
    
    def update(self, k, reward):
        self.t += 1
        self.choices[k] += 1
        self.cumulator[k] += reward
    
    def chooseArm(self):
        if self.t < self.K:
            return self.t
        else:
            ucb = self.cumulator/self.choices + np.sqrt(2*np.log(self.t)*self.xi*self.s2/self.choices)
            return np.argmax(ucb)
        

if __name__ == '__main__':

    s2 = float( sys.argv[1] )

    T = 1000
    repeat = 1000

    mean_regret = []
    s2_list = []

    regret = []
    for i in range(repeat):
        # print("T: {:4d} -- Repetition: {:4d}/{}".format(T, i+1, repeat), end='\r')
        bb = BernoulliBandit(2, [0.5, 0.6])
        alg = UCB(2, s2)
        for t in range(T):
            k = alg.chooseArm()
            reward = bb.pullArm(k)
            alg.update(k, reward)

        regret.append( bb.regret(alg.choices) )
    mean_regret.append( np.mean(regret) )
    s2_list.append(s2)
    
    np.savez('question2h1/mean_regret_{}'.format(s2), s2_list=s2_list, mean_regret=mean_regret)