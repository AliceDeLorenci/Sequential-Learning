import sys
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

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
    

class FollowTheLeader():
    def __init__(self, K):
        self.K = K                      # Number of arms
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
            return np.argmax(self.cumulator/self.choices)
        

if __name__ == '__main__':

    lT = int( sys.argv[1] )
    uT = int( sys.argv[2] )

    repeat = 1000

    mean_regret = []
    T_list = []
    for T in range(lT, uT+1):
        regret = []
        for i in range(repeat):
            # print("T: {:4d} -- Repetition: {:4d}/{}".format(T, i+1, repeat), end='\r')
            bb = BernoulliBandit(2, [0.5, 0.6])
            ftl = FollowTheLeader(2)
            for t in range(T):
                k = ftl.chooseArm()
                reward = bb.pullArm(k)
                ftl.update(k, reward)

            regret.append( bb.regret(ftl.choices) )
        mean_regret.append( np.mean(regret) )
        T_list.append(T)
    
    np.savez('question1d/mean_regret_{}_{}'.format(lT, uT), T_list=T_list, mean_regret=mean_regret)