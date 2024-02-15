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
    
    def regret(self, n):
        T = np.sum(n)
        return T * self.optimal_prob - np.sum( [n_k*self.success_probs[k] for k, n_k in enumerate(n)] )
    

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
        

if __name__ == '__main__':

    folder = "question3d"

    lT = int( sys.argv[1] )
    uT = int( sys.argv[2] )

    p = [0.5, 0.6]

    repeat = 10 # 1000

    mean_regret = []
    T_list = []
    for T in range(lT, uT+1):
        regret = []
        for i in range(repeat):
            # print("T: {:4d} -- Repetition: {:4d}/{}".format(T, i+1, repeat), end='\r')
            bb = BernoulliBandit(2, p)
            alg = UCBV(2)
            for t in range(T):
                k = alg.chooseArm()
                reward = bb.pullArm(k)
                alg.update(k, reward)

            regret.append( bb.regret(alg.n) )
        mean_regret.append( np.mean(regret) )
        T_list.append(T)
    
    np.savez(folder+'/mean_regret_{}_{}'.format(lT, uT), T_list=T_list, mean_regret=mean_regret)