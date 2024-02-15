import sys
import numpy as np

from algorithms import *
        
# Usage: python3 question2h.py <folder> <s2> <p1> <p2>
if __name__ == '__main__':

    # Folder to save the results
    folder = sys.argv[1]

    # Subgaussian parameter
    s2 = float( sys.argv[2] )

    # Success probabilities of the arms
    p1 = float( sys.argv[3] )
    p2 = float( sys.argv[4] )

    # Number of repetitions for each s2
    repeat = 1000

    K = 2           # Number of arms
    p = [p1, p2]    # Success probabilities of the arms
    T = 1000        # Time horizon  

    mean_regret = []
    s2_list = []


    regret = []

    for i in range(repeat):
        # print("T: {:4d} -- Repetition: {:4d}/{}".format(T, i+1, repeat), end='\r')
        bb = BernoulliBandit(2, p)
        alg = UCB(2, s2)
        regret.append( simulate(bb, alg, T) )

    mean_regret.append( np.mean(regret) )
    s2_list.append(s2)
    
    np.savez(folder+'/mean_regret_{}'.format(s2), s2_list=s2_list, mean_regret=mean_regret)