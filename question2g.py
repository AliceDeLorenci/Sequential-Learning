import sys
import numpy as np

from algorithms import *
        
# Usage: python3 question2g.py <folder> <lT> <uT> <p1> <p2> <s2>
# obs: the arguments <p1>, <p2> and <s2> git stare optional
if __name__ == '__main__':

    # Folder to save the results
    folder = sys.argv[1]

    # Command line arguments determine the range of T values: [lT, uT]
    lT = int( sys.argv[2] ) 
    uT = int( sys.argv[3] )

    # Success probabilities of the arms
    if len(sys.argv) > 4:
        p1 = float( sys.argv[4] )
        p2 = float( sys.argv[5] )
    else:
        p1 = 0.5
        p2 = 0.6
    
    # Subgaussian parameter
    if len(sys.argv) > 6:
        s2 = float( sys.argv[6] )
    else:
        s2 = 1/4

    # Number of repetitions for each T
    repeat = 1000

    K = 2           # Number of arms
    p = [p1, p2]    # Success probabilities of the arms

    mean_regret = []
    T_list = []

    for T in range(lT, uT+1):
        regret = []

        for i in range(repeat):
            # print("T: {:4d} -- Repetition: {:4d}/{}".format(T, i+1, repeat), end='\r')
            bb = BernoulliBandit(2, p)
            alg = UCB(2, s2)
            regret.append( simulate(bb, alg, T) )

        mean_regret.append( np.mean(regret) )
        T_list.append(T)
    
    np.savez(folder+'/mean_regret_{}_{}'.format(lT, uT), T_list=T_list, mean_regret=mean_regret)