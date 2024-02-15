import sys
import numpy as np

from algorithms import *
        
# Usage: python3 question5d.py <folder> <lT> <uT>
if __name__ == '__main__':

    # Folder to save the results
    folder = sys.argv[1]

    # Command line arguments determine the range of T values: [lT, uT]
    lT = int( sys.argv[2] ) 
    uT = int( sys.argv[3] )

    # Number of repetitions for each T
    repeat = 1000

    K = 2           # Number of arms
    p = [0.5, 0.6]  # Success probabilities of the arms

    mean_regret = []
    T_list = []

    for T in range(lT, uT+1):
        regret = []

        for i in range(repeat):
            # print("T: {:4d} -- Repetition: {:4d}/{}".format(T, i+1, repeat), end='\r')
            bb = BernoulliBandit(2, p)
            alg = UCBV(2)
            regret.append( simulate(bb, alg, T) )

        mean_regret.append( np.mean(regret) )
        T_list.append(T)
    
    np.savez(folder+'/mean_regret_{}_{}'.format(lT, uT), T_list=T_list, mean_regret=mean_regret)