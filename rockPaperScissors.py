import numpy as np

class EWAPlayer():
    def __init__(self, M=3, eta=1):
        self.M = M                      # Number of actions
        self.eta = eta                  # EWA parameter
        self.p = np.ones(M) / M         # Distribution over actions
        self.i = -1                     # Action

    def update(self, loss):
        self.p = self.p * np.exp(-self.eta * loss) / np.sum(self.p * np.exp(-self.eta * loss))

    def play(self):
        self.i = np.random.choice(self.M, p=self.p)
        return self.i, self.p
    
class OptimalAdversary():
    def __init__(self, N=3):
        self.N = N                      # Number of actions
        self.j = -1                     # Action

    def play(self, p, L):
        expectedReward = p.reshape((1,3)) @ L # expected adversary reward for each adversary action

        # obs1: it is important to randomly break the ties otherwise we will have a strong bias towards the first action
        # obs2: printing intermediate variables a precision problem was detected that was biasing the results
        # argmax = np.flatnonzero( np.abs( expectedReward - expectedReward.max() ) < 1e-18 )
        argmax = np.flatnonzero( expectedReward == expectedReward.max() )
        self.j = np.random.choice( argmax ) # adversary's action

        # print("p:", p)
        # print()
        # print(argmax)
        # print("j:", self.j)

        return self.j

class RPSFullInformation():
    
    def __init__(self, player, adversary):

        self.options = {0: "rock", 1: "paper", 2: "scissors"}    # equivalence between indexes and actions
        self.L = np.array( [[0, 1, -1], 
                            [-1, 0, 1], 
                            [1, -1 , 0]] ) # loss matrix
                
        self.player = player
        self.adversary = adversary

        self.playerLoss = []         # player's loss
        self.pHistory = []           # keep track of player's action distribution over time

        self.t = 0
    
    def play(self):
        self.t += 1

        i, p = self.player.play()
        self.pHistory.append(p)

        j = self.adversary.play(p, self.L)
        self.playerLoss.append( self.L[i, j] )

        self.player.update(self.L[:, j])
        

class EXP3Player():
    def __init__(self, K=3, eta=1):
        self.K = K                      # Number of actions
        self.eta = 1                    # EXP3 parameter
        self.p = np.ones(K) / K         # Distribution over actions
        self.a = -1                     # Action
    
    def update(self, loss):
        l_hat = np.zeros(self.K)
        l_hat[self.a] = loss / self.p[self.a]
        self.p = self.p * np.exp(-self.eta * l_hat) / np.sum(self.p * np.exp(-self.eta * l_hat))

    def play(self):
        self.a = np.random.choice(self.K, p=self.p)
        return self.a, self.p

        
class RPSBandit():
    
    def __init__(self, player, adversary):

        self.options = {0: "rock", 1: "paper", 2: "scissors"}    # equivalence between indexes and actions
        self.L = np.array( [[1/2, 1, 0], 
                            [0, 1/2, 1], 
                            [1, 0, 1/2]] ) # loss matrix
                
        self.player = player
        self.adversary = adversary

        self.playerLoss = []         # player's loss
        self.pHistory = []           # keep track of player's action distribution over time

        self.adversaryLoss = []      # adversary's loss
        self.qHistory = []           # keep track of adversary's action distribution over time

        self.t = 0
    
    def play(self):
        self.t += 1

        i, p = self.player.play()
        self.pHistory.append(p)

        j, q = self.adversary.play()
        self.qHistory.append(q)

        self.playerLoss.append( self.L[i, j] )
        self.adversaryLoss.append( self.L[j, i] )

        self.player.update( self.L[i, j] )
        self.adversary.update( self.L[j, i] )
        