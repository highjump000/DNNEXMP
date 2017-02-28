#created by bohuai jiang
#on 2017.2.10 10:00
from ENVIRONMENT import ENVIRONMENT
import numpy as np
import random

class EgreedyAgent:
    def __init__(self,ENVI=ENVIRONMENT(10,10)):
        self.envi = ENVI
        self.N_ACT = ENVI.NUMBER_OF_ACTION
    def run(self):
        T = 1000 # number of Try

        r = 0
        Q = self.envi.zero_init_Q();
        count = np.zeros([self.envi.height, self.envi.width, self.envi.NUMBER_OF_ACTION])
        e = 0.8  # exploration rate

        state = self.envi.I
        for t in range(T):
            a = self.greedyAction(e,state,Q)
            V = self.envi.Reward(state) # value function
            r = r + V
            Q[state[0]][state[1]][a] = (Q[state[0]][state[1]][a] * count[state[0]][state[1]][a] + V)/(count[state[0]][state[1]][a]+1)
            count[state[0]][state[1]][a] += 1

    def greedyAction(self,e,state,Q):
        if random.uniform(0,1) < e:
            return random.randint(0,self.N_ACT-1)
        else:
            return np.argmax(Q[state[0]][state[1]])