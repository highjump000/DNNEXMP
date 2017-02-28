#created by bohuai jiang
#on 2017/2/10 3:47
from ENVIRONMENT import ENVIRONMENT
import numpy as np

class MDP_PolicyIteration:
    def __init__(self,ENV = ENVIRONMENT(10,10)):
        self.env = ENV
        self.w = ENV.width
        self.h = ENV.height
        self.n_act = ENV.NUMBER_OF_ACTION
    def run(self):
        episode = 10000
        V = np.zeros([self.w,self.h])
        pi = np.zeros([self.w,self.h,float(1/self.n_act)])
        #P # transition matrix here all Probability are same set as 1
        for t in range(episode):
            for i in range(self.h): # map's hight
                for j in range(self.w): # map's width
                    sum_pi_v = 0
                    for k in range(self.n_act):
                        sum_pi_v += pi[i][j][k]
                    V[i][j] = sum_pi_v*(1/float(t)*self.env.Reward([i,j])+float(t-1)/float(t)*V(i,j))
