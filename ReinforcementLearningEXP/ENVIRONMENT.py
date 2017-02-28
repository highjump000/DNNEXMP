#created by bohuai jiang
# on 2/10/2017 09:47
# create agent environment
import numpy as np

class ENVIRONMENT:
    def __init__(self,width,height):
        self.width = width
        self.height = height
        self.I = [0,0] # initial state
        self.O = [width-1,height-1] # end state
        self.NUMBER_OF_ACTION = 4
    # show environment
    def show(self):
        for i in range(self.height):
            line = ''
            for j in range(self.width):
                if i == self.I[0] and j == self.I[1]:
                    line += '[I]'
                else:
                    if i == self.O[0] and j == self.O[1]:
                        line += '[O]'
                    else:
                        line += '[ ]'
            print line
    def get
    def getNextState(self,state,a):
        if a == 0: #go up
            if state[0] < self.height-1:
                state[0] += 1
        if a == 1: #go down
            if state[0] > 0:
                state[0] -= 1
        if a == 2: #go left
            if state[1] < self.width-1:
                state[1] += 1
        if a == 3: #go right
            if state[1] > 0:
                state[0] -= 1
        return state

    def Reward(self,state):
        if state == self.O:
            return 100
        else:
            return 0
    def zero_init_Q(self):
        Q = np.zeros([self.height,self.width,self.NUMBER_OF_ACTION])
        return Q