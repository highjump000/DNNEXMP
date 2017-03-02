# created by bohuai jiang
import numpy as np

class MazeEnv:

    def __init__(self):
        self.maze = np.load('maze.npy')
        self.my = len(self.maze)
        self.mx = len(self.maze[0])
        self.action = [0,1,2,3,4]#UP,DOWN,LEFT,RIGHT
        self.goalState = [0,0]
        self.startSate = [self.my-1,self.mx-1]

    #-- return total numbers of states
    def n_state(self):
        return self.mx*self.my

    #-- return number of action
    def n_action(self):
        return len(self.action)

    def reward(self,state):
        if state == self.goalState:
            return 100
        else:
            return 0

    def get_all_states(self):
        res= []
        for i in range(self.my):
            for j in range(self.mx):
                res.append([i,j])
        return res

    def get_all_reward(self):
        res = np.zeros([self.mx*self.my,1])
        res[0] = 100
        return res
    #-- return transition matrix for all states
    def getP(self,update):

        if update:
            print 'generating P...'
            P = np.zeros([self.n_state(),self.n_state()])
            states = self.get_all_states()
            for i in range(self.n_state()):
                for j in range(self.n_state()):
                    if self.isNeighbor(states[i],states[j]):
                        P[i,j] = 1
            np.save('P',P)
            print 'done'
        return np.load('P.npy')

    def getNeiboursState(self,state):
        P = self.getP(0)
        stats_table = self.get_all_states()
        index = self.my*state[0] + state[1]
        Neibours_index = np.nonzero(P[index]==1)[0]
        Nebours = []
        for i in range(len(Neibours_index)):
            Nebours.append(stats_table[Neibours_index[i]])
        return Nebours

    def getNeiboursIndex(self, index):
        P = self.getP(0)
        Neibours_index = np.nonzero(P[index] == 1)[0]
        return Neibours_index

    def isNeighbor(self,currentState,checkState):
        if self.isWall(checkState):
            return False
        elif np.abs(currentState[0]-checkState[0])== 1 and np.abs(currentState[1]-checkState[1])==0:
            return True
        elif np.abs(currentState[0] - checkState[0]) == 0 and np.abs(currentState[1] - checkState[1]) == 1:
            return True
        else:
            return False

    def isWall_idx(self,index):
        states_table = self.get_all_states();
        state = states_table[index]
        if self.maze[state[0]][state[1]] != 0:
            return False
        else:
            return True

    def isWall(self,state):
        if self.maze[state[0]][state[1]] != 0:
            return False
        else:
            return True