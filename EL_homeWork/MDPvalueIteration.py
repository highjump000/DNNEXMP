# created by bohuai jiang
# on 2017 2 28

import numpy as np
from MazeEnvironment import MazeEnv

#### load environment ####
ME = MazeEnv()
n_state = ME.n_state()
n_action = ME.n_action()
mx = ME.mx
my = ME.my
########### MDP ###########
#-- parameters initialization --#
gamma = 0.9
V = np.zeros([my*mx,1])
#pi = np.ones([my,mx,n_action])*(1/float(n_action))
iteration = 1000
P = ME.getP(0)
R = ME.get_all_reward()

init_state = ME.goalState

for i in xrange(n_state):
    if not ME.isWall_idx(i):
        p = 1
        nebours = ME.getNeiboursIndex(i)
        Value_list = []
        for j in range(len(nebours)):
            n_index = nebours[j]
            Value = gamma*float((R[i][0] + p*V[n_index]))
            Value_list.append(Value)
        V[i] = np.max(Value_list)
        #print V[i]

print V[1]
