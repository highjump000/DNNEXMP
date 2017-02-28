#created by bohuai jiang
#on 2/10/2017 9:56

from ENVIRONMENT import ENVIRONMENT

env = ENVIRONMENT(10,10)
env.show()
Q = env.zero_init_Q()
print Q[0][0]