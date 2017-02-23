# created by bohuai jiang
# on 2017/2/15

from numpy import random
import numpy as np
import matplotlib.pyplot as plt

# --- generated data ---　＃
k = 30
q = 0.6
a = np.array([random.normal(0,0.1,k)-q,random.normal(0,0.1,k)+q])
b = np.array([random.normal(0,0.1,k)+q,random.normal(0,0.1,k)+q])
c = np.array([random.normal(0,0.1,k)+q,random.normal(0,0.1,k)-q])
d = np.array([random.normal(0,0.1,k)-q,random.normal(0,0.1,k)-q])

plt.plot(a,'r.')
plt.plot(b,'r.')
plt.plot(c,'r.')
plt.plot(d,'r.')
plt.show()