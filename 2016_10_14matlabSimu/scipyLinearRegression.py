import numpy as np
from IF import IF
import matplotlib.pyplot as plt
from IF_raw import IF_raw
#address = '/home/buhuai/Documents/IF/all_in_one.csv';
address = '/home/buhuai/Documents/all_in_one.csv';
read = IF_raw(address);
#read = IF(address);

x = read.getXB();
x = np.matrix(x)
y = read.getY();
y = np.matrix(y)
H,W = x.shape
out = np.zeros((H, W, 1))


xT = x.transpose();


xTx = xT*x

W = np.linalg.inv(xT*x)*(xT*y);

T = x*W;

MSE = np.sum(np.power(T-y,2))/len(T)

print MSE

profit = read.getPerformance(W);
a = read.getAction(W)
plt.plot(np.cumsum(profit))
plt.show()



#a = np.matrix(x)

#QW = np.linalg.solve(x,y);
