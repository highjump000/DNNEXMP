#created by bohuai jiang
#on 2016/12/19
import matplotlib.pyplot as plt
import numpy as np
import DeepQ as dq
import NN_network as nn
import TradingFun as TF


from RandomWalk import RandomWalk


def getXY(period, p):
    datalength = len(p);
    datalen = datalength - period - 1;
    X = np.zeros([datalen, period]);
    Xb = np.ones([datalen, period + 1]);
    T = np.zeros([datalen, 1]);
    for i in range(period, datalength - 1):
        for j in xrange(period):
            X[i - period][j] = p[i - j];
            Xb[i - period][j] = p[i - j];
        T[i - period] = p[i + 1];
    return np.float32(X), np.float32(T);


# load real data
address= '../../../Documents/RB/all_in_one.csv'
data = TF.csvread(address);

dl = 100

X,T= getXY(3,data[:,-1]);
X = X[1:dl,:]
T = T[1:dl]

plt.plot(T)
plt.show()
# created a deepQ
#model = TDQ(X,T);

#model.goTDQ();
dq.DQ(X,T)