import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import TradingFun as TF;

from IF_raw import IF_raw
from RandomWalk import RandomWalk
from PureReg import PureReg


import NN_network as nn

# created by bohuai jiang
# 2016 12 26
# working on real data

def evalFunc(X,T,w,b,cost):
    X = nn.featureScaling(X, 1, -1);
    Y = nn.MLP(X, w, b);
    Y = sess.run(Y);
    Y = nn.defeatureScaling(Y, 1, -1, T);
    a_nn = np.argmax(Y, axis=1) - 1;
    profit = TF.TSL(a_nn, P, 15, 2, 0);
    return -1*sum(profit)

def getXY(period,p):
        datalength = len(p);
        datalen = datalength-period-1;
        X = np.zeros([datalen,period]);
        Xb = np.ones([datalen,period+1]);
        T = np.zeros([datalen,1]);
        for i in range(period,datalength-1):
            for j in xrange(period):
                X[i-period][j] = p[i-j];
                Xb[i-period][j] = p[i - j];
            T[i-period] = p[i+1];
        return np.float32(X),np.float32(T);

########## load data ##########
n_features = 30;
update = 0;
train = 0.5;


address= '/home/yohoo/Desktop/ubuntu_doc/Documents/RB/all_in_one.csv';
data = TF.csvread(address);
X,P= getXY(n_features,data[:,-1]);
data_len = len(X);
splitV = int(data_len*train);
if update:
    T = TF.getQtable(X,P,15,2)
    np.save('Tcu',T);

T = np.load('Tcu.npy')

# train set
train_X = X[0:splitV,:];
train_P = P[0:splitV];

train_T = T[0:splitV,:];

# test set
test_X = X[splitV:-1,:];
test_P = P[splitV:-1];

test_T = T[splitV:-1,:];

a = np.argmax(T,axis=1)-1;

#P = [100,90,80,69];
#a = [1  , 0, 0, 0]

profit = TF.TSL(a,P,15,2,0);
#profit = TF.MarketState(a,P,2)


TF.plotPerformance(profit,'org')

plt.figure().suptitle('price',fontsize = 20);
plt.plot(P)

########## create MLP ############

# generate stable initilization

# -- train -- #
sess = tf.InteractiveSession();

w,b, s, p = nn.MLCNN_init_out([1,30,1],3,[1],[[1,1]]);

print X.shape

w,b,sess = nn.train(train_X,train_T,w,b,[s,p],sess);

# -- in sample -- #

train_X = nn.featureScaling(train_X,1,-1);

xh,xw = nn.getWH(train_X);
train_Y = Model(train_X,w,b,xh,xw,[s,p]);

#train_Y = nn.MLP(train_X,w,b);

train_Y = sess.run(train_Y);

train_Y = nn.defeatureScaling(train_Y,1,-1,train_T);

MSE_nn = sess.run(nn.MSE(train_Y,train_T));

a_nn = np.argmax(train_Y,axis=1)-1;


profit_nn = TF.TSL(a_nn,train_P,15,2,0);
TF.plotPerformance(profit_nn,'nn in sample (stop lose)');
profit_nn = TF.MarketState(a_nn,train_P,2);
TF.plotPerformance(profit_nn,'nn in sample (market state)');

# -- out sample --#

test_X = nn.featureScaling(test_X,1,-1);
xh,xw = nn.getWH(test_X);
test_Y = Model(test_X,w,b,xh,xw,[s,p]);

test_Y = sess.run(test_Y);

test_Y = nn.defeatureScaling(test_Y,1,-1,train_T);

a_nn = np.argmax(test_Y,axis=1)-1;

profit_nn = TF.TSL(a_nn,test_P,15,2,0);
TF.plotPerformance(profit_nn,'nn out sample (stop lose)');
profit_nn = TF.MarketState(a_nn,test_P,2);
TF.plotPerformance(profit_nn,'nn out sample (market state)');


########### linear regression ########


print "MSE nn : ", MSE_nn

############ display profit #############



#plt.figure().suptitle('org',fontsize = 20);
#plt.plot(np.cumsum(profit))


plt.show()
