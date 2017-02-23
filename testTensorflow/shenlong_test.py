#created by bohuai jiang
#on 2017/2/20
#regression test
import TradingFun as TF
import numpy as np
import NN_network as nn
from keras.models import Sequential
from keras.layers import Convolution1D,MaxPooling1D,Activation,Flatten,Dropout,Dense,Merge
import matplotlib.pyplot as plt
from keras.regularizers import l2, activity_l2
import tensorflow as tf

def toOneHot(input):
    DL = len(input)
    output = np.zeros([DL,2])
    for i in range(DL):
        if input[i] > 0:
            output[i][1] = 1
        else:
            output[i][0] =1
    return output

def getSig(Y,a):
    length = len(Y)
    th_up = np.percentile(Y[:, 1], a)
    th_down = np.percentile(Y[:, 0], a)
    bsig = np.zeros(length)
    ssig = np.zeros(length)
    bsig[Y[:, 1] > th_up] = 1
    ssig[Y[:, 0] < th_down] = 1
    return bsig,ssig

def getSigRegression(Y):
    length = len(Y)
    a = TF.getA(Y)
    bsig = np.zeros(length)
    ssig = np.zeros(length)
    bsig[a == 1] = 1
    ssig[a == -1 ] = 1
    return bsig, ssig

def adjust(data,period,forwardOne):
    data.close = TF.CallibrationData(period,data.close,forwardOne=forwardOne)
    data.delta_close = TF.CallibrationData(period,data.delta_close,forwardOne=forwardOne)
    data.timestamps = TF.CallibrationData(period,data.timestamps,forwardOne=forwardOne)

    print len(data.close)
    print len(data.delta_close)
    print len(data.timestamps)
    return data
##########################################################

period = 30
update = 0
a = 10

#### load data

data = TF.getData(15,'rb',[2010,1],[2013,1])
close = data.close
X,P = TF.getXY(period,close,forwardOne=0)
# change X to diff
if update:
    T = TF.getQtable(P,15,0)
    np.save('T',T)
T = np.load('T.npy')

#X = np.diff(X,axis=1)
#X = np.reshape(X,[-1,period-1,1])


#### CREATE KERAS MODEL ####
model = nn.Keras_MLP(X,T,topo = [512,400,10])
sX,sT = nn.Keras_preprocess_FeatureScaling(X,T)
model = nn.Keras_train(sX,sT,model,lr=0.0003,show_performance=1,iteration=400,batch_size= -1,bestCount=10)
Y = model.predict(sX)
Y = nn.defeatureScaling(Y,1,-1,nn.getMaxMin(sT))

#### TO TRADING SIGNAL ####

bsig,ssig = getSigRegression(Y)
data = adjust(data,period,0)

print len(bsig)
print len(ssig)

a = TF.getA(Y)

plt.figure().suptitle('bsig')
plt.plot(bsig)

plt.figure().suptitle('ssig')
plt.plot(ssig)

plt.figure().suptitle('act')
plt.plot(a)

TF.plotPerformance(data,bsig,ssig,'rb')
profit = TF.TSL(a,data.close,15,2,1)
TF.plotPerformanceSimple(profit,'stp only')
plt.show()

