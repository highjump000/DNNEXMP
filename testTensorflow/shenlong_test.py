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
np.random.seed(1337)  # for reproducibility
def toOneHot(T):
    DL = len(T)
    out = np.zeros([DL,3])
    for i in range(DL):
        max_indx = np.argmax(T[i])
        out[i][max_indx] = 1
    return out
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

period = 80
update = 1

classification = 1
#### load data

data = TF.getData(15,'rb',[2010,1],[2013,1])
close = data.close
X,P = TF.getXY(period,close,forwardOne=0)
# change X to diff
if update:
    T = TF.getQtable(P,1,2)
    np.save('T',T)
T = np.load('T.npy')
#
X = np.diff(X,axis=1)
X = np.reshape(X,[-1,period-1,1])


#### CREATE KERAS MODEL ####
model = Sequential()
model.add(Convolution1D(20, 30, border_mode='valid', input_shape=[period-1,1]))
model.add(Activation('tanh'))
model.add(Flatten())
model.add(Dense(output_dim=10))
model.add(Activation('tanh'))
model.add(Dense(output_dim=3))
if classification:
    T = toOneHot(T)
    model = nn.Keras_train(X,T,model,lr=0.004,show_performance=1,iteration=30,batch_size= -1,bestCount=100,train_percent=0.7,loss='categorical_crossentropy',evalFun=nn.best_validation)
else :
    sT = nn.featureScaling(T,1,-1)
    model = nn.Keras_train(X,sT,model,lr=0.004,show_performance=1,iteration=30,batch_size= 2048,bestCount=100,train_percent=-1,evalFun=nn.best_validation)
Y = model.predict(X)
if classification==0:
    Y = nn.defeatureScaling(Y,1,-1,nn.getMaxMin(T))
#### TO TRADING SIGNAL ####

bsig,ssig = getSigRegression(Y)
data = adjust(data,period,0)

print len(bsig)
print len(ssig)

a = TF.getA(Y)


TF.plotPerformance(data,bsig,ssig,'rb')
profit = TF.TSL(a,data.close,15,2,1)
TF.plotPerformanceSimple(profit,'stp only')


#########################
#### out sample test ####
#########################
data = TF.getData(15,'rb',[2013,2],[2017,1])
close = data.close
X,P = TF.getXY(period,close,forwardOne=0)
X = np.diff(X,axis=1)
X = np.reshape(X,[-1,period-1,1])
Y = model.predict(X)
#### TO TRADING SIGNAL ####

bsig,ssig = getSigRegression(Y)
Y = nn.defeatureScaling(Y,1,-1,nn.getMaxMin(T))
data = adjust(data,period,0)

print len(bsig)
print len(ssig)

a = TF.getA(Y)


TF.plotPerformance(data,bsig,ssig,'rb')
profit = TF.TSL(a,data.close,15,2,1)
TF.plotPerformanceSimple(profit,'stp only')

#########################
#### out sample test ####
#########################
data = TF.getData(15,'rb',[2015,1],[2017,1])
close = data.close
X,P = TF.getXY(period,close,forwardOne=0)
X = np.diff(X,axis=1)
X = np.reshape(X,[-1,period-1,1])
Y = model.predict(X)
#### TO TRADING SIGNAL ####

bsig,ssig = getSigRegression(Y)
Y = nn.defeatureScaling(Y,1,-1,nn.getMaxMin(T))
data = adjust(data,period,0)

print len(bsig)
print len(ssig)

a = TF.getA(Y)


TF.plotPerformance(data,bsig,ssig,'rb')
profit = TF.TSL(a,data.close,15,2,1)
TF.plotPerformanceSimple(profit,'stp only')
plt.show()