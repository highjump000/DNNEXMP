#created by bohuai jiang
#on 2017/2/23
#test performace for simple neural net
import TradingFun as TF
import numpy as np
import NN_network as nn
from keras.models import Sequential
from keras.layers import Convolution1D,MaxPooling1D,Activation,Flatten,Dropout,Dense,Merge
import matplotlib.pyplot as plt
from keras.regularizers import l2, activity_l2
import tensorflow as tf
np.random.seed(1337)  # for reproducibility
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

period = 100
update = 0
a = 30

#### load data

data = TF.getData(15,'rb',[2010,1],[2013,1])
close = data.close
diff_close = np.diff(close)
# re-calibrate data
data.close = data.close[0:-1]
data.delta_close = data.delta_close[0:-1]
data.timestamps = data.timestamps[0:-1]


X,P = TF.getXY(period,diff_close,forwardOne=1)
T = toOneHot(P)

#### CREATE KERAS MODEL ####
model = nn.Keras_MLP(X,T,topo = [0])
X = nn.featureScaling(X,1,-1)
model = nn.Keras_train(X,T,model,loss='categorical_crossentropy',show_performance=1,train_percent=-1)
Y = model.predict(X)

#### TO TRADING SIGNAL ####

bsig,ssig = getSig(Y,a)
plt.figure().suptitle('bsig')
plt.plot(bsig)

data = adjust(data,period,1)


TF.plotPerformance(data,bsig,ssig,'rb')
plt.show()

