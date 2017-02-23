import  TradingFun as TF

import tensorflow as tf
import NN_network as nn

from elf.eldata import Data
from elf.units import PlainFilter, FixedRatioTrainStop
import elf.performance as perf
import elf.elutil as eu
import elf.elplot as ep
import numpy as np
import matplotlib.pyplot as plt


window_size = 30
name = 'rb'
def toBSsig(Qvalue):
    DL = len(Qvalue)
    bsig = np.zeros(DL)
    ssig = np.zeros(DL)
    a = TF.getA(Qvalue)
    for i in range(DL):
        if a[i] == 1:
            bsig[i] = 1
        if a[i] == -1:
            ssig[i] =1
    return bsig,ssig
def toOneHotKey(list):
    LD = len(list)
    out = np.zeros([LD,2])
    for i in range(LD):
        if list[i] > 0:
            out[i][1] =1
        else:
            out[i][0] =1
    return out

def correlation(y_true, y_pred):
    return -K.abs(K.sum((y_pred - K.mean(y_pred))*y_true, axis=-1))

def show_result(data, bsig, ssig,name):
    #plt.figure().suptitle(name)
    pf = PlainFilter()
    frs = FixedRatioTrainStop()
    #plt.plot(bsig-ssig)
    ps = pf.eval(data, [], [bsig, ssig])
    #plt.plot(ps, 'r')
    ps = frs.eval(data, [15], ps)
    #plt.plot(ps, 'k')
    #plt.grid()
    #plt.ylim([-1.5, 1.5])
    #plt.show()

    #ps = bsig - ssig

    accu_profits, net_accu_profits = perf.get_accu_profits(ps, data.delta_close, eu.get_unit_cost(name))
    draw_down = perf.get_draw_down(net_accu_profits)

    trading_times, win_rate, pf, md, ppt, total_profit = perf.get_performance(ps, data.delta_close,
                                                                              eu.get_unit_cost(name))
    title = 'TT=%d, WR=%.2f, PF=%.2f, MD=%.2f, PPT=%.2f, NTP=%.2f' % (trading_times, win_rate, pf,
                                                                      md, ppt, net_accu_profits[-1])

    f, ax = plt.subplots(3, 1, sharex=True)
    ep.plot_profits(ax[0], accu_profits, net_accu_profits, draw_down, data.timestamps, title=title)
    ax[1].plot(ps)
    ax[1].set_ylim([-1.5, 1.5])
    ax[1].grid()
    ax[2].plot(data.close)
    ax[2].grid()


# ------- read data ------ #
prefix = '/home/yohoo/Documents/ubuntu_doc/Documents/'
d = Data()
d.set_data_path(prefix+'min/', prefix+'log/')

d.set_name(name)
d.load_raw_data()
d.set_period(15)

data = d.extract_data_by_period([[2010, 1], [2013, 1]])
close_ = np.copy(data.close)
dc = np.copy(data.delta_close)




# ------ data preparation ----#
# --- get diff insample data
update = 1;
data_diff = np.diff(close_)
data_close = close_[1::]
trainX,trainP = TF.getXY(window_size,data_diff)
#trainT = toOneHotKey(trainP)
if update:
    trainT = TF.getQtable(trainP,15,2)
    np.save('trainT',trainT)
TrainT = np.load('trainT.npy')
### ----- tensorflow train---- ###
# tf param
CNN = 1
n_output = len(trainT[0])
if CNN:

    print 'out put size', n_output
    w,b,s,p = nn.modelSampleInit(window_size,n_output,topo=[30,10,1],kernal=[30,1,1],padding = [-1,-1,-1])
    sess = tf.InteractiveSession()
    w,b,sess = nn.MLCNN2D_train(trainX,trainT,w,b,s,p,sess,act_func=tf.tanh,cost_func=nn.MSE,show_performance=1,frame = 1)
    #sess.run(tf.global_variables_initializer())
    Y = sess.run(nn.modelSampleCNN(nn.featureScaling(trainX,1,-1),w,b,window_size,[s,p,n_output],act_func=tf.tanh))
    maxMinY = nn.getMaxMin(TrainT)
    Y = nn.defeatureScaling(Y,1,-1,maxMinY)
else:
    w, b = nn.MLP_init(window_size,n_output, topo=[30,10,1])
    sess = tf.InteractiveSession()
    w,b,sess = nn.MLP_train(trainX,trainT,w,b,sess,show_performance=1,batch_size=1500)

# ----------- insample test ----------- #
'''
bsig = np.zeros(len(Y))
ssig = np.zeros(len(Y))

th_up = np.percentile(Y[:, 1], 70)
th_down = np.percentile(Y[:, 0], 70)

bsig[Y[:, 1] > th_up] = 1
ssig[Y[:, 0] > th_down] = 1
'''
bsig,ssig = toBSsig(Y)


data.close = data.close[window_size+1:-1]
data.delta_close = data.delta_close[window_size+1:-1]
data.timestamps = data.timestamps[window_size+1:-1]

#print data.close.shape
#print len(bsig)
#print len(ssig)

show_result(data, bsig, ssig,'insample')

a = TF.TFReadSignal(bsig,ssig)
profit = TF.TSL(a,trainP,15,2,0)
TF.plotPerformance(profit,'insample')

# ----------- outsample test ----------- #
data = d.extract_data_by_period([[2013, 1], [2017, 1]])
close_ = np.copy(data.close)


data_diff = np.diff(close_)
testX,testP = TF.getXY(window_size,data_diff)
maxMinX = nn.getMaxMin(trainX)
maxMinY = nn.getMaxMin(trainT)
Y = sess.run(nn.modelSampleCNN(nn.featureScalingOutSample(testX,maxMinX,1,-1),w,b,window_size,[s,p,n_output],act_func=tf.tanh))
Y = nn.defeatureScaling(Y,1,-1,maxMinY)
'''
bsig = np.zeros(len(Y))
ssig = np.zeros(len(Y))

th_up = np.percentile(Y[:, 1], 70)
th_down = np.percentile(Y[:, 0], 70)

bsig[Y[:, 1] > th_up] = 1
ssig[Y[:, 0] > th_down] = 1
'''

bsig,ssig = toBSsig(Y)

data.close = data.close[window_size+1:-1]
data.delta_close = data.delta_close[window_size+1:-1]
data.timestamps = data.timestamps[window_size+1:-1]

show_result(data, bsig, ssig,'outsample')

a = TF.TFReadSignal(bsig,ssig)
profit = TF.TSL(a,testP,15,2,0)
TF.plotPerformance(profit,'outsample')

plt.show()
