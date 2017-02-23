from keras.models import Sequential
from keras import backend as K
from keras.layers import Dense, Activation, Flatten, Dropout,MaxPooling1D
from keras.layers.convolutional import Convolution1D
from keras.regularizers import l2, activity_l2

from elf.eldata import Data
from elf.units import PlainFilter, FixedRatioTrainStop
import elf.performance as perf
import elf.elutil as eu
import elf.elplot as ep
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
import TradingFun as TF
import NN_network as nn

window_size = 100
name = 'rb'

def toOneHot(input):
    DL = len(input)
    output = np.zeros([DL,2])
    for i in range(DL):
        if input[i] > 0:
            output[i][1] = 1
        else:
            output[i][0] =1
    return output

def correlation(y_true, y_pred):
    return -K.abs(K.sum((y_pred - K.mean(y_pred))*y_true, axis=-1))


def show_result(data, bsig, ssig):
    data.close = data.close[window_size:]
    data.delta_close = data.delta_close[window_size:]
    data.timestamps = data.timestamps[window_size:]

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


prefix = '/home/yohoo/Documents/Data/'
d = Data()
d.set_data_path(prefix+'min/', prefix+'log/')
d.set_name(name)
d.load_raw_data()
d.set_period(15)

data = d.extract_data_by_period([[2010, 1], [2013, 1]])
close_ = np.copy(data.close)

#mean_c = np.mean(close_)
#var_c = np.std(close_)
#close_ = (close_ - mean_c)/var_c

dc = np.copy(data.delta_close)

print max(dc), min(dc), np.mean(dc), np.std(dc)

# var_dc = np.std(dc)
# dc = dc/var_dc

length = len(close_)-window_size

train_x = np.zeros((length, window_size, 1))
train_y = np.zeros(length, dtype=int)

for i in xrange(length):
    train_x[i, :, 0] = close_[i+1:i+window_size+1] - close_[i:i+window_size]
    #train_x[i+length, :, 0] = -train_x[i, :, 0]
    #print train_x[i, -1, 0], dc[i+window_size]
    if dc[i+window_size] > 0:
        train_y[i] = 1
        #train_y[i+length] = 0
    else:
        train_y[i] = 0
        #train_y[i + length] = 1

train_y = np_utils.to_categorical(train_y, 2)

print 'end of preprocessing'

# print train_x.shape

model = Sequential()
model.add(Convolution1D(10, 30, border_mode='same', input_shape=train_x.shape[1:]))
#model.add(MaxPooling1D(pool_length=2))
model.add(Activation('relu'))
#model.add(Convolution1D(300,30,border_mode='same', input_shape=train_x.shape[1:]))
#model.add(MaxPooling1D(pool_length=2))
#model.add(Activation('sigmoid'))
#model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(output_dim=30,W_regularizer=l2(0.001)))
model.add(Activation('sigmoid'))
#model.add(Dropout(0.25))
model.add(Dense(output_dim=2))
model.add(Activation('softmax'))
#model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])


#model.fit(train_x, train_y, batch_size=32, nb_epoch=30)

model = nn.Keras_train(train_x,train_y,model,batch_size=32,iteration=50,show_performance=1,loss='categorical_crossentropy',train_percent=0.7)

# loss_and_metrics = model.evaluate(train_x, train_y, batch_size=32)

predict_y = model.predict(train_x[:length, :, :], batch_size=32)


bsig = np.zeros(length)
ssig = np.zeros(length)

print "=========="
print bsig
print ssig

a = 10

th_up = np.percentile(predict_y[:, 0], 100-a)
th_down = np.percentile(predict_y[:, 0], a)

bsig[predict_y[:, -1] > th_up] = 1
ssig[predict_y[:, -1] < th_down] = 1


show_result(data, bsig, ssig)
act = TF.TFReadSignal(bsig,ssig)
profit = TF.TSL(act,data.close[window_size:],15,2,0)
TF.plotPerformanceSimple(profit,'insample')

data = d.extract_data_by_period([[2013, 1], [2017, 1]])
close_ = np.copy(data.close)

dc = np.copy(data.delta_close)

length = len(close_)-window_size

test_x = np.zeros((length, window_size, 1))
test_y = np.zeros(length, dtype=int)

for i in xrange(length):
    test_x[i, :, 0] = close_[i+1:i+window_size+1] - close_[i:i+window_size]
    if dc[i+window_size] > 0:
        test_y[i] = 1
    else:
        test_y[i] = 0

test_y = np_utils.to_categorical(test_y, 2)
predict_y = model.predict(test_x, batch_size=32)

bsig = np.zeros(test_x.shape[0])
ssig = np.zeros(test_x.shape[0])
bsig[predict_y[:, -1] > th_up] = 1
ssig[predict_y[:, -1] < th_down] = 1

show_result(data, bsig, ssig)
act = TF.TFReadSignal(bsig,ssig)
profit = TF.TSL(act,data.close[window_size:],15,2,0)
TF.plotPerformanceSimple(profit,'outsample')
plt.show()
