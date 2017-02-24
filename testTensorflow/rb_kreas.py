from keras.models import Sequential
from keras import backend as K
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution1D
from keras.regularizers import l2, activity_l2

from pyelf.eldata import Data
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils

import TradingFun as TF
import NN_network as nn

window_size = 100
name = 'rb'

#np.random.seed(1337)  # for reproducibility
def correlation(y_true, y_pred):
    return -K.abs(K.sum((y_pred - K.mean(y_pred))*y_true, axis=-1))



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

#plt.plot(train_y)
#plt.show()
# print train_x.shape

model = Sequential()
model.add(Convolution1D(100, 30, border_mode='same', input_shape=train_x.shape[1:]))
model.add(Activation('tanh'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(output_dim=30, W_regularizer=l2(0.005)))
model.add(Activation('tanh'))
model.add(Dropout(0.25))
model.add(Dense(output_dim=2))
#model.add(Activation('softmax'))
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


#model.fit(train_x, train_y, batch_size=len(train_x), nb_epoch=20)

model = nn.Keras_train(train_x,train_y,model,lr=0.001,batch_size=1,train_percent=-1,show_performance=0,loss='categorical_crossentropy',iteration=1,evalFun=nn.best_validation)

# loss_and_metrics = model.evaluate(train_x, train_y, batch_size=wwww32)

predict_y = model.predict(train_x[:length, :, :])

bsig = np.zeros(length)
ssig = np.zeros(length)

a = 30
th_up = np.percentile(predict_y[:, 0], 100-a)
th_down = np.percentile(predict_y[:, 0], a)

bsig[predict_y[:, -1] > th_up] = 1
ssig[predict_y[:, -1] < th_down] = 1

data.close = data.close[window_size:]
data.delta_close = data.delta_close[window_size:]
data.timestamps = data.timestamps[window_size:]

TF.plotPerformance(data,bsig,ssig,'rb')


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

data.close = data.close[window_size:]
data.delta_close = data.delta_close[window_size:]
data.timestamps = data.timestamps[window_size:]

TF.plotPerformance(data,bsig,ssig,'rb')

plt.show()