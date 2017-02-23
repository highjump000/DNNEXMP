#created by bohuai jiang
#on 2017/2/13

from keras.models import Sequential
from keras.layers.core import Dense, Dropout ,Activation

from RandomWalk import RandomWalk
import matplotlib.pyplot as plt
import numpy as np
import TradingFun as TF
import NN_network as nn
import tensorflow as tf
#---- create data ----#
update = 0
if update:
    randw = RandomWalk(10000)
    data = randw.generated()

    np.save('rdw_data',data)

data = np.load('rdw_data.npy')
X,T = TF.getXY(30,data[0:10000])


#--- arithmetic method ----
w = TF.linearRegression(X,T)
Y = TF.linearRegression_getY(w,X)

#plt.figure().suptitle('result')
plt.plot(Y,label='arithmetic method')
#--- keras method ---#
sX = nn.featureScaling(X,1,-1)
sT = nn.featureScaling(T,1,-1)
model = Sequential()
model.add(Dense(1,input_shape=(30,)))
#model.compile(loss="mean_squared_error",optimizer='adadelta')
K = nn.Keras_train(sX,sT,model,batch_size=len(sT),iteration=100000,train_percent=.9999,show_performance=1)

Y = model.predict(sX)
Y = nn.defeatureScaling(Y,1,-1,nn.getMaxMin(T))

plt.plot(Y,label = 'keras')
plt.legend()
plt.show()
'''
#--- tensorflow ---#

iteration = 30

sess = tf.InteractiveSession()
w = nn.weight_variable([30,1])
b = nn.bias_variable([1])

input_x = tf.placeholder("float32",shape=[None,30])
input_t = tf.placeholder("float32",shape=[None,1])
tf_Y = nn.linearACT(input_x,w,b)
cost = nn.MSE(tf_Y,input_t)

opt = tf.train.GradientDescentOptimizer(10e-6).minimize(cost)
#opt = tf.train.AdagradOptimizer(1).minimize(cost)
#opt = tf.train.MomentumOptimizer
sess.run(tf.global_variables_initializer())

costV = []
for i in range(iteration):
    sess.run(opt,feed_dict={input_x:X,input_t:T})
    costV.append(sess.run(cost,feed_dict={input_x:X,input_t:T}))
    print 'iteration: ',i, ' MSE: ',costV[i]

Y = sess.run(tf_Y,feed_dict={input_x:X})
#plt.plot(Y,label='tensorflow')
#plt.legend()
plt.figure().suptitle('loss')
plt.plot(costV,'r')
plt.xlabel('iterations')
plt.ylabel('MSE')
plt.savefig('../../../../../Desktop/MSE.png')
plt.show()
'''
