import numpy as np
import tensorflow as tf
import random
from IF_raw import IF_raw
import matplotlib.pyplot as plt

address = '/home/buhuai/Documents/all_in_one.csv';
read = IF_raw(address);


X = read.feature_scalling(read.getX())
T = read.feature_scalling(read.getY())
Xb = read.getXNormailzedB()
dataLen = len(X)

T1 = np.zeros((dataLen,1))
T1_ = T[:,0];

for i in xrange(dataLen):
    T1[i] = T[i][0]

plt.scatter(range(len(T1)),T1)

# -----------
# prepare tensorflow
sess = tf.InteractiveSession()

# ------------
# input
#x = tf.placeholder("float",shape = [None, 30])
t = tf.placeholder("float",shape = [None,1])
xb = tf.placeholder("float",shape = [None, 31])

# ------------
# model
w = tf.Variable(tf.truncated_normal([31,1],stddev=0.01))
#b = tf.Variable(tf.constant(0.01,shape = [1]))

y = tf.matmul(xb,w);

# arithmetic method
XTX = tf.matmul(tf.transpose(xb),xb)

w_ = tf.matmul(tf.matrix_inverse(XTX),tf.matmul(tf.transpose(xb),t))

y_ = tf.matmul(xb,w_);



cost = tf.reduce_mean(tf.square(y-t))
opt = tf.train.GradientDescentOptimizer(5).minimize(cost)


sess.run(tf.initialize_all_variables())

for i in range(1000):
    opt.run(feed_dict={xb: Xb, t: T1})
    MSE = cost.eval(feed_dict={xb: Xb, t: T1})
    print ("step %d, MSE %g" % (i, MSE))

sess.run(tf.initialize_all_variables())
print sess.run(w,feed_dict={xb:Xb,t:T1})
print sess.run(w_,feed_dict={xb:Xb,t:T1})
plt.plot(y.eval(feed_dict={xb: Xb, t: T1}))
#plt.plot(y_.eval(feed_dict={xb: Xb, t: T1}))
plt.show()