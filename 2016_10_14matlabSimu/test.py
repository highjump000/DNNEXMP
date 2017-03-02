import numpy as np
import tensorflow as tf
import random
from IF_raw import IF_raw
import matplotlib.pyplot as plt

#def goNomalization(v):
#    mean = np.mean(v[:,0]);
#    var  = np.var(V[])

address = '/home/buhuai/Documents/all_in_one.csv';
read = IF_raw(address);


X = read.getXNormailzed()  # normalize exclude bias
T = read.getY()
Xb = read.getXNormailzedB() #normalize include bias
dataLen = len(X)

T1 = np.zeros((dataLen,1))
T1_ = T[:,0];

for i in xrange(dataLen):
    T1[i] = T[i][0]

#plt.scatter(range(len(T1)),T1)

# -----------
# prepare tensorflow
sess = tf.InteractiveSession()

# ------------
# input
x = tf.placeholder("float",shape = [None, 30])
t = tf.placeholder("float",shape = [None,3])
xb = tf.placeholder("float",shape = [None, 31])

# ------------
# model
w = tf.Variable(tf.truncated_normal([30,3],stddev=0.01))
b = tf.Variable(tf.constant(0.01,shape = [3]))

y = tf.matmul(x,w)+b;

# arithmetic method
XTX = tf.matmul(tf.transpose(xb),xb)

w_ = tf.matmul(tf.matrix_inverse(XTX),tf.matmul(tf.transpose(xb),t))

y_ = tf.matmul(xb,w_);

cost = tf.reduce_mean(tf.sqrt(tf.square(y-t)))
opt = tf.train.GradientDescentOptimizer(0.1).minimize(cost)
#opt = tf.train.AdadeltaOptimizer(1e-6).minimize(cost)

# run linear regression
sess.run(tf.initialize_all_variables())

for i in range(1000):
    opt.run(feed_dict={x: X, t: T})
    MSE = cost.eval(feed_dict={x: X, t: T})
    print ("step %d, MSE %g" % (i, MSE))

sess.run(tf.initialize_all_variables())
#print sess.run(y,feed_dict={x:X,t:T})
#print sess.run(y_,feed_dict={xb:Xb,t:T})
#plt.plot(y.eval(feed_dict={x: X, t: T}))
#plt.plot(y_.eval(feed_dict={xb: Xb, t: T1}))
#plt.show()

#profit = read.getPerformanceV2(sess.run(y,feed_dict={x:X,t:T}))
profit = read.getPerformanceV2(sess.run(y_,feed_dict={xb:Xb,t:T}))


plt.plot(np.cumsum(profit))
plt.show()
