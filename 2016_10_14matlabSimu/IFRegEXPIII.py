import numpy as np
import tensorflow as tf
import random
from IF_raw import IF_raw
import matplotlib.pyplot as plt

address = '/home/buhuai/Documents/all_in_one.csv';
read = IF_raw(address);

X = read.getX()
T = read.getY()
A = read.getOneHotLKey()

batchSize = 100
# -----------
# prepare tensorflow
sess = tf.InteractiveSession()

# ------------
# input
x = tf.placeholder("float",shape = [None, 30])
t = tf.placeholder("float",shape = [None,3])
a = tf.placeholder("float",shape = [None,3]) # one hot action

# ------------
# model
w = tf.Variable(tf.truncated_normal([30,3],stddev=0.01))
b = tf.Variable(tf.constant(0.01,shape = [3]))
z = 1;

y = tf.matmul(x,w)+b;

# -------------
# cost & optimization
y_action = tf.reduce_sum(tf.mul(y,a),reduction_indices=1)
maxT = tf.reduce_sum(tf.mul(t,a),reduction_indices=1)
cost = (tf.reduce_meantf.abs(maxT-y_action))
cost = (tf.min(readout,0)*z-lambda*z)
opt = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

sess.run(tf.initialize_all_variables())

#a = sess.run(diff,feed_dict={x:X,t:T})
#print a

for i in range(1000):
    opt.run(feed_dict={x:X,t:T,a:A})
    MSE = sess.run(cost, feed_dict = {x:X, t:T, a:A})
    print ("step %d, MSE %g" % (i, MSE))

# -------------
# evaluation
out = sess.run(y,feed_dict={x:X,t:T})
print out
profit = read.getProfit(out)

plt.plot(np.cumsum(profit))
plt.show()