import numpy as np
import tensorflow as tf
import random
from IF_raw import IF_raw
import matplotlib.pyplot as plt

address = '/home/buhuai/Documents/all_in_one.csv';
read = IF_raw(address);

X = read.getX()
T = read.getY()

dataLen = len(X)

T1 = np.zeros((dataLen,1))
T3 = np.zeros((dataLen,1))
for i in xrange(dataLen):
    T1[i] = T[i][0]
    T3[i] = T[i][2]
print T1
# ---------------
# prepare tensorflow
sess = tf.InteractiveSession()

# ---------------
# input
# two regression
x = tf.placeholder("float", shape = [None, 30])
t1 = tf.placeholder("float", shape = [None, 1])
t3 = tf.placeholder("float", shape = [None, 1])

# ----------------
# two separate model

# model 1
w1 = tf.Variable(tf.truncated_normal([30,1],stddev=0.01))
b1 = tf.Variable(tf.constant(0.01,shape = [1]))

y1 = tf.matmul(x , w1) + b1

# model 2
w2 = tf.Variable(tf.truncated_normal([30,1],stddev=0.01))
b2 = tf.Variable(tf.constant(0.01,shape = [1]))

y2 = tf.matmul(x , w2) + b2

# -------------------
# cost & optimization
cost1 = tf.reduce_mean(tf.square(y1-t1))
cost2 = tf.reduce_mean(tf.square(y2-t3))

#opt1 = tf.train.GradientDescentOptimizer(0.01).minimize(cost1)
#opt2 = tf.train.GradientDescentOptimizer(0.01).minimize(cost2)

opt1 = tf.train.AdamOptimizer(1e-6).minimize(cost1)
opt2 = tf.train.AdamOptimizer(1e-6).minimize(cost2)

sess.run(tf.initialize_all_variables())

for i in range(1000):
    #for i in range(dataLen):
    #    inX = [];
    #inX.append(X[i]);
    #    inT1 = []
    #    inT1.append(T1[i])
    #    inT3 = []
    #    inT3.append(T3[i])
    opt1.run(feed_dict = {x:X, t1:T1})
    opt2.run(feed_dict = {x:X, t3:T3})
    c1 = sess.run(cost1,feed_dict = {x: X, t1: T1})
    c2 = sess.run(cost2,feed_dict = {x: X, t3: T3})
    MSE = (c1+c2)/2
    print ("step %d, MSE %g" % (i, MSE))

# -------------
# evaluation
w1 = sess.run(w1,feed_dict = {x: X, t1: T1})
b1 = sess.run(b1,feed_dict = {x: X, t1: T1})
w3 = sess.run(w2,feed_dict = {x: X, t3: T3})
b3 = sess.run(b2,feed_dict = {x: X, t1: T1})

out = np.zeros((31,3))

for i in xrange(30):
    out[i][0] = w1[i]
    out[i][2] = w3[i]
out[-1][0] = b1
out[-1][2] = b3

out = np.matrix(np.asarray(out))
profit = read.getPerformance(out)
print np.matrix(read.getXB())*out
plt.plot(np.cumsum(profit))
plt.show()
