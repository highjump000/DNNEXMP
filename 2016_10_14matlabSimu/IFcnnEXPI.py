import numpy as np
import tensorflow as tf
import random
from IF_raw import IF_raw
from PureReg import PureReg
import matplotlib.pyplot as plt

#-- load data --#
address = '/home/buhuai/Documents/all_in_one.csv';
read = IF_raw(address);

X = read.getX3D()
T = read.getY()
batchSize = 100

H,W,D = X.shape

sess = tf.InteractiveSession()


x = tf.placeholder("float",shape=[None,30,1])
t = tf.placeholder("float",shape=[None,3])

# ---------------------------------
# build model
w_conv = tf.Variable(tf.truncated_normal([10,1,1],stddev=0.01))
b_conv = tf.Variable(tf.constant(0.01,shape = [1]))


w_fc = tf.Variable(tf.truncated_normal([5,3],stddev=0.01))
b_fc = tf.Variable(tf.constant(0.01,shape = [3]));

conv = tf.nn.conv1d(x, w_conv, stride = 5, padding='VALID')
h_conv =  tf.nn.sigmoid(conv + b_conv)
h_conv = tf.nn.relu(conv+b_conv)
h_conv_flat = tf.reshape(h_conv,[-1,5])

readout = tf.matmul(h_conv_flat,w_fc)+b_fc;

# ----------------------------------
# loss
cost = tf.reduce_mean(tf.square(t-readout));
#optimization = tf.train.AdamOptimizer(1e-6).minimize(cost);
optimization = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
# ----------------------------------
# optimization

sess.run(tf.initialize_all_variables())


for i in range(1000):
    perm = np.arange(H)
    np.random.shuffle(perm)

    #for j in range(batchSize):
    #    batch_xs[j,:,:] = X[perm[j],:,:]
    #    batch_ys[j,:] = T[perm[j],:]
    #optimization.run(feed_dict={x:batch_xs,t:batch_ys})
    optimization.run(feed_dict={x:X,t:T})
    afterCost = sess.run(cost, feed_dict={x: X, t: T})
    print ("step %d, training accuracy %g"%(i, afterCost))
    perCost = afterCost;

#print sess.run(w_conv, feed_dict={x: X, t: T})
# get profit
out = sess.run(readout,feed_dict={x: X, t: T});

profit = read.getPerformanceV2(out);
# get profit

plt.plot(np.cumsum(profit))
plt.show()