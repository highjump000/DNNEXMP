import numpy as np
import tensorflow as tf
import random
from IF_raw import IF_raw
import matplotlib.pyplot as plt
#-- load data --#
address = '/home/buhuai/Documents/all_in_one.csv';
read = IF_raw(address);

X = read.getX3DB()
T = read.getY()
batchSize = 100

H,W,D = X.shape

sess = tf.InteractiveSession()


x = tf.placeholder("float",shape=[None,31,1])
t = tf.placeholder("float",shape=[None,3])

# ---------------------------------
# build model
w_conv = tf.Variable(tf.truncated_normal([10,1,1],stddev=0.01))

conv = tf.nn.conv1d(x, w_conv, stride = 5, padding='VALID')
#h_conv =  tf.nn.sigmoid(conv)
h_conv = tf.nn.relu(conv)

pureReg_X = tf.reshape(h_conv,[-1,5])

XTX = tf.matmul(tf.transpose(pureReg_X),pureReg_X)

W = tf.matmul(tf.matrix_inverse(XTX),tf.matmul(tf.transpose(pureReg_X),t))

readout = tf.matmul(pureReg_X,W);

# ----------------------------------
# loss
cost = tf.reduce_mean(tf.square(t-readout));
#optimization = tf.train.AdamOptimizer(1e-6).minimize(cost);
optimization = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
# ----------------------------------
# optimization

sess.run(tf.initialize_all_variables())
# mini batch learning

print sess.run(XTX,feed_dict={x:X,t:T})

perCost = sess.run(cost,feed_dict={x:X,t:T});
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
    #if perCost <= afterCost:
    #    break
    perCost = afterCost;

#print sess.run(readout, feed_dict={x: X, t: T})
out = sess.run(readout, feed_dict={x: X, t: T});
#
act = read.getActionV2(out)

profit = read.getPerformanceV2(out);
# get profit
#save to file
np.set_printoptions(threshold=np.inf, linewidth=np.inf)  # turn off summarization, line-wrapping
with open('readout.txt', 'w') as f:
    f.write(np.array2string(out, separator=', '))

with open('action.txt', 'w') as f:
    f.write(np.array2string(act, separator=', '))

plt.plot(np.cumsum(profit))
plt.show()