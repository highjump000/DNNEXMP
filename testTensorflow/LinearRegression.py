from matplotlib.pyplot import plot,show,legend,figure
from numpy import linalg,random
import numpy as np
import tensorflow as tf
import NN_network as nn

update = 0
dlength = 1000
if update:
    T = np.float32(np.zeros([dlength,1]))
    x = np.float32(np.zeros([dlength,1]))
    xb = np.float32(np.ones([dlength,2]))
    for i in range(dlength):
        a = random.normal(1,3)
        b = random.normal(1, 0.1)
        x[i] = i
        T[i]=(3*a*i+b)
        xb[i][0] = i


#    T = np.asarray(T)
#    np.save('Xb',xb);
#    np.save('T',T);
#    np.save('X',x);

xb = np.load('Xb.npy');
T = np.load('T.npy');
x = np.load('X.npy');

plot(range(dlength),T,'.')

###--------- arithmetic method -----------###
w = linalg.lstsq(xb,T)[0]
Y = w[0]*x+w[1]
plot(Y,'r',label='arithmetic method')

mse = np.mean((Y-T)**2)
print 'arithmetic method: ',mse

###--------- classical tensorflow method ---------###
## preparation
lr = 0.01
batch_size = 100
sess = tf.InteractiveSession()
# hidden layer
w_init = tf.truncated_normal([1,1],stddev = 0.1)
w = tf.Variable(w_init)
b_init = tf.constant(1.0,shape = [1])
b = tf.Variable(b_init)


input_x = tf.placeholder("float32",shape=[None,1])
input_t = tf.placeholder("float32",shape=[None,1])
tf_Y = tf.matmul(input_x,w)+b
cost = tf.reduce_mean(tf.square(tf_Y-input_t))

## initialization
opt = tf.train.AdamOptimizer(lr).minimize(cost)
sess.run(tf.global_variables_initializer());
## train
iter = int(dlength/batch_size)
for iteration in range(iter):
    for epoch in range(1000):
        # shuffle
        batch_index = rd.sample(np.arange(0,dlength),batch_size)
        x_batch = x[batch_index]
        t_batch = T[batch_index]
        # mini-batch learning
        sess.run(opt,feed_dict={input_t:t_batch,input_x:x_batch})

Y = sess.run(tf_Y,feed_dict={input_t:T,input_x:x})
plot(Y,'g',label='classical tf')
mse = nn.MSE(Y,T)
print 'tensorflow result: ',sess.run(mse)
legend()
show()

'''
###--------- nn_networks method -------###
## preparation
sess = tf.InteractiveSession()
w,b = nn.MLP_init(1,1,topo=[10])
w,b,sess = nn.MLP_train(x,T,w,b,sess,batch_size=-1,train_precent=1,show_performance=0)
pre_x = nn.featureScaling(x,1,-1)
pre_Y = sess.run(nn.MLP(pre_x,w,b))
Y = nn.defeatureScaling(pre_Y,1,-1,[np.max(T),np.min(T)])

mse = nn.MSE(Y,T)
print 'nn_networks result: ',sess.run(mse)
plot(Y,'b',label= 'nn_network')
legend()
show()
'''
