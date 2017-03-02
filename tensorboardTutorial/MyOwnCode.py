
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt

# created data
def getData():
    l = 50;
    alpha = 100;
    X = np.zeros([l, 1]);
    T = np.zeros([l, 1]);
    for i in range(l):
        X[i] = i;
        T[i] = 3 * X[i] * random.random() + alpha * random.random();
    return X,T;

#######################
# tensorflow stuff
######################
def run():
    X,T = getData();

    sess = tf.InteractiveSession();

    cost = model(np.float32(X),np.float32(T));
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(0.01).minimize(cost)

    tf.global_variables_initializer().run();

    for i in range(1000):
        sess.run(train_step);
    plt.plot(T);
    plt.plot()

def model(X, T):
    y = nn_layer(X,1,1,'layer1');

    with tf.name_scope('mean_square_error'):
        cost = tf.reduce_mean(tf.square(y-T));
    return cost

def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.sigmoid):
    with tf.name_scope(layer_name):
        with tf.name_scope('weigths'):
            weights = weight_variable([input_dim,output_dim]);
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim]);
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
        activations = act(preactivate, name='activation')
        return activations;

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1);
    return tf.Variable(initial);

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape);
    return tf.Variable(initial);

def main(_):
    run();

if __name__ == '__main__':
    tf.app.run();