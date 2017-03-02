import tensorflow as tf

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01,shape = shape)
    return tf.Variable


#input data size 1x30
def createdNetWork(x,y):
    sess = tf.InteractiveSession()

    w_conv1 = weight_variable([1,10,1])
    b_conv1 = bias_variable([1])

    W_fc1 = weight_variable([5,3])
    b_fc1 = bias_variable([3])

    # return 1x5 data x 5

    #input layer
    X = tf.placeholder("float",[None,1,30]);

    #hidden layer
    h_conv1 = tf.nn.relu(tf.nn.conv2d(X,w_conv1,5,padding='VALID'))

    # readout layer
    readout = tf.matmul(h_conv1,W_fc1)+b_fc1

    ## optimization
    T = tf.placeholde("float",[None,1,3])
    cost = tf.reduce_mean(tf.sqare(T-readout))
    optimization = tf.train.AdamOptimizer(1e-6).minimize(cost)

    # optimization
    sess.run(tf.initialize_all_variables())

    for i in range(200):
        train_MSE = cost.eval(feed_dict = {X:x,T:y})
        print("step %d, training MSE %g"%(i,train_MSE));
