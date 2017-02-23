# created by bohuai jiang
# on 2016-12-19
# a easy version for creating a net work, store NN's trained model and debuging
# contains:
#           MLP
#                MLP_init : MLP random Weights & bias initialization
#                MLP      : run MLP model with given Weights & bias
#                MLP_get_model : get float type MLP Weights & bias
#           MLCNN
#

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random
import math as mat
from DynamicPlot import DynamicPlot
import TradingFun as TF
import copy as cp
import keras as k

# ------------------ cost function -----------------#
# some basic cost function
# you can defined your own
def MSE(Y,T):
    cost = tf.reduce_mean(tf.square(Y-T))
    return cost;

def crossEntropy(Y,T):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=T, logits=Y))
    return cost;

def logfunc(x, x2):
    return tf.mul(x, tf.log(tf.div(x, x2)))
# ----------------- activation ------------------------#
def linearACT(X,W,b):
    return tf.matmul(X,W)+b

# ----------------- sparse parameters -----------------#
def KL_Div(rho, rho_hat):
    invrho = tf.sub(tf.constant(1.), rho)
    invrhohat = tf.sub(tf.constant(1.), rho_hat)
    logrho = tf.add(logfunc(rho, rho_hat), logfunc(invrho, invrhohat))
    return logrho

#week OR one hidden layer only
def sparse_one_layer(X,W,b,hx,wx,model,rho,beta,rest):
     beta = tf.constant(beta)
     pre_p = tf.sigmoid(model(X,W[0:-1],b[0:-1],hx,wx,rest));
     rho_hat = tf.div(tf.reduce_sum(pre_p,0),tf.cast(hx,tf.float32));
     sparse = tf.mul(beta,tf.reduce_sum(KL_Div(rho,rho_hat)));
     return sparse

# ---------------- Regularization --------------#

# ----------------- set defalut evalutaion function -----------------------#
def defaultEVAl(X,T,w,b,cost):
    return cost;

############################## Multi-Layered Percepton ###############################
# created MLP network with given topology
# return weigth container & bias container
def MLP_init(n_feature,n_output,topo= [1]):

    n_hidden_layer = len(topo);
    weight_container = [];
    bias_container = [];

    if topo[0] != 0:
        #input layer
        weight_container.append(weight_variable([n_feature,topo[0]]));
        bias_container.append(bias_variable([topo[0]]));

        #hidden
        for i in range(1,n_hidden_layer):
            weight_container.append(weight_variable([topo[i-1], topo[i]]));
            bias_container.append(bias_variable([topo[i]]));

        #output layer
        weight_container.append(weight_variable([topo[-1], n_output]));
        bias_container.append(bias_variable([n_output]));

    else:
        # for signal leyer, either be linear regression or classification problems
        weight_container.append(weight_variable([n_feature, n_output]));
        bias_container.append(bias_variable([n_output]));

    return weight_container, bias_container

# get MLP readout,by given weights and biases
# weightes and biases : either be tf type or python type
# activation function : sigmoid as default
def MLP(input, weight_container, bias_container,act_func = tf.sigmoid):

    l = len(weight_container);
    if l != 1:
        #input layer
        preZ = tf.matmul(input,weight_container[0]) + bias_container[0];
        Z = act_func(preZ);

        #hidden layer
        for i in range(1,l-1):
            preZ = tf.matmul(Z,weight_container[i]) + bias_container[i];
            Z = act_func(preZ);

        # output layer
        Y = tf.matmul(Z,weight_container[-1])+bias_container[-1];
    else:
        Y = tf.matmul(input,weight_container[0])+bias_container[0];
    return Y

# train MLP
# optimizer : adapt (default)
# cost func : MSE (default)
def MLP_train(X,T,weight_container, bias_container, tf_session, optimizer= tf.train.AdamOptimizer,cost_func = MSE,
              show_command= 1, show_performance=0, batch_size = 0,max_iter = 4000,lowest_bounardy = 10e-6,train_precent = 0.7,frame=500, SAE= 0,RHO= 0.1,BETA = 3.0,lr = 0.0001):
    data_length = len(T);
    # ----- apply feature scaling ----- #
    preX = np.float32(featureScaling(X, 1, -1))
    if cost_func != crossEntropy:
        preT = np.float32(featureScaling(T, 1, -1))
    else :
        preT = T

    # ---- train & validation ---- #
    boundary = int(train_precent*data_length);
    train_X = preX[0:boundary,:];
    train_T = preT[0:boundary];
    ###
    if train_precent < 1:
        valid_X = preX[boundary:-1, :];
        valid_T = preT[boundary:-1];
    else:
        valid_X = train_X;
        valid_T = train_T;

    # ----- adjust batch_size ----- #
    if batch_size == 0:
        batch_size = int(data_length*0.05);
    if batch_size < 0:
        batch_size = len(train_X);
        max_iter = max_iter*10;

    lX,wX = getWH(train_X);
    lT,wT = getWH(train_T);

    # ----- initialise input for minibatch purpose ------ #
    input = tf.placeholder("float",shape=[None,wX]);
    output = tf.placeholder("float",shape=[None,wT])

    Y = MLP(input,weight_container, bias_container);

    BATCHSIZE = tf.placeholder("int32");

    if SAE:
        sparse = sparse_one_layer(input, weight_container, bias_container, BATCHSIZE, wX, modelSample, RHO, BETA,[]);
        PRECOST = cost_func(Y, output);
        cost = tf.add(cost_func(Y, output),
                  sparse_one_layer(input, weight_container, bias_container, BATCHSIZE, wX, modelSample, RHO, BETA,[]))
    else:
        cost = cost_func(Y, output);

    learning_rate = tf.placeholder("float");
    opt = optimizer(learning_rate).minimize(cost);

    tf_session.run(tf.global_variables_initializer());

    eval = np.inf
    bestLoc = 0

    loss_test = []
    loss_valid = []
    Xaxis = []

    i = 0
    epoch= max_iter

    # about learning rate #
    lr = lr
    pre_cost= 1;
    alpha = 0.001;
    lam = 0;
    r = 0.9;
    pre_eavl = 0;
    count_opt =0;
    # go gradient descent
    iter = int(boundary/batch_size);
    epoch = iter*epoch
    if show_command:
        print 'max interation : ' , epoch
    if show_performance:
        dplt = DynamicPlot();
        dplt.on_launch('loss')
        lines1 = dplt.requestLines('test','b-');
        lines2 = dplt.requestLines('validate','g-');
        lines3 = dplt.requestLines('best w b','r-');
    while i <= epoch:
            # minibatch
            batch_index = random.sample(np.arange(0,boundary),batch_size)
            X_batch = train_X[batch_index,:]
            T_batch = train_T[batch_index]

            tf_session.run(opt,feed_dict={learning_rate:lr,input:X_batch,output:T_batch,BATCHSIZE:batch_size})
            test_costV = tf_session.run(cost,feed_dict={input:train_X,output:train_T,BATCHSIZE:len(train_T)})
            valid_costV = tf_session.run(cost,feed_dict={input:valid_X,output:valid_T,BATCHSIZE:len(valid_T)})

            #if SAE :
               # print 'sparse : ' ,tf_session.run(sparse,feed_dict={input:train_X,output:train_T,BATCHSIZE:len(train_X)}),'cost :',tf_session.run(PRECOST,feed_dict={input:train_X,output:train_T,BATCHSIZE:len(train_X)});
               # evalV = (abs(test_costV - valid_costV) + 1) * max(valid_costV, test_costV) ;

            evalV = (abs(test_costV - valid_costV)+1)*max(valid_costV,test_costV)*10000

            if eval > evalV:
                eval = evalV
                final_w ,final_b = FC_get_model(weight_container,bias_container,tf_session);
                bestLoc = i;

            if i%frame == 0:
                loss_test.append(test_costV);
                loss_valid.append(valid_costV);
                Xaxis.append(i);


            slop = 1-(test_costV/pre_cost);

            if i%100 == 0 :
                if eval == evalV and i != 0:
                    lam = r*(slop**2)+(1-r)*lam;
                    v = alpha/mat.sqrt(lam)*slop;
                    lr = lr + v;
            if i%frame == 0:

                if show_performance:
                    dplt.on_running(lines1, Xaxis, loss_test);
                    dplt.on_running(lines2, Xaxis, loss_valid);
                    ends = max(np.max(loss_test), np.max(loss_valid))
                    starts = min(np.min(loss_test), np.min(loss_valid))
                    dplt.on_running(lines3, bestLoc, [starts, ends]);
                    dplt.draw();

                if pre_eavl == eval:
                    count_opt += 1;
                else:
                    count_opt = 0;
                pre_eavl = eval
                # ------- exist criteria ----- #
                if show_command :
                    print 'iteration : ',i, ' evaluation value : ',eval,' train cost : ',test_costV,' validate cost : ',valid_costV, ' lr : ',lr,'slop : ',slop,' exit count : ',count_opt
                if test_costV <= lowest_bounardy :
                    if show_command:
                        print 'reach exist critiera: lowest boundary'
                    break;
                if count_opt == 20:
                    if show_command:
                        print 'reach exist critiera: lowest boundary'
                    break;
                if slop == 0:
                    if show_command:
                        print 'reach exist critiera: slop equal 0'
                    break
            pre_cost = test_costV;
            i = i+1;
    if show_performance:
        dplt.done();
    return final_w,final_b,tf_session;
########################## Full connect network's wieght & bias #################################
# covert ternsor weight & bias to python type value & full connect network only
def FC_get_model(weight_container, bias_container,tf_session):

    l = len(weight_container);

    weight_out = [];
    bias_out = [];

    for i in xrange(l):
        weight_out.append(tf_session.run(weight_container[i]));
        bias_out.append(tf_session.run(bias_container[i]));

    return weight_out,bias_out;

########################## Multi-layer convolution neural network  ############################

# 2D convolution neural net
# note : doesnt contain
# input :
# inputSize - 2D array , size of input 'image'
# n_ouput  - integer, set num of output we want. size of ouput depends on kernals
# kernal - 2D array, specify kernals in every layers,
# return w,n,s,p
def MLCNN_init(inputSize,topo=[1],kernal=[[1,1]] ,stride = [1],padding= [-1],poolingON=1):

    feature_H = inputSize[0]
    feature_W = inputSize[1]
    feature_D = inputSize[2]

    n_layer = len(topo)
    weight_container = []
    bias_container = []

    #re-adjust kernal
    kernal,stride,padding,outSize = readjustKernal(feature_H,feature_W, kernal, stride ,padding)

    # input layer
    weight_container.append(weight_variable([kernal[0][0], kernal[0][1], feature_D, topo[0]]))
    bias_container.append(bias_variable([topo[0]]))

    # hidden layer
    for i in range(1,n_layer):
        weight_container.append(weight_variable([kernal[i][0],kernal[i][1],topo[i-1],topo[i]]));
        bias_container.append(bias_variable([topo[i]]));

    return weight_container,bias_container,stride,padding


# MLCNN with pooling layer
def MLCNN(input,weight_container,bias_container,stride,padding,poolon=0,act_func=tf.nn.relu):
    l = len(weight_container)
    # input layer
    preZ = conv2d(input, weight_container[0], stride[0], padding[0]) + bias_container[0]
    Z = act_func(preZ)
    if poolon:
        h_pool = max_pool(Z,stride = [2,2])

    # hidden layer
    for i in range(1,l):
        preZ = conv2d(h_pool, weight_container[i], stride[i], padding[i]) + bias_container[i]
        Z = act_func(preZ)
        if poolon:
            h_pool = max_pool(Z,stride = [2,2])
    return h_pool

# 2D convolution neural net
# with single layer network                                         weight_container
def MLCNN_init_out(inputSize,output,topo=[1], kernal=[[1,1]],stride = [1],padding = [-1]):

    feature_H = inputSize[0];
    feature_W = inputSize[1];
    feature_D = inputSize[2];

    n_layer = len(topo);
    weight_container = [];
    bias_container = [];

    #re-adjust kernal
    kernal,stride,padding,outSize = readjustKernal(feature_H,feature_W, kernal, stride ,padding);

    # input layer
    weight_container.append(weight_variable([kernal[0][0], kernal[0][1], feature_D, topo[0]]));
    bias_container.append(bias_variable([topo[0]]));

    # hidden layer
    for i in range(1,n_layer):
        weight_container.append(weight_variable([kernal[i][0],kernal[i][1],topo[i-1],topo[i]]));
        bias_container.append(bias_variable([topo[i]]));

    # output layer
    weight_container.append(weight_variable([outSize[0], outSize[1], topo[- 1], output]));
    bias_container.append(bias_variable([output]));

    return weight_container,bias_container,stride,padding;

# get MLCNN readout
def MLCNN_out(input, weight_container,bias_container,stride,padding,act_func = tf.nn.relu):

    l = len(weight_container);

    # input layer
    preZ = conv2d(input, weight_container[0], stride[0], padding[0]) + bias_container[0];
    Z = act_func(preZ);
    if l > 2:
        # hidden layer
        for i in range(1,l-1):
            preZ = conv2d(Z, weight_container[i], stride[i], padding[i]) + bias_container[i];
            Z = act_func(preZ);

    Y = conv2d(Z, weight_container[-1], [1,1], -1) + bias_container[-1];

    return Y;

################################   training method   ############################################
def modelSample(X,w,b,X_height,X_width,rest):
    Y = MLP(X,w,b);
    return Y;

def modelSampleInit(inputSize,outputSize,topo=[1],kernal=[1],stride = [1],padding = [-1]):
    layers = len(topo)
    kernals = np.ones([layers,2])
    for i in range(layers):
        kernals[i][1] = int(kernal[i])
    kernals = np.int64(kernals)
    return MLCNN_init_out([1,inputSize,1],outputSize,topo =topo,kernal = kernals,stride = stride, padding = padding)

def modelSampleCNN(X,w,b,xw,rest,MODEL = MLCNN_out,act_func= tf.nn.relu):
      s = rest[0];
      p = rest[1];
      out_size = rest[2]
      X = tf.reshape(X,[-1,1,xw,1])
      Y = MODEL(X,w,b,s,p,act_func=act_func);

      Y = tf.reshape(Y,[-1,out_size]);
      return Y;

def MLCNN2D_train_WBimg(X,T,weight_container, bias_container,s,p,tf_session,GET_MODEL=FC_get_model, optimizer= tf.train.AdamOptimizer,cost_func = MSE,
              show_command= 1, show_performance=0, batch_size = 0,max_iter = 4000,train_precent = 0.7,frame=500,act_func = tf.nn.relu):
    data_length = len(T)
    # ----- apply feature scaling ----- #
    preX = X
    preT = T

    # ---- train & validation ---- #
    boundary = int(train_precent*data_length)
    train_X = preX[0:boundary,:]
    train_T = preT[0:boundary,:]
    ###
    if train_precent < 1:
        valid_X = preX[boundary+1::, :]
        valid_T = preT[boundary+1::, :]
    else:
        valid_X = train_X
        valid_T = train_T

    # ----- adjust batch_size ----- #
    if batch_size == 0:
        batch_size = int(data_length*0.05)
    if batch_size < 0:
        batch_size = len(train_X)
        max_iter = max_iter*10

    lX, wX = getWH(train_X)
    lT, wT = getWH(train_T)

    # ----- initialise input for minibatch purpose ------ #
    input = tf.placeholder("float32",shape=[None,wX])
    output = tf.placeholder("float32",shape=[None,wT])

    Y = MLCNN_out(input,weight_container,bias_container,s,p,act_func=act_func)

    cost = cost_func(Y,output)

    learning_rate = tf.placeholder("float32")
    opt = optimizer(learning_rate).minimize(cost)

    tf_session.run(tf.global_variables_initializer())

    eval = np.inf
    bestLoc = 0

    loss_test = []
    loss_valid = []
    Xaxis = []

    i = 0
    epoch= max_iter

    # about learning rate #
    lr = 0.0001
    pre_cost= 1
    alpha = 0.001
    lam = 0
    r = 0.9
    pre_eavl = 0
    count_opt =0
    # go gradient descent
    iter = int(boundary/batch_size)
    epoch = iter*epoch
    if show_command:
        print 'max interation : ' , epoch
    if show_performance:
        dplt = DynamicPlot()
        dplt.on_launch('loss')
        lines1 = dplt.requestLines('test','b-')
        lines2 = dplt.requestLines('validate','g-')
        lines3 = dplt.requestLines('best w b','r-')
    while i <= epoch:
            # minibatch
            batch_index = random.sample(np.arange(0,boundary),batch_size)
            X_batch = train_X[batch_index,:]
            T_batch = train_T[batch_index,:]


            tf_session.run(opt,feed_dict={learning_rate:lr,input:X_batch,output:T_batch})
            test_costV = tf_session.run(cost,feed_dict={input:train_X,output:train_T})
            valid_costV = tf_session.run(cost,feed_dict={input:valid_X,output:valid_T})

            evalV = (abs(test_costV - valid_costV)+1)*max(valid_costV,test_costV)*10000;

            if eval > evalV:
                eval = evalV
                final_w ,final_b = GET_MODEL(weight_container,bias_container,tf_session);
                bestLoc = i;

            if i%frame == 0:
                loss_test.append(test_costV);
                loss_valid.append(valid_costV);
                Xaxis.append(i);


            slop = 1-(test_costV/pre_cost);

            if i%100 == 0 :
                if eval == evalV and i != 0:
                    lam = r*(slop**2)+(1-r)*lam;
                    v = alpha/mat.sqrt(lam)*slop;
                    lr = lr + v;
            if i%frame == 0:

                if show_performance:
                    dplt.on_running(lines1, Xaxis, loss_test);
                    dplt.on_running(lines2, Xaxis, loss_valid);
                    ends = max(np.max(loss_test), np.max(loss_valid))
                    starts = min(np.min(loss_test), np.min(loss_valid))
                    dplt.on_running(lines3, bestLoc, [starts, ends]);
                    dplt.draw();

                if pre_eavl == eval and i%frame==0:
                    count_opt += 1;
                else:
                    count_opt = 0;
                pre_eavl = eval
                # ------- exist criteria ----- #
                if show_command :
                    if cost_func != crossEntropy:
                        print 'iteration : ', i, ' evaluation value : ', eval, ' train cost : ', test_costV, ' validate cost : ', valid_costV, ' lr : ', lr, 'slop : ', slop, ' exit count : ', count_opt
                    else:
                        train_y = tf_session.run(Y,feed_dict={input:train_X,output:train_T})
                        correct_prediction_train = tf.equal(tf.argmax(train_T, 1), tf.argmax(train_y, 1))
                        valid_y = tf_session.run(Y, feed_dict={input: valid_X, output: valid_T})
                        correct_prediction_valid = tf.equal(tf.argmax(valid_T, 1), tf.argmax(valid_y, 1))
                        accuracy_train = tf_session.run(tf.reduce_mean(tf.cast(correct_prediction_train, tf.float32)))
                        accuracy_valid = tf_session.run(tf.reduce_mean(tf.cast(correct_prediction_valid, tf.float32)))
                        print 'iteration : ', i, ' evaluation value : ', eval, ' train cost : ', test_costV, ' validate cost : ', valid_costV, ' lr : ', lr, ' acc_train : ',accuracy_train,' acc_valid : ',accuracy_valid
                if count_opt == 20:
                    print 'reach exist critiera: lowest boundary'
                    break;
                if slop == 0:
                    print 'reach exist critiera: slop equal 0'
                    break
            pre_cost = test_costV;
            i = i+1;
    if show_performance:
        dplt.done();
    return final_w,final_b,tf_session;

def MLCNN2D_train(X,T,weight_container, bias_container,s,p,tf_session,MODEL=modelSampleCNN,GET_MODEL=FC_get_model, optimizer= tf.train.AdamOptimizer,cost_func = MSE,
              show_command= 1, show_performance=0, batch_size = 0,max_iter = 4000,train_precent = 0.7,frame=500,act_func = tf.nn.relu):
    data_length = len(T)
    # ----- apply feature scaling ----- #
    preX = np.float32(featureScaling(X, 1, -1))
    if cost_func != crossEntropy:
        preT = np.float32(featureScaling(T, 1, -1))
    else:
        preT = T

    # ---- train & validation ---- #
    boundary = int(train_precent*data_length)
    train_X = preX[0:boundary,:]
    train_T = preT[0:boundary,:]
    ###
    if train_precent < 1:
        valid_X = preX[boundary+1::, :]
        valid_T = preT[boundary+1::, :]
    else:
        valid_X = train_X
        valid_T = train_T

    # ----- adjust batch_size ----- #
    if batch_size == 0:
        batch_size = int(data_length*0.05)
    if batch_size < 0:
        batch_size = len(train_X)
        max_iter = max_iter*10

    lX, wX = getWH(train_X);
    lT, wT = getWH(train_T);

    # ----- initialise input for minibatch purpose ------ #
    input = tf.placeholder("float32",shape=[None,wX])
    output = tf.placeholder("float32",shape=[None,wT])

    Y = MODEL(input,weight_container,bias_container,wX,[s,p,wT],act_func=act_func)

    cost = cost_func(Y,output)

    learning_rate = tf.placeholder("float32")
    opt = optimizer(learning_rate).minimize(cost)

    tf_session.run(tf.global_variables_initializer())

    eval = np.inf
    bestLoc = 0

    loss_test = []
    loss_valid = []
    Xaxis = []

    i = 0
    epoch= max_iter

    # about learning rate #
    lr = 0.0001
    pre_cost= 1
    alpha = 0.001
    lam = 0
    r = 0.9
    pre_eavl = 0
    count_opt =0
    # go gradient descent
    iter = int(boundary/batch_size)
    epoch = iter*epoch
    if show_command:
        print 'max interation : ' , epoch
    if show_performance:
        dplt = DynamicPlot()
        dplt.on_launch('loss')
        lines1 = dplt.requestLines('test','b-')
        lines2 = dplt.requestLines('validate','g-')
        lines3 = dplt.requestLines('best w b','r-')
    while i <= epoch:
            # minibatch
            batch_index = random.sample(np.arange(0,boundary),batch_size)
            X_batch = train_X[batch_index,:]
            T_batch = train_T[batch_index,:]


            tf_session.run(opt,feed_dict={learning_rate:lr,input:X_batch,output:T_batch})
            test_costV = tf_session.run(cost,feed_dict={input:train_X,output:train_T})
            valid_costV = tf_session.run(cost,feed_dict={input:valid_X,output:valid_T})

            evalV = (abs(test_costV - valid_costV)+1)*max(valid_costV,test_costV)*10000;

            if eval > evalV:
                eval = evalV
                final_w ,final_b = GET_MODEL(weight_container,bias_container,tf_session);
                bestLoc = i;

            if i%frame == 0:
                loss_test.append(test_costV);
                loss_valid.append(valid_costV);
                Xaxis.append(i);


            slop = 1-(test_costV/pre_cost);

            if i%100 == 0 :
                if eval == evalV and i != 0:
                    lam = r*(slop**2)+(1-r)*lam;
                    v = alpha/mat.sqrt(lam)*slop;
                    lr = lr + v;
            if i%frame == 0:

                if show_performance:
                    dplt.on_running(lines1, Xaxis, loss_test);
                    dplt.on_running(lines2, Xaxis, loss_valid);
                    ends = max(np.max(loss_test), np.max(loss_valid))
                    starts = min(np.min(loss_test), np.min(loss_valid))
                    dplt.on_running(lines3, bestLoc, [starts, ends]);
                    dplt.draw();

                if pre_eavl == eval and i%frame==0:
                    count_opt += 1;
                else:
                    count_opt = 0;
                pre_eavl = eval
                # ------- exist criteria ----- #
                if show_command :
                    if cost_func != crossEntropy:
                        print 'iteration : ', i, ' evaluation value : ', eval, ' train cost : ', test_costV, ' validate cost : ', valid_costV, ' lr : ', lr, 'slop : ', slop, ' exit count : ', count_opt
                    else:
                        train_y = tf_session.run(Y,feed_dict={input:train_X,output:train_T})
                        correct_prediction_train = tf.equal(tf.argmax(train_T, 1), tf.argmax(train_y, 1))
                        valid_y = tf_session.run(Y, feed_dict={input: valid_X, output: valid_T})
                        correct_prediction_valid = tf.equal(tf.argmax(valid_T, 1), tf.argmax(valid_y, 1))
                        accuracy_train = tf_session.run(tf.reduce_mean(tf.cast(correct_prediction_train, tf.float32)))
                        accuracy_valid = tf_session.run(tf.reduce_mean(tf.cast(correct_prediction_valid, tf.float32)))
                        print 'iteration : ', i, ' evaluation value : ', eval, ' train cost : ', test_costV, ' validate cost : ', valid_costV, ' lr : ', lr, ' acc_train : ',accuracy_train,' acc_valid : ',accuracy_valid
                if count_opt == 20:
                    print 'reach exist critiera: lowest boundary'
                    break;
                if slop == 0:
                    print 'reach exist critiera: slop equal 0'
                    break
            pre_cost = test_costV;
            i = i+1;
    if show_performance:
        dplt.done();
    return final_w,final_b,tf_session;

############################ encoder method For trading ##################################
def getHiddenReadout(input,rest,model,size):
    X = input[-1];
    input = np.asarray(input);
    w = input[0:size];
    b = input[size:-1];

    w = np.asarray(w);
    b = np.asarray(b);
    xh = rest[0];
    xw = rest[1];
    r = rest[2];
    Y = tf.sigmoid(model(X,w[0:-1],b[0:-1],xh,xw,r));
    return Y;

def reward(Y,P,commission):
    n_of_act = len(Y[0]);
    a = np.argmax(Y,axis=1);
    # find best state
    out,r,f,profit = findBestActions(a,n_of_act,P,commission);
    return out,r,f,profit;

def testReward(Y,P,r,f,commission):
    a = np.argmax(Y,axis=1);
    a_ = to_signal(a,r,f);
    profit = TF.MarketState(a_,P,commission);
    reward = trading_valuation(profit);
    return reward,profit;

def to_signal(a,r,f):
    R = cp.copy(a); # for raise signal
    R[R==r] = -1;
    R[R!=-1] = 0;
    R[R==-1] = 1;
    F = cp.copy(a); # for fall signal
    F[F!=f] = 0;
    F[F==f] = -1;
    a_ = F+R;
    return a_

def findBestActions(a,n_of_act,T,commission):
    out = -np.inf;
    r = 0;
    f = 0;
    profit= [];
    for raiseSign in range(n_of_act):
        for fallSign in range(n_of_act):
            if raiseSign != fallSign:
                R = cp.copy(a); # for raise signal
                R[R==raiseSign] = -1;
                R[R!=-1] = 0;
                R[R==-1] = 1;
                F = cp.copy(a); # for fall signal
                F[F==fallSign] = -1;
                F[F!=-1] = 0;
                a_ = F+R;

                reward = TF.MarketState(a_,T,commission);
                compare = trading_valuation(reward)

                if out < compare:
                    out = compare;
                    profit = reward;
                    r = raiseSign;
                    f = fallSign;
    return out,r,f,profit;

def trading_valuation(profit):
    tp = sum(profit);
    nt = sum(profit!=0);
    pnt = nt - sum(profit<0);
    #ppt = tp/nt;
    risk = np.std(profit);
    if risk != 0:
        sharpe = tp/risk;
    else:
        sharpe = 0;
    if tp > 0:
        ts = abs(sharpe*tp);
    else:
        ts = -abs(sharpe*tp);

    return ts;

def placeholderMatrixList(ListMatrix):
    DL = len(ListMatrix);
    out = [];
    for i in range(DL):
        h,w = getWH(ListMatrix[i]);
        input = tf.placeholder("float32",shape=[h,w]);
        out.append(input);
    return out;

def placeholderList(List):
    DL = len(List);
    out = [];
    for i in range(DL):
        input = tf.placeholder("float32");
        out.append(input);
    return out;

def train_studyHiddenState(X,T,P,weight_container, bias_container,rest_container,tf_session,MODEL = modelSample, GET_MODEL=FC_get_model, optimizer= tf.train.AdamOptimizer,cost_func = MSE,
              show_command= 1, show_performance=0, batch_size = 0,max_iter = 4000,train_percent = 0.7,frame=500,commission = 2,RHO=0.01,BETA=3.):
    data_length = len(T);

    # ----- shuffle data ----- #
    shuffle_idx = random.sample(range(data_length),data_length);

    # ----- apply feature scaling ----- #
    preX = np.float32(featureScaling(X, 1, -1));
    preT = np.float32(featureScaling(T, 1, -1));

    preX_shuffle = preX[shuffle_idx];
    preT_shuffle = preT[shuffle_idx];
    # ---- train & validation ---- #
    boundary = int(train_percent*data_length);
    train_X = preX_shuffle[0:boundary,:];
    train_T = preT_shuffle[0:boundary];
    train_P = P[0:boundary];
    ###
    if train_percent < 1:
        valid_X = preX[boundary::, :];
        valid_T = preT[boundary::];
        valid_P = P[boundary::];
    else:
        valid_X = train_X;
        valid_T = train_T;
        valid_P = train_P;

    # ----- adjust batch_size ----- #
    if batch_size == 0:
        batch_size = int(data_length*0.05);
    if batch_size < 0:
        batch_size = len(train_X);
        max_iter = max_iter*10;

    lX,wX = getWH(train_X);
    lT,wT = getWH(train_T);

    # ----- initialise input for minibatch purpose ------ #
    input = tf.placeholder("float32",shape=[None,wX]);
    output = tf.placeholder("float32",shape=[None,wT])

    # about hidden readout #
    tf_session.run(tf.global_variables_initializer());
    w,b = GET_MODEL(weight_container,bias_container,tf_session)
    in_w = placeholderMatrixList(w);
    in_b = placeholderList(b);

    readout_input = (in_w+in_b)
    readout_input.append(input);

    BATCHSIZE = tf.placeholder("int32");

    hiddenReadout = getHiddenReadout(readout_input,[BATCHSIZE,wX,[],commission],MODEL,len(in_w))

    Y = MODEL(input,weight_container, bias_container,BATCHSIZE,wX,rest_container);


    cost = tf.add(cost_func(Y,output),sparse_one_layer(input,weight_container,bias_container,BATCHSIZE,wX,MODEL,RHO,BETA,rest_container))
    #cost = cost_func(Y,output)
    learning_rate = tf.placeholder("float32");
    opt = optimizer(learning_rate).minimize(cost);

    tf_session.run(tf.global_variables_initializer());

    eval = -np.inf;
    bestLoc = 0;

    loss_test = [];
    loss_valid = [];
    Xaxis = [];

    i = 0 ;
    epoch= max_iter;

    # about learning rate #
    lr = 0.0001;
    pre_cost= 1;
    alpha = 0.001;
    lam = 0;
    r = 0.9;
    pre_eavl = 0;
    count_opt =0;
    # go gradient descent
    iter = int(boundary/batch_size);
    epoch = iter*epoch;
    print 'max interation : ' , epoch
    if show_performance:
        dplt = DynamicPlot();
        dplt.on_launch('loss')
        lines1 = dplt.requestLines('test','b-');
        #lines2 = dplt.requestLines('validate','g-');
        lines3 = dplt.requestLines('best w b','r-');
        '''
        dplt_train = DynamicPlot();
        dplt_train.on_launch('test profit')
        profit_train = dplt_train.requestLines('profit','-');

        dplt_valid = DynamicPlot();
        dplt_valid.on_launch('valid profit')
        profit_valid = dplt_valid.requestLines('profit','-');
        '''
        dplt_overall = DynamicPlot();
        dplt_overall.on_launch('insample profit')
        profit_line = dplt_overall.requestLines('profit','-');

    while i <= epoch:
            #print i
            # minibatch
            batch_index = random.sample(np.arange(0,boundary),batch_size);
            X_batch = train_X[batch_index,:];
            T_batch = train_T[batch_index];

            tf_session.run(opt,feed_dict={learning_rate:lr,input:X_batch,output:T_batch,BATCHSIZE:batch_size});

            costV = tf_session.run(cost,feed_dict={input:train_X,output:train_T,BATCHSIZE:lX});

            w,b = GET_MODEL(weight_container,bias_container,tf_session)
            '''
            #------- test readout ----------#
            data_input_train = w+b;
            data_input_train.append(train_X)
            train_readout = tf_session.run(hiddenReadout,feed_dict={i: d for i, d in zip(readout_input,data_input_train)});
            train_costV,r,f,train_profit = reward(train_readout,train_P,commission);
            data_input = w+b;
            data_input.append(valid_X)
            valid_readout = tf_session.run(hiddenReadout,feed_dict={i: d for i, d in zip(readout_input,data_input)});
            valid_costV,valid_profit = testReward(valid_readout,valid_P,r,f,commission);
            '''

            data_input = w+b;
            data_input.append(preX);
            overall_readout = tf_session.run(hiddenReadout,feed_dict={i: d for i, d in zip(readout_input,data_input)});
            #overall_costV,overall_profit = testReward(overall_readout,P,r,f,commission);
            overall_costV,r,f, overall_profit = reward(overall_readout, P, commission);


            #plot
            if show_performance and i%frame == 0:
                '''
                dplt_train.on_running(profit_train,range(len(train_X)),np.cumsum(train_profit));
                dplt_train.draw();
                dplt_valid.on_running(profit_valid,range(len(valid_X)),np.cumsum(valid_profit));
                dplt_valid.draw();
                '''
                dplt_overall.on_running(profit_line,range(len(P)),np.cumsum(overall_profit))
                dplt_overall.draw()

            #evalV = (abs(train_costV - valid_costV)+1)*min(valid_costV,train_costV);
            evalV = overall_costV;


            if eval < evalV:
                eval = evalV
                final_w ,final_b = GET_MODEL(weight_container,bias_container,tf_session);
                bestLoc = i;

            if i%frame == 0:
                '''
                loss_test.append(train_costV);
                loss_valid.append(valid_costV);
                '''
                loss_test.append(overall_costV)
                Xaxis.append(i);
            if show_performance and i%frame == 0:
                dplt.on_running(lines1, Xaxis, loss_test);
                # dplt.on_running(lines2, Xaxis, loss_valid);
                # ends = max(np.max(loss_test), np.max(loss_valid))
                ends = np.max(loss_test)
                starts = min(np.min(loss_test), 0);
                dplt.on_running(lines3, bestLoc, [starts, ends]);
                dplt.draw();


            slop = 1-(costV/pre_cost);
            #print 'costV',costV
            #print 'pre_cost',pre_cost

            if i%100 == 0 :
                if eval == evalV and i != 0:
                    lam = r*(slop**2)+(1-r)*lam;
                    if  mat.sqrt(lam) != 0:
                        v = alpha/mat.sqrt(lam)*slop;
                    else:
                        v = 0;
                    lr = lr + v;
            if i%500 == 0:

                if pre_eavl == eval:
                    count_opt += 1;
                else:
                    count_opt = 0;
                pre_eavl = eval
                # ------- exist criteria ----- #
                if show_command :
                    print 'interation : ',i, ' evaluation value : ',eval, ' lr : ',lr,'slop : ',slop,' exit count : ',count_opt;
                if count_opt == 10:
                    print 'reach exist critiera: lowest boundary'
                    break;
                if slop == 0:
                    print 'reach exist critiera: slop equal 0'
                    break
            pre_cost = costV;
            i = i+1;
    if show_performance:
        dplt.done();
    return final_w,final_b,tf_session;

def getY(X,w,b,rest,sess,scaleON,model=modelSample):
    if scaleON:
        X = featureScaling(X,1,-1);
    hX, wX = getWH(X);
    Y = tf.sigmoid(model(X,w,b,hX,wX,rest));
    Y = sess.run(Y)
    #Y = defeatureScaling(Y,1,-1,T);
    return Y;

def getReward(X,T,w,b,rest,sess,model,train_percent,commission):
    data_length = len(T);
    # ----- apply feature scaling ----- #
    preX = np.float32(featureScaling(X, 1, -1));
    # ---- train & validation ---- #
    boundary = int(train_percent * data_length);
    train_X = preX[0:boundary, :];
    train_T = T[0:boundary];
    ###
    if train_percent < 1:
        valid_X = preX[boundary:-1, :];
        valid_T = T[boundary:-1];
    else:
        valid_X = train_X;
        valid_T = train_T;

    trainY = getY(train_X, w, b, rest, sess,0, model=model)
    train_reward,r,f,_ = reward(trainY,train_T,commission);

    validY = getY(valid_X, w, b, rest, sess,0, model=model)
    valid_reward, _ = testReward(validY,valid_T,r,f, commission);

    r = (abs(train_reward - valid_reward) + 1) * min(valid_reward, train_reward);

    return r;

def betterInitialization(times,X,T,P,rest,sess,model,model_init,TOPO,train_percent=0.7,commission=2):
    print 'searching for better initialization';
    final_w = [];
    final_b = [];
    eval = -np.inf;
    _,Xw = getWH(X);
    _,TW = getWH(T);
    for i in range(times):
        w,b = model_init(Xw,TW,topo = TOPO);
        sess.run(tf.global_variables_initializer())
        r = getReward(X,P,w,b,rest,sess,model,train_percent,commission);
        if r > eval:
            eval =r;
            final_w = w;
            final_b = b;
    print 'searching complete';
    return final_w,final_b;
#################################################################################################
# ----------------- create single layer customized net work ----------------- #
# generate auto decay kernal
#def autoDacayKernal1D(width,n_kernals,stride,paddings,decate_rate )
# auto adjust kernal function
def readjustKernal(height,width, kernals, stride,paddings, show_debug = 1,ispool=1):

    lk = len(kernals);
    lp = len(paddings);
    ls = len(stride);
    if lp == 1:
        paddings = np.ones(lk)*paddings[0];
    if ls == 1:
        stride = np.ones([lk,2])*stride[0];

    if show_debug:
        print '============= auto adjust kernals sizes =============='
    # --- adjust height
    if kernals[0][0] > height:
        if show_debug:
            print 'layer ', 0, 'kernal_h size :', kernals[0][0], 'is larger than its input_h size : ', height, ;

        if height != 1: kernals[0][0] = height - 1;
        else : kernals[0][0] = height;

        if show_debug:
            print' change to ', kernals[0][0];
    # --- adjust width
    if kernals[0][1] > width:
        if show_debug:
            print 'layer ', 0, 'kernal size :', kernals[0][1], 'is larger than its input size : ', width, ;

        if width != 1: kernals[0][1] = width - 1;
        else : kernals[0][1] = width;

        if show_debug:
            print ' change to ', kernals[0][1];
    # --- fist layer
    # height
    if paddings[0] == -1: # padding is "same"
        out_size_h = int(np.ceil(float(height-kernals[0][0]+1)/float(stride[0][0])));
        out_size_w = int(np.ceil(float(width - kernals[0][1] + 1) / float(stride[0][1])));
    else:
        out_size_h = int(np.ceil(float(height)/float(stride[0][0])));
        out_size_w = int(np.ceil(float(width) / float(stride[0][1])));
    if ispool:
        if out_size_h != 1: int(out_size_h / 2)
        if out_size_w != 1: out_size_w = int(out_size_w / 2)
    if show_debug:
        print 'layer 0 output size : ', [out_size_h,out_size_w];

    # --- hidden layer
    for i in range(1,lk):
        if kernals[i][0] > out_size_h:
            if show_debug:
                print 'layer ', i, 'kernal size :', kernals[i][0], 'is larger than its input size H : ', out_size_h,;
            # -- height
            if out_size_h > 1:kernals[i][0] = out_size_h-1;
            else:kernals[i][0] = out_size_h;
            if show_debug:
                print ' change to ', kernals[i][0];
        if kernals[i][1] > out_size_w:
            if show_debug:
                print 'layer', i, 'kernal size :', kernals[i][1], 'is larger than its input size w : ', out_size_w;
            # -- width
            if out_size_w > 1:kernals[i][1] = out_size_w-1;
            else: kernals[i][1] = out_size_w;
            if show_debug:
               print ' change to ', kernals[i][1];

        # height
        if paddings[i] == -1:  # padding is "same"
            out_size_h = int(np.ceil(float(out_size_h - kernals[i][0] + 1) / float(stride[i][0])));
            out_size_w = int(np.ceil(float(out_size_w - kernals[i][1] + 1) / float(stride[i][1])));
        else:
            out_size_h = int(np.ceil(float(out_size_h) / float(stride[i][0])));
            out_size_w = int(np.ceil(float(out_size_w) / float(stride[i][1])));
        if ispool:
            if out_size_h !=1 : int(out_size_h / 2)
            if out_size_w !=1 : out_size_w = int(out_size_w / 2)
        if show_debug:
            print 'layer',i,' output size : ',[out_size_h,out_size_w] ;
    if show_debug:
        print 'adjust kernal as : ', kernals
        print 'fianl out put size: ', [out_size_h,out_size_w]
        print '=============== auto adjust complete ================='
    return kernals,stride,paddings,[out_size_h,out_size_w];

# --------
def conv2d(x,W,stride,padding):
    if padding == -1:
        p = "VALID";
    else:
        p = "SAME";
    return tf.nn.conv2d(x,W,strides=[1,stride[0],stride[1],1],padding= p);

def conv1d(x,W,stride,padding):
    if padding == -1:
        p = "VALID";
    else :
        p = "SAME";
    return tf.nn.conv1d(x,W,stride,p);

def max_pool(x,stride=[2,2]):
    return tf.nn.max_pool(x,ksize = [1,stride[0],stride[1],1],strides =[1,stride[0],stride[1],1] ,padding="SAME")

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

# ------------------ pre-processing ------------------#

def featureScaling(input,MAX,MIN):

    size = input.shape;

    l = size[0];
    if len(size) == 2:
        w = size[1];
    else:
        w = 1;

    if w !=1 :
        out = np.zeros([l,w]);
        for i in xrange(w):
            if np.max(input[:, i]) - np.min(input[:, i]) != 0:
                out[:, i] = MIN + (input[:, i] - np.min(input[:, i]))*((MAX - MIN) / (np.max(input[:, i]) - np.min(input[:, i])));
    else:
        out = MIN + (input - np.min(input))*((MAX - MIN) / (np.max(input) - np.min(input)));

    return np.float32(out);

def featureScalingOutSample(input,maxMin,MAX,MIN):
    size = input.shape;

    l = size[0];
    if len(size) == 2:
        w = size[1];
    else:
        w = 1;

    if w != 1:
        out = np.zeros([l, w]);
        for i in xrange(w):
            if np.max(input[:, i]) - np.min(input[:, i]) != 0:
                maxv = maxMin[i][0]
                minv = maxMin[i][1]
                out[:, i] = MIN + (input[:, i] - minv) * ((MAX - MIN) / (maxv - minv))
    else:
        maxv = maxMin[0]
        minv = maxMin[1]
        out = MIN + (input - minv) * ((MAX - MIN) / (maxv - minv));

    return np.float32(out);

def featureScalingSingleV(input, MAX,MIN, observ):
    out = MIN + (input - np.min(observ)) * ((MAX - MIN) / (np.max(observ) - np.min(observ)));
    return out;

def getMaxMin(list):
    size = list.shape
    l = size[0]
    if len(size) == 2:
        w = size[1];
    else:
        w = 1;
    if w == 1:
        return [np.max(list),np.min(list)]
    else:
        out = np.zeros([w,2])
        for i in xrange(w):
            out[i][0] = np.max(list[:,i])
            out[i][1] = np.min(list[:,i])
        return out
def defeatureScaling(input,b,a,maxMin):
    size = input.shape;

    l = size[0];
    if len(size) == 2:
        w = size[1];
    else:
        w = 1;

    if w ==1 :
        Xmax = maxMin[0];
        Xmin = maxMin[1];
        out = (input-a)*(Xmax-Xmin)/(b-a)+Xmin;
    else :
        out = np.zeros([l,w])
        for i in xrange(w):
            Xmax = np.max(maxMin[i][0]);
            Xmin = np.min(maxMin[i][1]);
            out[:,i] = (input[:,i] - a) * (Xmax - Xmin) / (b - a) + Xmin;
    return np.float32(out);

def substract_Mean(input):

    size = input.shape;

    l = size[0];
    if len(size) == 2:
        w = size[1];
    else:
        w = 1;

    if w != 1:
        mean = np.mean(input,axis=0);
        input = input - mean;
    else:
        mean = np.mean(input);
        input = input - mean;

    return np.float32(input);

def normalization(input):

    size = input.shape;
    l = size[0];
    if len(size) == 2:
        w = size[1];
    else:
        w = 1;

    if w != 1:
        mean = np.mean(input, axis=0);
        std = np.std(input,axis=0);
        input = (input-mean)/std
    else:
        mean = np.mean(input);
        std = np.std(input);
        input = (input - mean) / std

    return input

def getWH(array):

    size = array.shape;
    l = size[0];
    if len(size) == 2:
        w = size[1];
    else:
        w = 1;

    return l,w;

#########################################################
##                   keras                             ##
#########################################################
### evaluation function in keras ####
def best_closeDistance(trainV,validV,isMin = 1):
    if isMin:
        return (abs(trainV - validV) + 0.1) * (max(validV, trainV))
    else:
        return ((1 / abs(trainV - validV+0.0001) + 0.1)) * (min(validV, trainV))

def best_validation(trainV,validV,isMin = 1):
    return validV
#### MLP ###
def Keras_MLP(X,T,topo=[1],actfunc= 'tanh', w_decay = [0.0]):
    n_layer = len(topo)
    input_w  = len(X[0])
    out_w = len(T[0])
    model = k.models.Sequential()
    if topo[0] != 0:
        #input layer
        model.add(k.layers.Dense(topo[0],input_shape=(input_w,),W_regularizer=k.layers.regularizers.l2(w_decay[0])))
        model.add(k.layers.Activation(actfunc))
        for i in range(1,n_layer):
            if i < (len(w_decay)-1):
                model.add(k.layers.Dense(topo[i],W_regularizer=k.layers.regularizers.l2(w_decay[i])))
            else:
                model.add(k.layers.Dense(topo[i]))
            model.add(k.layers.Activation(actfunc))
        #output layer
        model.add(k.layers.Dense(out_w))
    else:
        model.add(k.layers.Dense(out_w, input_shape=(input_w,)))
    return model

def Keras_train(X,T,keras_model,loss = 'MSE',lr = 0.0001,iteration = 20,show_performance = 0,train_percent = 0.7,
                batch_size = 128,lr_reduce_factor = 0.2,patience = 0,bestCount=8,verbose=0,evalFun= best_closeDistance):
    model = k.models.Sequential()
    model.add(keras_model)
    if loss is 'categorical_crossentropy':
        model.add(k.layers.Activation('softmax'))
    #optimizer
    adam = k.optimizers.Adam(lr = lr)
    #complie
    model.compile(loss = loss,optimizer = adam,metrics = ['accuracy'])

    if show_performance:
        dplt = DynamicPlot()
        if loss is 'categorical_crossentropy':
            dplt.on_launch('acc')
        else:
            dplt.on_launch('loss')
        lines1 = dplt.requestLines('train', 'b-')
        lines2 = dplt.requestLines('validation', 'g-')
        lines3 = dplt.requestLines('best w b', 'r-')

    Xaxis = []
    loss_train = []
    loss_val = []
    if loss is 'categorical_crossentropy':
        eval = 0
    else:
        eval = np.inf
    bestLoc = 0
    count = -1
    early_stopping = k.callbacks.EarlyStopping(monitor='val_loss', patience=patience)
    #keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)
    reduce_LR = k.callbacks.ReduceLROnPlateau(monitor='val_loss', factor = lr_reduce_factor,patience=int(patience*0.3))

    if batch_size < 0:
        batch_size = len(X)

    bestLoc = 0
    for i in range(iteration):
        count += 1
        if train_percent < 1 and train_percent > 0:
            histo = model.fit(X, T, batch_size=batch_size, nb_epoch=1, verbose=verbose, validation_split=(1-train_percent),callbacks=[early_stopping,reduce_LR])
        else:
            histo = model.fit(X, T, batch_size=batch_size, nb_epoch=1, verbose=verbose)

        #learningRate = histo.history.get('lr')
        #print 'learning rate : ' , learningRate[0]

        if loss is 'categorical_crossentropy':
            train_costV = histo.history.get('acc')[0]
            if train_percent < 1 and train_percent > 0:
                valid_costV = histo.history.get('val_acc')[0]
            else:
                valid_costV = train_costV
            evalV = evalFun(train_costV,valid_costV,isMin=0)

            print'Interation ', i, '/', iteration, 'evalV :', evalV, ' eval : ', eval,' train_eval : ',train_costV,' valid_eval : ',valid_costV, 'count : ',count
        else:
            train_costV = histo.history.get('loss')[0]
            if train_percent < 1 and train_percent > 0:
                valid_costV = histo.history.get('val_loss')[0]
            else:
                valid_costV = train_costV
            evalV = evalFun(train_costV,valid_costV,isMin=1)
            print'Interation ', i, '/', iteration, 'evalV :', evalV, ' eval : ', eval, ' train_eval : ', train_costV, ' valid_eval : ', valid_costV,'count : ',count

        loss_train.append(train_costV)
        loss_val.append(valid_costV)

        Xaxis.append(i)


        if (eval > evalV and loss is not 'categorical_crossentropy')or(eval < evalV and loss is 'categorical_crossentropy')and i>0 :
            eval = evalV
            model.save('nn_model.h5')
            #model_out = k.models.load_model('nn_model.h5')
            bestLoc = i
            count = -1
        if show_performance:
            dplt.on_running(lines1, Xaxis, loss_train)
            dplt.on_running(lines2, Xaxis, loss_val)
            ends = max(np.max(loss_train), np.max(loss_val))
            starts = min(np.min(loss_train), np.min(loss_val))
            dplt.on_running(lines3, bestLoc, [starts, ends])
            dplt.draw()

        if count == bestCount:
            break
        pre_cost = train_costV
    if show_performance:
        dplt.done()

    return k.models.load_model('nn_model.h5')


#### preprocess in keras ####
def Keras_preprocess_FeatureScaling(X,T,range=[1,-1]):
    resX = featureScaling(X,range[0],range[1])
    resT = featureScaling(T,range[0],range[1])
    return resX,resT
