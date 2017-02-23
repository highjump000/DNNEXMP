# created by bohuai jiang
# on 2017 2/7
import TradingFun as TF
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import NN_network as nn
#=================== load raw data =================#
PERIOD = 30
COMMISSION = 2
sptr = 15
update = 0
use_mirror = 1
ifclassification = 0
if update:
    address= '../../../Documents/RB/all_in_one.csv'
    data = TF.csvread(address)
    P = data[:,-1]
    X,T = TF.getXY(PERIOD,data[:,-1])
    if not use_mirror:
        TrainX,TrainP,TestX,TestP = TF.getTrainTest2D(0.7,X,T);
        TrainT = TF.getQtable(TrainP,sptr,COMMISSION)
    else :
        TrainX,TrainP,TestX,TestP = TF.getTrainTest2D_mirror(0.7,30,P)

        trT = TF.getQtable(TrainP[0:len(TrainP)/2],sptr,COMMISSION)
        trT_mirror = TF.getQtable(TrainP[len(TrainP)/2::],sptr,COMMISSION)
        TrainT = np.concatenate((trT,trT_mirror))
    TestT = TF.getQtable(TestP,sptr,COMMISSION)

    np.save('TrainX',TrainX)
    np.save('TrainT',TrainT)
    np.save('TrainP',TrainP)
    np.save('TestX',TestX)
    np.save('TestT',TestT)
    np.save('TestP',TestP)

TrainX = np.load('TrainX.npy')
TrainT = np.load('TrainT.npy')
TrainP = np.load('TrainP.npy')
TestX = np.load('TestX.npy')
TestT = np.load('TestT.npy')
TestP = np.load('TestP.npy')
'''
plt.figure().suptitle('train');
plt.plot(TrainT)
plt.figure().suptitle('test');
plt.plot(TestT);

plt.show()

# test
a = TF.getA(TrainT)
profit = TF.TSL(a,TrainP,sptr,COMMISSION,1)
TF.plotPerformance(profit,'desire reward')
plt.show()
'''
MLP = 0
if MLP:
    print 'using MLP'
    if ifclassification:
        TrainT = TF.toOneHotKey(TF.getA(TrainT),3)

    #====================build model =====================#
    sess = tf.InteractiveSession();
    #w,b = nn.MLP_init(30,3,topo = [0])
    w, b, s, p = nn.MLCNN_init([1, 30, 1], topo=[3], kernal=[[1, 30]])
    w = np.array([tf.reshape(w[0],[30,3])])

    if ifclassification:
        w,b,sess = nn.MLP_train(TrainX,TrainT,w,b,sess,show_performance=1,batch_size=1500,frame=1,cost_func = nn.crossEntropy)
    else:
        w, b, sess = nn.MLP_train(TrainX, TrainT, w, b, sess, show_performance=1, batch_size=32, frame=1,max_iter=100)
    #---- insample test ----#
    pre_Y = sess.run(nn.MLP(nn.featureScaling(TrainX,1,-1),w,b))
    maxMin = nn.getMaxMin(TrainT)
    if ifclassification:
        Y = pre_Y
    else:
        Y = nn.defeatureScaling(pre_Y, 1, -1, maxMin)
    a = TF.getA(Y)

    plt.figure().suptitle('in sample')
    plt.plot(a)

    profit = TF.TSL(a,TrainP,sptr,COMMISSION,1)
    TF.plotPerformance(profit,'in sample result')


    #---- outsample test ----#
    pre_Y = sess.run(nn.MLP(nn.featureScaling(TestX,1,-1),w,b))
    if ifclassification:
        Y = pre_Y
    else:
        Y = nn.defeatureScaling(pre_Y,1,-1,maxMin)
    a = TF.getA(Y)

    plt.figure().suptitle('out sample')
    plt.plot(a)

    profit = TF.TSL(a,TestP,sptr,COMMISSION,0)
    TF.plotPerformance(profit,'out sample result')
    plt.show()
else:
    #===================build cnn model===================#
    print 'using CNN'
    sess = tf.InteractiveSession()
    w,b,s,p = nn.modelSampleInit(30,3,topo=[100,10,30],kernal=[[1,8],[1,23],[1,1]])
    w,b,sess = nn.MLCNN2D_train(TrainX,TrainT,w,b,s,p,sess,show_performance=1,frame= 1,batch_size= 32,max_iter=100,act_func= tf.nn.relu)

    #---- insample test ----#
    pre_Y = sess.run(nn.modelSampleCNN(nn.featureScaling(TrainX,1,-1),w,b,len(TrainX),30,[s,p,3],act_func=tf.nn.relu))
    maxMin = nn.getMaxMin(TrainT)
    if ifclassification:
        Y = pre_Y
    else:
        Y = nn.defeatureScaling(pre_Y,1,-1,maxMin)
    a = TF.getA(Y)

    plt.figure().suptitle('in sample')
    plt.plot(a)

    profit = TF.TSL(a,TrainP,sptr,COMMISSION,1)
    TF.plotPerformance(profit,'in sample result')

    #---- outsample test ----#
    pre_Y = sess.run(nn.modelSampleCNN(nn.featureScaling(TestX,1,-1),w,b,len(TestX),30,[s,p,3],act_func=tf.sigmoid))
    if ifclassification:
        Y = pre_Y
    else:
        Y = nn.defeatureScaling(pre_Y,1,-1,maxMin)
    a = TF.getA(Y)

    plt.figure().suptitle('out sample')
    plt.plot(a)

    profit = TF.TSL(a,TestP,sptr,COMMISSION,0)
    TF.plotPerformance(profit,'out sample result')
    plt.show()
