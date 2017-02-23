#created by bohuai jiang
#on 2017/1/9de
#market state investigation model

import TradingFun as TF
import matplotlib.pyplot as plt
import NN_network as nn
import tensorflow as tf
import numpy as np
############ load raw data ###############
PERIOD = 30
COMMISSION = 2

address= '../../../Documents/RB/all_in_one.csv'
data = TF.csvread(address)
P = data[:,-1]
X,T = TF.getXY(PERIOD,data[:,-1])
TrainorgX,TrainorgT,TestX,TestT = TF.getTrainTest2D(0.7,X,T);
TrainX,TrainT,TestX,TestT = TF.getTrianTest2D_image(0.7,30,P)

plt.figure().suptitle('train');
plt.plot(TrainT)
plt.figure().suptitle('test');
plt.plot(TestT);

plt.show()

############ build model ##########

sess = tf.InteractiveSession()
w,b = nn.MLP_init(30,1,topo=[10,3])

#w,b = nn.betterInitialization(1,X,X,T,[],sess,nn.modelSample,nn.MLP_init,[3]);

#sess.run(tf.global_variables_initializer());
#w,b,sess = nn.train_studyHiddenState(TrainX,TrainT,TrainT,w,b,[],sess,show_performance=1,train_percent=0.7)
w,b,sess = nn.MLP_train(TrainX,TrainT,w,b,sess,train_precent=1,show_performance=1,SAE=1,RHO=0.0005,BETA = 10.0,batch_size=-1);


# insample
Y = nn.getY(nn.featureScaling(TrainorgX,1,-1),w[0:-1],b[0:-1],[],sess,1);

plt.figure().suptitle("insanmple")
plt.plot(Y,".");
plt.plot(nn.featureScaling(TrainorgT,1,-1),label= "price")


reward,r,f,profit = nn.reward(Y,TrainorgT,COMMISSION);
TF.plotPerformance(profit,"insample")
print "reward",reward


# outsample
Y = nn.getY(nn.featureScaling(TestX,1,-1),w[0:-1],b[0:-1],[],sess,1);
reward,profit = nn.testReward(Y,TestT,r,f,COMMISSION)
TF.plotPerformance(profit,"outsample")
print "reward",reward;


plt.figure().suptitle("outsanmple")
plt.plot(Y,".");
plt.plot(nn.featureScaling(TestT,1,-1),label= "price")
plt.show()


