import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
from TSL import TSL


class PureReg:
    X = [];
    T = [];
    def __init__(self,X,T):
        self.X = np.asarray(X);
        self.T = np.asarray(T);

    def PureReg(self):
        X = np.asarray(self.X);
        Y = np.asarray(self.T);
        # add bias
        X = self.addBias(X);


        X = np.matrix(X);
        Y = np.matrix(Y);
        XT = X.H

        W = np.linalg.solve((XT * X) ,(XT * Y));

        return W
    def addBias(self,X):
        outX = np.append(X,np.ones([len(X),1]),1);
        return outX;

    # gradient descent version of PreReg
    def PureRegTF(self):
        length,width = self.X.shape;
        #preprocess data
        X = self.featureScaling(self.X,1,-1);
        T = self.featureScaling(self.T,1,-1);
        # ----
        # prepare tensorflow
        sess = tf.Session();
        # build model
        #w = tf.Variable(tf.truncated_normal([width, 3], stddev=0.1))
        w = tf.Variable(tf.constant(0.0,shape = [width,1]));
        b = tf.Variable(tf.constant(1.0,shape = [1]));
        y = tf.matmul(X,w)+b;
        cost = tf.reduce_mean(tf.square(y-T));

        # set opt function
        learning_rate = tf.placeholder("float")
        opt = tf.train.RMSPropOptimizer(learning_rate).minimize(cost);
        # run gradient descent
        sess.run(tf.global_variables_initializer())
        eval = np.inf;

        loss = [];
        i = 0;
        epoch = 100;
        MAX = 20000;
        lr = 0.0001;
        while i <= epoch:
            sess.run(opt,feed_dict={learning_rate:lr})
            #print i
            if eval > sess.run(cost):
                eval = sess.run(cost);
                final_w = sess.run(w);
                final_b = sess.run(b);
            #else :
            #   break;
            loss.append(sess.run(cost));
            if i == epoch and eval == sess.run(cost):
                epoch = epoch + 200;
            if i%500 == 0:
                if eval == sess.run(cost):
                    lr = lr + lr;
                    print 'epoch : ',i
            if epoch == MAX:
                break;
            i = i + 1;
        #plot loss
        plt.figure().suptitle('loss',fontsize = 20);
        plt.plot(loss);
        return final_w,final_b,sess,eval

    def getYTF(self):
        w, b, sess, cost = self.PureRegTF();
        X = self.featureScaling(self.X, 1, -1);
        Y = tf.matmul(X, w) + b;
        Y = sess.run(Y);
        Y = self.featureScaling(Y,np.max(self.T),np.min(self.T));

        return Y;

    def getY(self):
        W = self.PureReg();
        X = np.asarray(self.X);
        X = self.addBias(X);
        Y = X * W;

        return Y;

    def plotCompareY(self):
        Y1 = self.getY();
        Y2 = self.getYTF();
        plt.figure()
        plt.plot(Y1,label='arithmatic');
        #plt.legend();
        #plt.figure()
        plt.plot(Y2,label='tensorflow');
        plt.legend();
        plt.show();


    # gradient descent version of show Performance
    def runInSampleTF(self):
        w,b,sess,cost = self.PureRegTF();
        X = self.featureScaling(self.X, 1, -1);
        Y = tf.matmul(X,w)+b;
        Y = sess.run(Y);
        a = Y.argmax(axis=1) - 1;
        MSE = cost;
        # -----------
        # TSL out8396e-06  -9.63285493e-06  -9.88787360e-06]
        profit = TSL(a, self.X[:, 0], 15, 2, 1)
        # -----------
        #fig = plt.figure();
        #plt.plot(np.cumsum(profit),label= 'tf');
        #fig.suptitle('profit', fontsize=20)
        return profit,MSE

    # this is compare version
    def runCompareInsample(self):
        p1,MSE1 = self.runInSample();
        p2,MSE2 = self.runInSampleTF();

        print MSE1 , MSE2

        plt.figure().suptitle('profit',fontsize=20);
        plt.plot(np.cumsum(p1),label= 'arithemtic');
        plt.plot(np.cumsum(p2),label= 'tensorflow');
        plt.legend();
        plt.show();

    def runInSample(self):
        W = self.PureReg();
        X = np.asarray(self.X);
        X = self.addBias(X);
        Y = X*W;
        a = Y.argmax(axis = 1)-1;
        # -----------
        #TSL out
        profit = TSL(a, X[:,0], 15, 2, 1);

        MSE = np.mean(np.mean(np.square(self.T - Y),axis=0));

        # -----------

        #plt.plot(np.cumsum(profit),label= 'arithemtic');
        return profit,MSE


    def runOutSample(self,W):
        X = np.matrix(np.asarray(self.X));
        X = self.addBias(X);
        Y = X * W;
        a = Y.argmax(axis=1) - 1;
        D_len = len(Y);
        profit = np.zeros(D_len);

        for i in xrange(D_len):
            profit[i] = self.T[a[i]];

        return profit

    def featureScaling(self,matrix,MAX,MIN):
        # convert to matrix
        matrix = np.float32(matrix);
        matrix = np.matrix(matrix);
        l,w = matrix.shape;

        for i in xrange(w):
            if np.max(matrix[:,i])-np.min(matrix[:,i]) == 0:
                matrix[:, i] = np.zeros([l,1]);
            else:
                matrix[:,i] = MIN + np.multiply((matrix[:,i]-np.min(matrix[:,i])),(MAX-MIN)/(np.max(matrix[:,i])-np.min(matrix[:,i])));

        return matrix