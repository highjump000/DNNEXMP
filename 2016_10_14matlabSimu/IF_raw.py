#created by bohuai jiang
#read IF.CSV filed
#return its X and Y
from numpy import genfromtxt
import numpy as np
import math as m
class IF_raw:

        Data=[];

        def __init__(self,address):
                self.Data = genfromtxt(address,delimiter = ',');

        def getX(self):
                X = [];
                for i in xrange(len(self.Data)):
                        close = self.Data[i][7];
                        Xrow = [];
                        for j in range(2,32):
                                Xrow.append(m.log(self.Data[i][j]/close));
                        #Xrow.append(1);
                        X.append(Xrow);
                return np.asarray(X);
        def getXB(self):
                X = [];
                for i in xrange(len(self.Data)):
                        close = self.Data[i][7];
                        Xrow = [];
                        for j in range(2,32):
                                Xrow.append(self.Data[i][j]);
                                #Xrow.append(m.log(self.Data[i][j]/close));
                        Xrow.append(1);
                        X.append(Xrow);
                return np.asarray(X);

        def getProfit(self,act):
                con = 'aggressive'
                nt = 300;
                pou = 0.000025;
                slippage = 100;
                H = len(self.Data);
                profit = np.zeros(H);
                for i in xrange(H):
                        inC = self.Data[i][32];
                        inB = self.Data[i][33];
                        inA = self.Data[i][34];
                        outC = self.Data[i][35];
                        outB = self.Data[i][36];
                        outA = self.Data[i][37];
                        if con == 'close':
                                profit[i] = (outC-inC)*nt*act[i]-((outC+inC)*nt*pou+slippage)*abs(act[i]);
                        if con == 'passive':
                                if act[i] == 1:
                                        profit[i] = (outA-inB)*nt-((outA+inB)*nt*pou+slippage);
                                if act[i] == -1:
                                        profit[i] = (inA-outB)*nt-((inA+outB)*nt*pou+slippage);
                        if con == 'aggressive':
                                if act[i] == 1:
                                        profit[i] = (outB-inA)*nt-((outB+inA)*nt*pou+slippage);
                                if act[i] == -1:
                                        profit[i] = (inB-outA)*nt-((outA+inB)*nt*pou+slippage);
                return profit;

        def getY(self):
                DL = len(self.Data);
                a1 = np.ones(DL)*-1;
                a3 = np.ones(DL);
                r1 = self.getProfit(a1);
                r2 = np.zeros(DL);
                r3 = self.getProfit(a3);
                T = [];
                T.append(r1);
                T.append(r2);
                T.append(r3);
                return np.asarray(T).transpose();

        def getPerformance(self,Q):
                a = self.getAction(Q);
                profit = np.asarray(self.getProfit(a));
                return profit;

        def getPerformanceV2(self,score):
                score = np.asarray(score);
                a = score.argmax(axis=1) - 1;
                profit = np.asarray(self.getProfit(a));
                return profit;

        def getAction(self,Q):
                X = self.getXB();
                Y = X * Q;
                a = Y.argmax(axis=1) - 1;
                return a;
        def getActionV2(self,score):
                socre = np.asarray(score)
                a = score.argmax(axis=1) - 1;
                return a;
        def getX3D(self):
                X = self.getX()
                H,W = X.shape
                out = np.zeros((H,W,1))
                for i in range(0,H):
                        out[i,:,0] = X[i,:];
                return out;

        def getX3DB(self):
                X = self.getXB()
                H,W = X.shape
                out = np.zeros((H,W,1))
                for i in range(0,H):
                        out[i,:,0] = X[i,:];
                return out;

        def getY3D(self):
                Y = self.getY()
                H, W = Y.shape
                out = np.zeros((H, W, 1))
                for i in range(0, H):
                        out[i, :, 0] = Y[i, :];
                return out;

        def getOneHotLKey(self):
                Y = self.getY();
                act = self.getActionV2(Y);
                out = [];
                for i in range(len(act)):
                        if act[i] == -1:
                                out.append([1,0,0]);
                        if act[i] == 0:
                                out.append([0,1,0]);
                        if act[i] == 1:
                                out.append([0,0,1]);
                return out

        def getXNormailzed(self):
                X = self.getX()
                n = len(X[0])

                mean = []
                var  = []
                for i in xrange(n):
                        mean.append(np.mean(X[i,:]))
                        var.append(np.var(X[i,:]))
                out = [];
                for i in xrange(n):
                        norma = (X[:,i] - mean[i])/var[i]
                        out.append(norma)
                out = np.asarray(out);
                return out.transpose();
        #normalization
        def normailzation(self,X):
                n = len(X[0])
                mean = []
                var = []
                for i in xrange(n):
                        mean.append(np.mean(X[i, :]))
                        var.append(np.var(X[i, :]))
                out = [];
                for i in xrange(n):
                        norma = (X[:, i] - mean[i]) / var[i]
                        out.append(norma)
                out = np.asarray(out);
                return out.transpose();
        #feature scalling
        def feature_scalling(self,X):
                n = len(X[0]);
                maxV = [];
                minV = [];
                for i in xrange(n):
                        maxV.append(np.max(X[i, :]))
                        minV.append(np.min(X[i, :]))
                out = [];
                for i in xrange(n):
                        norma = -1+(X[:, i] - minV[i])*(1+1) / (maxV[i]-minV[i])
                        out.append(norma)
                out = np.asarray(out);
                return out.transpose();

        def getXNormailzedB(self):
                X = self.getXB()
                n = len(X[0])

                mean = []
                var = []
                for i in xrange(n):
                        mean.append(np.mean(X[i, :]))
                        var.append(np.var(X[i, :]))
                out = [];
                for i in xrange(n):
                        norma = (X[:, i] - mean[i]) / var[i]
                        out.append(norma)
                out = np.asarray(out);
                return out.transpose();




