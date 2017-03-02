#created by bohuai jiang
#read IF.CSV filed
#return its X and Y
from numpy import genfromtxt
import numpy as np
import math as m
class IF:
        Data=[];
        def __init__(self,address):
                self.Data = genfromtxt(address,delimiter = ',');
        def getX(self):
                X = [];
                for i in xrange(len(self.Data)):
                        close = self.Data[i][7];
                        Xrow = [];
                        for j in range(2,7):
                                Xrow.append(m.log(close/self.Data[i][j]));
                        Xrow.append(1);
                        X.append(Xrow);
                return np.matrix(np.asarray(X));
        def getProfit(self,act):
                con = 'aggressive'
                nt = 300;
                pou = 0.000025;
                slippage = 100;
                H = len(self.Data);
                profit = np.zeros(H);
                for i in xrange(H):
                        inC = self.Data[i][7];
                        inB = self.Data[i][8];
                        inA = self.Data[i][9];
                        outC = self.Data[i][11];
                        outB = self.Data[i][12];
                        outA = self.Data[i][13];
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
                return np.matrix(np.asarray(T)).transpose();

        def getPerformance(self,Q):
                a = self.getAction(Q);
                profit = np.asarray(self.getProfit(a));
                return profit;
        def getAction(self,Q):
                X = self.getX();
                Y = X * Q;
                a = Y.argmax(axis=1) - 1;
                return a;