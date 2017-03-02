#created by bohuai jiang
# on 2016,12,12 random walk
import numpy as np
import math as mt

class RandomWalk:
    datalength = 0;
    sigma = 1;
    alpha = 0.7;
    p_0 = 20;
    beta_0 = 10;
    k = 3;
    commission = 2;
    def __init__(self,datalength):
        self.datalength = datalength;
    def generated(self):
        mu = 0.0;
        e = np.random.normal(mu, self.sigma,self.datalength);
        v = np.random.normal(mu, self.sigma,self.datalength);
        p  = np.zeros(self.datalength);
        beta = np.zeros(self.datalength);
        p[0] = self.p_0;
        beta[0] = self.beta_0;

        for t in range(1,self.datalength):
            p[t] = p[t-1] +beta[t-1] + self.k*e[t];
            beta[t] = self.alpha*beta[t-1]+v[t];
        return p;

    def getXY(self,period):
        p = self.generated();
        datalen = self.datalength-period-1;
        X = np.zeros([datalen,period]);
        T = np.zeros(datalen);
        for i in range(period,self.datalength-1):
            for j in xrange(period):
                idx = i-j;
                X[i-period][j] = p[i-j];
            #T[i-period][0] = p[i] - p[i+1] - self.commission;
            #T[i-period][2] = p[i+1]-p[i] -self.commission;
            T[i-period] = p[i+1];
        X = np.matrix(X);
        return X,T






