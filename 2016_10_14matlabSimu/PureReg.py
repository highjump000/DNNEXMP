import numpy as np

def PureReg(X,Y):
    X = np.asarray(X);
    Y = np.asarray(Y);
    dataSize,nfeatures = X.shape;
    #add bias
    X = addBias(X);
    XT = X.transpose();

    X = np.matrix(X);
    Y = np.matrix(Y);
    XT = np.matrix(XT);

    W = np.linalg.inv(XT*X)*(XT*Y);
    return W;

def addBias(X):
    out = [];
    for i in xrange(len(X)):
        temp = [];
        for j in xrange(len(X[0])):
            temp.append(X[i][j])
        temp.append(1);
        out.append(temp);
    return np.asarray(out);