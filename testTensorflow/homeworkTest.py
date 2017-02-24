#created by bohuai jiang on 2017/2/24

from keras import models
import numpy as np
from keras.utils import np_utils

def toReadable(X,T):
    X = np.reshape(X, [-1, 28, 28, 1])
    T = np_utils.to_categorical(T, 11)
    return X,T

## load data
org_X = np.load('X_exam.npy')
org_T = np.load('T_exam.npy')
names = np.load('label_name.npy')
## load model
model_name = 'hw2_kxl_model.h5'
model = models.load_model(model_name)

## over all accuracy
X,T = toReadable(org_X,org_T)
score = model.evaluate(X, T, verbose=0)
print'overall accuracy :', score[1]

## acc in detial
for i in range(11):
    idex = np.nonzero(org_T==i)[0]
    T = org_T[idex]
    X = org_X[idex,:]
    X, T = toReadable(X, T)
    score = model.evaluate(X, T, verbose=0)
    print("accuracy:%.4f class[%s] label[%d] data size=%d"%(score[1],names[i],i,len(idex)))
    #print' accuracy:', score[1],'class [',names[i],'],label',i
