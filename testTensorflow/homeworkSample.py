#created by bohuai jiang
#on 2017/2/14

import numpy as np
import TradingFun as TF
from keras.datasets import mnist
from keras.optimizers import  adadelta,adam
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import matplotlib.pyplot as plt
from DynamicPlot import DynamicPlot
from keras import backend as K
import copy
import keras
import NN_network as nn

######################################
##            train part            ##
######################################

nb_classes = 11
# -------- load data --------- #

X = np.load('X.npy')
T = np.load('T.npy')
labeles = np.load('label_name.npy')
# -- convert X to 2D -- #
img_rows, img_cols = 28, 28

trainX,trainT,testX,testT = TF.getTrainTest2D(0.9,X,T)

trainX = np.float32(np.reshape(trainX,[-1,28,28,1]))
testX  = np.float32(np.reshape(testX ,[-1,28,28,1]))

trainT = np_utils.to_categorical(trainT,nb_classes)
testT = np_utils.to_categorical(testT,nb_classes)

input_shape = (img_rows, img_cols,1)
#######################################
##            model build            ##
#######################################

def MSE(y,T):
    loss = K.mean(K.square(y-T),axis=1)
    return loss

model = Sequential()
# feature extraction layer
model.add(Convolution2D(4,5,5,border_mode='same',input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(12,5,5,border_mode='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Activation('relu'))
# classification model
model.add(Flatten())
model.add(Dense(100))
model.add(Activation('sigmoid'))
model.add(Dense(50))
model.add(Activation('sigmoid'))
model.add(Dense(nb_classes))
#model.add(Activation('softmax'))

#adadelta = adadelta(lr=1.5)

model = nn.Keras_train(trainX,trainT,model,show_performance=1,loss='categorical_crossentropy')



'''
dplt = DynamicPlot()
dplt.on_launch('loss')
lines1 = dplt.requestLines('test','b-')
lines2 = dplt.requestLines('validation','g-')
Xaxis = []
loss_test = []
loss_val = []

Adadelta = adadelta()

model.compile(loss= 'categorical_crossentropy',
               optimizer= Adadelta,
               metrics=['accuracy'])
#model.save('model.h5')

#model2 = load_model('model.h5')
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=1, min_lr=0.001)
#lrate = keras.callbacks.LearningRateScheduler()
for i in range(10):

    histo = model.fit(trainX,trainT, batch_size=128, nb_epoch=1,verbose=0,validation_split=0.3,callbacks=[reduce_lr])
    #print 'learning rate',histo.history.get('lr')
    loss_test.append(histo.history.get('loss')[0])
    loss_val.append(histo.history.get('val_loss')[0])
    Xaxis.append(i)
    #print loss_test
    dplt.on_running(lines1,Xaxis,loss_test)
    dplt.on_running(lines2,Xaxis,loss_val)
    dplt.draw()

dplt.done()
plt.show()
model.save('mnist_model.h5')
'''

############################################################################
##                         out sample test                                ##
############################################################################

score = model.evaluate(testX, testT, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])
#print model.predict_classes(testX[1:2],batch_size=1)

score = model.evaluate(testX, testT, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])
#print model2.predict_classes(testX[1:2],batch_size=1)