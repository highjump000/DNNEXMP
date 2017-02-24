# from https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py


# from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

#np.random.seed(1024)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
#from keras.utils.visualize_util import plot

from keras import backend as K
import h5py
from keras.models import load_model
import NN_network as nn

#######################################################################
##                     training parameters                           ##
#######################################################################
batch_size = 128
nb_classes = 11
nb_epoch = 100
#######################################################################
##                   extract data & reformat                         ##
#######################################################################
# input image dimensions
img_rows, img_cols = 28, 28
# size of pooling area for max pooling
pool_size = (2, 2)

# the data, shuffled and split between train and test sets
datafile_X = 'X.npy'
datafile_y = 'T.npy'

X_data = np.load(datafile_X)
y_data = np.load(datafile_y)

p = 0.7 #
numdata = X_data.shape[0]
X_train = X_data[:int(numdata*p),:]
y_train = y_data[:int(numdata*p),:]

X_test = X_data[int(numdata*p):,:]
y_test = y_data[int(numdata*p):,:]


if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# to one hot key
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
X = np.reshape(X_data,[-1,img_rows,img_rows,1])
T = np_utils.to_categorical(y_data,nb_classes)
##########################################################################
##                     build model on keras                             ##
##########################################################################

model = Sequential()
# feature extraction layer
model.add(Convolution2D(4,5,5,border_mode='same',input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(12,5,5,border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Convolution2D(20,5,5,border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
# classification model
model.add(Flatten())
model.add(Dense(200))
model.add(Activation('sigmoid'))
model.add(Dense(50))
model.add(Activation('sigmoid'))
model.add(Dense(nb_classes))
#smodel.add(Activation('softmax'))

#model.add(Activation('softmax'))



#model.compile(loss='categorical_crossentropy',
#              optimizer='adadelta',
#              metrics=['accuracy'])

#model = nn.Keras_train(X,T,model,show_performance=1)

model = nn.Keras_train_Simple(X_train,Y_train,X_test,Y_test,model,eval= 'acc',opt='adadelta',loss ='categorical_crossentropy',batch_size=batch_size,iteration=nb_epoch)
print 'done'
plt.savefig('/home/yohoo/Pictures/hw2.png')
plt.show()
#reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
#            patience=5, min_lr=0.001)

model.save('hw2_peng.h5')

'''
loss = hiso.history.get('loss')
acc =  hiso.history.get('acc')

#print (loss)
#print acc

plt.plot(loss)
plt.plot(acc)
plt.title('Train data')
plt.xlabel('epoch')
plt.ylabel('loss_acc')
plt.legend(["loss", "acc"])

plt.savefig('hw2_peng.png')
plt.close()
'''
############################################################################
##                         out sample test                                ##
############################################################################

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

