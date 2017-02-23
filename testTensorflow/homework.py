#created by bohuai jiang
#on 2017/2/14
import numpy as np
import scipy.io
import random
#--- read data ---#
mat = scipy.io.loadmat('../../../Documents/img/caltech101_silhouettes_28.mat')

X = mat.get('X')
label = mat.get('Y')[0]
name = mat.get('classnames')

#--- extract partial Data for homework ---#
homework_X = []
homework_T = []
homework_label = []

total = 0
for i in range(101):
    num = sum(label==i+1)
    if num >= 100:
        total += num
        print 'label ',name[0][i][0],' : ',sum(label==i+1)
        idx =np.nonzero(label==i+1)[0]
        homework_X.append(X[idx])
        homework_T.append(label[idx])
        homework_label.append(name[0][i][0])
print 'total length : ',total
DL = len(homework_T)
tempX = homework_X[0]
tempT = np.zeros(len(homework_T[0]))

#- format data

for i in range(1,DL):
    tempX = np.concatenate((tempX,homework_X[i]),axis=0)
    tT = np.ones(len(homework_T[i]))*i
    tempT = np.concatenate((tempT, tT), axis=0)
tempT = np.reshape(tempT,[-1,1])
homework_X = tempX
homework_T = tempT

#--- split data into test & validate ---#
DL = len(homework_X)
index = range(DL)
random_index = random.sample(index,DL)
homework_X = homework_X[random_index,:]
homework_T = homework_T[random_index]
#-- train 85% test 15% --#
train_percent = 0.85
boundary = int(train_percent*DL)
X = homework_X[0:boundary,:]
T = homework_T[0:boundary]
X_exam = homework_X[boundary+1::,:]
T_exam = homework_T[boundary+1::]

print 'T max     :' ,max(T)
print 'T exam max:' ,max(T_exam)
#-- save result --#
print 'data saved'
np.save('X',X)
np.save('T',T)
np.save('label_name',homework_label)
np.save('X_exam',X_exam)
np.save('T_exam',T_exam)