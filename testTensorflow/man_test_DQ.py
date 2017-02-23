#created by bohuai jiang
# test new deep Q
import numpy as np
from RandomWalk import RandomWalk;
import NN_network as nn
import matplotlib.pyplot as plt

########## load data ##########

n_features = 30;
updata = 1;
study_length = 100;

if updata:
    generator = RandomWalk(10000);
    price = generator.generated();
    X,T,Xb= generator.getXY(n_features)

    np.save('Xb',Xb);
    np.save('T',T);
    np.save('X',X);

Xb_org = np.load('Xb.npy');
T_org = np.load('T.npy');
X_org = np.load('X.npy');


# split data
Xb = Xb_org[0:study_length,:];
T = T_org[0:study_length];
X = X_org[0:study_length,:]

########## run deep Q ##############

