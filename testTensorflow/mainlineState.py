#created by bohuai jiang
#on 2017 1 20

import TradingFun as TF
import matplotlib.pyplot as plt


period = 30;
Xlen = 30;

address= '../../../Documents/RB/all_in_one.csv'
data = TF.csvread(address);
P = data[:,-1]
#PMa = TF.MA(P,period)
P0 = P[0:-period]
P1 = P[period::]

T = TF.getDifflabel(P0,P1,2);
X,_ = TF.getXY(Xlen,P0);

print len(P0),len(P)

act = TF.toActions(T)
print act

profit = TF.MarketState(act,P0,2)

TF.plotPerformance(profit,'desired')
plt.show()
