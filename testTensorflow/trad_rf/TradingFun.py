# created by bohuai jiang
# trading parckage
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
##############
from pyelf.units import PlainFilter, FixedRatioTrainStop
import pyelf.performance as perf
import pyelf.elutil as eu
import pyelf.elplot as ep
from pyelf.eldata import Data


############# about trading target ############
# get performance base on market state
def MarketState(a,price,commission):
    L = len(price)

    profit = np.zeros(L)

    dir = a[0]
    if dir != 0 :
        enter = 1
        enterPrice = price[0]
    else:
        enter = 0;

    for i in range(1,L):
        # normal out
        if a[i] == 0 and enter:
            enter = 0
            profit[i] = dir*(price[i]-enterPrice)-commission
            dir = 0
        # off-hands out
        if a[i] == -dir and enter:
            profit[i] = dir*(price[i]-enterPrice)-commission
            dir = a[i]
            enterPrice = price[i]

        if dir == 0 and a[i] != 0 and enter!=1:
            enter = 1
            enterPrice = price[i]
            dir = a[i]

    return profit

# stop loss
def TSL(a,price,stpr,commission,doOffHand):
    L = len(price);
    price_tracker = [];

    enterP = 0; #enter price
    enter = 0; #enter signal
    dir = 0;

    profit = np.zeros(L);

    for i in xrange(L):
        if enter:
            price_tracker.append(price[i]);

        #in market & off hands exit
        if a[i] != 0:
            if enter:
                if a[i] != dir:
                    enter = 0;
                    outP = price[i];
                    profit[i],dir,price_tracker = outMarket(dir,outP,enterP,commission);
                    if doOffHand:
                        dir = a[i];
                        enter = 1;
                        enterP = price[i];
                        price_tracker.append(price[i]);
            else:
                dir = a[i];
                enter = 1;
                enterP = price[i];
                price_tracker.append(price[i]);

        #TSL
        if price_tracker: # price_tracker must not empty
            currentP = price[i];

            #stopWin = max(price_tracker)-stpr*currentP/1000; #STL long
            #stopLose = min(price_tracker)-stpr*currentP/1000; #STL short

            stopWin = max(price_tracker)*(1-stpr/1000.0);  # STL long
            stopLose = min(price_tracker)*(1+stpr/1000.0); #STL short

            #long
            if (dir == 1) and (currentP < stopWin):
                enter = 0 ;
                profit[i], dir, price_tracker = outMarket(dir, currentP, enterP, commission);

            #short
            if (dir == -1) and (currentP > stopLose):
                enter = 0;
                profit[i], dir, price_tracker = outMarket(dir, currentP, enterP, commission);


    return profit;

def outMarket(dir,outP,enterP,commission):
    net_profit= dir*(outP-enterP)-commission;
    dir = 0;
    price_tracker = [];

    return net_profit,dir,price_tracker;

def TSLgetProfit(actions,T):
    DL = len(T)
    profit = [];
    for i in range(DL):
        profit.append(T[i][int(actions[i]+1)])
    return np.asarray(profit)
# stop loss for single trade
def TSLreward(action,price,step,stpr,commission):
        L = len(price);
        reward = [0];

        #case not entered
        if action == 0:
            if step == L:
                return reward,step,1;
            else:
                return reward,step,0;
        #case end of the data
        if step == L:
            return reward,step,1;

        enterP = price[step];
        price_tracker = [];
        price_tracker.append(enterP);

        out_idx = step;

        for i in range(step+1,L):
            price_tracker.append(price[i]);
            currentP = price[i];

            stopWin = np.max(price_tracker)*(1-stpr/1000.0);  # STL long
            stopLose = np.min(price_tracker)*(1+stpr/1000.0); #STL short

            # long
            if (action == 1) and (currentP <= stopWin):
                reward = action*(currentP-enterP)-commission;
                out_idx = i;
                return reward, out_idx,0;
            # short
            if (action == -1) and (currentP >= stopLose):
                reward = action * (currentP - enterP) - commission;
                out_idx = i
                return reward, out_idx, 0;
        return reward,L,1;
# for deep Q uses

def rewardFunc_state(action,step,enter,enterP,direct,P,commission):
    action = action -1;
    #if out market
    if direct != 0 and enter == 1:
        if action != direct:
            reward = (P[step][0]-enterP)*direct-commission
            direct = action;
            enter = 0;
            if  action != 0:
                enterP = P[step][0];
                enter = 1;
            return reward,enter,direct,enterP
    #if in market
    if direct == 0:
        if action != direct:
            direct = action;
            enter = 1;
            reward = 0;
            enterP = P[step][0]
            return reward,enter,direct,enterP
    return 0,enter,direct,enterP;

################### about modify given data #####################

def getQtable(P,stpr,commission):
        data_length = len(P);
        T = np.zeros([data_length,3]);
        for i in xrange(data_length):
            T1 = TSLreward(-1,P,i,stpr,commission)[0][0]
            T3 = TSLreward(1,P,i,stpr,commission)[0][0]
            print i, T1,T3
            T[i][0] = T1;
            T[i][2] = T3;
        return T

def getA(Qtable):
    data_len = len(Qtable)
    a = np.zeros(data_len)
    for i in range(data_len):
        a[i] = np.argmax(Qtable[i])-1
    return a;

def getProfitHF(a,T):
    profit = []
    LD = len(T)
    for i in range(LD):
        profit.append(T[i][int(a[i]+1)])
    return profit

def csvread(address):
    data = genfromtxt(address,delimiter=',');
    return np.asarray(data);
#not quite usefull
def getXY(period, p,forwardOne=1):
    datalength = len(p);
    datalen = datalength - period - 1;
    X = np.zeros([datalen, period]);
    T = np.zeros([datalen, 1]);
    for i in range(period, datalength - 1):
        for j in xrange(period):
            X[i - period][j] = p[i - j];
        if forwardOne:
            T[i - period] = p[i+1]
        else:
            T[i - period] = p[i]
    return np.float32(X), np.float32(T);

def CallibrationData(period,input,forwardOne =1):
    datalength = len(input)
    datalen = datalength - period - 1
    if forwardOne:
        return input[period+1:datalength]
    else:
        return input[period:datalength-1]


def getTrainTest2D(percent, X,T):
    Dlen = len(T);
    TrainLen = int(Dlen*percent);
    TrainX = X[0:TrainLen,:];
    TrainT = T[0:TrainLen];
    if TrainLen < Dlen:
        TestX = X[TrainLen+1:-1,:];
        TestT = T[TrainLen+1:-1];
    else:
        TestX = X;
        TestT = T;
    return TrainX,TrainT,TestX,TestT;

def getTrainTest2D_mirror(percent,period,P):
    Dlen = len(P)
    TrainLen = int(Dlen*percent)
    TrainP = P[0:TrainLen]
    imgP = TrainP[::-1] # image data
    if TrainLen < Dlen:
        TestP = P[TrainLen+1:-1]
    else:
        TestP = P
    trainX,trainT = getXY(period,TrainP)
    imgX,imgT = getXY(period,imgP)
    testX,testT = getXY(period,TestP)

    trainX = np.append(trainX,imgX,axis=0)
    trainT = np.append(trainT,imgT,axis=0)

    return trainX,trainT,testX,testT




# one hot key format
def toOneHotKey(actions,n_act):
    dl = len(actions)
    one_hot_key = np.zeros([dl,n_act])
    for i in range(dl):
        one_hot_key[i][int(actions[i]+1)] = 1
    return one_hot_key


def toActions(one_hot_ley):
    act = np.argmax(one_hot_ley,axis=1)
    act = act-1
    return act


########### about plot #########
# convert buy sell signal to TF readable signal
def TFReadSignal(bsig,ssig):
    lD = len(bsig)
    ssig[ssig == 1] =-1
    return bsig+ssig

def plotPerformanceSimple(profit,title):
    # param
    tp = sum(profit)
    nt = sum(profit != 0)
    if nt !=0 :
        ppt = tp/nt
    else:
        ppt = 0
    neg = sum(profit < 0)
    #print np.where(profit < 0)

    plt.figure().suptitle(title,fontsize = 20)
    plt.plot(np.cumsum(profit),label = 'tp : '+str(tp)+' ppt : '+str(ppt)+' nt : '+str(nt)+' neg : ' + str(neg))
    plt.legend(loc=4)

############# basic modification ############
def MA(LIST,STEP):
    LIST = np.asarray(LIST)
    Dl = len(LIST)
    res = np.zeros(Dl-STEP)

    for i in xrange(STEP,Dl):
        j = i - STEP
        res[j] = np.mean(LIST[j:i]);
    return res

############ simple ml ###########

def linearRegression(X,T):
    # convert to array
    X = np.asarray(X)
    T = np.asarray(T)

    # add bias to X
    Dl = len(T)
    bias = np.ones([Dl,1])
    Xb = np.concatenate((X,bias),1)

    # convert to matrix
    Xb = np.matrix(Xb)
    T = np.matrix(T)
    w = np.linalg.inv(Xb.transpose()*Xb)*(Xb.transpose()*T)
    return w

def linearRegression_getY(w,X):

    # add bias
    X = np.asarray(X)
    Dl = len(X)
    bias = np.ones([Dl,1])
    Xb = np.concatenate((X,bias),1)

    # convert to matrix
    Xb = np.matrix(Xb)

    Y = Xb*w
    return Y

#########################################
##       common eval func              ##
#########################################

def toBsigSsig(ps):
    DL = len(ps)
    bsig = np.zeros(DL)
    ssig = np.zeros(DL)
    bsig[ps == 1] = 1
    ssig[ps == -1] =1
    return bsig,ssig


def plotPerformance(data,bsig,ssig,name):
    pf = PlainFilter()
    frs = FixedRatioTrainStop()
    ps = eu.tp15(data.close, bsig, ssig)
    #bsig, ssig = toBsigSsig(ps)
    #ps = pf.eval(data, [], [bsig, ssig])


    #plt.figure().suptitle('ps')
    #plt.plot(ps)

    ps = frs.eval(data, [15], ps)

   # plt.figure().suptitle('ps2')
   # plt.plot(ps)

    accu_profits, net_accu_profits = perf.get_accu_profits(ps, data.delta_close, eu.get_unit_cost(name))
    draw_down = perf.get_draw_down(net_accu_profits)

    trading_times, win_rate, pf, md, ppt, total_profit = perf.get_performance(ps, data.delta_close,
                                                                              eu.get_unit_cost(name))

    title = 'TT=%d, WR=%.2f, PF=%.2f, MD=%.2f, PPT=%.2f, NTP=%.2f' % (trading_times, win_rate, pf,
                                                                     md, ppt, net_accu_profits[-1])

    f, ax = plt.subplots(3, 1, sharex=True)
    ep.plot_profits(ax[0], accu_profits, net_accu_profits, draw_down, data.timestamps, title=title,fonsize=8)
    ax[1].plot(ps)
    ax[1].set_ylim([-1.5, 1.5])
    ax[1].grid()
    ax[2].plot(data.close)
    ax[2].grid()

def getData(time_period,name,startmonth,endmonth):
    prefix = '/home/yohoo/Documents/Data/'
    d = Data()
    d.set_data_path(prefix + 'min/', prefix + 'log/')
    d.set_name(name)
    d.load_raw_data()
    d.set_period(time_period)
    return d.extract_data_by_period([startmonth,endmonth])