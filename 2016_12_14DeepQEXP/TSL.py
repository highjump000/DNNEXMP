#Trade stop lost
#created by bohuai jiang
#on 2016/12/12
import numpy as np


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