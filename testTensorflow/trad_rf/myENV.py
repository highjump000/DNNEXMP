from random import random
import gym
import TradingFun as TF
import numpy as np
from gym import spaces

class myENV(gym.Env):
    # input : startDate&endDate - data start/end '[year , month]'
    #         input_length - length of period, allow the agent to read
    #         period - time period in minute, min: 1 minute
    #         name - type of commodity
    #         input_format - preprocessed of input
    #                        'log_diff' = log(vend/v')
    #                        'diff' = (vend-v')/vend
    #                        'raw' = v
    def __init__(self,startDate,endDate,input_length,period = 15,name = 'rb', input_format = 'log_diff'):
        self.commission = 2
        #- read data
        data = TF.getData(period,name,startDate,endDate)
        open = data.open
        high = data.high
        low = data.low
        close = data.close
        vol = data.vol
        #- get input data with give input_length
        in_open, self.t_open = TF.getXY(input_length, open, forwardOne=0)
        in_high, self.t_high = TF.getXY(input_length, high, forwardOne=0)
        in_low, self.t_low = TF.getXY(input_length, low, forwardOne=0)
        in_close,self.t_close = TF.getXY(input_length,close,forwardOne=0)
        in_vol, self.t_vol = TF.getXY(input_length, vol, forwardOne=0)

        self.data_len = len(self.t_close)

        #- data preprocess
        if input_format is 'log_diff':
            self.open = self.log_diff(in_open)
            self.high = self.log_diff(in_high)
            self.low = self.log_diff(in_low)
            self.close = self.log_diff(in_close)
            self.vol = self.log_diff(in_vol)
        elif input_format is 'diff':
            self.open = self.diff(in_open)
            self.high = self.diff(in_high)
            self.low = self.diff(in_low)
            self.close = self.diff(in_close)
            self.vol = self.diff(in_vol)
        else:
            self.open = in_open
            self.high = in_high
            self.low = in_low
            self.close = in_close
            self.vol = in_vol
        #- adjust data
        self.data = self.adjust(data,period,0)

        self.actions = ["LONG","SHORT"]
        self.action_space = spaces.Discrete(len(self.actions))
        self.reset()
    #- inherit from gym
    def _reset(self):
        self.close_check = []
        self.totalReward = 0
        self.cumProfit = 0
        self.done = 0
        self.agentPosition = 0
        self.reward = 0
        self.bougths = []
        #
        self.position = 0
        self.getState()
        return self.state
    #- inherit from gym
    def _step(self, action):
        if self.done:
            return self.state, self.reward, self.done,{}

        self.reward = 0
        if self.actions[action] == "LONG":
            if self.bougths:   # if bougths is not empty
                if self.position == -1: # current position isnt LONG
                    for b in self.bougths:
                        self.reward -= b  # count SHORT reward
                    self.reward -= self.commission
                    self.bougths = [] #empty bougths
                else:
                    #self.reward = 1 if(self.bougths[-1]-self.commission)>0 else -1
                    self.reward = self.bougths[-1] - self.commission
                self.position = 1 # reset position
        if self.actions[action] == "SHORT":
            if self.bougths:
                if self.position == 1: # current position isnt SHORT
                    for b in self.bougths:
                        self.reward += b  # count SHORT reward
                    self.reward -= self.commission
                    self.bougths = [] #empty bougths
                else:
                    #self.reward = 1 if (-self.bougths[-1]-self.commission)>0 else -1
                    self.reward = -self.bougths[-1] - self.commission
                self.position = -1 # reset position

        if self.agentPosition == 0:
            price_different = 0
        else :
            price_different =  self.t_close[self.agentPosition][0] - self.t_close[self.agentPosition-1][0]
        self.bougths.append(price_different)

        self.agentPosition += 1

        if self.agentPosition >= len(self.t_close)-1:
            self.done = 1
            self.state = []
        else:
            self.getState()
        self.cumProfit = np.sum(self.bougths)
        self.close_check.append(self.t_close[self.agentPosition-1][0])
        return self.state,self.reward,self.done,{"dt": self.data.timestamps[self.agentPosition],"cum": self.cumProfit,"bougths": self.bougths,"close": self.close_check}
    #-- utility function --#

    #######################
    #    created state    #
    #######################

    def getState(self):
        tmp_state = []
        #--position information
        ppt = (np.sum(self.bougths)/len(self.bougths)) if len(self.bougths) > 0 else 1.
        size = len(self.bougths)
        tmp_state.append(np.array([[ppt,size,self.position]]))
        #--history information
        histor_info = np.reshape(self.close[self.agentPosition],[1,-1,1]) # reshape to 3D data
        tmp_state.append(np.array([histor_info]))
        self.state = tmp_state

    def adjust(self,data, period, forwardOne):
        data.close = TF.CallibrationData(period, data.close, forwardOne=forwardOne)
        data.delta_close = TF.CallibrationData(period, data.delta_close, forwardOne=forwardOne)
        data.timestamps = TF.CallibrationData(period, data.timestamps, forwardOne=forwardOne)

        print len(data.close)
        print len(data.delta_close)
        print len(data.timestamps)
        return data

    def diff(self,value):
        H = len(value)
        W = len(value[0])
        out = np.zeros([H, w - 1])
        for i in xrange(H):
            v = value[i][0]
            for j in xrange(1, W):
                out[i][j-1] = (v-value[i][j]) / value[i][j]
        return out

    def log_diff(self,value):
        H = len(value)
        W = len(value[0])
        out = np.zeros([H,W-1])
        for i in xrange(H):
            v = value[i][0]
            for j in xrange(1,W):
                out[i][j-1] = np.log(v/value[i][j])
        return out

    def _seed(self):
        return int(random() * 100)




