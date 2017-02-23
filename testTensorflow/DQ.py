#created by bohuai jiang
#on 2017 1 6
# my own deep Q function

# a deep Q frame
import NN_network as nn;
import tensorflow as tf;
import random
import numpy as np;
import TradingFun as TF;
from collections import deque
from DynamicPlot import DynamicPlot
import matplotlib.pyplot as plt

# Hyper Parameters for DQN
GAMMA = 0.8 # discount factor for target Q
INITIAL_EPSILON = 0.9  # starting value of epsilon
FINAL_EPSILON = 0.5  # final value of epsilon
LEARNING_RATE = 0.001
TOPO = [0]
class DQ:
    ######### nn stuff ############
    #----------------basic structure-----------------#
    def model_init(self,input_size,output_size):
        w,b = nn.MLP_init(input_size,output_size,TOPO);
        return w,b;

    def model_run(self,input,w,b):
        Y = nn.MLP(input,w,b);
        return Y;

    def costFunc(self,Y,one_hot_actions,T):
        Q_action = tf.reduce_sum(tf.mul(T, one_hot_actions), reduction_indices=1)
        cost = nn.MSE(Q_action,Y)
        return cost;
    ####

    def egreedy_action(self,Qvalue,epsilon,N_ACTION):
        if random.random() <= epsilon:
            return random.randint(0, N_ACTION - 1)
        else:
            return np.argmax(Qvalue);

    ######## trading Func ##########

    def rewardFunc(self,action,step,rest):
        #---extract rest container
        T = rest[0];
        stpr = rest[1];
        commission = rest[2];
        nt = rest[3];
        profit = rest[4];
        X = rest[5];

        #if action == 0:
        #   action = -1;

        #---get reward
        action = (action - 1);
        if action == 0:
            if step < len(X)-1:
                profit.append(0);
                return 0, 0, X[step+1,:],step+1,[nt,profit];
            else:
                profit.append(0);
                return 0, 1, X[step,:],step,[nt,profit];

        r,index,done = TF.TSLreward(action,T,step,stpr,commission);
        if index >= len(X):
            next_state = X[index-1, :]
        else:
            next_state = X[index, :]
        #---update&save trading param
        nt += 1;
        profit.append(r[0]);
        outrest = [nt,profit];
        if done:
            tp = sum(profit);
            ppt = tp/nt;
            #sharpe = tp/np.std(profit);
            return tp,done,next_state,index,outrest;
        else:
            return 0,done,next_state,index,outrest;

    def rewardFunc_state(self,action,step,enter,enterP,direct,P,commission):
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

    def  rewardCal(self,profit):
        tp = sum(profit);
        nt = sum(profit!= 0);

        if nt == 0:
            ppt = 0;
        else:
            ppt = tp/nt;

        if np.std(profit) == 0:
            sharpe = 0
        else:
            sharpe = tp/np.std(profit);
        return profit[-1];

    ############### Deep Q model 1 : give out market state ####################
    def TDQ_state(self, X, T, EPISODE=10000, MODEL_RUN=model_run, commission=2, train=0.7):
        # set test & validation
        DL = len(T);
        # trainL = int(DL*train);
        # train_X = X[]

        # parameters
        DATALEN, N_FEATURE = nn.getWH(X);
        N_ACTION = 3;
        epsilon = INITIAL_EPSILON;
        REPLAY_MEMORY = deque();
        TEST = 1

        REPLAY_SIZE = 3000  # DATALEN*3  # experience replay buffer size
        BATCH_SIZE = 500  # DATALEN/20  # size of minibatch

        print 'Batch_size : ', BATCH_SIZE

        # -------------- tensorflow stuff ----------------#
        sess = tf.InteractiveSession();
        w, b = self.model_init(N_FEATURE, N_ACTION);  # initialise model
        # -- set input
        state_input = tf.placeholder("float", [None, N_FEATURE]);
        action_input = tf.placeholder("float", [None, N_ACTION]);
        T_input = tf.placeholder("float", [None]);

        Y = self.model_run(state_input, w, b);
        cost = self.costFunc(Y, action_input, T_input);
        opt = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost);

        sess.run(tf.global_variables_initializer());
        # ----------- done -----------#


        # ---------------- plot stuff -----------------#
        dplt = DynamicPlot();
        dplt.on_launch('insample profit');
        line = dplt.requestLines('profit', '-');

        time = 0;
        start = 0;
        cost_out = [];

        dplt2 = DynamicPlot();
        dplt2.on_launch('cost');
        line2 = dplt2.requestLines('cost', 'r-');
        # ------------ done ----------#

        for episode in xrange(EPISODE):
            # calculate reward
            enterP = 0;
            enter = 0;
            direct = 0;
            print "gethering data ..."
            for step in range(DATALEN - 1):
                #
                # selected action
                state = X[step, :];
                Q_value = sess.run(Y, feed_dict={state_input: [state]})[0]
                action = self.egreedy_action(Q_value, epsilon, N_ACTION)
                # action = np.argmax(Q_value)

                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000  # update e
                # get reward
                reward, enter, direct, enterP = self.rewardFunc_state(action, step, enter, enterP, direct, T,
                                                                      commission)
                # print reward
                next_state = X[step + 1, :];
                # add to replay memory
                one_hot_action = np.zeros(N_ACTION)
                one_hot_action[actions] = 1
                if reward != 0:
                    done = 1;
                else:
                    done = 0;
                REPLAY_MEMORY.append((state, one_hot_action, reward, next_state, done))
                if len(REPLAY_MEMORY) > REPLAY_SIZE:
                    REPLAY_MEMORY.popleft()
                # train NN
                if len(REPLAY_MEMORY) > BATCH_SIZE:
                    # ---------------- train part ---------------#
                    minibatch = random.sample(REPLAY_MEMORY, BATCH_SIZE);
                    state_batch = [data[0] for data in minibatch]
                    action_batch = [data[1] for data in minibatch]
                    reward_batch = [data[2] for data in minibatch]
                    next_state_batch = [data[3] for data in minibatch]

                    y_batch = []
                    Q_value_batch = Y.eval(feed_dict={state_input: next_state_batch})
                    for i in range(0, BATCH_SIZE):
                        done = minibatch[i][4]
                        if done:
                            y_batch.append(reward_batch[i])
                        else:
                            y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))
                    sess.run(opt, feed_dict={T_input: y_batch, action_input: action_batch, state_input: state_batch})
                    costV = sess.run(cost,
                                     feed_dict={T_input: y_batch, action_input: action_batch, state_input: state_batch})
                    # ----------done---------#
                    # print costV
                    cost_out.append(costV);
                    time += 1;

                    Qvalue = sess.run(Y, feed_dict={state_input: X})
                    # print Qvalue
                    print sess.run(w[0][0])


                    # lenC = len(cost_out)
                    # print lenC
                    #  if lenC %1000 ==0 and lenC > 1:
                    #      cost_out = cost_out[lenC-1000:lenC];
                    #      start = time -1000
            # check
            print 'episode :', episode, ' done start test'
            Qvalue = sess.run(Y, feed_dict={state_input: X})
            actions = np.argmax(Qvalue, axis=1) - 1;
            actions

            profit = TF.MarketState(actions, T, commission);

            dplt.on_running(line, np.arange(0, len(profit)), np.cumsum(profit))
            dplt.draw()

            if len(cost_out) > 1:
                dplt2.on_running(line2, range(start, time), cost_out)
                dplt2.draw()

            total_reward = sum(profit);

            if episode == 0:
                print 'episode: ', episode, 'Evaluation Average tp Reward:', total_reward

        dplt.done()
        dplt2.done()

    ############### Deep Q model 2 : learning market state ####################

    def TDQ_state(self,X,T,EPISODE = 10000,MODEL_RUN=model_run,commission = 2,train = 0.7):
        #set test & validation
        DL = len(T);
        #trainL = int(DL*train);
        #train_X = X[]

        # parameters
        DATALEN,N_FEATURE = nn.getWH(X);
        N_ACTION = 3;
        epsilon = INITIAL_EPSILON;
        REPLAY_MEMORY = deque();
        TEST = 1

        REPLAY_SIZE = 3000#DATALEN*3  # experience replay buffer size
        BATCH_SIZE = 500#DATALEN/20  # size of minibatch

        print 'Batch_size : ',BATCH_SIZE

        #-------------- tensorflow stuff ----------------#
        sess = tf.InteractiveSession();
        w,b = self.model_init(N_FEATURE,N_ACTION); #initialise model
        #-- set input
        state_input = tf.placeholder("float", [None, N_FEATURE]);
        action_input = tf.placeholder("float", [None, N_ACTION]);
        T_input = tf.placeholder("float", [None]);

        Y = self.model_run(state_input, w, b);
        cost = self.costFunc(Y, action_input, T_input);
        opt = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost);

        sess.run(tf.global_variables_initializer());
        #----------- done -----------#


        #---------------- plot stuff -----------------#
        dplt = DynamicPlot();
        dplt.on_launch('insample profit');
        line = dplt.requestLines('profit', '-');

        time = 0;
        start = 0;
        cost_out = [];

        dplt2 = DynamicPlot();
        dplt2.on_launch('cost');
        line2 = dplt2.requestLines('cost', 'r-');
        #------------ done ----------#

        for episode in xrange(EPISODE):
            actions = [];
            # calculate reward
            enterP = 0;
            enter = 0;
            direct = 0 ;
            print "gethering data ..."
            for step in range(DATALEN-1):
                #print step
                # selected action
                state = X[step,:];
                Q_value = sess.run(Y,feed_dict={state_input:[state]})[0]
                action = self.egreedy_action(Q_value,epsilon,N_ACTION)
                #action = np.argmax(Q_value)

                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000 # update e
                # get reward
                reward,enter,direct,enterP = self.rewardFunc_state(action,step,enter,enterP,direct,T,commission)
                #print reward
                next_state = X[step+1,:];
                # add to replay memory
                one_hot_action = np.zeros(N_ACTION)
                one_hot_action[actions] = 1
                if reward != 0:
                    done = 1;
                else:
                    done = 0;
                REPLAY_MEMORY.append((state, one_hot_action, reward, next_state, done))
                if len(REPLAY_MEMORY) > REPLAY_SIZE:
                    REPLAY_MEMORY.popleft()
                #train NN
                if len(REPLAY_MEMORY) > BATCH_SIZE:
                    #---------------- train part ---------------#
                    minibatch = random.sample(REPLAY_MEMORY, BATCH_SIZE);
                    state_batch = [data[0] for data in minibatch]
                    action_batch = [data[1] for data in minibatch]
                    reward_batch = [data[2] for data in minibatch]
                    next_state_batch = [data[3] for data in minibatch]

                    y_batch = []
                    Q_value_batch = Y.eval(feed_dict={state_input: next_state_batch})
                    for i in range(0, BATCH_SIZE):
                        done = minibatch[i][4]
                        if done:
                            y_batch.append(reward_batch[i])
                        else:
                            y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))
                    sess.run(opt,feed_dict={T_input: y_batch,action_input: action_batch,state_input: state_batch})
                    costV = sess.run(cost,feed_dict={T_input: y_batch,action_input: action_batch,state_input: state_batch})
                    #----------done---------#
                    #print costV
                    cost_out.append(costV);
                    time += 1;

                    Qvalue = sess.run(Y, feed_dict={state_input: X})
                    #print Qvalue
                    print sess.run(w[0][0])


                #lenC = len(cost_out)
                #print lenC
              #  if lenC %1000 ==0 and lenC > 1:
              #      cost_out = cost_out[lenC-1000:lenC];
              #      start = time -1000
            #check
            print 'episode :',episode, ' done start test'
            Qvalue = sess.run(Y,feed_dict={state_input:X})
            actions= np.argmax(Qvalue,axis=1)-1;
            actions

            profit = TF.MarketState(actions,T,commission);

            dplt.on_running(line, np.arange(0,len(profit)), np.cumsum(profit))
            dplt.draw()

            if len(cost_out)>1:
                dplt2.on_running(line2, range(start,time), cost_out)
                dplt2.draw()

            total_reward = sum(profit);

            if episode == 0:
                print 'episode: ', episode, 'Evaluation Average tp Reward:', total_reward

        dplt.done()
        dplt2.done()

