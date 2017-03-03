import numpy as np

from myENV import myENV
from market_model_builder import MarketModelBuilder
from DynamicPlot import  DynamicPlot
import keras as k

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class ExperienceReplay(object):
    def __init__(self, max_memory=100, gamma=.9):
        self.max_memory = max_memory
        self.memory = list()
        self.discount = gamma

    def remember(self, states, game_over):
        print 'remember data'
        # memory[i] = [[state_t, action_t, reward_t, state_t+1], game_over?]
        self.memory.append([states, game_over])
        if len(self.memory) > self.max_memory:
            del self.memory[0]
    def get_batch(self, model, batch_size=10):
        print 'get batch'
        len_memory = len(self.memory)
        num_actions = model.output_shape[-1]
        inputs = []

        dim = len(self.memory[0][0][0])
        for i in xrange(dim):
            inputs.append([])

        targets = np.zeros((min(len_memory, batch_size), num_actions))
        for i, idx in enumerate(np.random.randint(0, len_memory, size=min(len_memory, batch_size))):
            state_t, action_t, reward_t, state_tp1 = self.memory[idx][0]
            game_over = self.memory[idx][1]

            for j in xrange(dim):
                inputs[j].append(state_t[j][0])

            #inputs.append(state_t)
            # There should be no target values for actions not taken.
            # Thou shalt not correct actions not taken #deep
            targets[i] = model.predict(state_t)[0]
            Q_sa = np.max(model.predict(state_tp1)[0])
            if game_over:  # if game_over is True
                targets[i, action_t] = reward_t
            else:
                # reward_t + gamma * max_a' Q(s', a')
                targets[i, action_t] = reward_t + self.discount * Q_sa
		
        #inputs = np.array(inputs)
        inputs = [np.array(inputs[i]) for i in xrange(dim)]

        return inputs, targets
class OnlineLearning(object):
    def __init__(self,gamma = 0.9):
        self.gamma = gamma
    def getXT(self,model,state,done):
        s_t,a_t,R,s = state
        T = model.predict(s_t)[0]
        if done:
            T[a_t] = R
        else:
            Q_v = np.max(model.predict(s)[0])
            T[a_t] = R + self.gamma*Q_v
        return s_t,np.array([T])

def train():

    env = myENV([2012,12],[2013,1],75,period=15)

    # parameters
    epsilon = .5  # exploration
    min_epsilon = 0.1
    epoch = env.data_len
    max_memory = 5000
    batch_size = 128
    gamma = 0.8

    from keras.optimizers import SGD
    model = MarketModelBuilder('dnq.h5').getModel()
    adam = k.optimizers.adam(lr=0.0001)
    model.compile(loss='mse', optimizer=adam)


    # Initialize experience replay object
    #exp_replay = ExperienceReplay(max_memory = max_memory, gamma=gamma)
    online_lear = OnlineLearning(gamma = gamma)
    # Train
    win_cnt = 0 # ?
    # plot performance
    plot_profit = DynamicPlot()
    plot_profit.on_launch('profit')
    plot_close = DynamicPlot()
    plot_close.on_launch('rb_close')
    plot_loss = DynamicPlot()
    plot_loss.on_launch('loss')
    l_profit = plot_profit.requestLines('accumulate profit')
    l_close = plot_close.requestLines('rb close','g-')
    l_loss = plot_loss.requestLines('MSE','r-')

    windowSize = 300

    for e in range(epoch):
        env.reset()
        done = 0
        # get initial input
        input_t = env.reset()
        cumReward = 0

        xaxis = []
        count = 0
        accuPro = []
        loss = []
        while not done:
            input_tm1 = input_t
            isRandom = False

            # get next action
            if np.random.rand() <= epsilon:
                action = np.random.randint(0, env.action_space.n, size=1)[0]

                isRandom = True
            else:
                q = model.predict(input_tm1)
                action = np.argmax(q[0])

            # apply action, get rewards and new state
            input_t, reward, done, info = env.step(action)
            cumReward += reward

            if env.actions[action] == "LONG" or env.actions[action] == "SHORT":
                color = bcolors.OKBLUE
                if isRandom:
                    color = bcolors.OKGREEN
            print "%s:\taction:%s\tcumReward:%.2f\tR:%.2f\tepsilon:%s" % (info["dt"], color + env.actions[action] + bcolors.ENDC, cumReward, reward,epsilon)

            # store experience
            #exp_replay.remember([input_tm1, action, reward, input_t], done)

            # adapt model
            #inputs, targets = exp_replay.get_batch(model, batch_size=batch_size)
            input,target = online_lear.getXT(model,[input_tm1, action, reward, input_t],done)
            histo = model.fit(input, target,batch_size = len(target),nb_epoch = 1,verbose=0)

            loss.append(np.mean(histo.history.get('loss')))

            #-- plot performance --#
            close_v = info["close"]
            if len(xaxis) > windowSize:
                accuPro.pop(0)
                close_v.pop(0)
                loss.pop(0)
                xaxis.pop(0)
            xaxis.append(count)
            accuPro.append(cumReward)
            plot_profit.on_running(l_profit,xaxis,accuPro)
            plot_profit.draw()
            plot_close.on_running(l_close,xaxis,close_v)
            plot_close.draw()
            plot_loss.on_running(l_loss,xaxis,loss)
            plot_loss\
                .draw()

            count += 1
        if cumReward > 0 and done:
            win_cnt += 1

        #print("Epoch {:03d}/{} | Loss {:.4f} | Win count {} | Epsilon {:.4f}".format(e, epoch, loss, win_cnt, epsilon))
        # Save trained model weights and architecture, this will be used by the visualization code
        model.save("model.h5")
        epsilon = max(min_epsilon, epsilon * 0.8)
