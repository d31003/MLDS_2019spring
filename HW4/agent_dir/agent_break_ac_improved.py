from agent_dir.agent import Agent
import scipy.misc
import numpy as np
from keras.models import Sequential, load_model, Input, Model
from keras.layers.core import Dense, Activation, Flatten, Reshape
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras import optimizers
import tensorflow as tf
import sys
import os
import h5py
import matplotlib.pyplot as plt

episode = 200000
LOSS_CLIPPING = 0.2
ENTROPY_LOSS = 1e-3
'''
def prepro(o,image_size=[80,80]):
    """
    Call this function to preprocess RGB image to grayscale image if necessary
    This preprocessing code is from
        https://github.com/hiwonjoon/tf-a3c-gpu/blob/master/async_agent.py
    
    Input: 
    RGB image: np.array
        RGB screen of game, shape: (210, 160, 3)
    Default return: np.array 
        Grayscale image, shape: (80, 80, 1)
    
    """
    o = 0.2126 * o[:, :, 0] + 0.7152 * o[:, :, 1] + 0.0722 * o[:, :, 2]
    y = o.astype(np.uint8)
    resized = scipy.misc.imresize(y, image_size)
    return np.expand_dims(resized.astype(np.float32),axis=2)
'''
def proximal_policy_optimization_loss(advantage, old_prediction):
    def loss(y_true, y_pred):
        prob = (y_true * 2 - 1) * (y_pred * 2 - 1)
        old_prob = (y_true * 2 - 1) * (old_prediction * 2 - 1)
        prob = (prob + 1) / 2
        old_prob = (old_prob + 1) / 2
        r = prob/(old_prob + 1e-10)
        return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantage) + ENTROPY_LOSS * -(prob * K.log(prob + 1e-10)))
    return loss


class Agent_AC(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        self.env = env
        super(Agent_AC,self).__init__(self.env)

        ##################
        # YOUR CODE HERE #
        ##################

        # GPU setting
        os.environ["CUDA_VISIBLE_DEVICES"] = '1'

        # skip warnings
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


        if args.test_pg_improved:
            #you can load your model here
            #dummy_value = np.zeros((1, 1))
            #dummy_action = dummy_value + 0.5
            # self.new = load_model('./pg_model/pg_improved1_model_h_5000.h5', custom_objects={'loss':proximal_policy_optimization_loss})
            print('loading trained model')
        else:
           
            self.optimizer = optimizers.Adam(lr=0.0001)
            self.new = self.new_model()
            self.old = self.old_model()
            
            '''
            
            '''
            #print(self.model.summary())


        self.env.reset()
    
    def new_model(self):
        state_input = Input(shape=(84,84,4))
        advantage = Input(shape=(1,))
        old_prediction = Input(shape=(1,))
        l1 = Dense(512, kernel_initializer='glorot_uniform', activation='relu')(state_input)
        fl = Reshape((84*84*512,))(l1)
        l2 = Dense(10, kernel_initializer='RandomNormal', activation='relu')(fl)
        out_actions = Dense(1, kernel_initializer='RandomNormal', activation='sigmoid')(l2)
        model = Model(inputs=[state_input, advantage, old_prediction], outputs=[out_actions])
        model.compile(loss=proximal_policy_optimization_loss(advantage, old_prediction),
            optimizer=self.optimizer, metrics=['accuracy'])
        # model.summary()
        # input()
        return model
    
    def old_model(self):
        model = Sequential()
        model.add(Dense(512, input_shape=(84,84,4), kernel_initializer='glorot_uniform'))
        model.add(Activation('relu'))
        model.add((Flatten()))
        #model.add(Dense(64, kernel_initializer='RandomNormal', activation='relu'))
        model.add(Dense(10, kernel_initializer='RandomNormal', activation='relu'))
        model.add(Dense(1, kernel_initializer='RandomNormal'))
        model.add(Activation('tanh'))
        model.compile(loss='mse', optimizer=self.optimizer, metrics=['accuracy'])
        return model

    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        #self.pre_screen = np.zeros((1,6400))



    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################


        # self.model = load_model('./pg_model/pg_model_600.h5')
        # self.model.compile(loss='binary_crossentropy',
        #     optimizer=self.optimizer, metrics=['accuracy'])
        # print(self.model.summary())          

        total_rewards = []
        seed = 11037
        self.env.seed(seed)
        loss_history = []
        past_30_mean = []

        for i in range(episode):
            print('-------------------------------')
            print('Run %d episodes'%(i+1))
            print('-------------------------------') 

            episode_reward = 0.0

            action_list = [] # length n
            observation_list = [] # length n+1 
            reward_list = []
            #observation_list.append(state)
            for i in range(40):
                state = self.env.reset()
                self.init_game_setting()
                done = False
                

                # first random step
                action = self.env.action_space.sample()
                state, _, _, _ = self.env.step(action)
                #observation_list.append(state)

                #playing one game
                while(not done):
                    action = self.make_action(state)
                    state, reward, done, info = self.env.step(action)
                    episode_reward += reward

                    reward_list.append(reward)
                    observation_list.append(state)

                    # Action
                    if action == 2:
                        action_list.append([0])
                    elif action == 3:
                        action_list.append([1])
                    else:
                        print('Error! Receive action: ', action)
                        input()
                reward_list[-1] = -1

            print(action_list)
            print(len(action_list), len(reward_list))

            # adding discount factor of reward
            tem = 0
            reward_list = np.array(reward_list, dtype='float64')
            for j in range(len(reward_list)-1, -1, -1):
                tem = 0.99 * tem
                if reward_list[j] == -1:
                    tem = -1
                elif reward_list[j] == 1:
                    tem = 1
                elif reward_list[j] == 0:
                    reward_list[j] += tem

            # Normalizing Reward
            reward_array = reward_list - np.mean(reward_list)
            reward_array = reward_array / np.std(reward_array)

            action_array = np.array(action_list, dtype='float64').reshape((-1,1))

            # Residual observation
            # input_ob = []
            # for j in range(len(observation_list)-2):
            #     input_ob.append(observation_list[j+1] - observation_list[j])
            
            input_ob = np.array(observation_list, dtype='float64').reshape((-1,84,84,4))

            total_rewards.append(episode_reward)

            dummy_value = np.zeros((len(input_ob), 1))
            dummy_action = dummy_value + 0.5
            # train model
            pred = self.new.predict([input_ob, dummy_value, dummy_action])
            pre_value = self.old.predict(input_ob)
            advantage = reward_array.reshape((-1,1)) - pre_value
            #print(reward_array.shape)
            #print(pre_value.shape)
            #print(advantage.shape)
            #advantage.reshape((1,))
            result_new = self.new.fit([input_ob, advantage, pred], [action_array], batch_size=100,
                epochs=1, verbose=1)#, sample_weight=reward_array
            
            result_old = self.old.fit(input_ob, reward_array, batch_size=100,
                epochs=1, sample_weight=reward_array, verbose=1)
            #loss_history.append(result_new.history['loss'])
            
            # save model and plot
            if i <= 30:    
                past_30_mean.append(np.mean(total_rewards))
                print('Episode Reward:', total_rewards[-1], '\tPast 30 Reward:', np.mean(total_rewards))
            else:
                past_30_mean.append(np.mean(total_rewards[-30:]))
                print('Episode Reward:', total_rewards[-1], '\tPast 30 Reward:', np.mean(total_rewards[-30:]))
            plt.plot(past_30_mean)
            plt.ylabel('Reward')
            plt.xlabel('Episode')
            plt.legend(['Past 30 episode'],loc='lower right')
            plt.savefig('./png/break_ac_improved_30_h_reward.png')
            plt.clf()

            if (i + 1) % 200 == 0:
                self.new.save('./ac_model/break_ac_improved_model_h_{}.h5'.format(i+1))
            # if (i + 1) % 10 == 0:
            #     plt.plot(total_rewards)
            #     plt.ylabel('Reward')
            #     plt.xlabel('Episode')
            #     plt.legend(['Each episode'],loc='lower right')
            #     plt.savefig('./png/pg_h_reward.png')
            #     plt.clf()
            #     plt.plot(loss_history)
            #     plt.ylabel('loss')
            #     plt.xlabel('episode')
            #     plt.legend(['train'],loc='upper right')
            #     plt.savefig('./png/pg_h_loss.png')
            #     plt.clf()


    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################

        # [right, left] = 2,3 = 0,1
        ob = observation.reshape((1,84,84,4))
        dummy_value = np.zeros((len(ob), 1))
        dummy_action = dummy_value + 0.5
        
        act = self.new.predict([ob, dummy_value, dummy_action])
        #print(act)
        #self.pre_screen = ob
        if act[0][0] <= np.random.uniform(0.0, 1.0):
            #print(2)
            return 2
        else:
            #print(3)
            return 3

    