from agent_dir.agent import Agent
import scipy.misc
import numpy as np
from keras.models import Sequential, load_model, Input, Model
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras import optimizers
import tensorflow as tf
import sys
import os
import h5py
import matplotlib.pyplot as plt

episode = 20000
LOSS_CLIPPING = 0.2
ENTROPY_LOSS = 1e-3

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
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'

        # skip warnings
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


        if args.test_pg_improved:
            #you can load your model here
            #dummy_value = np.zeros((1, 1))
            #dummy_action = dummy_value + 0.5
            # self.actor = load_model('./pg_model/pg_improved1_model_h_5000.h5', custom_objects={'loss':proximal_policy_optimization_loss})
            print('loading trained model')
        else:
            self.actor_lr = 0.0001
            self.critic_lr = 0.0001
            self.actor_optimizer = optimizers.Adam(lr=self.actor_lr)
            self.critic_optimizer = optimizers.Adam(lr=self.critic_lr)
            self.actor = self.build_actor()
            self.critic = self.build_critic()
            
            '''
            
            '''
            #print(self.model.summary())


        self.env.reset()
    
    def build_actor(self):
        model = Sequential()
        model.add(Dense(128, input_dim=6400, kernel_initializer='glorot_uniform'))
        model.add(Activation('relu'))
        model.add(Dense(2, kernel_initializer='RandomNormal', activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=self.actor_optimizer, metrics=['accuracy'])
        return model
    
    def build_critic(self):
        model = Sequential()
        model.add(Dense(128, input_dim=6400, kernel_initializer='glorot_uniform'))
        model.add(Activation('relu'))
        model.add(Dense(1, kernel_initializer='RandomNormal'))
        model.add(Activation('tanh'))
        model.compile(loss='mse', optimizer=self.critic_optimizer, metrics=['accuracy'])
        return model

    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        self.pre_screen = np.zeros((1,6400))



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

            state = self.env.reset()
            self.init_game_setting()
            done = False
            episode_reward = 0.0

            action_list = [] # length n
            observation_list = [] # length n+1 
            reward_list = []
            observation_list.append(prepro(state))

            # first random step
            action = self.env.action_space.sample()
            state, _, _, _ = self.env.step(action)
            observation_list.append(prepro(state))

            #playing one game
            while(not done):
                action = self.make_action(state)
                state, reward, done, info = self.env.step(action)
                episode_reward += reward

                reward_list.append(reward)
                observation_list.append(prepro(state))

                # Action
                if action == 2:
                    action_list.append([0])
                elif action == 3:
                    action_list.append([1])
                else:
                    print('Error! Receive action: ', action)
                    input()
            
            
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
            input_ob = []
            for j in range(len(observation_list)-2):
                input_ob.append(observation_list[j+1] - observation_list[j])
            
            input_ob = np.array(input_ob, dtype='float64').reshape((-1,6400))

            total_rewards.append(episode_reward)

            # train model
            value = self.critic.predict(input_ob)
            advantage = np.zeros((len(input_ob), 2))
            target = []
            for i in range(len(value) - 1):
                advantage[i][action_list[i]] = reward_array[i] + 0.99 * value[i+1] - value[i]
                target.append(reward_array[i] + 0.99 * value[i+1])
            
            advantage[i+1][action_list[i+1]] = reward_array[len(value)-1] - value[-1]
            target.append(reward_array[len(value)-1])

            #advantage = np.array(advantage).reshape((-1,1))
            target = np.array(target).reshape((-1,1))

            # print(reward_list)
            # print(advantage)
            # print(target)
            # input()
            
            result_actor = self.actor.fit(input_ob, advantage, batch_size=100,
                epochs=1, verbose=1)#, sample_weight=reward_array
            
            result_critic = self.critic.fit(input_ob, target, batch_size=100,
                epochs=1, verbose=1)
            
            
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
            plt.savefig('./png/pong_ac_30_reward.png')
            plt.clf()

            if (i + 1) % 200 == 0:
                self.actor.save('./ac_model/pong_ac_model_{}.h5'.format(i+1))
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
        ob = prepro(observation).reshape((1,6400))
        # dummy_value = np.zeros((len(ob), 1))
        # dummy_action = dummy_value + 0.5
        
        #act = self.actor.predict(ob - self.pre_screen)
        #print(act)
        policy = self.actor.predict(ob - self.pre_screen, batch_size=1).flatten()
        x = np.random.choice(2, 1, p=policy)[0]
        self.pre_screen = ob
        # print(x)
        # input()
        if x == 0:#act[0][0] <= np.random.uniform(0.0, 1.0)
            return 2
        else:
            return 3

    