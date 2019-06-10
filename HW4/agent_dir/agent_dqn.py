from agent_dir.agent import Agent
import scipy.misc
import numpy as np
from keras.models import Sequential, Model, load_model
from keras.layers.core import Dense, Activation, Flatten
from keras.layers import Input, Conv2D, Flatten, Dense, LeakyReLU, Multiply, Maximum, Add, merge, Lambda
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from collections import deque
import tensorflow as tf
import sys
import os
import h5py
import matplotlib.pyplot as plt


np.random.seed(326)
tf.reset_default_graph()
tf.set_random_seed(326)

class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        ##################
        # YOUR CODE HERE #
        ##################

        # GPU setting
        os.environ["CUDA_VISIBLE_DEVICES"] = '3'

        # skip warnings
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        
        self.env = env
        super(Agent_DQN,self).__init__(env)
        self.epsi = 0.8
        self.final_epsi = 0.025
        self.epsi_step = 0.005
        self.iter = 0
        self.replay_size = 10000
        self.num_actions = 4  #env.action_space.n
        self.batch_size = 32
        self.no_op_steps = 3
        self.total_reward = 0
        self.episode = 0
        self.total_loss = 0.0
        self.episode_limit = 40000
        self.plot_30 = []
        self.gamma = 0.99
        

        # Only for predict Q value of actions
        self.dummy_input = np.zeros((1,self.num_actions))
        self.dummy_batch = np.zeros((self.batch_size, self.num_actions))

        self.input_shape = (84, 84, 4)
        self.optimizer = optimizers.Adam(lr=0.00015)


        if args.test_dqn:
            #you can load your model here
            num = 10000
            self.q_net = load_model('./dqn_model/dqn_op_Q_{}.h5'.format(num))
            self.fixed_net = load_model('./dqn_model/dqn_op_Fixed_{}.h5'.format(num))
            print('loading trained model')
        else:
            self.q_net = self.make_model()
            self.fixed_net = self.make_model()
            print(self.q_net.summary())

        self.replay_deque = deque() # last_ob, action, reward, ob, terminal
        self.last_30_reward = deque()
        
        self.env.reset()




    def make_model(self):
        input_frame = Input(shape=self.input_shape)
        action_one_hot = Input(shape=(self.num_actions,))
        conv1 = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(input_frame)
        conv2 = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(conv1)
        conv3 = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(conv2)
        flat_feature = Flatten()(conv3)
        hidden_feature = Dense(512)(flat_feature)
        lrelu_feature = LeakyReLU()(hidden_feature)
        q_value_prediction = Dense(self.num_actions)(lrelu_feature)

        select_q_value_of_action = Multiply()([q_value_prediction,action_one_hot])
        target_q_value = Lambda(lambda x:K.sum(x, axis=-1, keepdims=True), output_shape=lambda_out_shape)(select_q_value_of_action)
        
        model = Model(inputs=[input_frame,action_one_hot], outputs=[q_value_prediction, target_q_value])
        
        # MSE loss on target_q_value only
        model.compile(loss=['mse','mse'], loss_weights=[0.0,1.0], optimizer=self.optimizer)

        return model


    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        
        self.total_reward = 0.0
        self.episode += 1


    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        
        while self.episode <= self.episode_limit:
            terminal = False
            observation = self.env.reset()

            # initial new game
            self.init_game_setting()

            for _ in range(np.random.randint(1, self.no_op_steps+1)):
                last_observation = observation
                observation, _, _, _ = self.env.step(0)  # Do nothing, 0=noop
            while not terminal:
                last_observation = observation
                action = self.make_action(last_observation, test=False)
                observation, reward, terminal, _ = self.env.step(action)
                #print("Make action", action, "reward", reward)
                self.run(last_observation, action, reward, terminal, observation)
                if terminal:
                    print("Episode", self.episode, "ends. Iter: ", self.iter)



    def make_action(self, observation, test=False):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################

        observation = observation.reshape((1,self.input_shape[0],self.input_shape[1],self.input_shape[2]))

        if not test:
            # exploration
            if self.epsi > np.random.uniform(0.0, 1.0) or self.iter < self.replay_size:
                action = np.random.randint(0,self.num_actions)
            else:
                action = np.argmax(self.q_net.predict([observation, self.dummy_input])[0])
            
            # Decrease epsilon
            if self.epsi > self.final_epsi and self.iter >= self.replay_size:
                self.epsi -= self.epsi_step
        else: # test
            if 0.005 > np.random.uniform(0.0, 1.0):
                action = np.random.randint(0,self.num_actions)
            else:
                action = np.argmax(self.q_net.predict([observation, self.dummy_input])[0])

        return action

    def run(self, last_ob, action, reward, terminal, ob):

        self.replay_deque.append((last_ob, action, reward, ob, terminal))

        # maintain replay size
        if len(self.replay_deque) > self.replay_size:
            self.replay_deque.popleft()
        else:
            self.episode = 0

        if self.iter >= self.replay_size:
            # Train network
            if self.iter % 4 == 0:
                print("Train Network")
                self.train_network()
            # Update target network
            if self.iter % 1000 == 0:
                print("Update")
                self.fixed_net.set_weights(self.q_net.get_weights())

        self.total_reward += reward
        self.iter += 1

        if terminal:
            self.last_30_reward.append(self.total_reward)
            if len(self.last_30_reward) > 30:
                self.last_30_reward.popleft()
            
            self.plot_30.append(np.mean(self.last_30_reward))

            # plot
            plt.plot(self.plot_30)
            plt.ylabel('Past 30 Reward')
            plt.xlabel('Episode')
            plt.savefig('./png/dqn_op_30_reward.png')
            plt.clf()

            # save model
            if self.episode % 1000 == 0 and self.episode > 0:
                self.q_net.save('./dqn_model/dqn_op_Q_{}.h5'.format(self.episode))
                self.fixed_net.save('./dqn_model/dqn_op_Fixed_{}.h5'.format(self.episode))




    def train_network(self, ddqn=False):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        terminal_batch = []
        y_batch = []

        x = np.random.choice(a=len(self.replay_deque), size=self.batch_size, replace=False)
        for data in x:
            state_batch.append(self.replay_deque[data][0])
            action_batch.append(self.replay_deque[data][1])
            reward_batch.append(self.replay_deque[data][2])
            next_state_batch.append(self.replay_deque[data][3])
            terminal_batch.append(self.replay_deque[data][4])

        reward_batch = list2np(reward_batch)

        # Convert True to 1, False to 0
        terminal_batch = np.array(terminal_batch) + 0

        # Q value from target network
        target_q_values_batch = self.fixed_net.predict([list2np(next_state_batch), self.dummy_batch])[0]
        
        if not ddqn:
            y_batch = reward_batch + (1 - terminal_batch) * self.gamma * np.max(target_q_values_batch, axis=-1)

        # one-hot action
        a_one_hot = np.zeros((self.batch_size,self.num_actions))
        for idx,ac in enumerate(action_batch):
            a_one_hot[idx,ac] = 1.0


        input_state = list2np(state_batch)
        result = self.q_net.fit([input_state, a_one_hot], [self.dummy_batch, y_batch],
            batch_size=32, epochs=1, verbose=1)

        #loss = self.q_net.train_on_batch([input_state,a_one_hot],[self.dummy_batch,y_batch])
        self.total_loss += result.history['loss'][0]



def list2np(listA):
    return np.array(listA, dtype='float64')

def lambda_out_shape(input_shape):
    shape = list(input_shape)
    shape[-1] = 1
    return tuple(shape)
