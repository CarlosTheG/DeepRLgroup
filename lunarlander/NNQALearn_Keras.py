''' EXPERIENCE REPLAY WOW '''
import gym
import numpy as np
import random
import keras
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras import optimizers
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import utils
from memory_bank import *
import tf_utils
import os
# matplotlib inline

# OPTIONS
save_name = './tmp/save'
# if not os.path.exists(save_name):
#     os.makedirs(save_name)
save = True
restore_name = 'save'
restore = False
VISUALIZE = False

# environment parameters
observation_dim = 8
action_dim = 4
# Set learning parameters
num_episodes = 1000
y = .98
epsilon_s = 0.1

games_per_update = 10
memory_size = 60000


env = gym.make('LunarLander-v2')


# set up logging
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()))
history = LossHistory()


model = Sequential()
#model.add(Dense(num_env_variables+num_env_actions, activation='tanh', input_dim=dataX.shape[1]))
model.add(Dense(512, activation='relu', input_dim=12))
model.add(Dense(250, activation='relu', input_dim=512))
model.add(Dense(1))

opt = optimizers.adam(lr=0.001)
model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])

memory_replay = MemoryReplay(memory_size, 8, 4)

#One hot encoding array
possible_actions = np.arange(0,4)
actions_1_hot = np.zeros((4,4))
actions_1_hot[np.arange(4),possible_actions] = 1


def get_policy(s):
    preds = np.empty(shape=[4], dtype=np.float32)
    for action in range(4):
        s_a = np.concatenate((s,actions_1_hot[action]), axis=0)
        predX = np.zeros(shape=(1,12))
        predX[0] = s_a

        #print("trying to predict reward at qs_a", predX[0])
        ex = predX[0].reshape(1,predX.shape[1])
        pred = model.predict(predX[0].reshape(1,predX.shape[1]))
        preds[action] = pred[0][0]
    return np.argmax(preds)

#create lists to contain total rewards and steps per episode
rList = [] # total rewards

for i in range(num_episodes):
    #Reset environment and get first new observation
    s = env.reset()
    #Reduce chance of random action as we train the model.
    e = epsilon_s/((i+1)/float(num_episodes))
    rAll = 0 # total reward
    lAll = 0
    j = 0
    r_prev = 0
    #The Q-Network
    game_data = []
    while j < 3000:
        j += 1
        # Chose action!
        if np.random.rand(1) < e:
            a = env.action_space.sample()
        else:
            a = get_policy(s)

        if VISUALIZE:
            env.render()
        s_a = np.concatenate((s,actions_1_hot[a]), axis=0)

        # take the action
        s,r,d,_ = env.step(a)
        rAll += r
        # add data
        data = [s_a, r]
        game_data.append(data)
        # break if done!
        if d == True:
            break
        if i % 100 == 0:
            env.render()
    rList.append(rAll)
    # manually compute bellman's for rewards
    game_len = len(game_data)
    zero_len = game_len-1
    for k in range(game_len):
        if k == 0:
            pass # reward at final step is reward from game
        else:
            # recurse with belman's equation within the game
            future_reward = y*game_data[zero_len-k+1][1]
            game_data[zero_len-k][1] = game_data[zero_len-k][1] + future_reward
    print("Game #",i, " steps = ", j ,"last reward", r," finished with headscore ", game_data[0][1])

    # add data to the experience_replay bank
    for s,r in game_data:
        memory_replay.add(s, r)

    # Train on data!
    if i % games_per_update == 0:
        X, r = memory_replay.get_training_data()
        model.fit(X,r.reshape([-1,1]), batch_size=32,epochs=2,verbose=2,
            callbacks=[tensorboard, history])

print (rList)
print (history.losses)

print("Saving weights")
model.save_weights("weights-QSAL.h5")

if VISUALIZE:
    reward = 0
    for i in range(50):
        s = env.reset()
        while True:
            a = model.predict([s])
            s,r,d,_ = env.step(a[0]) #observation, reward, done, info
            reward += r
            if d == True:
                break
            env.render()
            time.sleep(0.001)
    print(reward/50.0)
