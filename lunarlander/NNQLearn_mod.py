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
from memory_bank_v2 import *
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

VISUALIZE = True

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
model.add(Dense(512, activation='relu', input_dim=8))
model.add(Dense(250, activation='relu', input_dim=512))
model.add(Dense(4))

opt = optimizers.adam(lr=0.001)
model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])


memory_replay = MemoryReplay(memory_size, 8, 4)


#create lists to contain total rewards and steps per episode
rList = [] # total rewards

print('session initiated')

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
        formatted_input = utils.format_state(s)
        # Chose action!
        if np.random.rand(1) < e:
            a = env.action_space.sample()
        else:
            a = model.predict(np.array([formatted_input]))[0]
            a = np.argmax(a)

        # take the action
        s,r,d,_ = env.step(a)
        # add data
        data = [formatted_input, a, r]
        game_data.append(data)

        rAll += r
        if d == True:
            break
        if i % 50 == 0:
            env.render()

    # manually compute bellman's for rewards
    game_len = len(game_data)
    zero_len = game_len-1
    for j in range(game_len):
        if j == 0:
            pass # reward at final step is reward from game
        else:
            # recurse with belman's equation within the game
            future_reward = y*game_data[zero_len-j+1][2]
            game_data[zero_len-j][2] = game_data[zero_len-j][2] + future_reward
    print("Game #",i, " steps = ", j ,"last reward", r," finished with headscore ", game_data[0][2])


    # add data to the experience_replay bank
    for s,a,r in game_data:
        s,a,r = game_data[j]
        memory_replay.add(s, a, r)

    # Train on data!
    if i % games_per_update == 0:
        X, a, r = memory_replay.get_training_data()
        targets = model.predict(X)
        r[np.where(np.isnan(r))] = 0
        targets[:, a] = r
        model.fit(X,targets, batch_size=32,epochs=2,verbose=2,
                    callbacks=[tensorboard, history])

    rList.append(rAll)


print (rList)
print (history.losses)

if VISUALIZE:
    reward = 0
    for i in range(50):
        s = env.reset()
        while True:
            a = model.predict(np.array([s]))[0]
            action = np.argmax(a)
            s,r,d,_ = env.step(action) #observation, reward, done, info
            reward += r
            if d == True:
                break
            env.render()
            time.sleep(0.001)
    print(reward/50.0)
