from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras import optimizers
import numpy as np
import gym
import os


weigths_filename = "weights-QSAL.h5"

env = gym.make('LunarLander-v2')

model = Sequential()
#model.add(Dense(num_env_variables+num_env_actions, activation='tanh', input_dim=dataX.shape[1]))
model.add(Dense(512, activation='relu', input_dim=12))
model.add(Dense(250, activation='relu', input_dim=512))
model.add(Dense(1))
opt = optimizers.adam(lr=0.001)
model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])

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

#load previous model weights
if True:
    dir_path = os.path.realpath(".")
    fn = dir_path + "/"+weigths_filename
    print("filepath ", fn)
    if  os.path.isfile(fn):
        print("loading weights")
        model.load_weights(weigths_filename)
    else:
        print("File ",weigths_filename," does not exis. Retraining... ")


reward = 0
for i in range(50):
    s = env.reset()
    while True:
        action = get_policy(s)
        s,r,d,_ = env.step(action) #observation, reward, done, info
        reward += r
        if d == True:
            break
        env.render()
print(reward/50.0)
