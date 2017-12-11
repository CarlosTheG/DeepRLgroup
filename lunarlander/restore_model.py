from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras import optimizers
import gym
import os


weigths_filename = "LL-QL-v2-weights2.h5"

env = gym.make('LunarLander-v2')

model = Sequential()
#model.add(Dense(num_env_variables+num_env_actions, activation='tanh', input_dim=dataX.shape[1]))
model.add(Dense(512, activation='relu', input_dim=8))
model.add(Dense(250, activation='relu', input_dim=512))
model.add(Dense(4))

opt = optimizers.adam(lr=0.001)
model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])

#load previous model weights
if load_previous_weights:
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
        a = model.predict(np.array([s]))[0]
        action = np.argmax(a)
        s,r,d,_ = env.step(action) #observation, reward, done, info
        reward += r
        if d == True:
            break
        env.render()
        time.sleep(0.001)
print(reward/50.0)
