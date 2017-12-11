''' EXPERIENCE REPLAY WOW '''
import gym
import numpy as np
import random
import tensorflow as tf
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
tf.reset_default_graph()

#These lines establish the feed-forward part of the network used to choose actions
inputs = tf.placeholder(shape=[None,observation_dim + action_dim],dtype=tf.float32)
target = tf.placeholder(shape=[None,1],dtype=tf.float32)
# Build first layer
W1, b1, a1 = tf_utils.build_NN_layer(inputs, [12,500], 'layer1')
# Build second layer
W2, b2, a2 = tf_utils.build_NN_layer(a1, [500,250], 'layer2')
# Build shrinking third layer
W3, b3, Qout = tf_utils.build_NN_layer(a2, [250,1], 'layer3')
# Softmax prediction

#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
loss = tf.losses.mean_squared_error(Qout,target)
trainer = tf.train.AdamOptimizer(learning_rate=0.01)
updateModel = trainer.minimize(loss)

tf.summary.scalar('Loss',loss)
summary_op = tf.summary.merge_all()

init = tf.initialize_all_variables()

memory_replay = MemoryReplay(memory_size, 8, 4)

#One hot encoding array
possible_actions = np.arange(0,4)
actions_1_hot = np.zeros((4,4))
actions_1_hot[np.arange(4),possible_actions] = 1



#create lists to contain total rewards and steps per episode
rList = [] # total rewards

saver = tf.train.Saver()

with tf.Session() as sess:

    def get_policy(s):
        preds = np.empty(shape=[4], dtype=np.float32)
        for action in range(4):
            s_a = np.concatenate((s,actions_1_hot[action]), axis=0)
            predX = np.zeros(shape=(1,12))
            predX[0] = s_a

            #print("trying to predict reward at qs_a", predX[0])
            ex = predX[0].reshape(1,predX.shape[1])
            pred = sess.run(Qout, feed_dict={inputs:ex})
            preds[action] = pred[0][0]
        return np.argmax(preds)

    summary_writer = tf.summary.FileWriter('./tmp/summary', sess.graph)

    if restore:
        saver.restore(sess, './tmp/' + restore_name)
    else:
        sess.run(init)

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
            _, l = sess.run([updateModel,loss],
                feed_dict={inputs:X,target:r.reshape([-1, 1])})
            print ("Updated model with loss = ", l)

        summary = tf.Summary()
        summary.value.add(tag='Reward',simple_value=rAll)
        summary.value.add(tag='Total_Loss',simple_value=lAll)
        summary_writer.add_summary(summary, i)
        summary_writer.flush()

    print (rList)
    if VISUALIZE:
        reward = 0
        for i in range(50):
            s = env.reset()
            while True:
                formatted_input = utils.format_state(s)
                a,allQ = sess.run([predict,Qout],
                    feed_dict={inputs:[formatted_input.flatten()]})
                s,r,d,_ = env.step(a[0]) #observation, reward, done, info
                reward += r
                if d == True:
                    break
                env.render()
                time.sleep(0.001)
        print(reward/50.0)

    if save:
        save_path = saver.save(sess, save_name)
