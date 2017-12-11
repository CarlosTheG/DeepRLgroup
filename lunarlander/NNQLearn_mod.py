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

VISUALIZE = True

# Set learning parameters
num_episodes = 1000
y = .98
epsilon_s = 0.1

games_per_update = 10
memory_size = 60000


env = gym.make('LunarLander-v2')
tf.reset_default_graph()

#These lines establish the feed-forward part of the network used to choose actions
inputs = tf.placeholder(shape=[None,8],dtype=tf.float32)
nextQ = tf.placeholder(shape=[None,4],dtype=tf.float32)
# Build first layer
W1, b1, a1 = tf_utils.build_NN_layer(inputs, [8,500], 'layer1')
# Build second layer
W2, b2, a2 = tf_utils.build_NN_layer(a1, [500,250], 'layer2')
# Build shrinking third layer
W3, b3, Qout = tf_utils.build_NN_layer(a2, [250,4], 'layer3')
# Softmax prediction
predict = tf.argmax(Qout,1)

#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
cross_entropy_loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=nextQ, logits=Qout))
trainer = tf.train.AdamOptimizer(learning_rate=0.001)
updateModel = trainer.minimize(cross_entropy_loss)

tf.summary.scalar('Loss',cross_entropy_loss)
summary_op = tf.summary.merge_all()

init = tf.initialize_all_variables()

memory_replay = MemoryReplay(memory_size, 8, 4)


#create lists to contain total rewards and steps per episode
rList = [] # total rewards

saver = tf.train.Saver()

with tf.Session() as sess:

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
            formatted_input = utils.format_state(s)
            # Chose action!
            a = [0]
            if np.random.rand(1) < e:
                a[0] = env.action_space.sample()
            else:
                a = sess.run(predict, feed_dict={inputs:[formatted_input.flatten()]})

            # take the action
            s,r,d,_ = env.step(a[0])
            # add data
            data = [formatted_input, a[0], r]
            game_data.append(data)

            rAll += r
            if d == True:
                break
            # if i % 100 == 0:
            #     env.render()

        # manually compute bellman's for rewards
        game_len = len(game_data)
        zero_len = game_len-1
        for j in range(game_len):
            if j == 0:
                pass # reward at final step is reward from game
            else:
                # recurse with belman's equation within the game
                future_reward = y*game_data[zero_len-j+1][2]
                game_data[zero_len-j][2] = game_data[zero_len-j][2]

        gameX = np.array([item[0] for item in game_data])
        targets = sess.run(Qout, feed_dict={inputs:gameX})

        # add data to the experience_replay bank
        for j in range(game_len):
            s,a,r = game_data[j]
            target = targets[j]
            target[a] = r
            memory_replay.add(s, target)

        # Train on data!
        if i % games_per_update == 0:
            X, y = memory_replay.get_training_data()
            _, l = sess.run([updateModel,cross_entropy_loss],feed_dict={inputs:X,nextQ:y})
            print ("Updated model with loss = ", l)
            lAll += l

        summary = tf.Summary()
        summary.value.add(tag='Reward',simple_value=rAll)
        summary.value.add(tag='Total_Loss',simple_value=lAll)
        summary_writer.add_summary(summary, i)
        summary_writer.flush()

        rList.append(rAll)
        print ("Reward for round", i, "is :", rAll)

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
