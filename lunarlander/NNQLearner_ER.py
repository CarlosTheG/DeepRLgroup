''' EXPERIENCE REPLAY WOW '''
import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import utils
from replay_mem import *
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


env = gym.make('LunarLander-v2')
tf.reset_default_graph()

#These lines establish the feed-forward part of the network used to choose actions
inputs = tf.placeholder(shape=[1,8],dtype=tf.float32)
# Build first layer
W1, b1, a1 = tf_utils.build_NN_layer(inputs, [8,500], 'layer1')
# Build second layer
W2, b2, a2 = tf_utils.build_NN_layer(a1, [500,250], 'layer2')
# Build shrinking third layer
W3, b3, Qout = tf_utils.build_NN_layer(a2, [250,4], 'layer3')
# Softmax prediction
predict = tf.argmax(Qout,1)

#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
nextQ = tf.placeholder(shape=[1,4],dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.AdamOptimizer(learning_rate=0.001)
updateModel = trainer.minimize(loss)

tf.summary.scalar('Loss',loss)
summary_op = tf.summary.merge_all()

init = tf.initialize_all_variables()
# Set learning parameters
num_episodes = 5000
y = .99
epsilon_s = 0.1

RETRAIN_GAMES = 10
BATCH_SIZE = 30000
memory_size = 60000
experience_replay = ExperienceReplay(memory_size, 8)
# populate initial memory bank with random actions
s = env.reset()
for i in range(memory_size):
    a = env.action_space.sample()
    s1,r,d,_ = env.step(a)
    experience_replay.add(s, a, s1, r, d)
    s = s1
    if d == True:
        s = env.reset()



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
            s1,r,d,_ = env.step(a[0])
            r = utils.get_reward(r, s1, a)
            # add to memory bank
            experience_replay.add(s, a[0], s1, r, d)
            rAll += r
            s = s1
            if d == True:
                break
            env.render()

        if i % RETRAIN_GAMES:
            # Train on data!
            for _ in range(BATCH_SIZE):
                # fetch sample from memory
                s, a, s1, r, d = experience_replay.sample()
                # fetch prediction for state s
                formatted_input = utils.format_state(s)
                allQ = sess.run(Qout, feed_dict={inputs:[formatted_input.flatten()]})
                #Obtain the Q' values by feeding the new state through our network
                new_state = utils.format_state(s1)
                Q1 = sess.run(Qout,feed_dict={inputs:[new_state.flatten()]})
                #Obtain maxQ' and set our target value for chosen action.
                maxQ1 = np.max(Q1)
                targetQ = allQ
                targetQ[0,a] = r + y*maxQ1 # add the following to location of last action in targetQ: reward + discount rate*maxreward
                #Train our network using target and predicted Q values
                _, l = sess.run([updateModel,loss],feed_dict={inputs:[formatted_input],nextQ:targetQ})
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
