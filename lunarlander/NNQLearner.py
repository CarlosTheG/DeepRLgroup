import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import utils
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
W1, b1, a1 = tf_utils.build_NN_layer(inputs, [8,8], 'layer1')
# Build second layer
W2, b2, a2 = tf_utils.build_NN_layer(a1, [8,8], 'layer2')
# Build shrinking third layer
W3, b3, Qout = tf_utils.build_NN_layer(a2, [8,4], 'layer3')
# Softmax prediction
predict = tf.argmax(Qout,1)

#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
nextQ = tf.placeholder(shape=[1,4],dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.AdamOptimizer(learning_rate=0.01)
updateModel = trainer.minimize(loss)

tf.summary.scalar('Loss',loss)
summary_op = tf.summary.merge_all()

init = tf.initialize_all_variables()
# Set learning parameters
num_episodes = 2000
y = .99
epsilons = np.linspace(1, 0.1, num_episodes*20)


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
        e = epsilons[i]
        #Reduce chance of random action as we train the model.
        # e = 4./((i/200) + 10)
        rAll = 0 # total reward
        lAll = 0
        j = 0
        #The Q-Network
        while j < 3000:
            j += 1
            #Choose an action by greedily (with e chance of random action) from the Q-network
            formatted_input = utils.format_state(s)
            a,allQ = sess.run([predict,Qout],
                feed_dict={inputs:[formatted_input.flatten()]})
            if np.random.rand(1) < e:
                a[0] = env.action_space.sample()
            #Get new state and reward from environment
            s1,r,d,_ = env.step(a[0]) #observation, reward, done, info
            r = utils.get_reward(r, s1, a)
            #Obtain the Q' values by feeding the new state through our network
            new_state = utils.format_state(s1)
            Q1 = sess.run(Qout,feed_dict={inputs:[new_state.flatten()]})
            #Obtain maxQ' and set our target value for chosen action.
            maxQ1 = np.max(Q1)
            targetQ = allQ
            targetQ[0,a[0]] = r + y*maxQ1 # add the following to location of last action in targetQ: reward + discount rate*maxreward
            #Train our network using target and predicted Q values
            _,l = sess.run([updateModel,loss],feed_dict={inputs:[formatted_input.flatten()],nextQ:targetQ})
            rAll += r
            lAll += l
            s = s1
            if d == True:
                _, summary_str = sess.run([updateModel, summary_op],
                    feed_dict={inputs: [formatted_input.flatten()], nextQ: targetQ})
                summary_writer.add_summary(summary_str, i)
                summary_writer.flush()
                break

        summary = tf.Summary()
        summary.value.add(tag='Reward',simple_value=rAll)
        summary.value.add(tag='Total_Loss',simple_value=lAll)
        summary_writer.add_summary(summary, i)
        summary_writer.flush()
        rList.append(rAll)
        print ("Reward for round", i, "is :", rAll)

    print (rList)
    if VISUALIZE:
        for i in range(200):
            s = env.reset()
            while True:
                formatted_input = utils.format_state(s)
                a,allQ = sess.run([predict,Qout],
                    feed_dict={inputs:[formatted_input.flatten()]})
                s,r,d,_ = env.step(a[0]) #observation, reward, done, info
                if d == True:
                    break
                env.render()
                time.sleep(0.001)

    if save:
        save_path = saver.save(sess, save_name)
