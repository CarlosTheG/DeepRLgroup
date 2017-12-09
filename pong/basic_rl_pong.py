import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import utils
# matplotlib inline

# OPTIONS
save_name = 'save'
save = True
restore_name = 'save'
restore = False

output_to_action = {0:0,1:2,2:3}
env = gym.make('Pong-v0')
tf.reset_default_graph()

#These lines establish the feed-forward part of the network used to choose actions
inputs = tf.placeholder(shape=[1,160*160],dtype=tf.float32)
W = tf.Variable(tf.random_uniform([160*160,3],0,0.01))
Qout = tf.matmul(inputs,W)
predict = tf.argmax(Qout,1)

#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
nextQ = tf.placeholder(shape=[1,3],dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.AdamOptimizer(learning_rate=0.1)
updateModel = trainer.minimize(loss)

init = tf.initialize_all_variables()
# Set learning parameters
y = .99
e = 0.1

num_episodes = 2000
#create lists to contain total rewards and steps per episode
rList = [] # total rewards

saver = tf.train.Saver()

with tf.Session() as sess:
    if restore:
        saver.restore(sess, "/tmp/" + restore_name)
    else:
        sess.run(init)
    print('session initiated')
    for i in range(num_episodes):
        #Reset environment and get first new observation
        s = env.reset()
        rAll = 0 # total reward
        d = False # whether or not to reduce chance of random action after training model
        j = 0
        #The Q-Network
        while True:
            #Choose an action by greedily (with e chance of random action) from the Q-network
            formatted_input = utils.format_state(s, utils.GRAY)
            a,allQ = sess.run([predict,Qout],
                feed_dict={inputs:[formatted_input.flatten()]})
            if np.random.rand(1) < e:
                a[0] = np.random.choice(3,1)
            #Get new state and reward from environment
            s1,r,d,_ = env.step(output_to_action[a[0]]) #observation, reward, done, info
            r = utils.convert_reward(r, a)
            #Obtain the Q' values by feeding the new state through our network
            new_state = utils.format_state(s1, utils.GRAY)
            Q1 = sess.run(Qout,feed_dict={inputs:[new_state.flatten()]})
            #Obtain maxQ' and set our target value for chosen action.
            maxQ1 = np.max(Q1)
            targetQ = allQ
            targetQ[0,a[0]] = r + y*maxQ1 # add the following to location of last action in targetQ: reward + discount rate*maxreward
            #Train our network using target and predicted Q values
            _,W1 = sess.run([updateModel,W],
                feed_dict={inputs:[formatted_input.flatten()],nextQ:targetQ})
            rAll += r
            s = s1
            if d == True:
                #Reduce chance of random action as we train the model.
                e = 1./((i/50) + 10)
                break

            if i % 50 == 0:
                env.render()

        rList.append(rAll)
        print ("Reward for round", i, "is :", rAll)

    if save:
        save_path = saver.save(sess, "./tmp/" + save_name)

print (rList)
# plt.plot(rList)
