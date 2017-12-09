import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import utils
# matplotlib inline

def variable_summaries(var,name):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope(name):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('Mean', mean)
    with tf.name_scope('Stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('Stddev', stddev)
    tf.summary.scalar('Max', tf.reduce_max(var))
    tf.summary.scalar('Min', tf.reduce_min(var))
    tf.summary.histogram('Histogram', var)

def weight_variable(shape,name):
    '''
    Initialize weights
    :param shape: shape of weights, e.g. [w, h ,Cin, Cout] where
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters
    Cout: the number of filters
    :return: a tensor variable for weights with initial values
    '''

    # IMPLEMENT YOUR WEIGHT_VARIABLE HERE
    initial = tf.truncated_normal(shape, stddev=0.1)
    W = tf.Variable(initial)
    variable_summaries(W,name)

    return W

def bias_variable(shape,name):
    '''
    Initialize biases
    :param shape: shape of biases, e.g. [Cout] where
    Cout: the number of filters
    :return: a tensor variable for biases with initial values
    '''

    # IMPLEMENT YOUR BIAS_VARIABLE HERE
    initial = tf.constant(0.1, shape=shape)
    b = tf.Variable(initial)
    variable_summaries(b,name)

    return b

def conv2d(x, W, strides):
    '''
    Perform 2-D convolution
    :param x: input tensor of size [N, W, H, Cin] where
    N: the number of images
    W: width of images
    H: height of images
    Cin: the number of channels of images
    :param W: weight tensor [w, h, Cin, Cout]
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters = the number of channels of images
    Cout: the number of filters
    :return: a tensor of features extracted by the filters, a.k.a. the results after convolution
    '''

    # IMPLEMENT YOUR CONV2D HERE
    h_conv = tf.nn.conv2d(x, W, strides=strides, padding='SAME')

    return h_conv

def activation(actfun,input,name):
    act = actfun(input)
    variable_summaries(act,name)
    return act

# OPTIONS
save_name = 'save'
save = True
restore_name = 'save'
restore = False

output_to_action = {0:0,1:2,2:3}
env = gym.make('Pong-v0')
tf.reset_default_graph()

#These lines establish the feed-forward part of the network used to choose actions
inputs = tf.placeholder(shape=[1,160,160,1],dtype=tf.float32)
# W1 = tf.Variable(tf.random_uniform([160*160,2000],0,0.01))

wc1 = weight_variable([10,10,1,32],'Conv_1_Weights') # filter: width, height, numchannels, numfilters
bc1 = bias_variable([32],'Conv_1_Bias')
# hc1 = tf.nn.relu(conv2d(inputs, wc1, [1,4,4,1]) + bc1) # batch stride stride depth
hc1 = activation(tf.nn.relu,conv2d(inputs, wc1, [1,4,4,1]) + bc1,'Conv_1_Activation')

wc2 = weight_variable([5,5,32,32],'Conv_2_Weights') # filter: width, height, numchannels, numfilters
bc2 = bias_variable([32],'Conv_2_Bias')
hc2 = tf.nn.relu(conv2d(hc1, wc2, [1,2,2,1]) + bc2,'Conv_2_Activations') # batch stride stride depth

wfc1 = weight_variable([12800, 1024],'FC_1_Weights')
bfc1 = bias_variable([1024],'FC_1_Bias')
hc2_flat = tf.reshape(hc2, [-1,12800])
hfc1 = activation(tf.nn.relu,tf.matmul(hc2_flat, wfc1) + bfc1,'FC_1_Activation')

wfc2 = weight_variable([1024, 3],'Output_Weights')
bfc2 = bias_variable([3],'Output_Bias')
Qout = tf.matmul(hfc1, wfc2) + bfc2
predict = tf.argmax(Qout,1)

#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
nextQ = tf.placeholder(shape=[1,3],dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.AdamOptimizer(learning_rate=0.1)
updateModel = trainer.minimize(loss)

# summaries
variable_summaries(Qout,'Q_Out')
tf.summary.scalar('Loss',loss)
summary_op = tf.summary.merge_all()

init = tf.initialize_all_variables()

# Set learning parameters
y = .99
e = 0.1
num_episodes = 1
#create lists to contain total rewards and steps per episode
rList = [] # total rewards

saver = tf.train.Saver()

with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter('./tmp/summary', sess.graph)
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
        #The Q-Network
        while True:
            #Choose an action by greedily (with e chance of random action) from the Q-network
            formatted_input = utils.format_state(s, utils.GRAY)
            a,allQ = sess.run([predict,Qout],
                feed_dict={inputs:[formatted_input[:,:,np.newaxis]]})
            if np.random.rand(1) < e:
                a[0] = np.random.choice(3,1)
            #Get new state and reward from environment
            s1,r,d,_ = env.step(output_to_action[a[0]]) # observation, reward, done, info
            r = utils.convert_reward(r, a)
            #Obtain the Q' values by feeding the new state through our network
            new_state = utils.format_state(s1, utils.GRAY)
            Q1 = sess.run(Qout,feed_dict={inputs:[new_state[:,:,np.newaxis]]})
            #Obtain maxQ' and set our target value for chosen action.
            maxQ1 = np.max(Q1)
            targetQ = allQ
            targetQ[0,a[0]] = r + y*maxQ1 # add the following to location of last action in targetQ: reward + discount rate*maxreward
            # print (str(W1.eval()))
            rAll += r
            s = s1
            if d == True: # train network and generate summary
                _, summary_str = sess.run([updateModel, summary_op],
                                          feed_dict={inputs: [formatted_input[:, :, np.newaxis]], nextQ: targetQ})
                summary_writer.add_summary(summary_str, i)
                summary_writer.flush()
                #Reduce chance of random action as we train the model.
                e = 1./((i/50) + 10)
                break
            else: # just train network
                _ = sess.run([updateModel],feed_dict={inputs: [formatted_input[:, :, np.newaxis]], nextQ: targetQ})
            # if i == num_episodes-1:
            #     env.render()
        rList.append(rAll)
        print ("Reward for round", i, "is :", rAll)
    if save:
        save_path = saver.save(sess, "/tmp/model.ckpt")
    sess.close()


# plt.plot(rList)
