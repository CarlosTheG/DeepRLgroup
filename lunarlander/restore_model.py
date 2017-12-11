import tensorflow as tf
import gym
import tf_utils


env = gym.make('LunarLander-v2')
tf.reset_default_graph()

#These lines establish the feed-forward part of the network used to choose actions
inputs = tf.placeholder(shape=[1,8],dtype=tf.float32)
# Build first layer
W1, b1, a1 = tf_utils.build_NN_layer(inputs, [8,40], 'layer1')
# Build second layer
W2, b2, a2 = tf_utils.build_NN_layer(a1, [40,30], 'layer2')
# Build shrinking third layer
W3, b3, Qout = tf_utils.build_NN_layer(a2, [30,4], 'layer3')
# Softmax prediction
predict = tf.argmax(Qout,1)

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, "/tmp/model.ckpt")

    for i in range(50):
        s = env.reset()
        while True:
            formatted_input = utils.format_state(s)
            a = sess.run(predict, feed_dict={inputs:[formatted_input.flatten()]})
            s,r,d,_ = env.step(a[0]) #observation, reward, done, info
            if d == True:
                break
            env.render()
            time.sleep(0.01)
