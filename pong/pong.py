'''
Runs a Pong Q Learned network
'''

import gym
import matplotlib.pyplot as plt
import time
import utils

env = gym.make('Pong-v0')
state = env.reset()

# 6 actions
print (env.action_space)
# 210x160x3 input image
print (env.observation_space)

# for i in range(5):
#     #plt.imshow(s)
#     # take a random action. Returns state which is new game state
#     state, reward, done, _ = env.step(env.action_space.sample())
#     env.render()
#     plt.imshow(state)
# plt.show()
# env.close()



'''
0-1 nothing, 2 upward, 3 downward, 4 upward, 5 downward
** From this we should simplify our output to 3 actions [0, 2, 3] **
'''
def test_actions():
    for i in range(2, 4, 1):
        print ("testing action: ", i)
        for j in range(10):
            env.step(i)
            env.render()
            time.sleep(0.2)

'''
1 reward for winning a round
-1 for losing a round
'''
def test_rewards():
    for i in range(5000):
        state, reward, done, _ = env.step(env.action_space.sample())
        if reward > 0.0:
            print(reward)
            plt.imshow(state)
            plt.show()
