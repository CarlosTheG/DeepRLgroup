'''
Explore the lunar lander space
'''
import gym
import matplotlib.pyplot as plt
import time

env = gym.make('CarRacing-v0')
state = env.reset()

# 4 actions
print (env.action_space)
# 8 length vector
print (env.observation_space)

env.render()

j = 0
while False: #j < 3000:
    j += 1
    action = env.action_space.sample()
    print (action)
    state,reward,d,_ = env.step(action)
    #print (str(state))
    env.render()

# plt.imshow(state)
# plt.show()


'''
3 vector
    0th: turn angle (-0.5 left, 0.5 right), break into .1s
    1st: accelerator (0 none, 1 accel), break into .1s
    2nd: breaks (0 none, 1 break), break into .1s
'''
def test_actions():
    for i in range(-5, 5, 1):
        action = i / float(10)
        print ("testing action: ", action)
        env.reset()
        for j in range(100):
            state,r,d,_ = env.step([action,0,0])
            env.render()
            if d == True:
                break
test_actions()
'''
unsure...
'''
def test_rewards():
    env.reset()
    for i in range(100):
        state, reward, done, _ = env.step([0, 1, 0])
        #print (reward)
        env.render()
        if done:
            break
