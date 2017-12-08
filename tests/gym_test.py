'''
This should run correctly if gym is installed.
'''

import gym
env = gym.make('Pong-v0')
env.reset()
for _ in range(4000):
    env.render()
    _,_,d,_ = env.step(env.action_space.sample()) # take a random action
    if d == True:
        print ("Game Over!")
        break
