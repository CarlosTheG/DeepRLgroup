'''
This should run correctly if gym is installed.
'''

import gym
import time
env = gym.make('LunarLander-v2')

NUM_RESTARTS = 1
avg_time = 0.0
frames = 0.0
for _ in range(NUM_RESTARTS):
    env.reset()
    tick = time.time()
    for num in range(4000):
        #env.render()
        _,_,d,_ = env.step(env.action_space.sample()) # take a random action
        env.render()
        if d == True:
            frames += num
            break
    avg_time += (time.time() - tick) / float(NUM_RESTARTS)

print ("Done! Average runtime:",avg_time)
print ("frames per second", frames/(avg_time*10))
