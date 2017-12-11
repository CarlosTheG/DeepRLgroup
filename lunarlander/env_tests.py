'''
Explore the lunar lander space
'''
import gym
import matplotlib.pyplot as plt
import time

env = gym.make('LunarLander-v2')
state = env.reset()

# 4 actions
print (env.action_space)
# 8 length vector
print (env.observation_space)

env.render()

# while True:
#     state,reward,d,_ = env.step(env.action_space.sample())
#     print (str(state))
#     if d == True:
#         break
#     env.render()
#     time.sleep(0.1)


'''
0 nothing, 1 fire left engine, 2 fire main engine, 3 left engine
'''
def test_actions():
    for i in range(0, 4, 1):
        print ("testing action: ", i)
        env.reset()
        for j in range(200):
            state,r,d,_ = env.step(i)
            env.render()
            time.sleep(0.1)
            if d == True:
                break
            print (str(state))
test_actions()


'''
Firing main engine is -0.3 each frame
Crashing is -100, landing is +100
Each leg touching ground is +10
Given negative reward for how far horizontally the agent is from landing pad.
'''
def test_rewards():
    env.reset()
    for i in range(5000):
        state, reward, done, _ = env.step(3)
        state,reward,done,_ = env.step(2)
        if reward != 0.0:
            print(reward)
            env.render()
        if done:
            break
