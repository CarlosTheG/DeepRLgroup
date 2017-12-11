import tensorflow as tf
import numpy as np

def format_state(s, params=None):
    return s

def get_reward(r,s,a):
    r = 0
    # add reward for speed
    speed = min(2,np.sqrt(s[2]**2 + s[3]**2))
    r += (2 - speed)
    # add reward for distance to center
    dist_to_center = min(2, np.sqrt(s[0]**2 + (s[1]-0.5)**2))
    r += (2 - dist_to_center)
    return r
    # if s[1] < 0.5:
    #     r -= (speed*5)*((1-s[1])+1)
    #     if speed < 0.05:
    #         r += 5
    #     return r
    # else:
    #     return r + 0.3
