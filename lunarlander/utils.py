import tensorflow as tf
import numpy as np

def format_state(s, params=None):
    return s

def get_reward(r,s,a):
    return r
    # # add reward for speed
    # speed = min(2, np.sqrt(s[2]**2 + s[3]**2))
    # r += (2 - speed)**3
    # # add reward for distance to center
    # dist_to_land = min(2, np.sqrt(s[0]**2 + (s[1]-0.2)**2))
    # r += (2 - dist_to_land)**2
    # return r
