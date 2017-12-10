import tensorflow as tf
import numpy as np

def format_state(s, params=None):
    return s

def get_reward(r,s,a):
    r = 0
    speed = max(2,np.sqrt(s[2]**2 + s[3]**2))
    return (2 - speed)**4
    # if s[1] < 0.5:
    #     r -= (speed*5)*((1-s[1])+1)
    #     if speed < 0.05:
    #         r += 5
    #     return r
    # else:
    #     return r + 0.3
