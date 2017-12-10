import tensorflow as tf
import numpy as np

def format_state(s, params=None):
    return s

def get_reward(r,s,a):
    speed = np.sqrt(s[2]**2 + s[3]**2)
    if s[1] < 0.5:
        r = r - (speed*5)*((1-s[1])+1)
        if speed < 0.2:
            r += 5
    if a[0] == 0:
        return r - 0.5
    else:
        return r + 0.3
