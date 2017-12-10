import tensorflow as tf
import numpy as np

def format_state(s, params=None):
    return s

def get_reward(r,s,a):
    speed = np.sqrt(s[2]**2 + s[3]**2)
    r = r - speed
    if a[0] == 0:
        return r - 0.5
    else:
        return r + 0.3
