import tensorflow as tf

def format_state(s, params=None):
    return s

def get_reward(r,s,a):
    if a[0] == 2:
        return r + 0.4
    return r
