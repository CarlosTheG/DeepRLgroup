import tensorflow as tf

def format_state(s, params=None):
    return s

def get_reward(r,s,a):
    r += 1 # Free point for staying alive!
    if a[0] == 2:
        return r + 0.3
    return r
