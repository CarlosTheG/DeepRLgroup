import tensorflow as tf
import numpy as np

def format_state(s, params=None):
    return s

def get_reward(r,s,a):
    if r < -80:
        # was a crash landing
        r = -200
    elif r > 80:
        # was a soft landing!
        r = 200
    # add reward for speed
    speed = min(2,np.sqrt(s[2]**2 + s[3]**2))
    r += (2*(2 - speed))**5
    # add reward for distance to center
    dist_to_land = min(2, np.sqrt(s[0]**2 + (s[1]-0.2)**2))
    r += (2 - dist_to_land)**3
    # add reward for staying stable
    dist_to_straight = np.sqrt(s[4]**2 + s[5]**2)
    r -= (dist_to_straight+1)**2
    return r
