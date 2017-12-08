import numpy as np

ORIGINAL = 0
SHRUNKEN = 1
GRAY = 2
BINARY = 3
'''
Takes in the observation state from Pong-v0 and converts it to a model
friendly input by removing unneeded detail.

form parameter explanation:
    0: return the original state
    1: return the shrunken state
    2: shrink and grayscale
    3: shrink and convert to binary
'''
def format_state(state, form=ORIGINAL):
    if form == ORIGINAL:
        return state
    shrunken = state[34:194,:,:]
    if form == SHRUNKEN:
        return shrunken
    elif form == GRAY:
            gray = np.dot(shrunken[...,:3], [0.299, 0.587, 0.114])
            return gray
    raise Exception("BINARY FORM NOT IMPLEMENTED YET!!")
