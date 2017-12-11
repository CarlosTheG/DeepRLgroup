import numpy as np

class MemoryReplay:
    def __init__(self, memory_size, observation_dim, action_dim):
        self.memory_size = memory_size
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.insert_location = 0
        self.count = 0

        self.state = np.empty([self.memory_size, self.observation_dim],
            dtype=np.float32)
        self.action = np.empty([self.memory_size], dtype=np.uint8)
        self.reward = np.empty([self.memory_size], dtype=np.float32)


    def add(self, state, action, reward):
        i = self.insert_location
        self.count += 1

        self.state[i,:] = state
        self.action[i] = action
        self.reward[i] = reward
        # update insert location to next location in replay memory
        self.insert_location = (self.insert_location + 1) % self.memory_size


    def get(self, idx):
        return (self.state[idx,:], self.action[idx], self.reward[idx])

    def get_training_data(self, size=None):
        limit = min(self.memory_size, self.count)
        if size == None:
            size = limit*2
        idxs = np.random.randint(limit, size=size)
        X = np.empty([size, self.observation_dim], dtype=np.float32)
        a = np.empty([size], dtype=np.uint8)
        r = np.empty([size], dtype=np.float32)
        i = 0
        for j in idxs:
            s, action, reward = self.get(j)
            X[i, :] = s
            a[i] = action
            r[i] = reward
            i += 1
        return np.array(X), np.array(a), np.array(r)
