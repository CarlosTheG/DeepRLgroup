import numpy as np

class MemoryReplay:
    def __init__(self, memory_size, observation_dim, action_dim):
        self.memory_size = memory_size
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.insert_location = 0
        self.count = 0

        self.state = np.empty([self.memory_size, self.observation_dim+self.action_dim],
            dtype=np.float32)
        self.reward = np.empty([self.memory_size], dtype=np.float32)


    def add(self, state, reward):
        i = self.insert_location
        self.count += 1

        self.state[i,:] = state
        self.reward[i] = reward
        # update insert location to next location in replay memory
        self.insert_location = (self.insert_location + 1) % self.memory_size


    def get(self, idx):
        return (self.state[idx,:], self.reward[idx])

    def get_training_data(self, size=None):
        limit = min(self.memory_size, self.count)
        if size == None:
            size = limit
        idxs = np.random.randint(limit, size=size)
        X = np.empty([size, self.observation_dim+self.action_dim], dtype=np.float32)
        r = np.empty([size], dtype=np.float32)
        i = 0
        for j in idxs:
            s, reward = self.get(j)
            X[i, :] = s
            r[i] = reward
            i += 1
        return np.array(X), np.array(r)
