import numpy as np

class MemoryReplay:
    def __init__(self, memory_size, observation_dim, action_dim):
        self.memory_size = memory_size
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.insert_location = 0
        self.count = 0

        self.target = np.empty([self.memory_size, self.action_dim], dtype=np.float32)
        self.state = np.empty([self.memory_size, self.observation_dim], dtype=np.float32)


    def add(self, state, target):
        i = self.insert_location
        self.count += 1

        self.target[i,:] = target
        self.state[i,:] = state
        # update insert location to next location in replay memory
        self.insert_location = (self.insert_location + 1) % self.memory_size

    # def sample(self):
    #     limit = min(self.memory_size, self.count)
    #     idx = np.random.randint(limit)
    #     return (self.state[idx,:], self.action[idx], self.reward[idx])

    def get(self, idx):
        return (self.state[idx,:], self.target[idx,:])

    def get_training_data(self, size=None):
        limit = min(self.memory_size, self.count)
        if size == None:
            size = limit*2
        idxs = np.random.randint(limit, size=size)
        X = np.empty([self.memory_size, self.observation_dim], dtype=np.float32)
        y = np.empty([self.memory_size, self.action_dim], dtype=np.float32)
        for i in idxs:
            s, target = self.get(i)
            X[i, :] = s
            y[i, :] = target
        return np.array(X), np.array(y)
