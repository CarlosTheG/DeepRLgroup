import numpy as np

class ExperienceReplay:
    def __init__(self, memory_size, observation_dim):
        self.memory_size = memory_size
        self.insert_location = 0
        self.count = 0

        self.action = np.empty(self.memory_size, dtype=np.uint8)
        self.reward = np.empty(self.memory_size, dtype=np.float32)
        self.state = np.empty([self.memory_size, observation_dim], dtype=np.float32)
        self.next_state = np.empty([self.memory_size, observation_dim], dtype=np.float32)
        self.terminal = np.empty(self.memory_size, dtype=np.bool)


    def add(self, state, action, next_s, reward, done):
        i = self.insert_location
        self.count += 1

        self.action[i] = action
        self.reward[i] = reward
        self.state[i,:] = state
        self.next_state[i,:] = next_s
        self.terminal[i] = done
        # update insert location to next location in replay memory
        self.insert_location = (self.insert_location + 1) % self.memory_size

    def sample(self):
        idx = np.random.randint(self.memory_size)
        return (self.state[idx,:], self.action[idx], self.next_state[idx,:],
            self.reward[idx], self.terminal[idx])
