from collections import deque
import numpy as np
import os
import gzip
import pickle

class ReplayBuffer:

    # TODO: implement a capacity for the replay buffer (FIFO, capacity: 1e5 - 1e6)

    # Replay buffer for experience replay. Stores transitions.
    def __init__(self, capacity=1e4):
        self.capacity = capacity
        self.count = 0 
        self.report = True

        self.states = deque()
        self.actions = deque()
        self.next_states = deque()
        self.rewards = deque()
        self.dones = deque()
        

    def add_transition(self, state, action, next_state, reward, done):
        """
        This method adds a transition to the replay buffer.
        """
        if self.count == self.capacity :
            self.states.popleft()
            self.actions.popleft()
            self.next_states.popleft()
            self.rewards.popleft()
            self.dones.popleft()
            self.count-=1

        self.states.append(state)
        self.actions.append(action)
        self.next_states.append(next_state)
        self.rewards.append(reward)
        self.dones.append(1 if done==True else 0)
        self.count+=1
        if self.count%1000 == 0 and self.report:
            print("replay buffer at ",self.count)
            if self.count == self.capacity :
                print("buffer full")
                self.report=False
            

    def next_batch(self, batch_size):
        """
        This method samples a batch of transitions.
        """
        batch_indices = np.random.choice(np.arange(self.count), batch_size)
        # TODO : bad code, fix later
        batch_states = np.array([self.states[i] for i in batch_indices])
        batch_actions = np.array([self.actions[i] for i in batch_indices])
        batch_next_states = np.array([self.next_states[i] for i in batch_indices])
        batch_rewards = np.array([self.rewards[i] for i in batch_indices])
        batch_dones = np.array([self.dones[i] for i in batch_indices])
        return batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones
