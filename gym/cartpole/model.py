import numpy as np
import random
from collections import deque

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

class CartPoleModel:

    def __init__(self):
        self.model = Sequential()
        self.model.add(Dense(24, activation="relu", input_shape=(4,)))
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(2))
        self.model.compile(optimizer='nadam', loss='mse')
        self.history = deque(maxlen=10000)

    def get_action(self, state, epsilon=0, verbose=0):
        if random.random() < epsilon:
            chosen_action = random.choice((0, 1))
            if verbose: print(state, '!!', chosen_action)
            return chosen_action

        q_values = self.model.predict(np.reshape(state, [1, 4]))[0]
        chosen_action = np.argmax(q_values)
        if verbose: print(state, '=>', chosen_action, q_values)
        return chosen_action

    def remember(self, entry):
        self.history.append(entry)

    def load(self):
        try:
            self.model.load_weights('cartpole_model.hd5')
            print('Loaded!')
        except Exception as e:
            print(e)

    def save(self):
        self.model.save_weights('cartpole_model.hd5')

    def learn(self, batch_size=64, discount=.9, verbose=0):
        if len(self.history) < batch_size: return
        batch = random.sample(self.history, batch_size)

        foo = zip(*batch)
        states, actions, rewards, state_nexts, dones = foo
        states = np.reshape(states, [batch_size, 4])
        state_nexts = np.reshape(state_nexts, [batch_size, 4])
        state_preds = self.model.predict(states)
        state_next_preds = self.model.predict(state_nexts)

        for idx, action in enumerate(actions):
            q_learned = rewards[idx]
            if not dones[idx]:
                q_learned += discount * np.amax(state_next_preds[idx])
            state_preds[idx][action] = q_learned
            
        self.model.fit(states, state_preds, verbose=verbose)

