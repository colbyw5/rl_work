#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 10:35:31 2020

@author: colbywilkinson
"""

#utilities
import random
import numpy as np
from collections import deque
import pandas as pd
import pickle

# model
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

''' DDQN for PuckWorld '''

# importing PuckWorld environment
from ple.games.puckworld import PuckWorld
from ple import PLE 

class DDQN_agent:
    
    def __init__(self, state_space, action_space, memory_max = 10000,
                 discount = 0.95, epsilon = 1.0, epsilon_min = 0.01,
                 learning_rate = 0.001):
        
        self.state_space = state_space
        self.action_space = action_space
        self.memory = deque(maxlen = memory_max)
        self.gamma = discount
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = 0.9995
        self.learning_rate = learning_rate
        self.q_network = self._build_model()
        
    def _build_model(self):
        
        ''' neural network for Q-value function '''
        
        q_network = Sequential()
        q_network.add(Dense(6, input_dim = len(self.state_space),
                            activation = 'tanh'))
        q_network.add(Dense(6, activation = 'tanh'))
        q_network.add(Dense(len(self.action_space), activation = 'linear'))
        q_network.compile(loss = 'mse',
                          optimizer = Adam(lr = self.learning_rate))
        return q_network
        
    def memorize(self, state, action, reward, next_state, done):
        
        ''' add recent experience to memory to train q_agent'''
        
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        
        ''' select action using epsilon-greedy method '''
        
        if np.randomrand() < self.epsilon:
            return random.sample(self.state_space, 1)
        
        action_q = self.q_network.predict(state)
        return np.argmax(action_q[0])
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        