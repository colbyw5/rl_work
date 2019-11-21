#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 08 16:33:40 2019

@author: colbywilkinson
"""

#utilities
import random
import numpy as np
from collections import deque
import math

# model
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# custom function for score collection
#from scores.score_logger import ScoreLogger

# water world

# importing waterworld environment
from ple.games.waterworld import WaterWorld
from ple import PLE


# Model parameters

GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 20

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995

# creat DQN solver for function approximation

class DQNSolver:

    def __init__(self, observation_space, action_space):
        self.exploration_rate = EXPLORATION_MAX

        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = Sequential()
        self.model.add(Dense(24, input_shape=(observation_space,), activation="relu"))
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

    
# funciton to convert game state to np array

def process_state(state):
    location_vec = np.array([state['player_x'],
                          state['player_y'],
                          state['player_velocity_x'],
                          state['player_velocity_y']])
    
    range_vec = np.append(state['creep_dist']['BAD'], state['creep_dist']['GOOD'])
    
    
    
    bearing_vec = np.array()
    
    state_vec = np.append(location_vec, range_vec)
    
    return state_vec
    

    
    
def waterworld():
    
    # Set up WaterWorld Environment
    game = WaterWorld(width=500, height=500, num_creeps=1)
    p = PLE(game, display_screen=True, state_preprocessor=process_state)
    
    ''' update score logger for waterworld '''
    #score_logger = ScoreLogger(ENV_NAME)
    
    observation_space = p.state_dim[0]
 
    action_space = len(p.getActionSet())
    
    dqn_solver = DQNSolver(observation_space, action_space)
    
    run = 0
    
    while True:
        
        run += 1
        
        # create new game environment
        p.reset_game()
        p.init()
        state = p.getGameState()
        state = np.reshape(state, [1, observation_space])
        step = 0
        
        while True:
            
            step += 1
            ''' update render to display current run '''
            #env.render()
            
            agent_action = dqn_solver.act(state)
            
            action = p.getActionSet()[agent_action]
            
            reward = p.act(action)
            
            state_next = p.getGameState()
            
            terminal = p.game_over()
            
            
            ''' line below may be specific to cartpole '''
            #reward = reward if not terminal else -reward
            
            state_next = np.reshape(state_next, [1, observation_space])
            
            dqn_solver.remember(state, agent_action, reward, state_next, terminal)
            
            state = state_next
            
            if terminal:
                print ("Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(step))
                ''' update score logger for waterworld '''
                #ScoreLogger.add_score(step, run)
                break
            dqn_solver.experience_replay()


if __name__ == "__main__":
    waterworld()
