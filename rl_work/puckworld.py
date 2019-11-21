#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 16:33:40 2019

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
from ple.games.puckworld import PuckWorld
from ple import PLE


# Model parameters

GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 10

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
    
    good_range = math.sqrt((state['player_x'] - state['good_creep_x'])**2 + (state['player_y'] - state['good_creep_y'])**2)
    bad_range = math.sqrt((state['player_x'] - state['bad_creep_x'])**2 + (state['player_y'] - state['bad_creep_y'])**2)
    
    
    range_vec = np.append(good_range, bad_range)
    
    good_bearing = math.atan2((state['player_x'] - state['good_creep_x']), (state['player_y'] - state['good_creep_y']))
    bad_bearing = math.atan2((state['player_x'] - state['bad_creep_x']), (state['player_y'] - state['bad_creep_y']))
    
    bearing_vec = np.append(good_bearing, bad_bearing)
    
    state_vec = np.concatenate([location_vec, range_vec, bearing_vec])
    
    return state_vec
    

    
    
def puckworld():
    
    # Set up WaterWorld Environment
    game = PuckWorld(width=500, height=500)
    p = PLE(game, display_screen=True, state_preprocessor=process_state)
    
    ''' update score logger for PuckWorld '''
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
            
            state_next = np.reshape(state_next, [1, observation_space])
            
            dqn_solver.remember(state, agent_action, reward, state_next, terminal)
            
            state = state_next
            
            print(reward)
            print(step)
            
            if step > 200:
                print ("Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(reward))
                ''' update score logger for waterworld '''
                #ScoreLogger.add_score(step, run)
                break
            dqn_solver.experience_replay()


if __name__ == "__main__":
    puckworld()
