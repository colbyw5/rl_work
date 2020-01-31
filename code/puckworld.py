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
import pandas as pd
import pickle

# model
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

''' DQN for PuckWorld '''

# importing PuckWorld environment
from ple.games.puckworld import PuckWorld
from ple import PLE 

# Model parameters

GAMMA = 0.95
LEARNING_RATE = 0.01

MEMORY_SIZE = 10000
BATCH_SIZE = 100

EXPLORATION_MAX = 0.20
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.99995

# creat DQN solver for function approximation

class DQNSolver:

    def __init__(self, observation_space, action_space):
        
        self.exploration_rate = EXPLORATION_MAX
        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)
        
        self.model = Sequential()
        self.model.add(Dense(6, input_shape=(observation_space,), activation="relu"))
        self.model.add(Dense(6, activation="tanh"))
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
            self.model.fit(state, q_values, verbose=0, epochs = 1)
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

    

    

    
    
def puckworld(process_state, solved_score, solved_runs, display = False, max_runs = 1000):
    
    # Set up WaterWorld Environment
 
    game = PuckWorld(width=500, height=500)
    
    p = PLE(game, display_screen=display, state_preprocessor=process_state)
    
    observation_space = p.state_dim[0]
 
    action_space = len(p.getActionSet())
    
    dqn_solver = DQNSolver(observation_space, action_space)
    
    # setup collection items
    
    run = 0
    run_results = []
    exploration_rates = []
    runs = []
        
    solved = False
    
    history = deque(np.repeat(-np.inf, solved_runs), maxlen = solved_runs)

    while not solved:
        
        # create new game environment
        p.reset_game()
        p.init()
        state = np.reshape(p.getGameState(), [1, observation_space])
        step = 0
        
        # set up run results collection
        
        run_proceed = True
        
        run_rewards = []
        
        while run_proceed:
            
            
            # getting action: getting Q-value from agent, translate into action

            agent_action = dqn_solver.act(state)
            
            action = p.getActionSet()[dqn_solver.act(state)]
            
            # agent acts, reward is realized
            
            reward = p.act(action)
            
            run_rewards.append(reward)
            
            # getting next state based on action
            
            state_next = p.getGameState()
            
            # terminating current run at 500 steps
            
            terminal = step == 500
            
            # reshaping new state for memory, adding s, a, r, s', terminal to memory
            
            state_next = np.reshape(state_next, [1, observation_space])
            
            dqn_solver.remember(state, agent_action, reward, state_next, terminal)
            
            # updating current state
            
            state = state_next
            
            step += 1
            
            #current rule for run: 500 steps (green puck location updates)
            
            if step == 500:
                
                print ("Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(sum(run_rewards)))
                
                # save run, step, reward, exploration, 
                
                run += 1
                
                run_results.append(sum(run_rewards))
                
                exploration_rates.append(dqn_solver.exploration_rate)
                
                runs.append(run)
                
                history.append(sum(run_rewards))
                
                # solved: if the past 'solved_runs' runs are all greater than 'solved_score'
                
                solved = not any(run < solved_score for run in history)
                
                run_rewards = []
                
            # Update network every 10 steps
                
            # if np.mod(step, 10) == 0:
                
            dqn_solver.experience_replay()

                # if solved, save results to CSV, save model
            
            if solved or run == max_runs:
                
                # save results to csv
                
                print('SOLVED')
                
                run_proceed = False
                
                pd.DataFrame({'rewards': run_results,
                              'exploration': exploration_rates,
                              'run': runs}).to_csv('results.csv', index = False)
                
                # save model

                with open('solved.model', 'wb') as solved_model:
                    pickle.dump(dqn_solver, solved_model)
                    
                solved = True
                    
                break
                



if __name__ == "__main__":
    puckworld()
