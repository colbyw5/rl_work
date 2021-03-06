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

# model
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf

# importing PuckWorld environment
from ple.games.puckworld import PuckWorld
from ple import PLE 

''' Deep Q-learning for PuckWorld '''

class DQN_agent:
    
    def __init__(self, state_space, action_space, memory_max = 10000,
                 discount = 0.95, epsilon = 0.5, epsilon_min = 0.01,
                 learning_rate = 0.01, epsilon_decay = 0.9995):
        
        self.state_space = state_space
        self.action_space = action_space
        self.memory = deque(maxlen = memory_max)
        self.gamma = discount
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.q_network = self._build_model()
        
    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        
        error = y_true - y_pred
        cond  = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))
        
    def _build_model(self):
        
        ''' neural network for Q-value function '''
        
        q_network = Sequential()
        q_network.add(Dense(40, input_dim = self.state_space,
                            activation = 'tanh'))
        q_network.add(Dense(self.action_space, activation = 'linear'))
        q_network.compile(loss = self._huber_loss,
                          optimizer = Adam(lr = self.learning_rate))
        return q_network
        
    def memorize(self, state, action, reward, next_state, done):
        
        ''' add recent experience to memory to train q_agent'''
        
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state, actions):
        
        ''' select action using epsilon-greedy method '''
        
        if np.random.rand() < self.epsilon:
            
            action = random.sample(actions, 1)[0]
            
        else:
            
            action_q = self.q_network.predict(state)
        
            action = actions[np.argmax(action_q[0])]
        
        return action
    
    def replay(self, batch_size = 256, epochs = 1):
        
        batch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done, in batch:
            
            target = reward
            
            if not done:
                
                target = (reward + self.gamma * 
                          np.amax(self.q_network.predict(next_state)[0]))
                
            target_f = self.q_network.predict(state)
            
            target_f[0][action] = target
            
            self.q_network.fit(state, target_f, epochs = epochs, verbose = 0)
                
        if self.epsilon > self.epsilon_min:
            
            self.epsilon *= self.epsilon_decay
    
                    
    def load_agent(self, path):
        
        self.q_network.load_weights(path)
        
    def save_agent(self, path):
        
        self.q_network.save_weights(path)
        
        
        
        

def puckworld_dqn(process_state, display = False, max_iterations = 1000):
    
    # Set up WaterWorld Environment
 
    game = PuckWorld(width=100, height=100)
    
    p = PLE(game, display_screen=display, state_preprocessor = process_state)
    
    state_space = p.state_dim[0]
 
    action_space = len(p.getActionSet())
    
    agent = DQN_agent(state_space, action_space, memory_max = 1000,
                 discount = 0.9, epsilon = 0.2, epsilon_min = 0.01,
                 learning_rate = 0.01,  epsilon_decay = 0.9995)
    
    # setup collection items
    
    iteration = 0
    iteration_results = []
    exploration_rates = []
    iterations = []
    
    history = deque(np.repeat(-np.inf, max_iterations), maxlen = max_iterations)
    
    # create new game environment
    
    p.reset_game()
        
    p.init()

    while iteration <= max_iterations:
        
        state = np.reshape(p.getGameState(), [1, state_space])
        
        step = 0
        
        iteration_rewards = []
        
        while step <= 500:
            
            
            # getting action: getting Q-value from agent, translate into action

            action = agent.act(state, actions = p.getActionSet())
            
            # agent acts, reward is realized
            
            reward = round(p.act(action), 2)
            
            iteration_rewards.append(reward)
            
            # getting next state based on action
            
            state_next = p.getGameState()
            
            # terminating current run at 500 steps
            
            done = step == 500
            
            # reshaping new state for memory, adding s, a, r, s', terminal to memory
            
            state_next = np.reshape(state_next, [1, state_space])
            
            agent_action = p.getActionSet().index(action)
            
            agent.memorize(state, agent_action, reward, state_next, done)
            
            # updating current state
            
            state = state_next
            
            step += 1
            
            #current rule for run: 500 steps (green puck location updates)
            
            if step == 500:
                
                print ("Run: " + str(iteration) + ", exploration: " + str(agent.epsilon) + ", score: " + str(sum(iteration_rewards)))
                
                # save run, step, reward, exploration, 
                
                iteration_results.append(np.sum(iteration_rewards))
                
                exploration_rates.append(agent.epsilon)
                
                iterations.append(iteration)
                
                history.append(np.sum(iteration_rewards))
                
                iteration_rewards = []
                
            # Update network each step when memory length > batch size
            
            if (len(agent.memory) > 500) & (step % 50 == 0):
                
                agent.replay()

        
        # max iterations reached: save results to CSV, save model    
            
        if iteration == max_iterations:
            
            # save results to csv
            
            print('End Training')
            
            pd.DataFrame({'rewards': iteration_results,
                          'exploration': exploration_rates,
                          'run': iterations}).to_csv('./results.csv', index = False)
            
            # save model
            
            agent.save_agent(path = '../models/model.h5')
            
            break
        
        iteration += 1
        
        
 
    
