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
from keras.layers import Dropout
from keras.optimizers import Adam

# importing cartpole environment
import gym


''' Deep Q-learning for Cartpole '''

class DQN_agent:
    
    def __init__(self, state_space, action_space, memory_max = 1000000,
                 discount = 0.95, epsilon = 1.0, epsilon_min = 0.01,
                 learning_rate = 0.001, epsilon_decay = 0.995):
        
        self.state_space = state_space
        self.action_space = action_space
        self.memory = deque(maxlen = memory_max)
        self.gamma = discount
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.q_network = self._build_model()
        

    
        
    def _build_model(self):
        
        ''' neural network for Q-value function '''
        
        q_network = Sequential()
        q_network.add(Dense(20, input_dim = self.state_space,
                            activation = 'tanh'))
        q_network.add(Dropout(0.5))
        q_network.add(Dense(20, input_dim = self.state_space,
                            activation = 'tanh'))
        q_network.add(Dense(self.action_space, activation = 'linear'))
        q_network.compile(optimizer=Adam,
                          loss='mse')
        return q_network
    
        
    def memorize(self, state, action, reward, next_state, done):
        
        ''' add recent experience to memory to train q_agent'''
        
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state, actions):
        
        ''' select action using epsilon-greedy method '''
        
        if np.random.rand() < self.epsilon:
            
            action = random.randrange(self.action_space)
            
        else:
            
            action_q = self.q_network.predict(state)
        
            action = np.argmax(action_q[0])
        
        return action
    
    def replay(self, batch_size = 32, epochs = 1):
        
        batch = random.sample(self.memory, batch_size)
        
        batch_states = np.empty((0,self.state_space))
        
        batch_qs = np.empty((0, self.action_space))
        
        for state, action, reward, next_state, done, in batch:
            
            target = reward
            
            if not done:
                
                target = (reward + self.gamma * 
                          np.amax(self.target_network.predict(next_state)[0]))
                
            target_f = self.target_network.predict(state)
            
            target_f[0][action] = target
            
            batch_states = np.append(batch_states, state, axis = 0)
            
            batch_qs = np.append(batch_qs, target_f, axis = 0)
            
        self.q_network.fit(batch_states, batch_qs, epochs = epochs, verbose = 0)
                
        if self.epsilon > self.epsilon_min:
            
            self.epsilon *= self.epsilon_decay
    
                    
    def load_agent(self, path):
        
        self.q_network.load_weights(path)
        
    def save_agent(self, path):
        
        self.q_network.save_weights(path)
        
    
        
        

def cartpole_dqn(process_state, display = False, max_iterations = 1000):
    
    # Set up WaterWorld Environment
 
    env = gym.make("CartPole-v1")
    
    state_space = env.observation_space.shape[0]
 
    action_space = env.action_space.n
    
    agent = DQN_agent(state_space, action_space, memory_max = 1000000,
                 discount = 0.95, epsilon = 1, epsilon_min = 0.01,
                 learning_rate = 0.001,  epsilon_decay = 0.995)
    
    # setup collection items
    
    iteration = 0
    iteration_results = []
    exploration_rates = []
    iterations = []
    
    history = deque(np.repeat(-np.inf, max_iterations), maxlen = max_iterations)
    
    # create new game environment
    

    while iteration <= max_iterations:
        
        state = np.reshape(env.reset(), [1, state_space])
        
        step = 0
        
        iteration_rewards = []
        
        while step <= 500:
            
            step +=1
            
            # getting action: getting Q-value from agent, translate into action

            action = agent.act(state, actions = env.actions)
            
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
        
# run from terminal
if __name__ == "__main__":
    cartpole_dqn()
    
