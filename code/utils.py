#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 10:17:42 2020

@author: colbywilkinson
"""

# plotting packages
import pandas as pd
import matplotlib.pyplot as plt

# for state processor
from math import atan2, sqrt
import numpy as np

# for agent evaluation
from ple.games.puckworld import PuckWorld
from ple import PLE
from puck_dqn import DQN_agent



def state_process_1(state): 
    
    '''
    converts PuckWorld state into vector reflecting agent position, bearing 
    and range to target and bad puck
    '''
    
    agent_loc = np.array([state['player_x'], state['player_y']])
    
    good_range = sqrt((state['player_x'] - state['good_creep_x'])**2 + (state['player_y'] - state['good_creep_y'])**2)
    
    #bad_range = sqrt((state['player_x'] - state['bad_creep_x'])**2 + (state['player_y'] - state['bad_creep_y'])**2)
    
    #range_vec = np.array(good_range) // 1
    
    good_bearing = -atan2((state['good_creep_x'] - state['player_x']), state['good_creep_y']) - (state['player_y'])
    
    #bad_bearing = -atan2((state['good_creep_x'] - state['player_x']), (state['good_creep_y']) - (state['player_y']))
    
    bearing_vec = np.array([good_bearing, good_range]).round(decimals = 0)
    
    state_vec = np.concatenate([bearing_vec, agent_loc])
    
    return state_vec

def state_process_2(state, num_decimals = 1):
    
    '''
    converts PuckWorld state into vector of positons (agent, target, enemy)
    and velocity of agent and enemy
    '''
    
    return np.array(list(state.values())).round(num_decimals)

def evaluate_agent(agent_path, width, height, process_state):
    
    game = PuckWorld(width=width, height=height)
    p = PLE(game, display_screen=True, state_preprocessor=process_state)
    state_space = p.state_dim[0]
    action_space = len(p.getActionSet())
    
    agent = DQN_agent(state_space=state_space, action_space=action_space)
    agent.load_agent(path = agent_path)
    
    p.init()
    steps = 10000
    reward = 0.0
    rewards = []
    for step in range(steps):
        
        state = np.reshape(p.getGameState(), [1, state_space])
        agent_action = agent.act(state)
        action = p.getActionSet()[agent_action]
        #action = rng.choice([119, 97, 100, 115])
        reward = p.act(action)
        rewards.append(reward)
        
def plot_rewards(rewards_path, save = False, save_path = "./rewards.png"):
    
    puck_results = pd.read_csv(rewards_path)
    puck_results['25-MA'] = puck_results['rewards'].rolling(window=25).mean()
    puck_results['1000-MA'] = puck_results['rewards'].rolling(window=1000).mean()
    
    plt.figure(figsize=(12,5))
    plt.xlabel('ITERATION')
    plt.ylabel('REWARD')
    plt.plot(puck_results['rewards'], label = 'reward')
    plt.plot(puck_results['25-MA'], label = '25 Iteration Moving-Average')
    plt.plot(puck_results['1000-MA'], label = '1,000 Iteration Moving-Average')
    plt.legend(loc="lower right")
    
    if save:
        
        plt.savefig(save_path)
        
    
