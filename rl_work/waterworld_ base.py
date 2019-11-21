#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 10:30:58 2019

@author: colbywilkinson
"""

# importing waterworld environment
from ple.games.waterworld import WaterWorld
from ple import PLE


# import standard utlities
import numpy as np
import random as rng



# funciton to convert game state to np array

def process_state(state):
    location_vec = np.array([state['player_x'],
                          state['player_y'],
                          state['player_velocity_x'],
                          state['player_velocity_y']])
    
    distances_vec = np.append(state['creep_dist']['BAD'], state['creep_dist']['GOOD'])
    
    state_vec = np.append(location_vec, distances_vec)
    
    return state_vec
    
 

    
game = WaterWorld(width=500, height=500, num_creeps=5)
p = PLE(game, display_screen=True, state_preprocessor=process_state)
#agent = myAgentHere(input_shape=p.getGameStateDims(), allowed_actions=p.getActionSet())

p.init()
nb_frames = 1000
reward = 0.0
rewards = []
for i in range(nb_frames):
    if p.game_over():
        p.reset_game()
    state = p.getGameState()
    #action = agent.pickAction(reward, state)
    action = rng.choice([119, 97, 100, 115])
    reward = p.act(action)
    print(state)
    print(reward)


















