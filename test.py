#!/usr/bin/env python3

import gym
import arm_v0
import time
import pygame

env = gym.make('Arm_2D')

observation = env.reset()

time.sleep(1)

while True:

    env.render()
    action = env.action_space.sample()
    
    observation, reward, done, info = env.step(action)
    print(observation)
    
    if done:
        print('done')
        break
    time.sleep(0.1)

env.close()
