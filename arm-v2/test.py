import gym
import arm_2D_v2
import time
import pygame

env = arm_2D_v2.Arm_2D_v2()

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
