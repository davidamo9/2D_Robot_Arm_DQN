# 2D_Robot_Arm_DQN
NUS ME5406 Final Project

## testing code
activate python env: source .env/bin/activate
```
#!/usr/bin/env python3

import gym
import arm_v0
import time

env = gym.make('Arm_2D')

state = env.reset()
env.render(mode='human')

while True:
    
    action = env.action_space.sample()
    
    state, reward, done, info = env.step(action)
    print('state = {0}; reward = {1}'.format(state, reward))
    
    if done:
        print('done')
        break
    time.sleep(0.1)

env.close()
```
