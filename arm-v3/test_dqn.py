import time
import gym
import arm_2D_v3
import numpy as np
from keras import models


env = arm_2D_v3.Arm_2D_v3()
model = models.load_model('test_arm_model-v3-dqn.h5')

success = 0

for i in range(10):
    s = env.reset()
    score = 0
    while True:
        env.render()
        a = np.argmax(model.predict(np.array([s]))[0])
        s, reward, done, _ = env.step(a)
        score += reward
        if done:
            if score > 0:
                success += 1
            print('score:', score)
            break

print('Success rate: ', success)

env.close()
