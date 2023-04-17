import time
import gym
import arm_2D
import numpy as np
from tensorflow.keras import models


env = arm_2D.Arm_2D()
model = models.load_model('test_arm_model-v1-dqn.h5')

for i in range(5):
    s = env.reset()
    score = 0
    while True:
        env.render()
        time.sleep(0.01)
        a = np.argmax(model.predict(np.array([s]))[0])
        s, reward, done, _ = env.step(a)
        score += reward
        if done:
            print('score:', score)
            break
env.close()