####### changes made for beginners who want to run in terminal and view cartpole process #######
####### to run, when in terminal type python cartpole-random-beg.py                     #######

import gym
import numpy as np
import matplotlib.pyplot as plt
import time

def run_episode(env, parameters):
    observation = env.reset()
    totalreward = 0
    for t in range(200):
        action = 0 if np.matmul(parameters, observation) < 0 else 1
        observation, reward, done, info = env.step(action)
        totalreward += reward
        env.render()
        time.sleep(.1)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
    return totalreward

def train(submit):
    env = gym.make('CartPole-v0')
    if submit:
        env.monitor.start('cartpole-experiments/', force = True)

    counter = 0
    bestparams = None
    bestreward = 0
    for _ in range(10000):
        counter += 1
        parameters = np.random.rand(4) * 2 - 1
        reward = run_episode(env, parameters)
        if reward > bestreward:
            bestreward = reward
            bestparams = parameters
            if reward == 200:
                print("you've reached 200 reward points!")
                env.close()
                break

    if submit:
        for _ in range(100):
            run_episode(env, bestparams)
        env.monitor.close()

    return counter



if __name__ == "__main__":
    train(False)
