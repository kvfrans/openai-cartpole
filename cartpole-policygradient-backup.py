import math
import random
import gym
import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out

update_step = 0.1
value_update_step = 0.1
discount = 0.9
totals = np.zeros(4)
occurences = 0

def run_episode(env, parameters, value):
    global totals
    global discount
    global update_step
    global value_update_step
    global occurences

    observation = env.reset()
    totalreward = 0
    for _ in xrange(200):
        # env.render()
        # print "running iteration"
        # calculate policy
        probs = softmax(np.matmul(parameters,observation))

        # take a step
        action = 0 if random.uniform(0,1) < probs[0] else 1
        old_observation = observation
        observation, reward, done, info = env.step(action)
        totalreward += reward


        # update average observation
        totals += observation
        occurences += 1

        # get observations and error
        stateval = np.matmul(observation,value)
        oldval = np.matmul(old_observation,value)
        tderror = reward + discount*stateval - oldval

        # update value/critic
        value = value + value_update_step*tderror

        # update policy
        score = old_observation - (totals / occurences)
        parameters = parameters + update_step*score*tderror

        print parameters

        if done:
            break
    return parameters, totalreward, value

env = gym.make('CartPole-v0')
parameters = np.random.rand(2,4) * 2 - 1
# env.monitor.start('cartpole-hill/', force=True)
value = np.random.rand(4) * 2 - 1
for _ in xrange(100000):
    parameters, reward, value = run_episode(env, parameters, value)
    # print reward
    if reward == 200:
        break

for _ in xrange(100):
    run_episode(env,parameters, value)
# env.monitor.close()
