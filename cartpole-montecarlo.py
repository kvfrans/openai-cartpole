import math
import random
import gym
import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out

def getCompatibleFeatures(observation, action, parameters):
    policy = -1.0 * softmax(np.matmul(parameters,observation))
    policy[action] += 1.0
    observation2 = np.repeat(observation.reshape((4,1)), 2, axis=1)
    return np.dot(observation2, np.diag(policy)).T # This is probably a slow way to do it

update_step = 0.01

def run_episode(env, parameters):
    global update_step

    observation = env.reset()
    totalreward = 0
    transitions = []

    # run episode
    for _ in xrange(200):
        # env.render()

        # calculate policy
        probs = softmax(np.matmul(parameters,observation))

        # take a step
        action = 0 if random.uniform(0,1) < probs[0] else 1
        old_observation = observation
        observation, reward, done, info = env.step(action)
        transitions.append((old_observation, action, reward))
        totalreward += reward

        if done:
            break

    change = np.zeros((2,4))
    # iterate through all transitions
    for index, trans in enumerate(transitions):
        obs, action, reward = trans

        # calculate the reward after taking this action
        future_reward = 0
        future_transitions = len(transitions) - index
        for index2 in xrange(future_transitions):
            future_reward += transitions[(index2) + index][2]

        # print totals

        # update policy
        traces = getCompatibleFeatures(observation,action,parameters)
        change += traces * future_reward
    print change
    change = change / len(transitions)
    parameters += change*update_step


    return parameters, totalreward

env = gym.make('CartPole-v0')
parameters = np.random.rand(2,4) * 2 - 1
traces = np.zeros((4,2))
# env.monitor.start('cartpole-hill/', force=True)
for _ in xrange(100000):
    parameters, reward = run_episode(env, parameters)
    print reward
    # print parameters
    if reward == 200:
        break

for _ in xrange(100):
    run_episode(env,parameters)
# env.monitor.close()
