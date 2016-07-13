import gym
import numpy as np
import matplotlib.pyplot as plt

def run_episode(env, parameters):
    observation = env.reset()
    totalreward = 0
    counter = 0
    for _ in xrange(200):
        # env.render()
        action = 0 if np.matmul(parameters,observation) < 0 else 1
        observation, reward, done, info = env.step(action)
        totalreward += reward
        counter += 1
        if done:
            break
    return totalreward

def train(submit):
    env = gym.make('CartPole-v0')
    if submit:
        env.monitor.start('cartpole-hill/', force=True)

    episodes_per_update = 5
    noise_scaling = 0.1
    parameters = np.random.rand(4) * 2 - 1
    bestreward = 0
    counter = 0

    for _ in xrange(2000):
        counter += 1
        newparams = parameters + (np.random.rand(4) * 2 - 1)*noise_scaling
        # print newparams
        # reward = 0
        # for _ in xrange(episodes_per_update):
        #     run = run_episode(env,newparams)
        #     reward += run
        reward = run_episode(env,newparams)
        # print "reward %d best %d" % (reward, bestreward)
        if reward > bestreward:
            # print "update"
            bestreward = reward
            parameters = newparams
            if reward == 200:
                break

    if submit:
        for _ in xrange(100):
            run_episode(env,parameters)
        env.monitor.close()
    return counter


r = train(submit=False)
print r
