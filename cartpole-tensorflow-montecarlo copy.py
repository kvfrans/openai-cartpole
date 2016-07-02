import tensorflow as tf
import numpy as np
import random
import gym
import math

def softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out

def policy_gradient():
    params = tf.placeholder("float",[4,2])
    state = tf.placeholder("float",[None,4])
    actions = tf.placeholder("float",[None,2])
    linear = tf.matmul(state,params)
    probabilities = tf.nn.softmax(linear)
    good_probabilities = tf.reduce_sum(tf.mul(probabilities, actions),reduction_indices=[1])
    eligibility = tf.log(good_probabilities)
    loss = tf.reduce_sum(eligibility)
    return params, state, actions, tf.gradients(loss,[params])

total = 0
counts = 0

def run_episode(env, parameters, sess):
    global total
    global counts

    observation = env.reset()
    totalreward = 0
    states = []
    actions = []
    for _ in xrange(200):
        # calculate policy
        probs = softmax(np.matmul(observation,parameters))
        action = 0 if random.uniform(0,1) < probs[0] else 1
        # record the transition
        states.append(observation)
        actionblank = np.zeros(2)
        actionblank[action] = 1
        actions.append(actionblank)
        # take the action in the environment
        observation, reward, done, info = env.step(action)
        totalreward += reward
        if done:
            break

    total += totalreward
    counts += 1

    pl_params, pl_state, pl_actions, grad = policy_gradient()
    gradient = sess.run(grad, feed_dict={pl_params: parameters, pl_state: states, pl_actions: actions})[0]
    baseline = (total / counts)
    parameters = parameters + gradient*0.01* (totalreward - baseline)
    print "totalreward %d baseline %d" % (totalreward, baseline)

    return parameters, totalreward

env = gym.make('CartPole-v0')
parameters = np.random.rand(4,2) * 2 - 1
sess = tf.InteractiveSession()
for _ in xrange(100000):
    parameters, reward = run_episode(env, parameters, sess)
    # print parameters
    if reward == 200:
        break
