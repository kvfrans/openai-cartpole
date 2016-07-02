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
    state = tf.placeholder("float",[4])
    state_reshaped = tf.reshape(state,[1,4])
    actions = tf.placeholder("float",[2])
    actions_reshaped = tf.reshape(actions,[1,2])
    linear = tf.matmul(state_reshaped,params)
    probabilities = tf.nn.softmax(linear)
    good_probabilities = tf.reduce_sum(tf.mul(probabilities, actions_reshaped),reduction_indices=[1])
    eligibility = tf.log(good_probabilities)
    loss = tf.reduce_sum(eligibility)
    return params, state, actions, tf.gradients(loss,[params])

def value_gradient():
    values = tf.placeholder("float",[4])
    values_reshaped = tf.reshape(values,[4,1])
    state = tf.placeholder("float",[4])
    state_reshaped = tf.reshape(state,[1,4])
    newvals = tf.placeholder("float")
    calculated = tf.matmul(state_reshaped,values_reshaped)
    diffs = calculated - newvals
    loss = tf.nn.l2_loss(diffs)
    return values, state, newvals, tf.gradients(loss,[values])

def value(values, observation):
    return np.matmul(values,observation)

def run_episode(env, sess, parameters, values):
    discount = 0.9

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
        # take the action in the environment
        old_observation = observation
        observation, reward, done, info = env.step(action)
        totalreward += reward

        # td error
        tderror = reward + discount*value(values,observation) - value(values,old_observation)
        newval = value(values,old_observation) + 0.1*tderror

        # update policy gradient
        pl_params, pl_state, pl_actions, grad = policy_gradient()
        pol_gradient = sess.run(grad, feed_dict={pl_params: parameters, pl_state: old_observation, pl_actions: actionblank})[0]
        parameters = parameters + 0.01*pol_gradient*tderror

        vl_values, vl_state, vl_newvals, vl_grad = value_gradient()
        val_gradient = sess.run(vl_grad, feed_dict={vl_values: values, vl_state: old_observation, vl_newvals:newval})[0]
        values = values + 0.01*val_gradient

        if done:
            break

    print "totalreward %d" % totalreward

    return parameters, values, totalreward

env = gym.make('CartPole-v0')
parameters = np.random.rand(4,2) * 2 - 1
values = np.random.rand(4) + 0.01
sess = tf.InteractiveSession()
for _ in xrange(100000):
    parameters, values, reward = run_episode(env, sess, parameters, values)
    # print parameters
    # if reward == 200:
    #     break
