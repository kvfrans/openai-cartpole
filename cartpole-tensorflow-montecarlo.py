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
    advantages = tf.placeholder("float",[None,1])
    linear = tf.matmul(state,params)
    probabilities = tf.nn.softmax(linear)
    good_probabilities = tf.reduce_sum(tf.mul(probabilities, actions),reduction_indices=[1])
    eligibility = tf.log(good_probabilities) * advantages
    loss = tf.reduce_sum(eligibility)
    return params, state, actions, advantages, tf.gradients(loss,[params])

def value_gradient():
    values = tf.placeholder("float",[4,1])
    state = tf.placeholder("float",[None,4])
    newvals = tf.placeholder("float",[None,1])
    calculated = tf.matmul(state,values)
    diffs = calculated - newvals
    # regularization = tf.nn.l2_loss(values)*0.001
    loss = tf.nn.l2_loss(diffs)
    return values, state, newvals, tf.gradients(loss,[values]), loss

def run_episode(env, parameters, values, sess, policy):
    observation = env.reset()
    totalreward = 0
    states = []
    actions = []
    advantages = []
    transitions = []
    update_vals = []
    for _ in xrange(200):
        if policy:
            env.render()
        # calculate policy
        probs = softmax(np.matmul(observation,parameters))
        action = 0 if random.uniform(0,1) < probs[0] else 1
        # record the transition
        states.append(observation)
        actionblank = np.zeros(2)
        actionblank[action] = 1
        actions.append(actionblank)
        # take the action in the environment
        old_observation = observation
        observation, reward, done, info = env.step(action)
        transitions.append((old_observation, action, reward))
        totalreward += reward

        if done:
            break
    for index, trans in enumerate(transitions):
        obs, action, reward = trans

        # calculate discounted monte-carlo return
        future_reward = 0
        future_transitions = len(transitions) - index
        decrease = 1
        for index2 in xrange(future_transitions):
            future_reward += transitions[(index2) + index][2] * decrease
            decrease = decrease * 0.9

        # what is the current value of this state
        currentval = np.matmul(obs,values)

        # advantage: how much better was this action than normal
        advantages.append(future_reward - currentval)

        # update the value function towards new return
        update_vals.append(future_reward)

    # print "ep done"

    # update value function
    update_vals_vector = np.expand_dims(update_vals, axis=1)
    values_vector = np.expand_dims(values, axis=1)
    vl_values, vl_state, vl_newvals, vl_grad, vl_loss = value_gradient()
    val_gradient = sess.run(vl_grad, feed_dict={vl_values: values_vector, vl_state: states, vl_newvals:update_vals_vector})[0][:,0]
    values = values - 0.001*val_gradient

    if policy:
        advantages_vector = np.expand_dims(advantages, axis=1)
        pl_params, pl_state, pl_actions, pl_advantages, grad = policy_gradient()
        gradient = sess.run(grad, feed_dict={pl_params: parameters, pl_state: states, pl_actions: actions, pl_advantages: advantages_vector})[0]
        parameters = parameters + gradient*0.01
        print "totalreward %d" % (totalreward)

    return parameters, totalreward, values

env = gym.make('CartPole-v0')
parameters = np.random.rand(4,2) * 2 - 1
values = np.random.rand(4) * 2 - 1
sess = tf.InteractiveSession()
for i in xrange(100000):
    if i < 100:
        parameters, reward, values = run_episode(env, parameters, values, sess, policy=False)
        print i
    else:
        parameters, reward, values = run_episode(env, parameters, values, sess, policy=True)
    if reward == 200:
        print "reward 200"
        break
