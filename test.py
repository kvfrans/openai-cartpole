import gym
import numpy as np

MAX_MODELS   = 10000  # models to try
MAX_EPISODES = 20     # episodes per model
MAX_STEPS    = 200    # steps per episode
OUTPUT_DIR   = '/tmp/cartpole-random-guess'

def run_episode(env, model):
  ''' play through an episode and return the reward '''
  ob = env.reset()
  done = False
  reward = 0.0
  for s in xrange(MAX_STEPS):
    action = int(np.dot(model, ob) > 0)
    ob, r, done, _ = env.step(action)
    reward += r
    if done:
      break
  return reward


if __name__ == '__main__':
  env = gym.make('CartPole-v0')
  best_r, best_m = None, None
  env.monitor.start(OUTPUT_DIR, force=True)

  # try differrent models
  for i in xrange(MAX_MODELS):
    m = np.random.rand(4) * 2 - 1  # uniform random [-1, 1)
    reward = sum(run_episode(env, m) for _ in xrange(MAX_EPISODES))
    print i, reward, m
    if reward > best_r:
      best_r, best_m = reward, m
    if reward == MAX_EPISODES * MAX_STEPS:
      # exit early on a perfect score, we will end up choosing this model
      break

  # evaluate the best model over 100 episodes
  print 'BEST', best_m
  for _ in xrange(100):
    run_episode(env, best_m)

  env.monitor.close()
