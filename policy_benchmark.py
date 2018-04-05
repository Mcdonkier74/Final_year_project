import gym
from gym import wrappers
from environment import Environment
from environment import Agent
from policy_gradient2 import Policy_Gradient
import numpy
import matplotlib.pyplot as plt
import tensorflow as tf


def train(agent, env, sess, num_episodes=NUM_EPISODES):
  history = []
  for i in xrange(NUM_EPISODES):
    if i % 100:
      print "Episode {}".format(i + 1)
    cur_state = env.reset()
    episode = []
    for t in xrange(MAX_STEPS):
      action = policy_gradient2.Policy_Gradient.get_action(cur_state, sess)
      next_state, reward, done, info = env.step(action)
      if done:
        reward = FAIL_PENALTY
        episode.append([cur_state, action, next_state, reward, done])
        print("Episode finished after {} timesteps".format(t + 1))
        print agent.get_policy(cur_state, sess)
        history.append(t + 1)
        break

      episode.append([cur_state, action, next_state, 1, done])
      cur_state = next_state
      if t == MAX_STEPS - 1:
        history.append(t + 1)
        print("Episode finished after {} timesteps".format(t + 1))
    if i % TRAIN_EVERY_NUM_EPISODES == 0:
      print 'train at episode {}'.format(i)
      agent.learn(episode, sess, EPOCH_SIZE)
  return agent, history


#agent = Policy_Gradient(lr=LEARNING_RATE,
#                       gamma=DISCOUNT_FACTOR,
#                       state_size=4,
#                       action_size=2,
#                       n_hidden_1=10,
#                       n_hidden_2=10)
#
#env = Environment(s_fname=s_fname, i_idx=i_idx)
#env = wrappers.Monitor(env, '/data_0725_0926.zip', force=True)


with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  agent, history = train(agent, env, sess)


window = 10
avg_reward = [numpy.mean(history[i*window:(i+1)*window]) for i in xrange(int(len(history)/window))]
f_reward = plt.figure(1)
plt.plot(numpy.linspace(0, len(history), len(avg_reward)), avg_reward)
plt.ylabel('Rewards')
plt.xlabel('Episodes')
f_reward.show()
print 'press enter to continue'
raw_input()

