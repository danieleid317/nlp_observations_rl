"""The intentions for this projecyt are to have the state/observations for the
model to be inputted as latent representations of the state via image feature vectors
and as descriptive text of the state of the environment

In this project i aimed at using embedded environment descriptions as opposed to
raw state data. By using embedded text descriptions we can see that the model is able
to use input of various forms.  This could help in more complex projects
because it could help by giving directions or providing higher level context
about the environment.  I ran two seperate tests on this project.  (expecting
similar results beforehand because they both are optimizing against the same loss)
One test was giving the text as environment descriptions and once as sufficient instructions
given a particular state. They performed almost equally well which shows that
the model can optimize for both individually. In future tests i hope to be able to
look into whether the input state can be some type of aggregation of environment
descriptions by a model such as that of show,attend,and tell, And instructions(provided
by a model that can select actions based on state expected next state fed to a model
that can output instructions given action, state and s+1)

"""

import abc
import tensorflow as tf
import numpy as np
import random

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

tf.compat.v1.enable_v2_behavior()
print('not loaded yet')
import tensorflow_hub as hub
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
print('loaded')
"""pretraining the text embedding model seemed like more work than it was
worth for a task with such a low dimensional state space. so i opted to use
texts that were semantically dissimilar for training instead of retraining
  a text embedding model"""

embedded = embed(['my house is so full of rooms','right turn at the intersection',
'tuna live in the ocean','outer space contains jupiter and pluto and the sun','done'])

"""As the first version, I found it simplest to give the latent semantic representations based
on hard coded logic. In future versions this could be done simply by having the latent vectors
output by an image annotation model (in the case of an autonomus navgation bot) in order to have
situational awareness.  The annotations would ideally provide enough context so the model could learn
based on the state of the environment. Unlike in this case where logic maps directly to directions(in set_state)
expecting the model only to learn what actions to take given a specific input vector z.
In theory the two should be able to learn identically, but differ only in practice.
 A model that takes directions to 'stop'(or go... or any other action) and then has to learn
  to 'stop', is virtually identical to a model that is given a state representation and has to
  learn that given a particular state it should result in a 'stop' action.
"""

#similar env to other rl projects with added use of text embeddings for obsv
class PathFinder(py_environment.PyEnvironment):
  def __init__(self):
    self._action_spec = array_spec.BoundedArraySpec(
        shape=(), dtype=np.int32, minimum=0, maximum=3, name='action')
    self._observation_spec = array_spec.BoundedArraySpec(
        shape=(1,8,), dtype=np.int32, minimum=0, name='observation')
    self._state = 0
    self._episode_ended = False
    self.step_count = 0
    self.grid = 0

  def init_grid(self):
    #start and target states are randomized for each episode making this env stochastic
    xs = random.sample(range(0,6),2)
    ys = random.sample(range(0,6),2)
    observation = [xs[0],ys[0],xs[1],ys[1]]
    return observation

  def set_state(self):
    #conditionals to set state as env description
    state = embedded[4][:4]#[0,0,0,0]
    #navigate x instructions
    if self.grid[0] < self.grid[2]:
      state = embedded[0][:4]#[0,0,0,1]
    elif self.grid[0] > self.grid[2]:
      state = embedded[1][:4]#[0,0,1,0]
    #navigate y instructions
    if self.grid[1] > self.grid[3]:
      state = embedded[2][:4]#[0,1,0,0]
    elif self.grid[1] < self.grid[3]:
      state = embedded[3][:4]#[1,0,0,0]
    x = tf.reshape(tf.concat([state,self.grid],0),[-1])
    return x

  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec

  def _reset(self):
    self._episode_ended = False
    self.grid = self.init_grid()
    self._state = self.set_state()
    self.step_count = 0
    return ts.restart(np.array([self._state], dtype=np.int32))

  def _step(self, action):
    self.step_count+=1
    if self._episode_ended:
      # The last action ended the episode. Ignore the current action and start
      # a new episode.
      return self.reset()

    if action == 0:
      if self.grid[1] == 0:
        pass
      else:
        self.grid[1] -= 1
        self._state = self.set_state()
    elif action == 1:#east
      if self.grid[0] == 5:
        pass
      else:
        self.grid[0] += 1
        self._state = self.set_state()
    elif action == 2:#south
      if self.grid[1] == 5:
        pass
      else:
        self.grid[1] += 1
        self._state = self.set_state()
    elif action == 3:#west
      if self.grid[0] == 0:
        pass
      else:
        self.grid[0] -= 1
        self._state = self.set_state()

    # termination condition for maxsteps
    if self.step_count >= 45 :
      self._episode_ended = True
      reward = -500
      return ts.termination(np.array([self._state], dtype=np.int32), reward)
    #if found target
    if self.grid[1] == self.grid[3] and self.grid[0] == self.grid[2]:
      reward = 1000
      self._episode_ended = True
      return ts.termination(np.array([self._state], dtype=np.int32), reward)

    #penalize for timestep
    else:
      reward = -10
      return ts.transition(np.array([self._state], dtype=np.int32), reward=reward, discount=0.9997)

env = PathFinder()
utils.validate_py_environment(env,episodes=2)

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network, q_rnn_network
import tf_agents
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
tf.compat.v1.enable_v2_behavior()

from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.networks import actor_distribution_network


num_iterations = 5000 # @param {type:"integer"}

collect_episodes_per_iteration = 2  # @param {type:"integer"}
replay_buffer_max_length = 2000  # @param {type:"integer"}

learning_rate = 1e-3  # @param {type:"number"}
log_interval = 25  # @param {type:"integer"}
fc_layer_params = (8,8,)
num_eval_episodes = 5  # @param {type:"integer"}
eval_interval = 50  # @param {type:"integer"}

train_py_env = PathFinder()
eval_py_env = PathFinder()

eval_tf_env = tf_py_environment.TFPyEnvironment(eval_py_env)
train_tf_env = tf_py_environment.TFPyEnvironment(train_py_env)

actor_net = actor_distribution_network.ActorDistributionNetwork(
    train_tf_env.observation_spec(),
    train_tf_env.action_spec(),
    fc_layer_params=fc_layer_params)
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

train_step_counter = tf.compat.v2.Variable(0)

agent = reinforce_agent.ReinforceAgent(
    train_tf_env.time_step_spec(),
    train_tf_env.action_spec(),
    actor_network=actor_net,
    optimizer=optimizer,
    normalize_returns=True,
    train_step_counter=train_step_counter)
agent.initialize()

eval_policy = agent.policy
collect_policy = agent.collect_policy

def compute_avg_return(environment, policy, num_episodes=5):

  total_return = 0.0
  for _ in range(num_episodes):

    time_step = environment.reset()
    episode_return = 0.0

    while not time_step.is_last():

      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)
      episode_return += time_step.reward
    total_return += episode_return

  avg_return = total_return / num_episodes
  return avg_return.numpy()[0]
print(compute_avg_return(eval_tf_env,agent.policy))

random_policy = random_tf_policy.RandomTFPolicy(train_tf_env.time_step_spec(),
                                                train_tf_env.action_spec())

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=1,
    max_length=replay_buffer_max_length)

def collect_episode(environment, policy, num_episodes):

  episode_counter = 0
  environment.reset()

  while episode_counter < num_episodes:
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)

    # Add trajectory to the replay buffer
    replay_buffer.add_batch(traj)

    if traj.is_boundary():
      episode_counter += 1

print('started training')
# (Optional) Optimize by wrapping some of the code in a graph using TF function.
agent.train = common.function(agent.train)

# Reset the train step
agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(eval_tf_env, agent.policy, num_eval_episodes)
returns = [avg_return]

for _ in range(2500):

  # Collect a few episodes using collect_policy and save to the replay buffer.
  collect_episode(
      train_tf_env, agent.collect_policy, collect_episodes_per_iteration)

  # Use data from the buffer and update the agent's network.
  experience = replay_buffer.gather_all()
  train_loss = agent.train(experience)
  replay_buffer.clear()

  step = agent.train_step_counter.numpy()

  if step % log_interval == 0:
    print('step = {0}: loss = {1}'.format(step, train_loss.loss))

  if step % eval_interval == 0:
    avg_return = compute_avg_return(eval_tf_env, agent.policy, num_eval_episodes)
    print('step = {0}: Average Return = {1}'.format(step, avg_return))
    returns.append(avg_return)

"""
The agent is able to converge on an optimal polcy in every test case. There were
two additional tests run using sparse and dense/shaped rewards.  The only difference
is the amount of experience needed until convergence. """
