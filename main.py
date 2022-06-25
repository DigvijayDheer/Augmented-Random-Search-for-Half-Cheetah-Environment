# Augmented Random Search for Half-Cheetah Environment

# Importing the libraries 
import os
import numpy as np
import gym
from gym import wrappers
import pybullet_envs

# Setting the Hyper Parameters

class Hyper_Parameters():
    
    def __init__(self):
        self.number_of_steps = 1000
        self.episode_length = 1000
        self.learning_rate = 0.02
        self.number_of_directions = 16
        self.number_of_best_directions = 16
        assert self.number_of_best_directions <= self.number_of_directions
        self.noise = 0.03
        self.seed = 1
        self.environment_name = 'HalfCheetahBulletEnv-v0'
        
# Normalizing the states

class Normalizer():
    
    def __init__(self, number_of_inputs):
        self.number_of_states = np.zeros(number_of_inputs)
        self.mean = np.zeros(number_of_inputs)
        self.numerator = np.zeros(number_of_inputs)
        self.variance = np.zeros(number_of_inputs)
        
    def observe(self, new_state):
        self.number_of_states += 1.
        last_mean = self.mean.copy()
        self.mean += (new_state - self.mean) / self.number_of_states
        self.numerator += (new_state - last_mean) * (new_state - self.mean)
        self.variance = (self.numerator / self.number_of_states).clip(min = 1e-2)
        
    def normalize(self, inputs):
        observed_mean = self.mean
        observed_standard_deviation = np.sqrt(self.variance)
        return (inputs - observed_mean) / observed_standard_deviation

# Building the AI

class Policy():
    
    def __init__(self, input_size, output_size):
        self.theta = np.zeros((output_size, input_size))
        
    def evaluate(self, input, delta = None, direction = None):
        if direction is None:
            return self.theta.dot(input)
        elif direction == "positive":
            return (self.theta + hp.noise*delta).dot(input)
        else:
            return (self.theta - hp.noise*delta).dot(input)

    def sample_deltas(self):
        return [np.random.randn(*self.theta.shape) for i in range(hp.number_of_directions)]
        
    def update(self, rollouts, sigma_reward):
        step = np.zeros(self.theta.shape)
        for pos_reward, neg_reward, perturbation in rollouts:
            step += (pos_reward - neg_reward) * perturbation
        self.theta += hp.learning_rate / (hp.number_of_best_directions * sigma_reward) * step

# Exploring the policy on one specific direction and over one episode

def explore(env, normalizer, policy, direction = None, delta = None):
    state = env.reset()
    done = False
    number_of_action_plays = 0.
    sum_of_rewards = 0
    while not done and number_of_action_plays < hp.episode_length:
        normalizer.observe(state)
        state = normalizer.normalize(state)
        action = policy.evaluate(state, delta, direction)
        state, reward, done, _ = env.step(action)
        reward = max(min(reward, 1), -1)
        sum_of_rewards += reward
        number_of_action_plays += 1
    return sum_of_rewards
    
# Training the AI

def train(env, policy, normalizer, hp):

    for step in range(hp.number_of_steps):
        
        #Initializing the perturbation deltas and positive/negative rewards
        deltas = policy.sample_deltas()
        positive_rewards = [0] * hp.number_of_directions
        negative_rewards = [0] * hp.number_of_directions
    
        # Getting the positive rewards in the positive directions
        for k in range(hp.number_of_directions):
            positive_rewards[k] = explore(env, normalizer, policy, direction = "positive", delta = deltas[k])
            
        # Getting the negative rewards in the negative/opposite directions
        for k in range(hp.number_of_directions):
            negative_rewards[k] = explore(env, normalizer, policy, direction = "negative", delta = deltas[k])
            
        # Gathering all the positive/negative rewards to compute the standard deviation of these rewards
        all_rewards = np.array(positive_rewards + negative_rewards)
        sigma_reward = all_rewards.std()
        
        # Sorting the rollouts by the max(pos_reward, neg_reward) and selecting the best directions 
        scores = {k : max(pos_reward, neg_reward) for k,(pos_reward, neg_reward) in enumerate(zip(positive_rewards, negative_rewards))}
        order = sorted(scores.keys(), key = lambda x : scores[x])[:hp.number_of_best_directions]
        rollouts = [(positive_rewards[k], negative_rewards[k], deltas[k]) for k in order]
    
        # Updating the policy
        policy.update(rollouts, sigma_reward)
        
        # Printing the final reward of the policy after the update
        reward_evaluation = explore(env, normalizer, policy)
        print('Step: ', step, 'Reward: ', reward_evaluation)
    
# Runnig the code

def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path
work_dir = mkdir('experiment', 'augmented_random_search')
monitor_dir = mkdir(work_dir, 'monitor')
    
hp = Hyper_Parameters()
np.random.seed(hp.seed)
env = gym.make(hp.environment_name)
env = wrappers.Monitor(env, monitor_dir, force = True)
number_of_inputs = env.observation_space.shape[0]
number_of_outputs = env.action_space.shape[0]
policy = Policy(number_of_inputs, number_of_outputs)
normalizer = Normalizer(number_of_inputs)
train(env, policy, normalizer, hp)
