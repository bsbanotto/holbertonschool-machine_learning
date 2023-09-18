#!/usr/bin/env python3
"""
This script will take our trained agent weights and play a game
"""
import tensorflow as tf

import gym
import numpy as np

from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import GreedyQPolicy

from keras.models import Sequential
from keras.layers import Dense, Flatten, Reshape, Conv2D, MaxPooling2D
from keras.optimizers import Adam

tf.logging.set_verbosity(tf.logging.ERROR)

# Create variables and values for our play environment
ATARI_ENV = 'Breakout-v0'
play_env = gym.make(ATARI_ENV)
np.random.seed(42)
play_env.seed(42)
nb_play_actions = play_env.action_space.n
window_size = 3

# Build our play model (same as train)
play_model = Sequential()

play_model.add(Reshape((210, 160 * window_size, 3),
                       input_shape=(window_size, 210, 160, 3)))
play_model.add(Conv2D(filters=(32), kernel_size=(3, 3), activation='relu'))
play_model.add(MaxPooling2D((2, 2)))
play_model.add(Conv2D(filters=(64), kernel_size=(3, 3), activation='relu'))
play_model.add(MaxPooling2D((2, 2)))
play_model.add(MaxPooling2D((2, 2)))
play_model.add(Flatten())
play_model.add(Dense(128, activation='relu'))
play_model.add(Dense(256, activation='relu'))
play_model.add(Dense(nb_play_actions, activation='linear'))

# Build our play agent
# Note policy here is GreedyQPolicy vs EpsGreedyQPolicy in Training agent
memory = SequentialMemory(limit=1000, window_length=3)
policy = GreedyQPolicy()
play_dqn = DQNAgent(model=play_model, nb_actions=nb_play_actions,
                    memory=memory, nb_steps_warmup=100,
                    target_model_update=10, policy=policy)
play_dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# Load weights from our trained model
play_dqn.load_weights('./Breakout-v4_weights.h5')

# Play the game using our player agent
play_dqn.test(play_env, nb_episodes=5, visualize=True,
              nb_max_episode_steps=500)
