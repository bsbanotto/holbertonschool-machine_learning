#!/usr/bin/env python3
"""
This script will train an agent to play Atari Breakout (poorly)
"""
import tensorflow as tf

import gym
import numpy as np

from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy

from keras.models import Sequential
from keras.layers import Dense, Flatten, Reshape, Conv2D, MaxPooling2D
from keras.optimizers import Adam

# This section defines some variable values
ATARI_ENV = 'Breakout-v0'
train_env = gym.make(ATARI_ENV)
np.random.seed(42)
train_env.seed(42)

nb_steps_fit = 1750000
nb_steps_warmup = nb_steps_fit / 3
update = 10

nb_train_actions = train_env.action_space.n
window_size = 3

# This builds our training model
train_model = Sequential()
train_model.add(Reshape((210, 160 * window_size, 3),
                        input_shape=(window_size, 210, 160, 3)))
train_model.add(Conv2D(filters=(32), kernel_size=(3, 3), activation='relu'))
train_model.add(MaxPooling2D((2, 2)))
train_model.add(Conv2D(filters=(64), kernel_size=(3, 3), activation='relu'))
train_model.add(MaxPooling2D((2, 2)))
train_model.add(MaxPooling2D((2, 2)))
train_model.add(Flatten())
train_model.add(Dense(128, activation='relu'))
train_model.add(Dense(256, activation='relu'))
train_model.add(Dense(nb_train_actions, activation='linear'))

# This builds and compiles our agent
memory = SequentialMemory(limit=1000, window_length=3)
policy = EpsGreedyQPolicy(eps=0.1)
train_dqn = DQNAgent(model=train_model, nb_actions=nb_train_actions,
                     memory=memory, nb_steps_warmup=nb_steps_warmup,
                     target_model_update=update, policy=policy)
train_dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# This trains our agent
train_dqn.fit(train_env, nb_steps=nb_steps_fit, visualize=False, verbose=1)

# Lastly, save our model weights
train_dqn.save_weights('policy.h5', overwrite=True)
