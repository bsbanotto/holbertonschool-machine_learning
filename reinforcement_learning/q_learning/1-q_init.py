#!/usr/bin/env python3
"""
Module to initialize a Q-table filled with zeros
"""
import numpy as np


def q_init(env):
    """
    Initialize the Q-table
    Args:
        env: The FrozenLakeEnv instance
    Returns:
        The Q-table as a numpy.ndarray of zeros
    """
    nb_states = env.observation_space.n
    nb_actions = env.action_space.n
    return np.zeros((nb_states, nb_actions))


if __name__ == "__main__":
    load_frozen_lake = __import__('0-load_env').load_frozen_lake

    env = load_frozen_lake()
    Q = q_init(env)
    print(Q.shape)
    env = load_frozen_lake(is_slippery=True)
    Q = q_init(env)
    print(Q.shape)
    desc = [['S', 'F', 'F'], ['F', 'H', 'H'], ['F', 'F', 'G']]
    env = load_frozen_lake(desc=desc)
    Q = q_init(env)
    print(Q.shape)
    env = load_frozen_lake(map_name='4x4')
    Q = q_init(env)
    print(Q.shape)
