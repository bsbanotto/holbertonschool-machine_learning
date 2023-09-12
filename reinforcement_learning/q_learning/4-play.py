#!/usr/bin/env python3
"""
Visualize a trained agent playing an episode
"""
import numpy as np


# Task 4. Play
def play(env, Q, max_steps=100):
    """
    Function that has th trained agent play an episode
    Args:
        env: FrozenLakeEnv instance
        Q: numpy.ndarray shape (state, action) containing the trained Q-table
        max_steps: Maximum number of steps in the episode
    Each state of the board should be displayed via the console
    We should always exploit the Q-table
    Returns the total rewards for the episode
    """
    total_reward = 0
    state = env.reset()
    print(env.render(), end='')
    for _ in range(max_steps):
        action = np.argmax(Q[state])
        state, reward, terminated, _ = env.step(action)
        total_reward += reward
        print(env.render(), end='')
        if terminated:
            break
    return total_reward


if __name__ == "__main__":
    load_frozen_lake = __import__('0-load_env').load_frozen_lake
    q_init = __import__('1-q_init').q_init
    train = __import__('3-q_learning').train

    np.random.seed(0)
    desc = [['S', 'F', 'F'], ['F', 'H', 'H'], ['F', 'F', 'G']]
    env = load_frozen_lake(desc=desc)
    Q = q_init(env)

    Q, total_rewards = train(env, Q)
    print(play(env, Q))
