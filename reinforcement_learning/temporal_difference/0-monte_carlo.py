#!/usr/bin/env python3
"""
This module contains a function that performs the Monte Carlo algorithm
"""
import numpy as np
import gym


def run_episode(env, max_steps, policy):
    """
    Runs an episode of the environment
    Args:
        env: the openAI environment instance
        max_steps: maximum number of steps per episode

    Return:
        episode_results: np.ndarray of integers shape (state, reward)
    """
    state = env.reset()
    episode_results = []

    # Run each episode until we reach max_steps
    for step in range(max_steps):
        action = policy(state)
        next_state, reward, done, _ = env.step(action)
        episode_results.append([state, reward])
        if done:
            break

        state = next_state

    return np.array(episode_results, dtype=int)


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1,
                gamma=0.99):
    """
    Args:
        env: the openAI environment instance
        V: np.ndarray shape (s,) containing the value estimate
        policy: a function that takes in a state and returns the next action
        episodes: total number of episodes to train over
        max_steps: maximum number of steps per episode
        alpha: learning rate
        gamma: discount rate

    Returns:
        V: the updated value estimate
    """
    # Loop through our episodes
    for episode in range(episodes):
        cumulative_reward = 0
        episode_results = run_episode(env, max_steps, policy)
        # Perform Monte Carlo Algorithm from finish to start
        for time in reversed(range(0, len(episode_results))):
            state, reward = episode_results[time]
            cumulative_reward = gamma * cumulative_reward + reward
            if state not in episode_results[:episode, 0]:
                V[state] = V[state] + alpha * (cumulative_reward - V[state])

    return V


if __name__ == "__main__":
    np.random.seed(0)

    env = gym.make('FrozenLake8x8-v0')
    LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3

    def policy(s):
        p = np.random.uniform()
        if p > 0.5:
            if s % 8 != 7 and env.desc[s // 8, s % 8 + 1] != b'H':
                return RIGHT
            elif s // 8 != 7 and env.desc[s // 8 + 1, s % 8] != b'H':
                return DOWN
            elif s // 8 != 0 and env.desc[s // 8 - 1, s % 8] != b'H':
                return UP
            else:
                return LEFT
        else:
            if s // 8 != 7 and env.desc[s // 8 + 1, s % 8] != b'H':
                return DOWN
            elif s % 8 != 7 and env.desc[s // 8, s % 8 + 1] != b'H':
                return RIGHT
            elif s % 8 != 0 and env.desc[s // 8, s % 8 - 1] != b'H':
                return LEFT
            else:
                return UP

    V = np.where(env.desc == b'H', -1, 1).reshape(64).astype('float64')
    np.set_printoptions(precision=4)
    env.seed(0)
    print(monte_carlo(env, V, policy).reshape((8, 8)))
