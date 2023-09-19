#!/usr/bin/env python3
"""
Module to perform the TD(λ) algorithm
"""
import numpy as np
import gym


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    """
    Args:
        env: the openAI environment instance
        V: numpy.ndarray of shape (s,) containing the value estimate
        policy: function that takes in state and returns  next action to take
        lambtha: eligibility trace factor
        episodes: total number of episodes to train over
        max_steps: maximum number of steps per episode
        alpha: learning rate
        gamma: discount rate
    Returns:
        V: the updated value estimate
    """
    eligibility_trace = np.zeros_like(V)
    for episode in range(episodes):
        state = env.reset()

        for step in range(max_steps):
            action = policy(state)
            next_state, reward, done, _ = env.step(action)

            # TD error
            delta = reward + (gamma * V[next_state] - V[state])

            # Update eligibility trace
            eligibility_trace *= (gamma * lambtha)
            eligibility_trace[state] += 1

            # Update value estimate
            V += delta * alpha * eligibility_trace

            state = next_state

            if done or step > max_steps:
                break

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
    print(td_lambtha(env, V, policy, 0.9).reshape((8, 8)))
