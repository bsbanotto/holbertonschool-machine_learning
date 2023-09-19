#!/usr/bin/env python3
"""
This module performs SARSA(λ)
"""
import gym
import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Args:
        env: the openAI environment instance
        Q: numpy.ndarray of shape (s,a) containing the Q table
        lambtha: eligibility trace factor
        episodes: total number of episodes to train over
        max_steps: maximum number of steps per episode
        alpha: learning rate
        gamma: discount rate
        epsilon: initial threshold for epsilon greedy
        min_epsilon: minimum value that epsilon should decay to
        epsilon_decay: decay rate for updating epsilon between episodes

    Returns:
        Q: the updated Q table
    """
    eligibility_trace = np.zeros_like(Q)
    for episode in range(episodes):
        state = env.reset()
        episode_done = False

        action = epsilon_greedy(Q, state, epsilon)

        epsilon = max(min_epsilon, epsilon - epsilon_decay)

        for step in range(max_steps):
            n_s, reward, episode_done, _ = env.step(action)
            n_a = epsilon_greedy(Q, n_s, epsilon)

            # Calculate TD error
            td_error = reward + gamma * Q[n_s][n_a] - Q[state][action]

            # Update eligibility trace
            eligibility_trace *= lambtha * gamma
            eligibility_trace[state][action] = 1.0

            # Update Q values
            Q += alpha * td_error * eligibility_trace

            state = n_s
            action = n_a

            if episode_done:
                break

    return Q


def epsilon_greedy(Q, state, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        return np.random.choice(len(Q[state]))
    else:
        return np.argmax(Q[state])


if __name__ == "__main__":
    np.random.seed(0)
    env = gym.make('FrozenLake8x8-v0')
    Q = np.random.uniform(size=(64, 4))
    np.set_printoptions(precision=4)
    print(sarsa_lambtha(env, Q, 0.9))
