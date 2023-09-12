#!/usr/bin/env python3
"""
Module to perform Q-learning
"""
import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Function that performs Q-learning
    Args:
        env: the FrozenLakeEnv instance
        Q: np.ndarray shape(state, action) containing the Q-table
        episodes: the total number of episodes to train over
        max_steps: maximum number of steps per episode
        alpha: learning rate
        gamma: discount rate
        epsilon: the initial threshold for epsilon greedy
        min_epsilon: the minimum value that epsilon should decay to
        epsilon_decay: decay rate for updating epsilon between episodes
    When the agent falls in a hole, the reward should be updated to -1
    Returns:
        Q: the updated Q-table
        total_rewards: list containing the rewards per episode
    """
    total_rewards = []

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)

            new_state, reward, terminated, _ = env.step(action)

            if reward == 1.0 and terminated is True:
                episode_reward += reward
            if reward == 0.0 and terminated is True:
                episode_reward -= 1
                reward = -1
            if reward == 0.0 and step + 1 == max_steps:
                episode_reward += 0

            # Bellman Equation
            s = state
            a = action
            ns = new_state
            r = reward
            g = gamma
            Q[s, a] = (1 - alpha) * Q[s, a] + alpha * (r + g * max(Q[ns]))

            state = new_state

            if terminated:
                break

        epsilon = min_epsilon + (1 - min_epsilon) * np.exp(-epsilon_decay *
                                                           episode)
        total_rewards.append(episode_reward)

    return Q, total_rewards


if __name__ == "__main__":
    load_frozen_lake = __import__('0-load_env').load_frozen_lake
    q_init = __import__('1-q_init').q_init

    np.random.seed(0)
    desc = [['S', 'F', 'F'], ['F', 'H', 'H'], ['F', 'F', 'G']]
    env = load_frozen_lake(desc=desc)
    Q = q_init(env)

    Q, total_rewards = train(env, Q)
    print(Q)
    split_rewards = np.split(np.array(total_rewards), 10)
    for i, rewards in enumerate(split_rewards):
        print((i+1) * 500, ':', np.mean(rewards))
