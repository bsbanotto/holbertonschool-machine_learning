#!/usr/bin/env python3
"""
Implementation of epsilon-greedy algorithm
"""
import numpy as np


# Task 2. Epsilon Greedy
def epsilon_greedy(Q, state, epsilon):
    """
    Function that uses epsilon-greedy to determine the next action
    Args:
        Q: numpy.ndarray shape(state, action) containing the q-table
        state: current state
        epsilon: epsilon to use for the calculation

    Should sample p with numpy.random.uniform to determine if the algorithm
        should explore or exploit

    If exploring, should pick the next action with numpy.random.randint from
        all possible actions
    Returns:
        the next action index
    """
    p = np.random.uniform()

    # Explore
    if p < epsilon:
        # random from 0 - 4 because we have 4 direction choices
        action = np.random.randint(0, 4)
    # Exploit
    else:
        action = np.argmax(Q[state])
    return action


if __name__ == "__main__":
    load_frozen_lake = __import__('0-load_env').load_frozen_lake
    q_init = __import__('1-q_init').q_init

    desc = [['S', 'F', 'F'], ['F', 'H', 'H'], ['F', 'F', 'G']]
    env = load_frozen_lake(desc=desc)
    Q = q_init(env)
    Q[7] = np.array([0.5, 0.7, 1, -1])
    np.random.seed(0)
    print(epsilon_greedy(Q, 7, 0.5))
    np.random.seed(1)
    print(epsilon_greedy(Q, 7, 0.5))
