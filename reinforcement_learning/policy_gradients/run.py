#!/usr/bin/env python3
"""
Main run file for Cart Pole policy gradient training and running
"""
import gym
render = __import__('render').render


env = gym.make('CartPole-v1')
scores = render(env=env,
                nb_episodes=10000,
                alpha=0.000045,
                gamma=0.98,
                show_result=True,
                episodes_to_render=[0, 250, 500, 1000, 2500, 5000, 10000],
                save_gifs=False,  # When True creates new gifs WILL BE SLOW
                )
env.close()
