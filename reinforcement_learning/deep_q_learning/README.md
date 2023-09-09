This is a README for project Deep Q Learning

In this project, we train a Deep Q-network using reinforcement learning to
play Atari's Breakout (and hopefully win, or at least perform as well as a
toddler)

## Conda Instructions
I'm including these because it's my first time creating a conda environment
and although this is probably silly, I want to remember this time forever

Create (breakout) conda environment using this command from the environment directory
`conda env create -f environment.yml`

Activate the deep_q environment 
`conda activate deep_q`

Install requirements using this command fron the environment directory
`pip install -r requirements.txt`

To use this environment in a Jupyter Notebook
1. Install ipykernel
`conda install -c anaconda ipykernel`
2. Send this environment to ipykernel
`python -m ipykernel install --user --name=deep_q`
3. When you start the notebook, select the deep_q kernel

## Task 0. Breakout
Write a python script  `train.py`  that utilizes  `keras`,  `keras-rl`, and  `gym`  to train an agent that can play Atari’s Breakout:

-   Your script should utilize  `keras-rl`‘s  `DQNAgent`,  `SequentialMemory`, and  `EpsGreedyQPolicy`
-   Your script should save the final policy network as  `policy.h5`

Write a python script  `play.py`  that can display a game played by the agent trained by  `train.py`:

-   Your script should load the policy network saved in  `policy.h5`
-   Your agent should use the  `GreedyQPolicy`
