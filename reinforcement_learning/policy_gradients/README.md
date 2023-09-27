# This is a README for the Policy Gradients project

## Implementing a Policy Gradient Control Algorithm for Cart Pole Game via Reinforcement Learning in this Project

To run this file in a conda environment:

```bash
git clone https://github.com/bsbanotto/holbertonschool-machine_learning/tree/main/reinforcement_learning/policy_gradients
cd <cloned_repo>
conda env create -f environment.yml
conda activate policy_grad
python3 ./run.py
```

There are 4 mandatory tasks in this project as follows

## Task 0. Simple Policy Function

Write a function that computes to policy with a weight of a matrix.

- Prototype:  `def policy(matrix, weight):`

## Task 1. Compute the Monte-Carls policy gradient

By using the previous function created  `policy`, write a function that computes the Monte-Carlo policy gradient based on a state and a weight matrix.

- Prototype:  `def policy_gradient(state, weight):`
  - `state`: matrix representing the current observation of the environment
  - `weight`: matrix of random weight
- Return: the action and the gradient (in this order)

## Task 2. Implement the training

By using the previous function created  `policy_gradient`, write a function that implements a full training.

- Prototype:  `def train(env, nb_episodes, alpha=0.000045, gamma=0.98):`
  - `env`: initial environment
  - `nb_episodes`: number of episodes used for training
  - `alpha`: the learning rate
  - `gamma`: the discount factor
- Return: all values of the score (sum of all rewards during one episode loop)

Since the training is quite long, please print the current episode number and the score after each loop. To display these information on the same line, you can use  `end="\r", flush=False`  of the print function.

With the following main file, you should have this result plotted:

![reward per episode](https://raw.githubusercontent.com/bsbanotto/holbertonschool-machine_learning/main/reinforcement_learning/policy_gradients/output.png)

## Task 3. Animate iteration

Update the prototype of the  `train`  function by adding a last optional parameter  `show_result`  (default:  `False`).

When this parameter is  `True`, render the environment every 1000 episodes computed.

**After 0 episodes**  
**Score: 19**

![Result after 0 episodes](https://raw.githubusercontent.com/bsbanotto/holbertonschool-machine_learning/main/reinforcement_learning/policy_gradients/episode_gifs/episode_0.gif)

**After 250 episodes:**  
**Score: 40**

![Result after 250 episodes](https://raw.githubusercontent.com/bsbanotto/holbertonschool-machine_learning/main/reinforcement_learning/policy_gradients/episode_gifs/episode_250.gif)

**After 500 episodes:**  
**Score: 32**

![Result after 500 episodes](https://raw.githubusercontent.com/bsbanotto/holbertonschool-machine_learning/main/reinforcement_learning/policy_gradients/episode_gifs/episode_500.gif)

**After 1000 episodes:**  
**Score: 22**

![Result after 1000 episodes](https://raw.githubusercontent.com/bsbanotto/holbertonschool-machine_learning/main/reinforcement_learning/policy_gradients/episode_gifs/episode_1000.gif)

**After 2500 episodes:**  
**Score: 441**

![Result after 2500 episodes](https://raw.githubusercontent.com/bsbanotto/holbertonschool-machine_learning/main/reinforcement_learning/policy_gradients/episode_gifs/episode_2500.gif)

**After 5000 episodes:**  
**Score: 500(max allowable**)

![Result after 5000 episodes](https://raw.githubusercontent.com/bsbanotto/holbertonschool-machine_learning/main/reinforcement_learning/policy_gradients/episode_gifs/episode_5000.gif)
