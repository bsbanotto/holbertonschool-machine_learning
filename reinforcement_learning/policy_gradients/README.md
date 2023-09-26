# This is a README for the Policy Gradients project

## In this project, I implement a reinforcement learning policy gradient control algorithm to train an agent to play Cart Pole

To run this file in a conda environment:

```bash
git clone https://github.com/bsbanotto/holbertonschool-machine_learning/tree/main/reinforcement_learning/policy_gradients
cd <cloned_repo>
conda env create -f environment.yml
conda activate policy_grad
python3 ./render.py
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

![reward per episode](https://s3.eu-west-3.amazonaws.com/hbtn.intranet/uploads/medias/2020/12/e2fff0551f5173b824a8ee1b2e67aff72d7309e2.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA4MYA5JM5DUTZGMZG%2F20230922%2Feu-west-3%2Fs3%2Faws4_request&X-Amz-Date=20230922T151337Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=c9215750b66c5870a33af0bd8da310297ad7866508c87d0a7ff564843314d46d)

## Task 3. Animate iteration

Update the prototype of the  `train`  function by adding a last optional parameter  `show_result`  (default:  `False`).

When this parameter is  `True`, render the environment every 1000 episodes computed.

**Result after 0 episodes:**

![Result after 0 episodes](https://github.com/bsbanotto/holbertonschool-machine_learning/blob/main/reinforcement_learning/policy_gradients/episode_gifs/episode_0.gif)

**Result after 250 episodes:**

![Result after 250 episodes](https://github.com/bsbanotto/holbertonschool-machine_learning/blob/main/reinforcement_learning/policy_gradients/episode_gifs/episode_250.gif)

**Result after 500 episodes:**

![Result after 500 episodes](https://github.com/bsbanotto/holbertonschool-machine_learning/blob/main/reinforcement_learning/policy_gradients/episode_gifs/episode_500.gif)

**Result after 1000 episodes:**

![Result after 1000 episodes](https://github.com/bsbanotto/holbertonschool-machine_learning/blob/main/reinforcement_learning/policy_gradients/episode_gifs/episode_1000.gif)

**Result after 2500 episodes:**

![Result after 2500 episodes](https://github.com/bsbanotto/holbertonschool-machine_learning/blob/main/reinforcement_learning/policy_gradients/episode_gifs/episode_2500.gif)

**Result after 5000 episodes:**

![Result after 5000 episodes](`link here`)
