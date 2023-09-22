This is a README for project Policy Gradients

There are 4 mandatory tasks in this project as follows

## Task 0. Simple Policy Function
Write a function that computes to policy with a weight of a matrix.

-   Prototype:  `def policy(matrix, weight):`

## Task 1. Compute the Monte-Carls policy gradient
By using the previous function created  `policy`, write a function that computes the Monte-Carlo policy gradient based on a state and a weight matrix.

-   Prototype:  `def policy_gradient(state, weight):`
    -   `state`: matrix representing the current observation of the environment
    -   `weight`: matrix of random weight
-   Return: the action and the gradient (in this order)

## Task 2. Implement the training
By using the previous function created  `policy_gradient`, write a function that implements a full training.

-   Prototype:  `def train(env, nb_episodes, alpha=0.000045, gamma=0.98):`
    -   `env`: initial environment
    -   `nb_episodes`: number of episodes used for training
    -   `alpha`: the learning rate
    -   `gamma`: the discount factor
-   Return: all values of the score (sum of all rewards during one episode loop)

Since the training is quite long, please print the current episode number and the score after each loop. To display these information on the same line, you can use  `end="\r", flush=False`  of the print function.

With the following main file, you should have this result plotted:

![](https://s3.eu-west-3.amazonaws.com/hbtn.intranet/uploads/medias/2020/12/e2fff0551f5173b824a8ee1b2e67aff72d7309e2.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA4MYA5JM5DUTZGMZG%2F20230922%2Feu-west-3%2Fs3%2Faws4_request&X-Amz-Date=20230922T151337Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=c9215750b66c5870a33af0bd8da310297ad7866508c87d0a7ff564843314d46d)

## Task 3. Animate iteration
Update the prototype of the  `train`  function by adding a last optional parameter  `show_result`  (default:  `False`).

When this parameter is  `True`, render the environment every 1000 episodes computed.

**Result after few episodes:**

![](https://s3.eu-west-3.amazonaws.com/hbtn.intranet/uploads/medias/2020/12/51a3d986d9c96960ddd0c009f7eaac5a2ce9f549.gif?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA4MYA5JM5DUTZGMZG%2F20230922%2Feu-west-3%2Fs3%2Faws4_request&X-Amz-Date=20230922T151337Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=845f1b79e3808fb592df5e48157a701162a1ca80ffbfdf59ca0e2fe6c00869d7)

**Result after more episodes:**

![](https://s3.eu-west-3.amazonaws.com/hbtn.intranet/uploads/medias/2020/12/8dadd3f7918aa188cde1b5c6ac2aafddac8a081f.gif?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA4MYA5JM5DUTZGMZG%2F20230922%2Feu-west-3%2Fs3%2Faws4_request&X-Amz-Date=20230922T151337Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=c066b22c8ab5de8ac279538b119c9e86234fc8f718721de20cb4fc76cb86e76c)

**Result after 10000 episodes:**

![](https://s3.eu-west-3.amazonaws.com/hbtn.intranet/uploads/medias/2020/12/da9d7deed16c5c9aec05e26bf14cf8b76e70dcce.gif?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA4MYA5JM5DUTZGMZG%2F20230922%2Feu-west-3%2Fs3%2Faws4_request&X-Amz-Date=20230922T151337Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=38cbd5ae7a310f20ad802fdb7b443c257ce39f83c2baa54bc811c4b9af181f62)