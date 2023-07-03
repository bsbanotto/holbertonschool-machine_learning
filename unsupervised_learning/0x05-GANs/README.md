This is a README for project Generative Adversarial Networks

There are 6 mandatory tasks in this project as follows:

## Task 0. Generator
-   Write a subclass that defines the generator:  `class Generator(nn.Module):`
-   Define the  `__init__`  construct with these parameters:  `(self, input_size, hidden_size, output_size)`
-   Make sure you define the feed-forward function inside of the class  `def forward(self, x):`
-   The network should have three layers and two  `tanh`  activation functions after the first and second layer.
-   The layers and activation functions should be contained inside of a  `nn.Sequential`  wrapper class.

## Task 1. Discriminator
-   Write a subclass that defines the generator:  `class Discriminator(nn.Module):`
-   Define the  `__init__`  construct with these parameters:  `(self, input_size, hidden_size, output_size)`
-   Make sure you define the feed-forward function inside of the class  `def forward(self, x):`
-   The network should have three layers and three sigmoid activation functions after each layer.
-   The layers and activation functions should be contained inside of a  `nn.Sequential`  class.

## Task 2. Sample Z
Write a function  `def sample_Z(mu, sigma, sampleType):`  that creates input for the generator and discriminator:

-   `mu`  Should be the mean of the distribution
-   `sigma`  Should be the standard deviation of the distribution
-   `sampleType`  Should be a variable that selects which model to sample for.
    -   The variable should accept a  `"G"`  or  `"D"`  as string values.
-   The input data for discrimintator should be from a normal distribution (it will also need random sampling in the training phase).
-   The input data for generator should be random sampling.
-   The function should return a  `torch.Tensor`  type for both generator and discriminator if the parameters are correct.
    -   It should return 0 otherwise.

## Task 3. Train Discriminator
Write a function called  `def train_dis(Gen,Dis, dInputSize, gInputSize, mbatchSize, steps, optimizer, crit):`

-   The  `Gen`, and  `Dis`  are the Discriminator and Generator Objects.
-   `dInputSize`  is the input size of Discriminator input data.
-   `gInputSize`  is the input size of Generator input data.
-   `mbatchSize`  should be the batch size for training.
-   `steps`  should be the number of steps for training.
-   `optimizer`  should be a stochastic gradient descent optimizer object.
-   The function should return the two item methods that belong to loss entropy class. for real and fake
-   The  `crit`  should be a BCEloss function.
-   Should use both random noise, and normal distribution for sampling
-   The 4 moments should be used in processing the sample.
-   The function should return the error estimate of the fake and real data, along with the fake and real data sets of type  `torch.Tensor()`.

## Task 4. Train Generator
Write a function called  `def train_gen(Gen,Dis, gInputSize, mbatchSize, steps, optimizer, crit):`

-   The  `Gen`, and  `Dis`  are the Discriminator and Generator Objects.
-   `gInputSize`  is the input size of Generator input data.
-   `mbatchSize`  should be the batch size for training.
-   `steps`  should be the number of steps for training.
-   `optimizer`  should be a stochastic gradient descent optimizer object.
-   The function should return the two item methods that belong to loss entropy class. for real and fake
-   The  `crit`  should be a BCEloss method.
-   Only random noise should be used for sampling
-   The 4 moments should be used in processing the sample.
-   The function should return the error of the fake data, and the fake data set of type  `torch.Tensor()`

## Task 5. Train GAN
Write a function  `def train_gan():`  that trains a GAN:

-   The learning rate should equal 1e-3.
-   The batch size should be 512
-   The number of iterations should be 5000.
-   The number of steps for both the descriminator and generator should be 20.
-   The discriminator and generator should only have 20 steps
-   You should use both Discriminator and Generator classes inside of function.
-   The function should return the fake generated distribution from the Generator of type  `torch.Tensor()`.
