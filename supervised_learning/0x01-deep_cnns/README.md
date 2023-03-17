This is a README for project 0x01. Deep Convolutional Architectures

There are 8 mandatory tasks in this project as follows.

## Task 0. Inception Block
<img src="https://github.com/bsbanotto/holbertonschool-machine_learning/blob/main/supervised_learning/0x01-deep_cnns/README_pics/Task0.png" />
Write a function  `def inception_block(A_prev, filters):`  that builds an inception block as described in  [Going Deeper with Convolutions (2014)](https://intranet.hbtn.io/rltoken/zgrvk17O1Br8DXMepaPGeA "Going Deeper with Convolutions (2014)"):

-   `A_prev`  is the output from the previous layer
-   `filters`  is a tuple or list containing  `F1`,  `F3R`,  `F3`,`F5R`,  `F5`,  `FPP`, respectively:
    -   `F1`  is the number of filters in the 1x1 convolution
    -   `F3R`  is the number of filters in the 1x1 convolution before the 3x3 convolution
    -   `F3`  is the number of filters in the 3x3 convolution
    -   `F5R`  is the number of filters in the 1x1 convolution before the 5x5 convolution
    -   `F5`  is the number of filters in the 5x5 convolution
    -   `FPP`  is the number of filters in the 1x1 convolution after the max pooling
-   All convolutions inside the inception block should use a rectified linear activation (ReLU)
-   Returns: the concatenated output of the inception block

## Task 1. Inception Network
Write a function  `def inception_network():`  that builds the inception network as described in  [Going Deeper with Convolutions (2014)](https://intranet.hbtn.io/rltoken/zgrvk17O1Br8DXMepaPGeA "Going Deeper with Convolutions (2014)"):

-   You can assume the input data will have shape (224, 224, 3)
-   All convolutions inside and outside the inception block should use a rectified linear activation (ReLU)
-   You may use  `inception_block = __import__('0-inception_block').inception_block`
-   Returns: the keras model
<img src="https://github.com/bsbanotto/holbertonschool-machine_learning/blob/main/supervised_learning/0x01-deep_cnns/README_pics/Task1.png" />

## Task 2. Identity Block
<img src="https://github.com/bsbanotto/holbertonschool-machine_learning/blob/main/supervised_learning/0x01-deep_cnns/README_pics/Task2.png" />
Write a function  `def identity_block(A_prev, filters):`  that builds an identity block as described in  [Deep Residual Learning for Image Recognition (2015)](https://intranet.hbtn.io/rltoken/NNWT9AFrWl1VKtWeRFwRPQ "Deep Residual Learning for Image Recognition (2015)"):

-   `A_prev`  is the output from the previous layer
-   `filters`  is a tuple or list containing  `F11`,  `F3`,  `F12`, respectively:
    -   `F11`  is the number of filters in the first 1x1 convolution
    -   `F3`  is the number of filters in the 3x3 convolution
    -   `F12`  is the number of filters in the second 1x1 convolution
-   All convolutions inside the block should be followed by batch normalization along the channels axis and a rectified linear activation (ReLU), respectively.
-   All weights should use he normal initialization
-   Returns: the activated output of the identity block

## Task 3. Projection Block
<img src="https://github.com/bsbanotto/holbertonschool-machine_learning/blob/main/supervised_learning/0x01-deep_cnns/README_pics/Task3.png" />
Write a function  `def projection_block(A_prev, filters, s=2):`  that builds a projection block as described in  [Deep Residual Learning for Image Recognition (2015)](https://intranet.hbtn.io/rltoken/NNWT9AFrWl1VKtWeRFwRPQ "Deep Residual Learning for Image Recognition (2015)"):

-   `A_prev`  is the output from the previous layer
-   `filters`  is a tuple or list containing  `F11`,  `F3`,  `F12`, respectively:
    -   `F11`  is the number of filters in the first 1x1 convolution
    -   `F3`  is the number of filters in the 3x3 convolution
    -   `F12`  is the number of filters in the second 1x1 convolution as well as the 1x1 convolution in the shortcut connection
-   `s`  is the stride of the first convolution in both the main path and the shortcut connection
-   All convolutions inside the block should be followed by batch normalization along the channels axis and a rectified linear activation (ReLU), respectively.
-   All weights should use he normal initialization
-   Returns: the activated output of the projection block

## Task 4. ResNet-50
Write a function  `def resnet50():`  that builds the ResNet-50 architecture as described in  [Deep Residual Learning for Image Recognition (2015)](https://intranet.hbtn.io/rltoken/NNWT9AFrWl1VKtWeRFwRPQ "Deep Residual Learning for Image Recognition (2015)"):

-   You can assume the input data will have shape (224, 224, 3)
-   All convolutions inside and outside the blocks should be followed by batch normalization along the channels axis and a rectified linear activation (ReLU), respectively.
-   All weights should use he normal initialization
-   You may use:
    -   `identity_block = __import__('2-identity_block').identity_block`
    -   `projection_block = __import__('3-projection_block').projection_block`
-   Returns: the keras model
<img src="https://github.com/bsbanotto/holbertonschool-machine_learning/blob/main/supervised_learning/0x01-deep_cnns/README_pics/Task4.png" />

## Task 5. Dense Block
<img src="https://github.com/bsbanotto/holbertonschool-machine_learning/blob/main/supervised_learning/0x01-deep_cnns/README_pics/Task5.png" />
Write a function  `def dense_block(X, nb_filters, growth_rate, layers):`  that builds a dense block as described in  [Densely Connected Convolutional Networks](https://intranet.hbtn.io/rltoken/jP7TPnaM8LDvX04aNWpq1w "Densely Connected Convolutional Networks"):

-   `X`  is the output from the previous layer
-   `nb_filters`  is an integer representing the number of filters in  `X`
-   `growth_rate`  is the growth rate for the dense block
-   `layers`  is the number of layers in the dense block
-   You should use the bottleneck layers used for DenseNet-B
-   All weights should use he normal initialization
-   All convolutions should be preceded by Batch Normalization and a rectified linear activation (ReLU), respectively
-   Returns: The concatenated output of each layer within the Dense Block and the number of filters within the concatenated outputs, respectively

## Task 6. Transition Layer
Write a function  `def transition_layer(X, nb_filters, compression):`  that builds a transition layer as described in  [Densely Connected Convolutional Networks](https://intranet.hbtn.io/rltoken/jP7TPnaM8LDvX04aNWpq1w "Densely Connected Convolutional Networks"):

-   `X`  is the output from the previous layer
-   `nb_filters`  is an integer representing the number of filters in  `X`
-   `compression`  is the compression factor for the transition layer
-   Your code should implement compression as used in DenseNet-C
-   All weights should use he normal initialization
-   All convolutions should be preceded by Batch Normalization and a rectified linear activation (ReLU), respectively
-   Returns: The output of the transition layer and the number of filters within the output, respectively

## Task 7. DenseNet-121
Write a function  `def densenet121(growth_rate=32, compression=1.0):`  that builds the DenseNet-121 architecture as described in  [Densely Connected Convolutional Networks](https://intranet.hbtn.io/rltoken/jP7TPnaM8LDvX04aNWpq1w "Densely Connected Convolutional Networks"):

-   `growth_rate`  is the growth rate
-   `compression`  is the compression factor
-   You can assume the input data will have shape (224, 224, 3)
-   All convolutions should be preceded by Batch Normalization and a rectified linear activation (ReLU), respectively
-   All weights should use he normal initialization
-   You may use:
    -   `dense_block = __import__('5-dense_block').dense_block`
    -   `transition_layer = __import__('6-transition_layer').transition_layer`
-   Returns: the keras model
<img src="https://github.com/bsbanotto/holbertonschool-machine_learning/blob/main/supervised_learning/0x01-deep_cnns/README_pics/Task7.png" />
