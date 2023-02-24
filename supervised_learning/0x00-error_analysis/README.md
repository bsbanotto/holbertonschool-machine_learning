This is a README for project 0x00. Error Analysis

There are 7 mandatory tasks in this project as follows:

## Task 0. Create Confusion<br>
Write the function  `def create_confusion_matrix(labels, logits):`  that creates a confusion matrix:

-   `labels`  is a one-hot  `numpy.ndarray`  of shape  `(m, classes)`  containing the correct labels for each data point
    -   `m`  is the number of data points
    -   `classes`  is the number of classes
-   `logits`  is a one-hot  `numpy.ndarray`  of shape  `(m, classes)`  containing the predicted labels
-   Returns: a confusion  `numpy.ndarray`  of shape  `(classes, classes)`  with row indices representing the correct labels and column indices representing the predicted labels

To accompany the following main file, you are provided with  [labels_logits.npz](https://s3.eu-west-3.amazonaws.com/hbtn.intranet.project.files/holbertonschool-ml/labels_logits.npz "labels_logits.npz"). This file does not need to be pushed to GitHub, nor will it be used to check your code.

## Task 1. Sensitivity<br>
Write the function  `def sensitivity(confusion):`  that calculates the sensitivity for each class in a confusion matrix:

-   `confusion`  is a confusion  `numpy.ndarray`  of shape  `(classes, classes)`  where row indices represent the correct labels and column indices represent the predicted labels
    -   `classes`  is the number of classes
-   Returns: a  `numpy.ndarray`  of shape  `(classes,)`  containing the sensitivity of each class

## Task 2. Precision<br>
Write the function  `def precision(confusion):`  that calculates the precision for each class in a confusion matrix:

-   `confusion`  is a confusion  `numpy.ndarray`  of shape  `(classes, classes)`  where row indices represent the correct labels and column indices represent the predicted labels
    -   `classes`  is the number of classes
-   Returns: a  `numpy.ndarray`  of shape  `(classes,)`  containing the precision of each class

## Task 3. Specificity<br>
Write the function  `def specificity(confusion):`  that calculates the specificity for each class in a confusion matrix:

-   `confusion`  is a confusion  `numpy.ndarray`  of shape  `(classes, classes)`  where row indices represent the correct labels and column indices represent the predicted labels
    -   `classes`  is the number of classes
-   Returns: a  `numpy.ndarray`  of shape  `(classes,)`  containing the specificity of each class

## Task 4. F1 Score<br>
Write the function  `def f1_score(confusion):`  that calculates the F1 score of a confusion matrix:

-   `confusion`  is a confusion  `numpy.ndarray`  of shape  `(classes, classes)`  where row indices represent the correct labels and column indices represent the predicted labels
    -   `classes`  is the number of classes
-   Returns: a  `numpy.ndarray`  of shape  `(classes,)`  containing the F1 score of each class
-   You must use  `sensitivity = __import__('1-sensitivity').sensitivity`  and  `precision = __import__('2-precision').precision`  create previously

## Task 5. Dealing with Error<br>
In the text file  `5-error_handling`, write the lettered answer to the question of how you should approach the following scenarios. Please write the answer to each scenario on a different line. If there is more than one way to approach a scenario, please use CSV formatting and place your answers in alphabetical order (ex.  `A,B,C`):

Scenarios:

```
1. High Bias, High Variance
2. High Bias, Low Variance
3. Low Bias, High Variance
4. Low Bias, Low Variance
```

Approaches:

```
Always fix High Bias first, Then fix High Variance

A. Train more                       |   High Bias
B. Try a different architecture     |   High Anything
C. Get more data                    |   High Variance
D. Build a deeper network           |   High Bias
E. Use regularization               |   High Variance
F. Nothing                          |   All is well - Low Bias and Low Variance
```

## Task 6. Compare and Contrast<br>
Given the following training and validation confusion matrices and the fact that human level performance has an error of ~14%, determine what the most important issue is and write the lettered answer in the file  `6-compare_and_contrast`

<img src="https://github.com/bsbanotto/holbertonschool-machine_learning/blob/main/supervised_learning/0x00-error_analysis/readme_photos/Training%20Matrix.png" />

<img src="https://github.com/bsbanotto/holbertonschool-machine_learning/blob/main/supervised_learning/0x00-error_analysis/readme_photos/Validation%20Matrix.png"/>

Most important issue:

```
## A. High Bias
B. High Variance
C. Nothing
```
