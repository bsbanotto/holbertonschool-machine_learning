This is a README for Dimensionality Reduction. There are 2 mandatory tasks in
this project

## Task 0. PCA
Write a function  `def pca(X, var=0.95):`  that performs PCA on a dataset:

-   `X`  is a  `numpy.ndarray`  of shape  `(n, d)`  where:
    -   `n`  is the number of data points
    -   `d`  is the number of dimensions in each point
    -   all dimensions have a mean of 0 across all data points
-   `var`  is the fraction of the variance that the PCA transformation should maintain
-   Returns: the weights matrix,  `W`, that maintains  `var`  fraction of  `X`‘s original variance
-   `W`  is a  `numpy.ndarray`  of shape  `(d, nd)`  where  `nd`  is the new dimensionality of the transformed  `X`

## Task 1. PCA v2
Write a function  `def pca(X, ndim):`  that performs PCA on a dataset:

-   `X`  is a  `numpy.ndarray`  of shape  `(n, d)`  where:
    -   `n`  is the number of data points
    -   `d`  is the number of dimensions in each point
-   `ndim`  is the new dimensionality of the transformed  `X`
-   Returns:  `T`, a  `numpy.ndarray`  of shape  `(n, ndim)`  containing the transformed version of  `X`
