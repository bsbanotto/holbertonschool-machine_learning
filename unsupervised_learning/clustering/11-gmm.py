#!/usr/bin/env python3
"""
Calculate the GMM from a dataset using sklearn.mixture
"""
import sklearn.mixture


def gmm(X, k):
    """
    X: numpy.ndarray of shape (n, d) containing the dataset
    k: number of clusters
    Returns pi, m, S, clss, bic
        pi: numpy.ndarray of shape (k,) containing the cluster priors
        m: numpy.ndarray of shape (k, d) containing the centroid means
        S: numpy.ndarray of shape (k, d, d) containing the covariance matrices
        clss: numpy.ndarray of shape (n,) containing the cluster indeces for
            each data point
        bic: numpy.ndarray of shape (kmax - kmin + 1) containing the BIC value
            for each cluster size tested
    """
    GMM = sklearn.mixture.GaussianMixture(k).fit(X)

    pi = GMM.weights_
    m = GMM.means_
    S = GMM.covariances_
    clss = GMM.predict(X)
    bic = GMM.bic(X)

    return pi, m, S, clss, bic
