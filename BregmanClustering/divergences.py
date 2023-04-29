#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 16:52:18 2023

@author: maximilien

This code is taken from 
https://github.com/juselara1/bregclus/blob/main/bregclus/divergences.py
"""


import numpy as np
from scipy.special import kl_div
import warnings
warnings.filterwarnings("ignore")

def kullbackLeibler_binaryMatrix( X, M ):
    essai = np.where( X == 0, -np.log( 1-M ), np.log(X/M) )
    return np.sum( essai )

#Bernoulli
def logistic_loss(X,M):
    #total = log_loss(X.flatten(),M.flatten())
    total = np.where( X == 0, -np.log( 1-M ), np.log(X/M) ).sum()
    return total

#Multinomial
def KL_div(X,M):
    total = np.sum(kl_div(X.flatten(),M.flatten()))
    return total

#Exponential
def itakura_saito_loss(X,M):
    total = np.sum((X/M - np.log(X/M) - 1))
    return total

#Poisson
def relative_entropy(X,M):
    total = np.sum(X*np.log(X/M) + M - X)
    return total

#gaussian
def euclidean_distance(X,M):
    return np.linalg.norm(X-M)

def _euclidean_vectorized(X,Y):
    """
    Computes a pairwise Euclidean distance between two matrices: D_ij=||x_i-y_j||^2.
    Parameters
    ----------
        X: array-like, shape=(batch_size, n_features)
           Input batch matrix.
        Y: array-like, shape=(n_clusters, n_features)
           Matrix in which each row represents the mean vector of each cluster.
    Returns
    -------
        D: array-like, shape=(batch_size, n_clusters)
           Matrix of paiwise dissimilarities between the batch and the cluster's parameters.
    """
    
    # same computation as the _old_euclidean function, but a new axis is added
    # to X so that Y can be directly broadcast, speeding up computations
    return np.sqrt(np.sum((np.expand_dims(X, axis=1)-Y)**2, axis=-1))

# expose the vectorized version as the default one
euclidean = _euclidean_vectorized
dist_to_phi_dict = {
        'gaussian': euclidean_distance,
        'bernoulli': logistic_loss,
        'multinomial':KL_div,
        'exponential': itakura_saito_loss,
        'poisson': relative_entropy
    }