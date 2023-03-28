#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 16:52:18 2023

@author: maximilien

This code is taken from 
https://github.com/juselara1/bregclus/blob/main/bregclus/divergences.py
"""


import numpy as np
import functools
import warnings
warnings.filterwarnings("ignore")

def distance_function_vec(func):
    """
    This decorates any distance function that expects two vectors as input. 
    """
    @functools.wraps(func)
    def wraped_distance(X, Y):
        """
        Computes a pairwise distance between two matrices.
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
        
        # naive implementation (without vectorization idk if it possible to do it in a generic way... maybe with jax)
        # probably it can be faster if we directly use numpy instead of list of compreensions to numpy
        
        # builds the np.array (batch_size, n_cluster) by testing the func for all combinations of XxY.
        return np.array([[func(sample, cluster_center) for cluster_center in Y] for sample in X])

    return wraped_distance

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


def kl( a, b ):
    if a>1 or a<0 or b >= 1 or b <= 0:
        raise TypeError( 'Kullback Leibler divergence cannot be computed' )
        
    if a == 0:
        return (1-a) * np.log( (1-a)/(1-b) )
    elif a==1:
        return a * np.log( a / b )
    else:
        return a * np.log( a / b ) + (1-a) * np.log( (1-a)/(1-b) )
    
    
def kl_vec( a, b ):
    return np.log( (a/b)**(a) ) + np.log( ( (1-a)/(1-b) )**(1-a) )



def euclidean_matrix ( X, means, Z ):
    M = Z@means@Z.T
    return np.linalg.norm( X - M, ord = 'fro' )


def kullbackLeibler_binaryMatrix( X, M ):
    essai = np.where( X == 0, -np.log( 1-M ), np.log(X/M) )
    return np.sum( essai )

def logistic_loss(X,M):
    X_flatten = X.flatten()
    M_flatten = M.flatten()
    total = np.sum(X_flatten*np.log(X_flatten/M_flatten) + (1-X_flatten)*np.log((1-X_flatten)/(1-M_flatten)))
    return total    


# expose the vectorized version as the default one
euclidean = _euclidean_vectorized

def _mahalanobis_vectorized(X, Y, cov):
    
    diff = np.expand_dims(np.expand_dims(X, axis=1)-Y, axis=-1)
    return np.sum(np.squeeze(((np.linalg.pinv(cov)@diff)*diff)), axis=-1)

def _squared_manhattan_vectorized(X, Y):
    
    return np.sum(np.abs(np.expand_dims(X, axis=1)-Y), axis=-1)**2