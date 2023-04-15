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

def kullbackLeibler_binaryMatrix( X, M ):
    essai = np.where( X == 0, -np.log( 1-M ), np.log(X/M) )
    return np.sum( essai )

#Multinomial
def KL_divergence(X,M):
    total = np.sum(X*np.log(X/M) + (1-X)*np.log((1-X)/(1-M)))
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
    return np.sum((X - M)**2)

dist_to_phi_dict = {
        'gaussian': euclidean_distance,
        'multinomial': KL_divergence,
        'exponential': itakura_saito_loss,
        'poisson': relative_entropy
    }