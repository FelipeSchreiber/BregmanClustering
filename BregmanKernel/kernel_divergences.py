#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 16:52:18 2023

@author: maximilien, Felipe Schreiber Fernandes

This code is taken in part from power k means bregman

"""

import numpy as np
from sklearn.metrics import *
import warnings
warnings.filterwarnings("ignore")

"""
DIVERGENCES DEFINITIONS
#DISTRIBUTION NAME -- DIVERGENCE NAME
d(X,Y) = phi(X) - phi(Y) + <X-Y, grad(phi)(Y)> 

SEE:
 "Clustering with Bregman Divergences" page 1725
"""

#Bernoulli | Logistic loss
#as stated in the paper page 1709

def hamming_(X,M):
    #total = hamming_loss(X,M)
    total = np.linalg.norm(X-M,ord=1)/len(X)
    return 1-total

def jaccard_(X,M):
    X = X.astype(bool)
    M = M.astype(bool)
    intersection = np.logical_and(X, M)
    union = np.logical_or(X, M)
    if union.sum() == 0:
        raise ValueError("UNION EMPTY") 
    # total = np.linalg.norm(X-M,ord=1)/np.logical_or(X , M).sum()
    return intersection.sum() / float(union.sum())

def logistic_loss(X,M):
    # total = np.where( X == 0, -np.log( 1-M ), -np.log(M) )
    total = np.log(1 + np.exp(- (2*X - 1) * ( np.log(M/(1-M)) ) ))
    return np.exp(-total.mean())

#Multinomial | KL-divergence
def KL_div(X,M):
    total = X*np.log(X/M)
    return np.exp(-total.mean())

#Exponential | Itakura-Saito Loss
def itakura_saito_loss(X,M):
    total = (X/M - np.log(X/M) - 1)
    return np.exp(-total.mean())

#Poisson | Generalized I-divergence
def generalized_I_divergence(X,M):
    total = X*np.log(X/M) + M - X
    return np.exp(-total.mean())

#gaussian | Squared Euclidean distance
def euclidean_distance(X,M):
    total = (0.5*(X-M)**2)
    return np.exp(-total.mean())

dist_to_divergence_dict = {
        'raw':None,
        'gaussian': euclidean_distance,
        'bernoulli': logistic_loss,
        'multinomial':KL_div,
        'exponential': itakura_saito_loss,
        'poisson': generalized_I_divergence,
        "jaccard":jaccard_,
        "hamming":hamming_
    }


"""
PHI DEFINITIONS
#DISTRIBUTION NAME -- DIVERGENCE NAME

SEE:
 "Clustering with Bregman Divergences" page 1725
"""

#Bernoulli | Logistic loss
def phi_bernoulli(X):
    total = X*np.log(X) + (1-X)*np.log(1-X)
    return total.sum()

#Multinomial | KL-divergence
def phi_multinomial(X):
    total = X*np.log(X)
    ## - X * log(N)
    return total.sum()

#Exponential | Itakura-Saito Loss
def phi_exponential(X):
    total = -np.log(X) - 1
    return total.sum()

#Poisson | Generalized I-divergence
def phi_poisson(X):
    total = X*np.log(X) - X
    return total

#gaussian | Squared Euclidean distance
def phi_gaussian(X):
    return 0.5*(X**2).sum() #* 1/(σ2) 

dist_to_phi_dict = {
        'gaussian': phi_gaussian,
        'bernoulli': phi_bernoulli,
        'multinomial': phi_multinomial,
        'exponential': phi_exponential,
        'poisson': phi_poisson 
    }

"""
PSI DEFINITIONS
#DISTRIBUTION NAME -- DIVERGENCE NAME
p(ψ,θ)(x) = exp(〈x, θ〉- ψ(θ))

SEE:
 "Clustering with Bregman Divergences" page 1725
"""

#Bernoulli | Logistic loss
def psi_bernoulli(θ):
    total = np.log(1 + np.exp(θ))
    return total.sum()

#Multinomial | KL-divergence
def psi_multinomial(θ):
    total = np.log(1 + np.sum(np.exp(θ)))#*N
    return total

#Exponential | Itakura-Saito Loss
def psi_exponential(θ):
    total = -np.log(-θ)
    return total.sum()

#Poisson | Generalized I-divergence
def psi_poisson(θ):
    total = np.exp(θ)
    return total.sum()

#gaussian | Squared Euclidean distance
def psi_gaussian(θ):
    return 0.5*(θ**2).sum() 

dist_to_psi_dict = {
        'gaussian': psi_gaussian,
        'bernoulli': psi_bernoulli,
        'multinomial': psi_multinomial,
        'exponential': psi_exponential,
        'poisson': psi_poisson 
    }


def rbf_kernel(X,M):
    return np.exp(-np.norm(X-M,dim=-1))

net_simmilarity_scores = {
    "jaccard":jaccard_,
    "hamming":hamming_
}

"""
psi*
"""

def legendre_gaussian(μ):
    return (μ**2/2).sum()

def legendre_bernoulli(μ):
    return (µ*np.log(µ) + (1 - μ)*np.log(1 - µ)).sum()