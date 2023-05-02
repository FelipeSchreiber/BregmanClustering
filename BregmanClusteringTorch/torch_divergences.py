#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 16:52:18 2023

@author: maximilien, Felipe Schreiber Fernandes

This code is taken from power k means bregman

"""

import torch
from torch.nn.functional import kl_div
from torch.nn import BCELoss
import warnings
warnings.filterwarnings("ignore")

#Bernoulli
def logistic_loss(X,M):
    print(X,M)
    total = BCELoss(X,M,reduce=False)
    #total = torch.where( X == 0, -torch.log( 1-M ), torch.log(X/M) )
    return total

#Multinomial
def KL_div(X,M):
    total = kl_div(X,M,reduction="none")
    return total

#Exponential
def itakura_saito_loss(X,M):
    total = (X/M - torch.log(X/M) - 1)
    return total

#Poisson
def relative_entropy(X,M):
    total = X*torch.log(X/M) + M - X
    return total

#gaussian
def euclidean_distance(X,M):
    return (X-M)**2

dist_to_phi_dict = {
        'gaussian': euclidean_distance,
        'bernoulli': logistic_loss,
        'multinomial':KL_div,
        'exponential': itakura_saito_loss,
        'poisson': relative_entropy
    }

def rbf_kernel(X,M):
    return torch.exp(-torch.norm(X-M,dim=-1))

""" 
def kullbackLeibler_binaryMatrix( X, M ):
    essai = torch.where( X == 0, -torch.log( 1-M ), torch.log(X/M))
    return essai

def euclidean_(X,M):
    return torch.square(X - M)

def phi_kl( a ):
    return torch.log(1+torch.exp(a))

def phi_euclidean( a ):
    return torch.square(a)

def dist_to_phi(dist):
    dist_to_phi_dict = {
        'gaussian': 'euclidean',
        'multinomial': 'kl_div',
        'exponential': 'itakura_saito',
        'poisson': 'relative_entropy',
        'gamma': 'gamma'
    }
    return dist_to_phi_dict[dist] """


'''
this function is structured weirdly: first 2 entries (phi, gradient of phi) can handle n x m theta matrix
last entry, only designed to work in iterative bregman update function, only works with 1 x m theta matrix and thus returns an m x m hessian
'''
def get_phi(name):
    phi_dict = {
        'euclidean': [lambda theta: torch.sum(theta**2, axis=1), lambda theta: 2*theta, lambda theta: 2*torch.eye(theta.size()[1], dtype=torch.float64)],
        'kl_div': [lambda theta: torch.sum(theta * torch.log(theta), axis=1), lambda theta: torch.log(theta) + 1, lambda theta: torch.eye(theta.size()[1], dtype=torch.float64) * 1/theta],
        'itakura_saito': [lambda theta: torch.sum(-torch.log(theta) - 1, axis=1), lambda theta: -1/theta, lambda theta: torch.eye(theta.size()[1]) / (theta**2)],
        'relative_entropy': [lambda theta: torch.sum(theta * torch.log(theta) - theta, axis=1), lambda theta: torch.log(theta), lambda theta: torch.eye(theta.size()[1]) / theta],
        'gamma': [lambda theta, k: torch.sum(-k + k * torch.log(k/theta), axis=1), lambda theta, k: -k/theta, lambda theta, k: k * torch.eye(theta.size()[1]) / (theta**2)]
    }
    return phi_dict[name]

#x, theta are both k-dimensional
def bregman_divergence(phi_list, x, theta):
    phi = phi_list[0]
    gradient = phi_list[1]

    bregman_div = phi(x) - phi(theta) - torch.dot(gradient(theta), x-theta)
    return bregman_div

#X is n x m, y is k x m, output is n x k containing all the pairwise bregman divergences
def pairwise_bregman(X, Y, phi_list, shape=None):
    phi = phi_list[0]
    gradient = phi_list[1]

    if shape:
        phi_X = phi(X, shape)[:, None]
        phi_Y = phi(Y, shape)[None, :]
    else:
        phi_X = phi(X)[:, None]
        phi_Y = phi(Y)[None, :]

    X = X[:, None]
    Y = Y[None, :]


    if shape:
        pairwise_distances = phi_X - phi_Y - torch.sum((X - Y) * gradient(Y, shape), axis=-1)
    else:
        pairwise_distances = phi_X - phi_Y - torch.sum((X - Y) * gradient(Y), axis=-1)

    return torch.clamp(pairwise_distances, min=1e-12, max=1e6)



