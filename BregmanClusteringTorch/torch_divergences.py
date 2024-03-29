#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 16:52:18 2023

@author: maximilien, Felipe Schreiber Fernandes

This code is taken in part from power k means bregman

"""

import torch
from torch.nn.functional import kl_div
import warnings
# from torch.autograd import grad
# from torch.autograd.functional import jacobian
from torch.func import grad,vmap
warnings.filterwarnings("ignore")


"""
DIVERGENCES DEFINITIONS
#DISTRIBUTION NAME -- DIVERGENCE NAME
d(X,Y) = phi(X) - phi(Y) + <X-Y, grad(phi)(Y)> 

SEE:
 "Clustering with Bregman Divergences" page 1725
 footnote for bernoulli
"""

#Bernoulli | Logistic loss
def logistic_loss(X,M):
    # total = torch.where( X == 0, -torch.log( 1-M ), -torch.log(M) )
    total = torch.log(1 + torch.exp(- (2*X - 1) * ( torch.log(M/(1-M)) ) ))
    return total

#Multinomial | KL-divergence
def KL_div(X,M):
    total = kl_div(X,M,reduction="none")
    return total

#Exponential | Itakura-Saito Loss
def itakura_saito_loss(X,M):
    total = (X/M - torch.log(X/M) - 1)
    return total

#Poisson | Generalized I-divergence
def generalized_I_divergence(X,M):
    total = X*torch.log(X/M) + M - X
    return total

#gaussian | Squared Euclidean distance
def euclidean_distance(X,M):
    return 0.5*(X-M)**2

dist_to_divergence_dict = {
        'gaussian': euclidean_distance,
        'bernoulli': logistic_loss,
        'multinomial':KL_div,
        'exponential': itakura_saito_loss,
        'poisson': generalized_I_divergence
    }


"""
PHI DEFINITIONS
#DISTRIBUTION NAME -- DIVERGENCE NAME

SEE:
 "Clustering with Bregman Divergences" page 1725
"""

#Bernoulli | Logistic loss
def phi_bernoulli(X):
    total = X*torch.log(X) + (1-X)*torch.log(1-X)
    # one_minus = 1 - X
    # total = torch.log(torch.pow(X,X)) + torch.log(torch.pow(one_minus,one_minus))
    return total

#Multinomial | KL-divergence
def phi_multinomial(X):
    total = X*torch.log(X)
    ## - X * log(N)
    return total

#Exponential | Itakura-Saito Loss
def phi_exponential(X):
    total = -torch.log(X) - 1
    return total

#Poisson | Generalized I-divergence
def phi_poisson(X):
    total = X*torch.log(X) - X
    return total

#gaussian | Squared Euclidean distance
def phi_gaussian(X):
    return 0.5*(X**2) #* 1/(σ2) 

## Takes a func R^m -> R^m and reduce 
def make_phi_with_reduce(reduce_func,phi):
    def compose_phi(X):
        return reduce_func(phi(X))
    return compose_phi

## Takes a func (R^{n x d}, R^{n x d} ) -> R^n, specified by div, and reduce -> R
def make_div_with_reduce(reduce_func,div):
    def compose_div(X,Y):
        return reduce_func(div(X,Y))
    return compose_div

## This func applies (PRECOMPUTED) bregman div to every pair of X,Y
## X is N x M and Y is K x M 
def make_att_div(reduce_func,div):
    def pairwise_div(X,Y):
        total = div(
            X[:,None],#.expand(-1,n_clusters,-1),
            Y[None,:]#.expand(N,-1,-1)
        )
        # total has shape N x K x D. Applies reduce in the last axis.
        return reduce_func(total,dim=-1)
    return pairwise_div

#x, theta are both m-dimensional. Creates the divergence func that maps x,theta -> scalar
## Apply definition D_φ(X,Y) = φ(x) - φ(y) - <x - y, φ'(y)> 
def make_breg_div(phi):
    ## phi R^m -> R
    ## grad_phi R^m -> R^m
    grad_phi = grad(phi)
    def bregman_divergence(x, theta):
        bregman_div = phi(x) - phi(theta) - torch.dot(grad_phi(theta), x-theta)
        return bregman_div
    return bregman_divergence

## given X in R^{N x M} and Θ in R^{N x M}, computes the bregman divergence for each
# pair X_{i:}, Θ_{i:}, returns a vector R^N, and finally reduce the vector,
# returning a scalar 
def make_pair_breg(reduce_func,breg_div):
    vectorized_breg = vmap(breg_div)
    def pair_breg(X,Y):
        return reduce_func(vectorized_breg(X,Y))
    return pair_breg

# This func is similar to make_att_div, but instead of taking the
#  precomputed bregman divergence, it computes using the definition
#  D_φ(X,Y) = φ(x) - φ(y) - <x - y, φ'(y)>.
def make_pairwise_breg(breg_div):
    vectorized_breg = vmap(breg_div)
    def pairwise_breg(X,Y):
        n_clusters = Y.shape[0]
        N = X.shape[0]
        X_ = torch.repeat_interleave(X, n_clusters, dim=0)
        Y_ = Y.repeat(N,1)
        return vectorized_breg(X_,Y_).reshape(N,n_clusters)
    return pairwise_breg

# This func is the same as make_pairwise_breg, but taken from
# power k means bregman https://github.com/avellal14/bregman_power_kmeans.
def make_pairwise_breg2(phi):
    vectorized_grad = vmap(grad(phi))
    vectorized_phi = vmap(phi)
    # breg_div = make_breg_div(phi)
    #X is n x m, y is k x m, output is n x k containing all the pairwise bregman divergences
    def pairwise_bregman(X, Y):
        ## phi R^m -> R
        ## grad_phi R^m -> R^m
        ## vmap(grad_phi) R^(k x m) -> R^(k x m)
        grad_phi = vectorized_grad(Y)
        phi_X = vectorized_phi(X)[:, None]
        phi_Y = vectorized_phi(Y)[None, :]

        X = X[:, None]
        Y = Y[None, :]
        ## X - Y is n x k x m
        ## (X - Y) x vmap(grad_phi) -> n x k
        pairwise_distances = phi_X - phi_Y - torch.sum((X - Y) * grad_phi[None, :], axis=-1)
        return torch.clamp(pairwise_distances, min=1e-12, max=1e6)
    return pairwise_bregman

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
    total = torch.log(1 + torch.exp(θ))
    return total.sum()

#Multinomial | KL-divergence
def psi_multinomial(θ):
    total = torch.log(1 + torch.sum(torch.exp(θ)))#*N
    return total

#Exponential | Itakura-Saito Loss
def psi_exponential(θ):
    total = -torch.log(-θ)
    return total.sum()

#Poisson | Generalized I-divergence
def psi_poisson(θ):
    total = torch.exp(θ)
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
    return torch.exp(-torch.norm(X-M,dim=-1))


### THIS SECTION WAS TAKEN FROM power k means bregman
'''
this function is structured weirdly: first 2 entries (phi, gradient of phi) can handle n x m theta matrix
last entry, only designed to work in iterative bregman update function, only works with 1 x m theta matrix and thus returns an m x m hessian
'''

"""
def get_phi(name):
    phi_dict = {
        'euclidean': [lambda theta: torch.sum(theta**2, axis=1), lambda theta: 2*theta, lambda theta: 2*torch.eye(theta.size()[1], dtype=torch.float64)],
        'kl_div': [lambda theta: torch.sum(theta * torch.log(theta), axis=1), lambda theta: torch.log(theta) + 1, lambda theta: torch.eye(theta.size()[1], dtype=torch.float64) * 1/theta],
        'itakura_saito': [lambda theta: torch.sum(-torch.log(theta) - 1, axis=1), lambda theta: -1/theta, lambda theta: torch.eye(theta.size()[1]) / (theta**2)],
        'relative_entropy': [lambda theta: torch.sum(theta * torch.log(theta) - theta, axis=1), lambda theta: torch.log(theta), lambda theta: torch.eye(theta.size()[1]) / theta],
        'gamma': [lambda theta, k: torch.sum(-k + k * torch.log(k/theta), axis=1), lambda theta, k: -k/theta, lambda theta, k: k * torch.eye(theta.size()[1]) / (theta**2)]
    }
    return phi_dict[name]

#X is n x m, y is k x m, output is n x k containing all the pairwise bregman divergences
#shape=gamma_shape
def pairwise_bregman(X, Y, phi, shape=None):
    ## phi R^m -> R
    ## grad_phi R^m -> R^m
    ## vmap(grad_phi) R^(k x m) -> R^(k x m)
    grad_phi = vmap(grad(phi))(Y)
    if shape:
        phi_X = phi(X, shape)[:, None]
        phi_Y = phi(Y, shape)[None, :]
    else:
        phi_X = phi(X)[:, None]
        phi_Y = phi(Y)[None, :]

    X = X[:, None]
    Y = Y[None, :]
    ## X - Y is n x k x m
    ## (X - Y) x vmap(grad_phi) -> n x k
    if shape:
        pairwise_distances = phi_X - phi_Y - torch.sum((X - Y) * grad_phi[None, :], axis=-1)
    else:
        pairwise_distances = phi_X - phi_Y - torch.sum((X - Y) * grad_phi[None, :], axis=-1)

    return torch.clamp(pairwise_distances, min=1e-12, max=1e6)
"""

