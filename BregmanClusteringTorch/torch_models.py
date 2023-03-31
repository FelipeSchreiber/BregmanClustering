#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 16:59:19 2023

@author: maximilien, Felipe Schreiber Fernandes
felipesc@cos.ufrj.br
"""

import scipy as sp
from sklearn.base import BaseEstimator, ClusterMixin
from BregmanClustering.models import *
from .torch_divergences import *
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
from torch_geometric.utils import *
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import warnings
warnings.filterwarnings("ignore")

class SoftBregmanClusteringTorch( BaseEstimator, ClusterMixin ):
    def __init__( self, n_clusters, 
                 graph_divergence = kullbackLeibler_binaryMatrix, attribute_divergence = euclidean_, 
                 initializer = 'chernoff', 
                 graph_initializer = "spectralClustering", attribute_initializer = 'GMM', 
                 n_iters = 25, init_iters=100,
                 normalize_=True, thresholding=True,
                 reduce_by = torch.sum
                ):
        """
        Bregman Hard Clustering Algorithm for partitioning graph with node attributes
        Parameters
        ----------
        n_clusters : INT
            Number of clustes.
        graph_divergence, attribute_divergence : function
            Pairwise divergence function. The default is euclidean.
        n_iters : INT, optional
            Number of clustering iterations. The default is 25.
        graph_initialize, attribute_initializer : STR, optional
            Specifies if the centroids are initialized at random "rand", K-Means++ "kmeans++", or a pretrained K-Means model "pretrained". The default is "rand".
        init_iters : INT, optional
            Number of iterations for K-Means++. The default is 100.
        Returns
        -------
        None.
        """
        self.n_clusters = n_clusters
        self.graph_divergence = graph_divergence
        self.attribute_divergence = attribute_divergence
        self.n_iters = n_iters
        self.initializer = initializer
        self.graph_initializer = graph_initializer
        self.attribute_initializer = attribute_initializer
        self.init_iters = init_iters
        self.graphDistribution = 'bernoulli'
        self.attributeDistribution = 'gaussian'
        self.normalize_ = normalize_
        self.thresholding = thresholding
        self.reduce_by = reduce_by
        self.N = 0
        self.row_indices = torch.arange(2)

    def spectralEmbedding( self, X ):
        if (X<0).any():
            X = rbf_kernel(X[:,None],X[None,:])
        L, V = torch.linalg.eig(X)
        U = V[:,-self.n_clusters:]
        return U
    
    def computeAttributeMeans( self, Y, Z ):
        attribute_means = (Z.T@Y)/(Z.sum(dim=0) + 10 * torch.finfo(Z.dtype).eps)[:, None]
        return attribute_means

    def computeGraphMeans(self,X,tau):
        graph_means = torch.zeros((self.n_clusters,self.n_clusters))
        tau_sum = tau.sum(0)
        weights = torch.tensordot(tau, tau, dims=((), ()))
        """
        weights[i,q,j,l] = tau[i,q]*tau[j,l]
        desired output:
        weights[q,l,i,j] = tau[i,q]*tau[j,l]
        """
        weights = weights.permute(1,3,0,2)
        for q in range(self.n_clusters):
            for l in range(self.n_clusters):
                graph_means[q,l]=torch.sum(weights[q,l]*X)/torch.sum(weights[q,l])
        #graph_means/=((tau_sum.reshape((-1, 1)) * tau_sum) - tau.T @ tau)
        torch.nan_to_num(graph_means,out=graph_means)
        return graph_means 
    
    def likelihoodAttributes(self,Y,Z):
        M = self.computeAttributeMeans(Y,Z)
        total = torch.sum( self.attribute_divergence(Y,Z@M) )
        return total

    def likelihoodGraph(self,X,Z):
        graph_mean = self.computeGraphMeans(X,Z)
        return 1/2 * torch.sum( self.graph_divergence( X, Z @ graph_mean @ Z.T ) )
 
    def VE_step(self,X,Y,tau):
        """
        Inputs: 
        X: adjacency matrix
        Y: attributes matrix
        tau: membership matrix
        """
        #N = X.shape[0]
        pi = tau.mean(0)
        """
        Compute net divergences for every pair X[i,j], mu[k,l]
        """
        net_divergences_elementwise = self.graph_divergence(X.reshape(-1,1),\
                                             self.graph_means.reshape(-1,1))\
                                            .reshape((self.N,self.N,self.n_clusters,self.n_clusters))
        """
        net_divergences has shape N x N x K x K
        tau has shape N x K
        the result must be N x K
        result[i,k] = sum_j sum_l tau[j,l] * net_div[i,j,k,l]
        tensordot performs the multiplication and sum over specified axes.
        "j" appears at axes 0 for tau and at axes 1 for net_divergence
        "l" appears at axes 1 for tau and at axes 3 for net_divergence
        """
        net_divergence_total = torch.tensordot(tau, net_divergences_elementwise, dims=[(0,1),(1,3)])
        #print(net_divergence_total)
        att_divergence_total = self.reduce_by(
                                                self.attribute_divergence(Y[:,None],\
                                                self.attribute_means[None,:]),\
                                                dim=-1
                                            )
        if self.normalize_:
            net_divergence_total -= phi_kl(X).sum(dim=1)[:,None]
            att_divergence_total -= phi_euclidean( Y ).sum(dim=1)[:,None]
        # print(att_divergence_total,net_divergence_total)
        temp = pi[None,:]*torch.exp(-net_divergence_total -att_divergence_total)
        if self.thresholding:
            max_ = torch.argmax(temp,dim=1)
            tau = torch.zeros((self.N,self.n_clusters))
            tau[self.row_indices,max_] = torch.ones(self.N)
            return tau
        tau = temp/(temp.sum(dim=1)[:,None])
        return tau

    def M_Step(self,X,Y,tau):
        att_means = self.computeAttributeMeans(Y,tau)
        graph_means = self.computeGraphMeans(X,tau)
        return att_means,graph_means
    
    def fit(self,X,Y,Z_init=None):
        """Perform one run of the SBM algorithm with one random initialization.
        Parameters
        ----------
        X : scipy.sparse.csr_matrix, shape=(n,n)
            Matrix to be analyzed
        indices_ones : Non zero indices of the data matrix.
        n : Number of rows in the data matrix.
        """
        old_ll = -torch.inf
        self.N = X.shape[0]
        self.row_indices = torch.arange(self.N)
        if Z_init is None:
            model = BregmanNodeAttributeGraphClustering(n_clusters=self.n_clusters)
            model.initialize( X, Y )
            model.assignInitialLabels( X, Y )  
            self.predicted_memberships = torch.tensor(model.predicted_memberships)
        else:
            self.predicted_memberships = torch.tensor(Z_init)
        #init_labels = self.predicted_memberships
        X = torch.tensor(X)
        Y = torch.tensor(Y)
        self.attribute_means = self.computeAttributeMeans(Y,self.predicted_memberships)
        self.graph_means = self.computeGraphMeans(X,self.predicted_memberships)
        new_tau = tau = self.predicted_memberships
        iter_ = 0 
        while True:
            print(iter_)
            new_tau = self.VE_step(X,Y,tau)
            self.attribute_means,self.graph_means = self.M_Step(X,Y,new_tau)
            if torch.allclose(tau,new_tau) or iter_ > self.n_iters:
                break
            iter_  += 1
            tau = new_tau
        self.predicted_memberships = new_tau
        return self

    def predict(self, X, Y):
        """
        Prediction step.
        Parameters
        ----------
        X : ARRAY
            Input data matrix (n, n) of the node interactions
        Y : ARRAY
            Input data matrix (n, m) of the attributes of the n nodes (each attribute has m features).
        Returns
        -------
        z: Array
            Assigned cluster for each data point (n, )
        """
        return torch.argmax(self.predicted_memberships,dim=1).numpy()
