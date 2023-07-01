#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 16:59:19 2023
@author: maximilien, Felipe Schreiber Fernandes
felipesc@cos.ufrj.br
"""

import numpy as np
import scipy as sp
from sklearn.preprocessing import normalize
from sklearn.base import BaseEstimator, ClusterMixin
from .divergences import *
from .phi import *
from BregmanInitializer.init_cluster import *
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import pairwise_kernels, paired_distances, pairwise_distances
from sklearn.mixture import GaussianMixture
from sklearn.manifold import SpectralEmbedding
from tqdm import tqdm
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import normalize, MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

def fromVectorToMembershipMatrice( z, n_clusters = 2 ):
    if len( set ( z ) ) > n_clusters:
        raise TypeError( 'There is a problem with the number of clusters' )
    n = len( z )
    Z = np.zeros( ( n, n_clusters ) )
    for i in range( n ):
        Z[ i, z[i] ] = 1
    return Z

def frommembershipMatriceToVector( Z ):
    z = np.argmax(Z,axis=1)
    return z

class BregmanKernelClustering( BaseEstimator, ClusterMixin ):
    def __init__( self, n_clusters, 
                 edgeDistribution = "bernoulli",
                 attributeDistribution = "gaussian",
                 weightDistribution = "gaussian",
                 n_iters = 25):
        self.n_clusters = n_clusters
        self.n_iters = n_iters
        self.edgeDistribution = edgeDistribution
        self.attributeDistribution = attributeDistribution
        self.weightDistribution = weightDistribution
        self.edge_divergence = dist_to_divergence_dict[self.edgeDistribution]
        self.weight_divergence = dist_to_divergence_dict[self.weightDistribution]
        self.attribute_divergence = dist_to_divergence_dict[self.attributeDistribution]

    
    def fit( self, A, X, Y):
        """
        Training step.
        Parameters
        ----------
        Y : ARRAY
            Input data matrix (n, m) of n samples and m features.
        X : ARRAY
            Input (n,n,d) tensor with edges. If a edge doesnt exist, is filled with NAN 
        A : ARRAY
            Input (n,n) matrix encoding the adjacency matrix
        Returns
        -------
        TYPE
            Trained model.
        """
        self.N = A.shape[0]
        self.edge_index = np.nonzero(A)
        X_ = X[self.edge_index[0],self.edge_index[1],:]
        
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
        return frommembershipMatriceToVector( self.predicted_memberships )