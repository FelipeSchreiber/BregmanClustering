#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 16:59:19 2023
@author: maximilien, Felipe Schreiber Fernandes
felipesc@cos.ufrj.br
"""

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from .kernel_divergences import *
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.kernel_approximation import Nystroem
import warnings
warnings.filterwarnings("ignore")

class BregmanKernelClustering( BaseEstimator, ClusterMixin ):
    def __init__( self, n_clusters, 
                 edgeDistribution = "bernoulli",
                 attributeDistribution = "gaussian",
                 weightDistribution = "gaussian",
                 n_iters = 25, full_kernel=False, n_components=None):
        self.n_clusters = n_clusters
        self.n_iters = n_iters
        self.edgeDistribution = edgeDistribution
        self.attributeDistribution = attributeDistribution
        self.weightDistribution = weightDistribution
        self.edge_divergence = dist_to_divergence_dict[self.edgeDistribution]
        self.weight_divergence = dist_to_divergence_dict[self.weightDistribution]
        self.attribute_divergence = dist_to_divergence_dict[self.attributeDistribution]
        self.full_kernel = full_kernel
        self.n_components = n_components
        if n_components is None:
            self.n_components = n_clusters

    def make_single_riemannian_metric(self,att_feats,net_feats,gamma=None):
        if gamma is None:
            gamma = 1.0 / att_feats

        def riemannian_metric(row1,row2):
            net_d = self.edge_divergence(row1[:net_feats],row2[:net_feats])/(net_feats)
            att_d = gamma*self.attribute_divergence(row1[-att_feats:],row2[-att_feats:])
            distance = np.exp(-(net_d + att_d)) 
            return distance
        
        return riemannian_metric
    
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
        self.model = None
        X_ = X[self.edge_index[0],self.edge_index[1],:]
        H = np.hstack((A,A.T))
        H_and_att = np.hstack((H,Y))
        metric = self.make_single_riemannian_metric(Y.shape[1],H.shape[1],gamma=None)
            
        if self.full_kernel:
            self.model = SpectralClustering(n_clusters=self.n_clusters,\
                                     affinity=metric,
                                    assign_labels='discretize',random_state=0).fit(H_and_att)
        else:
            feature_map_nystroem = Nystroem(kernel=metric,\
                                            random_state=42,\
                                            n_components=self.n_components)
            
            data_transformed = feature_map_nystroem.fit_transform(H_and_att)
            self.model = KMeans(n_clusters=self.n_clusters,\
                                random_state=0,\
                                n_init="auto")\
                                .fit(data_transformed)
        self.labels_ = self.model.labels_
        
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
        return self.model.labels_