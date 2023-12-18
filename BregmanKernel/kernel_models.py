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
from sklearn.manifold import SpectralEmbedding
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import csr_array 
import warnings
warnings.filterwarnings("ignore")

class BregmanKernelClustering( BaseEstimator, ClusterMixin ):
    def __init__( self, n_clusters, 
                 edgeSimilarity = "jaccard",
                 attributeDistribution = "gaussian",
                 weightDistribution = "gaussian",
                 single_metric=False,
                 n_components=None,
                 use_nystrom=False
                ):
        self.n_clusters = n_clusters
        self.edgeSimilarity = edgeSimilarity
        self.attributeDistribution = attributeDistribution
        self.weightDistribution = weightDistribution
        self.edge_sim = dist_to_divergence_dict[self.edgeSimilarity]
        self.weight_divergence = dist_to_divergence_dict[self.weightDistribution]
        self.attribute_divergence = dist_to_divergence_dict[self.attributeDistribution]
        self.single_metric = single_metric
        self.n_components = n_components
        if n_components is None:
            self.n_components = n_clusters
        self.use_nystrom = use_nystrom

    def make_single_riemannian_metric(self,att_feats,net_feats,gamma=None):
        att_sim_func = self.attribute_divergence
        net_sim_func = self.edge_sim

        def riemannian_metric(row1,row2):
            net_d = net_sim_func(row1[:net_feats],row2[:net_feats])
            att_d = att_sim_func(row1[-att_feats:],row2[-att_feats:])
            simmilarity = (net_d+att_d)/2
            return simmilarity
        
        return riemannian_metric
    
    def spectralEmbedding(self, X , metric):
        X = pairwise_kernels(X,metric=metric)
        U = SpectralEmbedding(n_components=self.n_clusters,\
								affinity="precomputed")\
								.fit_transform(X)
        return U
    
    ## only compute the simmilarity for the connected nodes
    def fit_efficient(self, edge_index, Y):
        w = np.zeros(len(edge_index[0]))
        self.N = Y.shape[0]
        for i, (u,v) in enumerate(zip(edge_index[0],edge_index[1])):
            w[i] = self.attribute_divergence(Y[u,:],Y[v,:])
        sim_matrix = csr_array((w, (edge_index[0], edge_index[1])), \
                               shape=(self.N, self.N))
        data_transformed = SpectralEmbedding(n_components=self.n_clusters,\
								affinity="precomputed")\
								.fit_transform(sim_matrix.toarray())
        data_transformed = MinMaxScaler().fit_transform(data_transformed)
        self.model = KMeans(n_clusters=self.n_clusters,\
                                random_state=0,\
                                n_init="auto")\
                                .fit(data_transformed)
        self.labels_ = self.model.labels_

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
        # X_ = X[self.edge_index[0],self.edge_index[1],:]
        H = np.hstack((A,A.T))
        H_and_att = np.hstack((H,Y))
        metric = self.make_single_riemannian_metric(Y.shape[1],H.shape[1],gamma=None)            
        
        if self.use_nystrom:
                feature_map_nystroem = Nystroem(kernel=metric,\
                                                random_state=42,\
                                                n_components=self.n_components)
                
                data_transformed = feature_map_nystroem.fit_transform(H_and_att)
                self.model = KMeans(n_clusters=self.n_clusters,\
                                random_state=0,\
                                n_init="auto")\
                                .fit(data_transformed)
        else:
            data_transformed = None
            att_metric = self.attribute_divergence
            net_metric = self.edge_sim
            net_sim_matrix = None

            if self.edgeSimilarity == "raw":
                net_sim_matrix = A
            else:
                net_sim_matrix = pairwise_kernels(H, metric=net_metric, filter_params=True)
            
            att_sim_matrix = pairwise_kernels(Y, metric=att_metric, filter_params=True)
            
            if self.single_metric:
                single_sim_matrix = (att_sim_matrix + net_sim_matrix)/2
                data_transformed = SpectralEmbedding(n_components=self.n_clusters,\
								affinity="precomputed")\
								.fit_transform(single_sim_matrix)
                # self.model = SpectralClustering(n_clusters=self.n_clusters,\
                #                      affinity=metric,
                #                     assign_labels='discretize',random_state=0).fit(H_and_att)

            else:
                att_transformed = SpectralEmbedding(n_components=self.n_clusters,\
								affinity="precomputed")\
								.fit_transform(att_sim_matrix)
                net_transformed = SpectralEmbedding(n_components=self.n_clusters,\
								affinity="precomputed")\
								.fit_transform(net_sim_matrix)
                data_transformed = np.hstack([net_transformed,att_transformed])
            data_transformed = MinMaxScaler().fit_transform(data_transformed)
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