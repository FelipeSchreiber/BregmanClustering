# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Fri Feb 17 16:59:19 2023
# @author: maximilien, Felipe Schreiber Fernandes
# felipesc@cos.ufrj.br
# """

# import numpy as np
# import scipy as sp
# from sklearn.preprocessing import normalize
# from sklearn.base import BaseEstimator, ClusterMixin
# from .divergences import *
# from .phi import *
# from BregmanInitializer.init_cluster import *
# from sklearn.metrics import accuracy_score
# from sklearn.metrics.pairwise import pairwise_kernels, paired_distances, pairwise_distances
# from sklearn.mixture import GaussianMixture
# from sklearn.manifold import SpectralEmbedding
# from tqdm import tqdm
# from sklearn.cluster import SpectralClustering
# from sklearn.preprocessing import normalize, MinMaxScaler
# import warnings
# warnings.filterwarnings("ignore")

# def fromVectorToMembershipMatrice( z, n_clusters = 2 ):
#     if len( set ( z ) ) > n_clusters:
#         raise TypeError( 'There is a problem with the number of clusters' )
#     n = len( z )
#     Z = np.zeros( ( n, n_clusters ) )
#     for i in range( n ):
#         Z[ i, z[i] ] = 1
#     return Z

# def frommembershipMatriceToVector( Z ):
#     z = np.argmax(Z,axis=1)
#     return z

# class BregmanKernelClustering( BaseEstimator, ClusterMixin ):
#     def __init__( self, n_clusters, 
#                  edgeDistribution = "bernoulli",
#                  attributeDistribution = "gaussian",
#                  weightDistribution = "gaussian",
#                  n_iters = 25):
#         self.n_clusters = n_clusters
#         self.n_iters = n_iters
#         self.edgeDistribution = edgeDistribution
#         self.attributeDistribution = attributeDistribution
#         self.weightDistribution = weightDistribution
#         self.edge_divergence = dist_to_divergence_dict[self.edgeDistribution]
#         self.weight_divergence = dist_to_divergence_dict[self.weightDistribution]
#         self.attribute_divergence = dist_to_divergence_dict[self.attributeDistribution]

#     def make_single_riemannian_metric(self,n_features,gamma=None,att_dist_=None):
#         if gamma is None:
#             gamma = 1.0 / n_features
        
#         if att_dist_ is None:
#             def att_dist_func(row1,row2):
#                 d = euclidean_distances(row1.reshape(1, -1),row2.reshape(1, -1))
#                 return gamma*d
#             att_dist_ = att_dist_func
        
#         def riemannian_metric(row1,row2):
#             net_d = hamming_loss(row1[:N],row2[:N])
#             att_d = att_dist_(row1[N:],row2[N:])
#             distance = np.exp(-(net_d + att_d)) 
#             return distance
        
#         return riemannian_metric
#     def fit( self, A, X, Y):
#         """
#         Training step.
#         Parameters
#         ----------
#         Y : ARRAY
#             Input data matrix (n, m) of n samples and m features.
#         X : ARRAY
#             Input (n,n,d) tensor with edges. If a edge doesnt exist, is filled with NAN 
#         A : ARRAY
#             Input (n,n) matrix encoding the adjacency matrix
#         Returns
#         -------
#         TYPE
#             Trained model.
#         """
#         self.N = A.shape[0]
#         self.edge_index = np.nonzero(A)
#         X_ = X[self.edge_index[0],self.edge_index[1],:]
#         metric = make_riemannian_metric(H.shape[1],X_np.shape[1],att_dist_=hamming_loss)
#             H_and_att = np.hstack((H,X_np))
            
#             SC2 = None
#             if attributes.shape[0] > 1000:
#                 feature_map_nystroem = Nystroem(kernel=metric , random_state=42, n_components=n_components)
#                 data_transformed = feature_map_nystroem.fit_transform(H_and_att)
#                 SC2 = KMeans(n_clusters=K, random_state=0, n_init="auto").fit(data_transformed)

#             else:
#                 SC2 = SpectralClustering(n_clusters=K,\
#                                      affinity=metric,
#                                     assign_labels='discretize',random_state=0).fit(H_and_att)
        
#     def predict(self, X, Y):
#         """
#         Prediction step.
#         Parameters
#         ----------
#         X : ARRAY
#             Input data matrix (n, n) of the node interactions
#         Y : ARRAY
#             Input data matrix (n, m) of the attributes of the n nodes (each attribute has m features).
#         Returns
#         -------
#         z: Array
#             Assigned cluster for each data point (n, )
#         """
#         return frommembershipMatriceToVector( self.predicted_memberships )