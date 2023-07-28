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
from joblib import effective_n_jobs
from sklearn.utils.parallel import Parallel, delayed
from sklearn.base import BaseEstimator, ClusterMixin
from .divergences import *
# from .phi import *
from BregmanInitializer.init_cluster import *
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import pairwise_kernels, paired_distances, pairwise_distances
from sklearn.mixture import GaussianMixture
from sklearn.manifold import SpectralEmbedding
from tqdm import tqdm
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import normalize, MinMaxScaler
import os
import tempfile
from BregmanTests.utils import gen_even_slices
import warnings
warnings.filterwarnings("ignore")

def singleAssignmentContainer(self,X_, H, nodes):
    for node in nodes:
        self.Z[ node ] = self.singleNodeAssignment( X_, H, node )

def singlecomputeTotalDivContainer(self,X, H, nodes):
    for node in nodes:
        self.Ztilde[node,:] = self.computeTotalDiv(node,X,self.predicted_memberships,H)
    
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

class BregmanGraphPartitioning( BaseEstimator, ClusterMixin ):
    
    def __init__( self, n_clusters, divergence = logistic_loss, 
                 n_iters = 25, initializer="spectralClustering", init_iters=100 ):
        """
        Bregman Hard Clustering Algorithm
        Parameters
        ----------
        n_clusters : INT
            Number of clustes.
        divergence : function
            Pairwise divergence function. The default is euclidean_distance.
        n_iters : INT, optional
            Number of clustering iterations. The default is 1000.
        has_cov : BOOL, optional
            Specifies if the divergence requires a covariance matrix. The default is False.
        initializer : STR, optional
            Specifies if the centroids are initialized at random "rand", K-Means++ "kmeans++", or a pretrained K-Means model "pretrained". The default is "rand".
        init_iters : INT, optional
            Number of iterations for K-Means++. The default is 100.
        pretrainer : MODEL, optional
            Pretrained K-Means model to use as pretrainer.
        Returns
        -------
        None.
        """
        self.n_clusters = n_clusters
        self.divergence = divergence
        self.n_iters = n_iters
        self.initializer = initializer
        self.init_iters = init_iters

    def fit(self, X):
        """
        Training step.
        Parameters
        ----------
        X : ARRAY
            Input data matrix (n, m) of n samples and m features.
        Returns
        -------
        TYPE
            Trained model.
        """
        self.create_params( X )
        convergence = True
        iteration = 0
        while convergence:
            new_memberships = self.assignments( X )
            new_means = self.reestimateMeans( X, new_memberships )
            iteration += 1
            if np.array_equal( new_memberships, self.predicted_memberships) or np.linalg.norm( new_means - self.means ) < 0.05 or iteration >= self.n_iters:
                convergence = False
                
            self.predicted_memberships = new_memberships
            self.means = new_means
        print( 'number of iterations : ', iteration )
        return self
    
    def create_params(self, X):
        if self.initializer=="rand":
            self.params = self.init_params( X )
        elif self.initializer=="spectralClustering":
            self.predicted_memberships = self.spectralClustering( X )
            self.means = self.reestimateMeans( X, self.predicted_memberships )
        else:
            raise TypeError( 'The initializer provided is not correct' )

    def init_params(self, X):
        idx = np.arange(X.shape[0])
        np.random.shuffle(idx)
        # TODO: to implement
        return X[ idx[ :self.n_clusters ] ]
    
    def spectralClustering( self, X ):
        sc = SpectralClustering( n_clusters = self.n_clusters, affinity = 'precomputed', assign_labels = 'kmeans' )
        z_init = sc.fit_predict( X )
        Z = fromVectorToMembershipMatrice( z_init )
        self.predicted_memberships = Z
        return self.predicted_memberships
    
    def reestimateMeans( self, X, Z ):
        normalisation = np.linalg.inv ( Z.T @ Z )
        return normalisation @ Z.T @ X @ Z @ normalisation

    def assignments( self, X ):
        z = np.zeros( X.shape[ 0 ], dtype = int )
        for node in tqdm( range( len( z ) ) ):
            z[ node ] = self.singleNodeAssignment( X, node )
        return fromVectorToMembershipMatrice( z )
    
    def singleNodeAssignment( self, X, node ):
        L = np.zeros( self.n_clusters )
        for k in range( self.n_clusters ):
            Ztilde = self.predicted_memberships.copy()
            Ztilde[ node, : ] = 0
            Ztilde[ node, k ] = 1
            M = Ztilde @ self.means @ Ztilde.T
            L[ k ] = logistic_loss( X, M )
            #print( L[k] )
        return np.argmin( L )
    
    def predict(self, X):
        """
        Prediction step.
        Parameters
        ----------
        X : ARRAY
            Input data matrix (n, m) of n samples and m features.
        Returns
        -------
        y: Array
            Assigned cluster for each data point (n, )
        """
        return frommembershipMatriceToVector( self.predicted_memberships )

    


class BregmanHard(BaseEstimator, ClusterMixin):
    #This is a copy paste from the code of the original paper on Bregman Clustering, found
    # https://github.com/juselara1/bregclus

    def __init__(self, n_clusters, divergence=euclidean_distance, n_iters=1000, has_cov=False,
                 initializer="rand", init_iters=100, pretrainer=None):
        """
        Bregman Hard Clustering Algorithm
        Parameters
        ----------
        n_clusters : INT
            Number of clustes.
        divergence : function
            Pairwise divergence function. The default is euclidean_distance.
        n_iters : INT, optional
            Number of clustering iterations. The default is 1000.
        has_cov : BOOL, optional
            Specifies if the divergence requires a covariance matrix. The default is False.
        initializer : STR, optional
            Specifies if the centroids are initialized at random "rand", K-Means++ "kmeans++", or a pretrained K-Means model "pretrained". The default is "rand".
        init_iters : INT, optional
            Number of iterations for K-Means++. The default is 100.
        pretrainer : MODEL, optional
            Pretrained K-Means model to use as pretrainer.
        Returns
        -------
        None.
        """
        self.n_clusters = n_clusters
        self.divergence = divergence
        self.n_iters = n_iters
        self.has_cov = has_cov
        self.initializer = initializer
        self.init_iters = init_iters
        self.pretrainer = pretrainer

    def fit(self, X):
        """
        Training step.
        Parameters
        ----------
        X : ARRAY
            Input data matrix (n, m) of n samples and m features.
        Returns
        -------
        TYPE
            Trained model.
        """
        self.create_params(X)
        for _ in range(self.n_iters):
            H = self.assignments(X)
            self.reestimate(X, H)
        return self

    def create_params(self, X):
        if self.initializer=="rand":
            self.params = self.init_params(X)
        elif self.initializer=="kmeans++":
            self.params = self.kmeanspp(X)
        else:
            self.params = self.use_pretrainer(X)
        if self.has_cov:
            self.cov = self.init_cov(X)

    def init_params(self, X):
        idx = np.arange(X.shape[0])
        np.random.shuffle(idx)
        return X[idx[:self.n_clusters]]
    
    def kmeanspp(self, X):
        idx = np.arange(X.shape[0])
        np.random.shuffle(idx)
        selected = idx[:self.n_clusters]
        init_vals = X[idx[:self.n_clusters]]

        for i in range(self.init_iters):
            clus_sim = euclidean_distance(init_vals, init_vals)
            np.fill_diagonal(clus_sim, np.inf)

            candidate = X[np.random.randint(X.shape[0])].reshape(1, -1)
            candidate_sims = euclidean_distance(candidate, init_vals).flatten()
            closest_sim = candidate_sims.min()
            closest = candidate_sims.argmin()
            if closest_sim>clus_sim.min():
                replace_candidates_idx = np.array(np.unravel_index(clus_sim.argmin(), clus_sim.shape))
                replace_candidates = init_vals[replace_candidates_idx, :]

                closest_sim = euclidean_distance(candidate, replace_candidates).flatten()
                replace = np.argmin(closest_sim)
                init_vals[replace_candidates_idx[replace]] = candidate
            else:
                candidate_sims[candidate_sims.argmin()] = np.inf
                second_closest = candidate_sims.argmin()
                if candidate_sims[second_closest] > clus_sim[closest].min():
                    init_vals[closest] = candidate
        return init_vals

    def use_pretrainer(self, X):
        self.params = self.pretrainer.cluster_centers_

    def init_cov(self, X):
        dists = euclidean_distance(X, self.params)
        H = np.argmin(dists, axis=1)
        covs = []
        for k in range(self.n_clusters):
            #covs.append(np.expand_dims(np.cov(X[H==k].T), axis=0))
            covs.append(np.expand_dims(np.eye(X.shape[1]), axis=0))
        return np.concatenate(covs, axis=0)

    def assignments(self, X):
        if self.has_cov:
            H = self.divergence(X, self.params, self.cov)
        else:
            H = self.divergence(X, self.params)
        return np.argmin(H, axis=1)

    def reestimate(self, X, H):
        for k in range(self.n_clusters):
            X_k = X[H==k]
            if X_k.shape[0] != 0:
                self.params[k] = np.mean(X_k, axis=0)
                if self.has_cov:
                    X_mk = X-self.params[k]
                    self.cov[k] = np.einsum("ij,ik->jk", X_mk, X_mk)/X_k.shape[0]

    def predict(self, X):
        """
        Prediction step.
        Parameters
        ----------
        X : ARRAY
            Input data matrix (n, m) of n samples and m features.
        Returns
        -------
        y: Array
            Assigned cluster for each data point (n, )
        """
        return self.assignments(X)

    
class BregmanNodeAttributeGraphClustering( BaseEstimator, ClusterMixin ):
    def __init__( self, n_clusters, 
                 edgeDistribution = "bernoulli",
                 attributeDistribution = "gaussian",
                 initializer = 'chernoff', 
                 graph_initializer = "spectralClustering", attribute_initializer = 'GMM', 
                 n_iters = 25, init_iters=100 ):
        """
        Bregman Hard Clustering Algorithm for partitioning graph with node attributes
        Parameters
        ----------
        n_clusters : INT
            Number of clustes.
        edge_divergence, attribute_divergence : function
            Pairwise divergence function. The default is euclidean_distance.
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
        self.n_iters = n_iters
        self.initializer = initializer
        self.graph_initializer = graph_initializer
        self.attribute_initializer = attribute_initializer
        self.init_iters = init_iters
        ## Variable that stores which initialization was chosen
        self.graph_init = False
        self.edgeDistribution = edgeDistribution
        self.attributeDistribution = attributeDistribution
        self.edge_divergence = dist_to_divergence_dict[self.edgeDistribution]
        self.attribute_divergence = dist_to_divergence_dict[self.attributeDistribution]
        self.edge_index = None 

    def fit( self, X, Y, Z_init=None ):
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
        self.N = X.shape[0]
        if Z_init is None:
            self.initialize( X, Y )
            self.assignInitialLabels( X, Y )
        else:
            self.predicted_memberships = Z_init
        self.attribute_means = self.computeAttributeMeans(Y,self.predicted_memberships)
        self.graph_means = self.computeGraphMeans(X,self.predicted_memberships)
        convergence = True
        iteration = 0
        while convergence:
            new_memberships = self.assignments( X, Y )

            self.attribute_means = self.computeAttributeMeans( Y, new_memberships )
            self.graph_means = self.computeGraphMeans( X, new_memberships )
            
            iteration += 1
            if accuracy_score( frommembershipMatriceToVector(new_memberships), frommembershipMatriceToVector(self.predicted_memberships) ) < 0.02 or iteration >= self.n_iters:
                convergence = False
            self.predicted_memberships = new_memberships
        print( 'number of iterations : ', iteration)
        return self
    
    def initialize( self, X, Y ):
        if self.attribute_initializer == 'GMM':
            model = GaussianMixture(n_components=self.n_clusters)
            model.fit( Y )
            self.memberships_from_attributes = fromVectorToMembershipMatrice( model.predict( Y ), n_clusters = self.n_clusters )
            self.attribute_model_init = model
        else:
            raise TypeError( 'The initializer provided for the attributes is not correct' )
            
        if self.graph_initializer == 'spectralClustering':
            U = self.spectralEmbedding(X)
            model = GaussianMixture(n_components=self.n_clusters)
            model.fit(U)
            self.memberships_from_graph = fromVectorToMembershipMatrice( model.predict( U ),\
                                                                            n_clusters = self.n_clusters )
            self.graph_model_init = model
            #self.graph_means = self.computeGraphMeans(X,self.memberships_from_graph)
        else:
            raise TypeError( 'The initializer provided for the graph is not correct' )
    
    def AIC_initializer(self,X,Y):
        U = self.spectralEmbedding(X)
        net_null_model = GaussianMixture(n_components=1,).fit(U)
        null_net = net_null_model.aic(U)
        net_model = self.graph_model_init
        fitted_net = net_model.aic(U)
        AIC_graph = fitted_net - null_net

        att_null_model = GaussianMixture(n_components=1).fit(Y)
        null_attributes = att_null_model.aic(Y)
        att_model = self.attribute_model_init
        fitted_attributes = att_model.aic(Y)
        AIC_attribute = fitted_attributes - null_attributes
        
        if AIC_graph < AIC_attribute:
            self.predicted_memberships = self.memberships_from_graph
            self.graph_init = True
            print( 'Initialisation chosen from the graph')
        else:
            self.predicted_memberships = self.memberships_from_attributes
            self.graph_init = False
            print( 'Initialisation chosen from the attributes' )
        return self

    def chernoff_initializer(self,X,Y):
        n = Y.shape[0]
        if self.graphChernoffDivergence( X, self.memberships_from_graph ) > \
                self.attributeChernoffDivergence( Y, self.memberships_from_attributes ) / n:
            self.predicted_memberships = self.memberships_from_graph
            self.graph_init = True
            print( 'Initialisation chosen from the graph')
        else:
            self.predicted_memberships = self.memberships_from_attributes
            self.graph_init = False
            print( 'Initialisation chosen from the attributes' )         
        return self
    
    def assignInitialLabels( self, X, Y ):
        if self.initializer == 'random':
            z =  np.random.randint( 0, self.n_clusters, size = X.shape[0] )
            self.predicted_memberships = fromVectorToMembershipMatrice( z, n_clusters = self.n_clusters )
        
        elif self.initializer == "AIC":
            self.AIC_initializer(X,Y)
        
        ## Chernoff divergence
        elif self.initializer == "chernoff":
            self.chernoff_initializer(X,Y)
        
    def spectralEmbedding(self, X ):
        if (X<0).any():
            X = pairwise_kernels(X,metric='rbf')
        U = SpectralEmbedding(n_components=self.n_clusters,\
								affinity="precomputed")\
								.fit_transform(X)
        return U
    
    def computeAttributeMeans( self, Y, Z ):
        attribute_means = np.dot(Z.T, Y)/(Z.sum(axis=0) + 10 * np.finfo(Z.dtype).eps)[:, np.newaxis]
        return attribute_means
    
    def computeGraphMeans( self, A, Z ):
        normalisation = np.linalg.pinv ( Z.T @ Z )
        return normalisation @ Z.T @ A @ Z @ normalisation
    
    def chernoffDivergence( self, a, b, t, distribution = 'bernoulli' ):
        if distribution.lower() == 'bernoulli':
            return (1-t) * a + t *b - a**t * b**(1-t)
    
    def graphChernoffDivergence( self, X, Z ):
        graph_means = self.computeGraphMeans( X , Z )
        n = Z.shape[ 0 ]
        pi = np.zeros( self.n_clusters )
        for c in range( self.n_clusters ):
            cluster_c = [ i for i in range( n ) if Z[i,c] == 1 ]
            pi[ c ] = len(cluster_c) / n
            
        if self.edgeDistribution == 'bernoulli':
            res = 10000
            for a in range( self.n_clusters ):
                for b in range( a ):
                    div = lambda t : - (1-t) * np.sum(  [ pi[c] * self.chernoffDivergence( graph_means[a,c], graph_means[b,c], t ) for c in range( self.n_clusters ) ] )
                    minDiv = sp.optimize.minimize_scalar( div, bounds = (0,1), method ='bounded' )
                    if - minDiv['fun'] < res:
                        res = - minDiv['fun']
        return res
    
    def attributeChernoffDivergence( self, Y, Z ):
        res = 10000
        attribute_means = self.computeAttributeMeans( Y, Z )
        for a in range( self.n_clusters ):
            for b in range( a ):
                div = lambda t : - t * (1-t)/2 * np.linalg.norm( attribute_means[a] - attribute_means[b] )
                minDiv = sp.optimize.minimize_scalar( div, bounds = (0,1), method ='bounded' )
                if - minDiv['fun'] < res:
                    res = - minDiv['fun']

        return res
    
    def likelihood( self, X, Y, Z ):
        graphLikelihood = self.likelihoodGraph(X,Z)
        attributeLikelihood = self.likelihoodAttributes(Y,Z)
        return graphLikelihood + attributeLikelihood
    
    def likelihoodGraph(self, X, Z):
        graph_mean = self.computeGraphMeans(X,Z)
        return 1/2 * np.sum( self.edge_divergence( X, Z @ graph_mean @ Z.T ) )
    
    def likelihoodAttributes( self, Y, Z):
        M = self.computeAttributeMeans(Y,Z)
        total = np.sum( paired_distances(Y,Z@M) )
        return total 
    
    def assignments( self, X, Y ):
        z = np.zeros( X.shape[ 0 ], dtype = int )
        H = pairwise_distances(Y,self.attribute_means,metric=self.attribute_divergence)
        for node in range( len( z ) ):
            z[ node ] = self.singleNodeAssignment( X, H, node )
        return fromVectorToMembershipMatrice( z, n_clusters = self.n_clusters )        
    
    def singleNodeAssignment( self, X, H, node ):
        L = np.zeros( self.n_clusters )
        for q in range( self.n_clusters ):
            Ztilde = self.predicted_memberships.copy()
            Ztilde[ node, : ] = 0
            Ztilde[ node, q ] = 1
            M = Ztilde @ self.graph_means @ Ztilde.T
            """
            X has shape n x n x d
            E has shape k x k x d
            
            the edge divergence computes the difference between node i (from community q) edges and the means
            given node j belongs to community l:
            
            sum_j phi_edge(e_ij, E[q,l,:])  
            """
            att_div = H[node,q]
            graph_div = self.edge_divergence( X[node,:], M[node,:] )
            L[ q ] = att_div + 0.5*graph_div
        return np.argmin( L )
    
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

## reduce_by and divergence_precomputed are for compatibility only with torch models
class BregmanNodeEdgeAttributeGraphClustering( BaseEstimator, ClusterMixin ):
    def __init__( self, n_clusters, 
                 edgeDistribution = "bernoulli",
                 attributeDistribution = "gaussian",
                 weightDistribution = "gaussian",
                 initializer = 'chernoff', 
                 graph_initializer = "spectralClustering", attribute_initializer = 'GMM', 
                 n_iters = 25, init_iters=100,
                 reduce_by=None,
                 divergence_precomputed=True,
                 use_random_init=False):
        """
        Bregman Hard Clustering Algorithm for partitioning graph with node attributes
        Parameters
        ----------
        n_clusters : INT
            Number of clustes.
        edge_divergence, attribute_divergence : function
            Pairwise divergence function. The default is euclidean_distance.
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
        self.n_iters = n_iters
        self.initializer = initializer
        self.graph_initializer = graph_initializer
        self.attribute_initializer = attribute_initializer
        self.init_iters = init_iters
        ## Variable that stores which initialization was chosen
        self.graph_init = False
        self.edgeDistribution = edgeDistribution
        self.attributeDistribution = attributeDistribution
        self.weightDistribution = weightDistribution
        self.edge_divergence = dist_to_divergence_dict[self.edgeDistribution]
        self.weight_divergence = dist_to_divergence_dict[self.weightDistribution]
        self.attribute_divergence = dist_to_divergence_dict[self.attributeDistribution]
        self.edge_index = None 
        self.use_random_init = use_random_init
        ## strategy denotes how the algorithm will handle the weight means when
        ## p_{a,b} = 0

        ## strategy = 0 is simply ignore the communities with p_{a,b} = 0
        ## strategy = 1 is to consider the means equal to zero
        ## strategy = 2 is to consider the means equal to the global mean
        ## strategy = 3 ignore only the weight divergence
    def fit( self, A, X, Y, Z_init=None):
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
        if Z_init is None:
            self.initialize( A, X, Y)
            self.assignInitialLabels( X, Y )
        else:
            self.predicted_memberships = Z_init
        #init_labels = self.predicted_memberships
        self.attribute_means = self.computeAttributeMeans(Y,self.predicted_memberships)
        self.edge_means = self.computeEdgeMeans(A,self.predicted_memberships)
        self.weight_means = self.computeWeightMeans(A, X, self.predicted_memberships)
        convergence = True
        iteration = 0
        while convergence:
            new_memberships = self.assignments( A, X, Y )

            self.attribute_means = self.computeAttributeMeans( Y, new_memberships )
            self.edge_means = self.computeEdgeMeans( A, new_memberships )
            self.weight_means = self.computeWeightMeans(A, X, new_memberships)
            
            iteration += 1
            if accuracy_score( frommembershipMatriceToVector(new_memberships), frommembershipMatriceToVector(self.predicted_memberships) ) < 0.02 or iteration >= self.n_iters:
                convergence = False
            self.predicted_memberships = new_memberships
        return self
    
    def initialize( self, A, X, Y ):
        model = BregmanInitializer(self.n_clusters,initializer=self.initializer,
                                    edgeDistribution = self.edgeDistribution,
                                    attributeDistribution = self.attributeDistribution,
                                    weightDistribution = self.weightDistribution)
        if self.edge_index is None:
            self.edge_index = np.nonzero(A)
        Z_init = None
        if self.use_random_init == True:
            Z_init = fromVectorToMembershipMatrice(np.random.randint(self.n_clusters,size=self.N),
                                                                        self.n_clusters)
        # model.initialize( X, Y , self.edge_index, Z_init=Z_init)
        model.initialize(  A, X, Y, Z_init=Z_init)
        self.predicted_memberships = model.predicted_memberships
        self.memberships_from_graph = model.memberships_from_graph
        self.memberships_from_attributes = model.memberships_from_attributes
        self.graph_init = model.graph_init

    def assignInitialLabels( self, X, Y ):
        return self
        
    def spectralEmbedding(self, X ):
        if (X<0).any():
            X = pairwise_kernels(X,metric='rbf')
        U = SpectralEmbedding(n_components=self.n_clusters,\
								affinity="precomputed")\
								.fit_transform(X)
        return U
    
    def computeAttributeMeans( self, Y, Z ):
        attribute_means = np.dot(Z.T, Y)/(Z.sum(axis=0) + 10 * np.finfo(Z.dtype).eps)[:, np.newaxis]
        return attribute_means
    
    def computeEdgeMeans( self, A, Z ):
        normalisation = np.linalg.pinv ( Z.T @ Z )
        return normalisation @ Z.T @ A @ Z @ normalisation
    
    def computeWeightMeans( self, A, X, Z):
        weights = np.tensordot(Z, Z, axes=((), ()))
        """
        weights[i,q,j,l] = tau[i,q]*tau[j,l]
        desired output:
        weights[q,l,i,j] = tau[i,q]*tau[j,l]
        """
        weights = np.transpose(weights,(1,3,0,2))[:,:,self.edge_index[0],self.edge_index[1]]
        X_ = X[self.edge_index[0],self.edge_index[1],:]
        """
        X is a |E| x d tensor
        weights is a k x k x |E|
        desired output: 
        out[q,l,d] = sum_e X[e,d] * weights[q,l,e]
        """
        weight_means = np.tensordot( weights,\
                                    X_,\
                                    axes=[(2),(0)] )/(np.sum(weights,axis=-1)[:,:,np.newaxis]) 

        return weight_means
    
    def likelihood( self, X, Y, Z ):
        graphLikelihood = self.likelihoodGraph(X,Z)
        attributeLikelihood = self.likelihoodAttributes(Y,Z)
        return graphLikelihood + attributeLikelihood
    
    def likelihoodGraph(self, X, Z):
        graph_mean = self.computeEdgeMeans(X,Z)
        return 1/2 * np.sum( self.edge_divergence( X, Z @ graph_mean @ Z.T ) )
    
    def likelihoodAttributes( self, Y, Z):
        M = self.computeAttributeMeans(Y,Z)
        total = np.sum( paired_distances(Y,Z@M) )
        return total 
    
    def assignments( self, A, X, Y ):
        z = np.zeros( X.shape[ 0 ], dtype = int )
        H = pairwise_distances(Y,self.attribute_means,metric=self.attribute_divergence)
        for node in range( len( z ) ):
            z[ node ] = self.singleNodeAssignment( A, X, H, node )
        return fromVectorToMembershipMatrice( z, n_clusters = self.n_clusters )        
    
    def singleNodeAssignment( self, A, X, H, node ):
        L = np.zeros( self.n_clusters )
        edge_indices_in = np.argwhere(self.edge_index[1] == node).flatten()
        v_indices_in = self.edge_index[0][edge_indices_in]
        edge_indices_out = np.argwhere(self.edge_index[0] == node).flatten()
        v_indices_out = self.edge_index[1][edge_indices_out]
        for q in range( self.n_clusters ):
            Ztilde = self.predicted_memberships.copy()
            Ztilde[ node, : ] = 0
            Ztilde[ node, q ] = 1
            z_t = np.argmax(Ztilde,axis=1)
            M_out = self.edge_means[np.repeat(q, self.N),z_t]
            M_in = self.edge_means[z_t,np.repeat(q, self.N)]
            E = self.weight_means
            """
            X has shape n x n x d
            E has shape k x k x d
            
            the edge divergence computes the difference between node i (from community q) edges and the means
            given node j belongs to community l:
            
            sum_j phi_edge(e_ij, E[q,l,:])  
            """
            att_div = H[node,q]
            edge_div = self.edge_divergence( A[node,:], M_out ) \
                            + self.edge_divergence( A[:,node], M_in ) \
                            - 2*self.edge_divergence(A[node,node],M_in[q])
                
            ## compute weight divergence
            weight_div = 0
            contains_nan = False
            E_ = E[q,z_t[v_indices_out],:]
            if np.isnan(E_).any():
                weight_div = np.inf
                contains_nan = True
            not_nan_idx = np.argwhere(~np.isnan(E_).any(axis=1)).flatten()
            E_without_nan = E_[not_nan_idx,:]
            if (len(v_indices_out) > 0) and (len(not_nan_idx) > 0) and (not contains_nan):
                weight_div += np.sum( paired_distances(X[node,v_indices_out,:][not_nan_idx,:],\
                                                            E_without_nan,\
                                                            metric=self.weight_divergence))
                
            ## same as before, but now for edges coming in node
            E_ = E[z_t[v_indices_in],q,:]
            if np.isnan(E_).any():
                weight_div = np.inf
                contains_nan = True
            not_nan_idx = np.argwhere(~np.isnan(E_).any(axis=1)).flatten()
            E_without_nan = E_[not_nan_idx,:]
            if (len(v_indices_in) > 0) and (len(not_nan_idx) > 0) and (not contains_nan):
                weight_div += np.sum( paired_distances(X[v_indices_in,node,:][not_nan_idx,:],\
                                                            E_without_nan,\
                                                            metric=self.weight_divergence))
            L[ q ] = att_div + (weight_div + edge_div)
        return np.argmin( L )
    
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

class BregmanNodeEdgeAttributeGraphClusteringEfficient( BaseEstimator, ClusterMixin ):
    def __init__( self, n_clusters, 
                 edgeDistribution = "bernoulli",
                 attributeDistribution = "gaussian",
                 weightDistribution = "gaussian",
                 initializer = 'chernoff', 
                 graph_initializer = "spectralClustering", attribute_initializer = 'GMM', 
                 n_iters = 25, init_iters=100,
                 reduce_by=None,
                 divergence_precomputed=True,
                 use_random_init=False):
        """
        Bregman Hard Clustering Algorithm for partitioning graph with node attributes
        Parameters
        ----------
        n_clusters : INT
            Number of clustes.
        edge_divergence, attribute_divergence : function
            Pairwise divergence function. The default is euclidean_distance.
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
        self.n_iters = n_iters
        self.initializer = initializer
        self.graph_initializer = graph_initializer
        self.attribute_initializer = attribute_initializer
        self.init_iters = init_iters
        ## Variable that stores which initialization was chosen
        self.graph_init = False
        self.edgeDistribution = edgeDistribution
        self.attributeDistribution = attributeDistribution
        self.weightDistribution = weightDistribution
        self.edge_divergence = dist_to_divergence_dict[self.edgeDistribution]
        self.weight_divergence = dist_to_divergence_dict[self.weightDistribution]
        self.attribute_divergence = dist_to_divergence_dict[self.attributeDistribution]
        self.edge_index = None 
        self.use_random_init = use_random_init
        self.n_jobs = effective_n_jobs(-1)

    def precompute_edge_divergences(self):
        self.precomputed_edge_div = pairwise_distances(np.array([0,1]).reshape(-1,1),\
                                             self.edge_means.reshape(-1,1),\
                                             metric=self.edge_divergence)\
                                            .reshape(
                                                        (2,\
                                                        self.n_clusters,self.n_clusters
                                                        )
                                                    )    
    def index_to_mask(self,v_idx):
        all_indices = np.zeros(self.N, dtype=bool)
        all_indices[v_idx] = True
        return all_indices
    
    def fit( self, edge_index, X, Y, Z_init=None):
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
        self.N = Y.shape[0]
        self.edge_index = edge_index
        X_ = None
        ## CASE X is N x N x 1: pass to |E| x 1 
        if X.shape[0] == X.shape[1]:
            X_ = X[self.edge_index[0],self.edge_index[1],:]
        else:           
            X_ = X
        if Z_init is None:
            self.initialize( self.edge_index, X_, Y)
            self.assignInitialLabels( X_, Y )
        else:
            self.predicted_memberships = Z_init
        path = tempfile.mkdtemp()
        Zpath = os.path.join(path,'z.mmap')
        self.Z = np.memmap(Zpath, dtype=int, shape=self.N, mode='w+')
        #init_labels = self.predicted_memberships
        self.attribute_means = self.computeAttributeMeans(Y,self.predicted_memberships)
        self.edge_means = self.computeEdgeMeans(self.predicted_memberships)
        self.weight_means = self.computeWeightMeans( X_, self.predicted_memberships)
        self.precompute_edge_divergences()
        convergence = True
        iteration = 0
        while convergence:
            new_memberships = self.assignments_joblib( X_, Y )

            self.attribute_means = self.computeAttributeMeans( Y, new_memberships )
            self.edge_means = self.computeEdgeMeans(new_memberships)
            self.weight_means = self.computeWeightMeans( X_, new_memberships)
            self.precompute_edge_divergences()    
            iteration += 1
            if accuracy_score( frommembershipMatriceToVector(new_memberships), frommembershipMatriceToVector(self.predicted_memberships) ) < 0.02 or iteration >= self.n_iters:
                convergence = False
            self.predicted_memberships = new_memberships
        return self
    
    def initialize( self, edge_index, X, Y ):
        model = BregmanInitializer(self.n_clusters,initializer=self.initializer,
                                    edgeDistribution = self.edgeDistribution,
                                    attributeDistribution = self.attributeDistribution,
                                    weightDistribution = self.weightDistribution)
        if self.edge_index is None:
            self.edge_index = edge_index
        Z_init = None
        if self.use_random_init == True:
            Z_init = fromVectorToMembershipMatrice(np.random.randint(self.n_clusters,size=self.N),
                                                                        self.n_clusters)
        model.initialize( X, Y , self.edge_index, Z_init=Z_init)
        # model.initialize(  A, X, Y, Z_init=Z_init)
        self.predicted_memberships = model.predicted_memberships
        self.memberships_from_graph = frommembershipMatriceToVector(model.memberships_from_graph)
        self.memberships_from_attributes = frommembershipMatriceToVector(model.memberships_from_attributes)
        self.graph_init = model.graph_init

    def assignInitialLabels( self, X, Y ):
        return self
        
    def spectralEmbedding(self, X ):
        if (X<0).any():
            X = pairwise_kernels(X,metric='rbf')
        U = SpectralEmbedding(n_components=self.n_clusters,\
								affinity="precomputed")\
								.fit_transform(X)
        return U
    
    def computeAttributeMeans( self, Y, Z ):
        attribute_means = (Z.T @ Y)/(Z.sum(axis=0) + 10 * np.finfo(Z.dtype).eps)[:, np.newaxis]
        return attribute_means
    
    # def computeEdgeMeans( self, A, Z ):
    #     normalisation = np.linalg.pinv ( Z.T @ Z )
    #     return normalisation @ Z.T @ A @ Z @ normalisation

    def computeEdgeMeans(self,tau):
        weights = np.tensordot(tau, tau, axes=((), ()))
        """
        weights[i,q,j,l] = tau[i,q]*tau[j,l]
        desired output:
        weights[q,l,i,j] = tau[i,q]*tau[j,l]
        """
        weights = np.transpose(weights,(1,3,0,2))
        """
        weights is a k x k x N x N tensor
        desired output: 
        out[q,l] = sum_e weights[q,l,e]
        """
        edge_means = weights[:,:,self.edge_index[0],self.edge_index[1]].sum(axis=-1)/\
            weights.sum(axis=(-1,-2))
        
        return edge_means 
    
    def computeWeightMeans( self,X_, Z):
        weights = np.tensordot(Z, Z, axes=((), ()))
        """
        weights[i,q,j,l] = tau[i,q]*tau[j,l]
        desired output:
        weights[q,l,i,j] = tau[i,q]*tau[j,l]
        """
        weights = np.transpose(weights,(1,3,0,2))[:,:,self.edge_index[0],self.edge_index[1]]
        """
        X is a |E| x d tensor
        weights is a k x k x |E|
        desired output: 
        out[q,l,d] = sum_e X[e,d] * weights[q,l,e]
        """
        weight_means = np.tensordot( weights,\
                                    X_,\
                                    axes=[(2),(0)] )/(np.sum(weights,axis=-1)[:,:,np.newaxis]) 
        
        if (self.edge_means==0).any():
            null_model = X_.mean(axis=0)
            undefined_idx = np.where(self.edge_means==0)
            weight_means[undefined_idx[0],undefined_idx[1],:] = null_model
        return weight_means
    
    def likelihood( self, X, Y, Z ):
        graphLikelihood = self.likelihoodGraph(X,Z)
        attributeLikelihood = self.likelihoodAttributes(Y,Z)
        return graphLikelihood + attributeLikelihood
    
    def likelihoodGraph(self, X, Z):
        graph_mean = self.computeEdgeMeans(Z)
        return 1/2 * np.sum( self.edge_divergence( X, Z @ graph_mean @ Z.T ) )
    
    def likelihoodAttributes( self, Y, Z):
        M = self.computeAttributeMeans(Y,Z)
        total = np.sum( paired_distances(Y,Z@M) )
        return total 
    
    def assignments( self, X_, Y ):
        z = np.zeros( Y.shape[ 0 ], dtype = int )
        H = pairwise_distances(Y,self.attribute_means,metric=self.attribute_divergence)
        for node in range( len( z ) ):
            z[ node ] = self.singleNodeAssignment( X_, H, node )
        return fromVectorToMembershipMatrice( z, n_clusters = self.n_clusters )

    def assignments_joblib(self, X_,Y):
        H = pairwise_distances(Y,self.attribute_means,metric=self.attribute_divergence)
        Parallel(backend="threading",n_jobs=self.n_jobs)\
            (delayed(singleAssignmentContainer)(self, X_, H, ranges)\
              for ranges in gen_even_slices(self.N,self.n_jobs) )        
        return fromVectorToMembershipMatrice( self.Z, n_clusters = self.n_clusters )

    def singleNodeAssignment( self, X_, H, node ):
        L = np.zeros( self.n_clusters )
        edge_indices_in = np.argwhere(self.edge_index[1] == node).flatten()
        v_idx_in = self.edge_index[0][edge_indices_in]
        
        edge_indices_out = np.argwhere(self.edge_index[0] == node).flatten()
        v_idx_out = self.edge_index[1][edge_indices_out]
        
        mask_in = self.index_to_mask(v_idx_in)
        mask_out = self.index_to_mask(v_idx_out)
        
        v_idx_in_comp = np.where(~mask_in)
        v_idx_out_comp = np.where(~mask_out)

        for q in range( self.n_clusters ):
            z_t = self.predicted_memberships.argmax(axis=1)
            z_t[node] = q
            E = self.weight_means
            """
            X has shape |E| x d
            E has shape k x k x d
            
            the edge divergence computes the difference between node i (from community q) edges and the means
            given node j belongs to community l:
            
            sum_j phi_edge(e_ij, E[q,l,:])  
            """
            att_div = H[node,q]
            edge_div = self.precomputed_edge_div[1,z_t[v_idx_in],q].sum()\
                    + self.precomputed_edge_div[1,q,z_t[v_idx_out]].sum()\
                    + self.precomputed_edge_div[0,z_t[v_idx_in_comp],q].sum()\
                    + self.precomputed_edge_div[0,q,z_t[v_idx_out_comp]].sum()\
                    - 2*self.precomputed_edge_div[0,q,q]

            ## compute weight divergence
            weight_div = 0
            contains_nan = False
            E_ = E[q,z_t[v_idx_out],:]
            if np.isnan(E_).any():
                weight_div = np.inf
                contains_nan = True
            not_nan_idx = np.argwhere(~np.isnan(E_).any(axis=1)).flatten()
            E_without_nan = E_[not_nan_idx,:]
            if (len(v_idx_out) > 0) and (len(not_nan_idx) > 0) and (not contains_nan):
                weight_div += np.sum( paired_distances(X_[edge_indices_out,:],\
                                                            E_without_nan,\
                                                            metric=self.weight_divergence))
                
            ## same as before, but now for edges coming in node
            E_ = E[z_t[v_idx_in],q,:]
            if np.isnan(E_).any():
                weight_div = np.inf
                contains_nan = True
            not_nan_idx = np.argwhere(~np.isnan(E_).any(axis=1)).flatten()
            E_without_nan = E_[not_nan_idx,:]
            if (len(v_idx_in) > 0) and (len(not_nan_idx) > 0) and (not contains_nan):
                weight_div += np.sum( paired_distances(X_[edge_indices_in,:],\
                                                            E_without_nan,\
                                                            metric=self.weight_divergence))
            L[ q ] = att_div + (weight_div + edge_div)
        return np.argmin( L )
    
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

class BregmanClusteringMemEfficient( BaseEstimator, ClusterMixin ):
    def __init__( self, n_clusters, 
                 edgeDistribution = "bernoulli",
                 attributeDistribution = "gaussian",
                 weightDistribution = "gaussian",
                 initializer = 'chernoff', 
                 graph_initializer = "spectralClustering", attribute_initializer = 'GMM', 
                 n_iters = 25, init_iters=100,
                 reduce_by=None,
                 divergence_precomputed=True,
                 use_random_init=False):
        """
        Bregman Hard Clustering Algorithm for partitioning graph with node attributes
        Parameters
        ----------
        n_clusters : INT
            Number of clustes.
        edge_divergence, attribute_divergence : function
            Pairwise divergence function. The default is euclidean_distance.
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
        self.n_iters = n_iters
        self.initializer = initializer
        self.graph_initializer = graph_initializer
        self.attribute_initializer = attribute_initializer
        self.init_iters = init_iters
        ## Variable that stores which initialization was chosen
        self.graph_init = False
        self.edgeDistribution = edgeDistribution
        self.attributeDistribution = attributeDistribution
        self.weightDistribution = weightDistribution
        self.edge_divergence = dist_to_divergence_dict[self.edgeDistribution]
        self.weight_divergence = dist_to_divergence_dict[self.weightDistribution]
        self.attribute_divergence = dist_to_divergence_dict[self.attributeDistribution]
        self.edge_index = None 
        self.use_random_init = use_random_init
        self.n_jobs = effective_n_jobs(-1)

    def precompute_edge_divergences(self):
        self.precomputed_edge_div = pairwise_distances(np.array([0,1]).reshape(-1,1),\
                                             self.edge_means.reshape(-1,1),\
                                             metric=self.edge_divergence)\
                                            .reshape(
                                                        (2,\
                                                        self.n_clusters,self.n_clusters
                                                        )
                                                    )    
    def index_to_mask(self,v_idx):
        all_indices = np.zeros(self.N, dtype=bool)
        all_indices[v_idx] = True
        return all_indices
    
    def fit( self, edge_index, X, Y, Z_init=None):
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
        self.N = Y.shape[0]
        self.edge_index = edge_index
        X_ = None
        ## CASE X is N x N x 1: pass to |E| x 1 
        if X.shape[0] == X.shape[1]:
            X_ = X[self.edge_index[0],self.edge_index[1],:]
        else:           
            X_ = X
        if Z_init is None:
            self.initialize( self.edge_index, X_, Y)
            self.assignInitialLabels( X_, Y )
        else:
            self.predicted_memberships = Z_init
        path = tempfile.mkdtemp()
        Zpath = os.path.join(path,'z.mmap')
        self.Z = np.memmap(Zpath, dtype=int, shape=self.N, mode='w+')
        #init_labels = self.predicted_memberships
        self.num_edges = np.ones((self.n_clusters,self.n_clusters))
        self.attribute_means = self.computeAttributeMeans(Y,self.predicted_memberships)
        self.edge_means = self.computeEdgeMeans(self.predicted_memberships)
        self.weight_means = self.computeWeightMeans( X_, self.predicted_memberships)
        self.precompute_edge_divergences()
        convergence = True
        iteration = 0
        while convergence:
            new_memberships = self.assignments( X_, Y )

            self.attribute_means = self.computeAttributeMeans( Y, new_memberships )
            self.edge_means = self.computeEdgeMeans(new_memberships)
            self.weight_means = self.computeWeightMeans( X_, new_memberships)
            self.precompute_edge_divergences()    
            iteration += 1
            if accuracy_score( frommembershipMatriceToVector(new_memberships), frommembershipMatriceToVector(self.predicted_memberships) ) < 0.02 or iteration >= self.n_iters:
                convergence = False
            self.predicted_memberships = new_memberships
        return self
    
    def initialize( self, edge_index, X, Y ):
        model = BregmanInitializer(self.n_clusters,initializer=self.initializer,
                                    edgeDistribution = self.edgeDistribution,
                                    attributeDistribution = self.attributeDistribution,
                                    weightDistribution = self.weightDistribution)
        if self.edge_index is None:
            self.edge_index = edge_index
        Z_init = None
        if self.use_random_init == True:
            Z_init = fromVectorToMembershipMatrice(np.random.randint(self.n_clusters,size=self.N),
                                                                        self.n_clusters)
        model.initialize( X, Y , self.edge_index, Z_init=Z_init)
        # model.initialize(  A, X, Y, Z_init=Z_init)
        self.predicted_memberships = model.predicted_memberships
        self.memberships_from_graph = frommembershipMatriceToVector(model.memberships_from_graph)
        self.memberships_from_attributes = frommembershipMatriceToVector(model.memberships_from_attributes)
        self.graph_init = model.graph_init

    def assignInitialLabels( self, X, Y ):
        return self
        
    def spectralEmbedding(self, X ):
        if (X<0).any():
            X = pairwise_kernels(X,metric='rbf')
        U = SpectralEmbedding(n_components=self.n_clusters,\
								affinity="precomputed")\
								.fit_transform(X)
        return U
    
    def computeAttributeMeans( self, Y, Z ):
        attribute_means = (Z.T @ Y)/(Z.sum(axis=0) + 10 * np.finfo(Z.dtype).eps)[:, np.newaxis]
        return attribute_means

    def computeEdgeMeans(self,Z):
        edge_means = np.zeros((self.n_clusters,self.n_clusters))
        C = Z.argmax(axis=1).astype(int)
        for i,j in zip(self.edge_index[0],self.edge_index[1]):
            edge_means[C[i],C[j]] += 1
        self.num_edges = edge_means 
        m = Z.sum(axis=0)
        D = np.outer(m, m)
        edge_means /= D
        return edge_means
    
    def computeWeightMeans( self,X_, Z):
        weight_means = np.zeros((self.n_clusters,self.n_clusters,X_.shape[1]))
        C = Z.argmax(axis=1).astype(int)
        for iter_,(i,j) in enumerate(zip(self.edge_index[0],self.edge_index[1])):
            weight_means[C[i],C[j],:] += X_[iter_,:] 
        weight_means /= self.num_edges[:,:,np.newaxis]
        if (self.edge_means==0).any():
            null_model = X_.mean(axis=0)
            undefined_idx = np.where(self.edge_means==0)
            weight_means[undefined_idx[0],undefined_idx[1],:] = null_model
        return weight_means
    
    def likelihood( self, X, Y, Z ):
        graphLikelihood = self.likelihoodGraph(X,Z)
        attributeLikelihood = self.likelihoodAttributes(Y,Z)
        return graphLikelihood + attributeLikelihood
    
    def likelihoodGraph(self, X, Z):
        graph_mean = self.computeEdgeMeans(Z)
        return 1/2 * np.sum( self.edge_divergence( X, Z @ graph_mean @ Z.T ) )
    
    def likelihoodAttributes( self, Y, Z):
        M = self.computeAttributeMeans(Y,Z)
        total = np.sum( paired_distances(Y,Z@M) )
        return total 
    
    def assignments( self, X_, Y ):
        z = np.zeros( Y.shape[ 0 ], dtype = int )
        H = pairwise_distances(Y,self.attribute_means,metric=self.attribute_divergence)
        for node in range( len( z ) ):
            z[ node ] = self.singleNodeAssignment( X_, H, node )
        return fromVectorToMembershipMatrice( z, n_clusters = self.n_clusters )

    def assignments_joblib(self, X_,Y):
        H = pairwise_distances(Y,self.attribute_means,metric=self.attribute_divergence)
        Parallel(backend="threading",n_jobs=self.n_jobs)\
            (delayed(singleAssignmentContainer)(self, X_, H, ranges)\
              for ranges in gen_even_slices(self.N,self.n_jobs) )        
        return fromVectorToMembershipMatrice( self.Z, n_clusters = self.n_clusters )

    def singleNodeAssignment( self, X_, H, node ):
        L = np.zeros( self.n_clusters )
        edge_indices_in = np.argwhere(self.edge_index[1] == node).flatten()
        v_idx_in = self.edge_index[0][edge_indices_in]
        
        edge_indices_out = np.argwhere(self.edge_index[0] == node).flatten()
        v_idx_out = self.edge_index[1][edge_indices_out]
        
        mask_in = self.index_to_mask(v_idx_in)
        mask_out = self.index_to_mask(v_idx_out)
        
        v_idx_in_comp = np.where(~mask_in)
        v_idx_out_comp = np.where(~mask_out)

        for q in range( self.n_clusters ):
            z_t = self.predicted_memberships.argmax(axis=1)
            z_t[node] = q
            E = self.weight_means
            """
            X has shape |E| x d
            E has shape k x k x d
            
            the edge divergence computes the difference between node i (from community q) edges and the means
            given node j belongs to community l:
            
            sum_j phi_edge(e_ij, E[q,l,:])  
            """
            att_div = H[node,q]
            edge_div = self.precomputed_edge_div[1,z_t[v_idx_in],q].sum()\
                    + self.precomputed_edge_div[1,q,z_t[v_idx_out]].sum()\
                    + self.precomputed_edge_div[0,z_t[v_idx_in_comp],q].sum()\
                    + self.precomputed_edge_div[0,q,z_t[v_idx_out_comp]].sum()\
                    - 2*self.precomputed_edge_div[0,q,q]

            ## compute weight divergence
            weight_div = 0
            contains_nan = False
            E_ = E[q,z_t[v_idx_out],:]
            if np.isnan(E_).any():
                weight_div = np.inf
                contains_nan = True
            not_nan_idx = np.argwhere(~np.isnan(E_).any(axis=1)).flatten()
            E_without_nan = E_[not_nan_idx,:]
            if (len(v_idx_out) > 0) and (len(not_nan_idx) > 0) and (not contains_nan):
                weight_div += np.sum( paired_distances(X_[edge_indices_out,:],\
                                                            E_without_nan,\
                                                            metric=self.weight_divergence))
                
            ## same as before, but now for edges coming in node
            E_ = E[z_t[v_idx_in],q,:]
            if np.isnan(E_).any():
                weight_div = np.inf
                contains_nan = True
            not_nan_idx = np.argwhere(~np.isnan(E_).any(axis=1)).flatten()
            E_without_nan = E_[not_nan_idx,:]
            if (len(v_idx_in) > 0) and (len(not_nan_idx) > 0) and (not contains_nan):
                weight_div += np.sum( paired_distances(X_[edge_indices_in,:],\
                                                            E_without_nan,\
                                                            metric=self.weight_divergence))
            L[ q ] = att_div + (weight_div + edge_div)
        return np.argmin( L )
    
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

class BregmanNodeEdgeAttributeGraphClusteringSoft( BaseEstimator, ClusterMixin ):
    def __init__( self, n_clusters, 
                 edgeDistribution = "bernoulli",
                 attributeDistribution = "gaussian",
                 weightDistribution = "gaussian",
                 initializer = 'chernoff', 
                 graph_initializer = "spectralClustering", attribute_initializer = 'GMM', 
                 n_iters = 100, init_iters=100,
                 reduce_by=None,
                 divergence_precomputed=True,
                 use_random_init=False):
        """
        Bregman Hard Clustering Algorithm for partitioning graph with node attributes
        Parameters
        ----------
        n_clusters : INT
            Number of clustes.
        edge_divergence, attribute_divergence : function
            Pairwise divergence function. The default is euclidean_distance.
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
        self.n_iters = n_iters
        self.initializer = initializer
        self.graph_initializer = graph_initializer
        self.attribute_initializer = attribute_initializer
        self.init_iters = init_iters
        ## Variable that stores which initialization was chosen
        self.graph_init = False
        self.edgeDistribution = edgeDistribution
        self.attributeDistribution = attributeDistribution
        self.weightDistribution = weightDistribution
        self.edge_divergence = dist_to_divergence_dict[self.edgeDistribution]
        self.weight_divergence = dist_to_divergence_dict[self.weightDistribution]
        self.attribute_divergence = dist_to_divergence_dict[self.attributeDistribution]
        self.edge_index = None 
        self.use_random_init = use_random_init
        self.n_jobs = effective_n_jobs(-1)

    def precompute_edge_divergences(self):
        if(np.isnan(self.edge_means).any()):
            raise ValueError ("GOT NAN EDGE MEANS")
        self.precomputed_edge_div = pairwise_distances(np.array([0,1]).reshape(-1,1),\
                                             self.edge_means.reshape(-1,1),\
                                             metric=self.edge_divergence)\
                                            .reshape(
                                                        (2,\
                                                        self.n_clusters,self.n_clusters
                                                        )
                                                    )    
    def index_to_mask(self,v_idx):
        all_indices = np.zeros(self.N, dtype=bool)
        all_indices[v_idx] = True
        return all_indices
    
    def fit( self, edge_index, X, Y, Z_init=None):
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
        self.N = Y.shape[0]
        self.edge_index = edge_index
        X_ = None
        ## CASE X is N x N x 1: pass to |E| x 1 
        if X.shape[0] == X.shape[1]:
            X_ = X[self.edge_index[0],self.edge_index[1],:]
        else:           
            X_ = X
        self.node_indices = np.arange(self.N)
        if Z_init is None:
            self.initialize( edge_index, X_, Y)
            self.assignInitialLabels( X_, Y )
        else:
            self.memberships_from_graph = self.memberships_from_attributes \
                = self.predicted_memberships = Z_init
        #init_labels = self.predicted_memberships
        path = tempfile.mkdtemp()
        Zpath = os.path.join(path,'z.mmap')
        self.Ztilde = np.memmap(Zpath, dtype=float, shape=(self.N,self.n_clusters), mode='w+')
        self.M_projection(X_,Y,self.predicted_memberships)
        convergence = False
        iteration = 0
        old_log_prob = np.inf
        while not convergence:
            Z_new = self.E_projection(X_, Y)
            new_log_prob = self.logprob(X_,Y)
            self.M_projection( X_,Y,Z_new)
            convergence = self.stop_criterion(X_,Y,\
                                              self.predicted_memberships,Z_new,\
                                                old_log_prob,new_log_prob,\
                                                iteration)
            self.predicted_memberships = Z_new
            old_log_prob = new_log_prob
            iteration += 1 
        return self
    
    def initialize( self, edge_index, X_, Y ):
        model = BregmanInitializer(self.n_clusters,initializer=self.initializer,
                                    edgeDistribution = self.edgeDistribution,
                                    attributeDistribution = self.attributeDistribution,
                                    weightDistribution = self.weightDistribution)
        if self.edge_index is None:
            self.edge_index = edge_index
        Z_init = None
        if self.use_random_init == True:
            Z_init = fromVectorToMembershipMatrice(np.random.randint(self.n_clusters,size=self.N),
                                                                        self.n_clusters)
        model.initialize( X_, Y , self.edge_index, Z_init=Z_init)
        # model.initialize(  A, X_, Y, Z_init=Z_init)
        self.predicted_memberships = model.predicted_memberships
        self.memberships_from_graph = frommembershipMatriceToVector(model.memberships_from_graph)
        self.memberships_from_attributes = frommembershipMatriceToVector(model.memberships_from_attributes)
        self.graph_init = model.graph_init

    def assignInitialLabels( self, X, Y ):
        return self
        
    def spectralEmbedding(self, X ):
        if (X<0).any():
            X = pairwise_kernels(X,metric='rbf')
        U = SpectralEmbedding(n_components=self.n_clusters,\
								affinity="precomputed")\
								.fit_transform(X)
        return U
    
    
    def computeAttributeMeans( self, Y, Z ):
        nk = Z.sum(axis=0) + 10 * np.finfo(Z.dtype).eps
        attribute_means = np.dot(Z.T, Y) / nk[:, np.newaxis]
        # attribute_means = np.dot(Z.T, Y)/(Z.sum(axis=0))[:, np.newaxis]
        #+ 10 * np.finfo(Z.dtype).eps
        if (np.isnan(attribute_means).any()):
            if (Z == 0).any():
                print( "Z == 0")
            raise ValueError ("att means contains Nan")
        return attribute_means
    
    def computeEdgeMeans(self,tau):
        weights = np.tensordot(tau, tau, axes=((), ()))
        """
        weights[i,q,j,l] = tau[i,q]*tau[j,l]
        desired output:
        weights[q,l,i,j] = tau[i,q]*tau[j,l]
        """
        weights = np.transpose(weights,(1,3,0,2))
        """
        weights is a k x k x N x N tensor
        desired output: 
        out[q,l] = sum_e weights[q,l,e]
        """
        edge_means = weights[:,:,self.edge_index[0],self.edge_index[1]].sum(axis=-1)/\
            weights.sum(axis=(-1,-2))

        # if (np.isnan(edge_means).any()):
        #     raise ValueError ("edge means contains Nan")
        return edge_means 
    
    def computeWeightMeans( self, X_, Z):
        weights = np.tensordot(Z, Z, axes=((), ()))
        """
        weights[i,q,j,l] = tau[i,q]*tau[j,l]
        desired output:
        weights[q,l,i,j] = tau[i,q]*tau[j,l]
        """
        weights = np.transpose(weights,(1,3,0,2))[:,:,self.edge_index[0],self.edge_index[1]]
        """
        X is a |E| x d tensor
        weights is a k x k x |E|
        desired output: 
        out[q,l,d] = sum_e X[e,d] * weights[q,l,e]
        """
        weight_means = np.tensordot( weights,\
                                    X_,\
                                    axes=[(2),(0)] )/(np.sum(weights,axis=-1)[:,:,np.newaxis]) 
        # if (np.isnan(weight_means).any()):
        #     print("W means contains Nan")
        return weight_means
    
    def computeTotalDiv(self,node,X_,Z,H):
        L = np.zeros( self.n_clusters )
        edge_indices_in = np.argwhere(self.edge_index[1] == node).flatten()
        v_idx_in = self.edge_index[0][edge_indices_in]
        
        edge_indices_out = np.argwhere(self.edge_index[0] == node).flatten()
        v_idx_out = self.edge_index[1][edge_indices_out]
        
        mask_in = self.index_to_mask(v_idx_in)
        mask_out = self.index_to_mask(v_idx_out)
        
        v_idx_in_comp = np.where(~mask_in)
        v_idx_out_comp = np.where(~mask_out)

        for q in range( self.n_clusters ):
            z_t = Z.argmax(axis=1)
            z_t[node] = q
            E = self.weight_means
            """
            X has shape n x n x d
            E has shape k x k x d
            
            the edge divergence computes the difference between node i (from community q) edges and the means
            given node j belongs to community l:
            
            sum_j phi_edge(e_ij, E[q,l,:])  
            """
            att_div = H[node,q]
            edge_div = self.precomputed_edge_div[1,z_t[v_idx_in],q].sum()\
                    + self.precomputed_edge_div[1,q,z_t[v_idx_out]].sum()\
                    + self.precomputed_edge_div[0,z_t[v_idx_in_comp],q].sum()\
                    + self.precomputed_edge_div[0,q,z_t[v_idx_out_comp]].sum()\
                    - 2*self.precomputed_edge_div[0,q,q]

            ## compute weight divergence
            weight_div = 0
            contains_nan = False
            E_ = E[q,z_t[v_idx_out],:]
            if np.isnan(E_).any():
                weight_div = np.inf
                contains_nan = True
            not_nan_idx = np.argwhere(~np.isnan(E_).any(axis=1)).flatten()
            E_without_nan = E_[not_nan_idx,:]
            if (len(v_idx_out) > 0) and (len(not_nan_idx) > 0) and (not contains_nan):
                weight_div += np.sum( paired_distances(X_[edge_indices_out,:],\
                                                            E_without_nan,\
                                                            metric=self.weight_divergence))
                
                if (not np.array_equal(E_without_nan.shape,X_[edge_indices_out].shape)):
                    raise ValueError ("ERROR mishape")
                
            ## same as before, but now for edges coming in node
            E_ = E[z_t[v_idx_in],q,:]
            if np.isnan(E_).any():
                weight_div = np.inf
                contains_nan = True
            not_nan_idx = np.argwhere(~np.isnan(E_).any(axis=1)).flatten()
            E_without_nan = E_[not_nan_idx,:]
            if (len(v_idx_in) > 0) and (len(not_nan_idx) > 0) and (not contains_nan):
                weight_div += np.sum( paired_distances(X_[edge_indices_in,:],\
                                                            E_without_nan,\
                                                            metric=self.weight_divergence))
                
                if (not np.array_equal(E_without_nan.shape,X_[edge_indices_in].shape)):
                    raise ValueError ("ERROR mishape")
            
            L[ q ] = att_div + weight_div + edge_div
        if (np.isnan(L).any()):
            raise ValueError ("L contains Nan")
        return L

    def q_exp(self,x,q):
        return np.power(1 + (1-q)*x, 1/(1-q))
    
    def E_projection(self, X, Y):
        Ztilde = np.zeros( (self.N,self.n_clusters), dtype = float)
        H = pairwise_distances(Y,self.attribute_means,metric=self.attribute_divergence)
        
        if np.isnan(H).any():
            # if np.isnan(Y).any():
            #     print("Att contains nan")
            # if np.isnan(self.attribute_means).any():
                # print("Att means contains nan")
            raise ValueError("H contains nan")
        
        # for node in range(self.N):
        #     Ztilde[node,:] = self.computeTotalDiv(node,X,self.predicted_memberships,H)
        Ztilde = self.computeTotalDiv_joblib(X, H)

        c = Ztilde.min(axis=1)
        Ztilde -= c[:,None]
        soft_assign = self.communities_weights.reshape(1, -1)*np.exp(-Ztilde)    
        return normalize(soft_assign, axis=1, norm='l1')
    
    def computeTotalDiv_joblib(self, X, H):
        Parallel(backend="threading",n_jobs=self.n_jobs)\
            (delayed(singlecomputeTotalDivContainer)(self, X, H, ranges)\
              for ranges in gen_even_slices(self.N,self.n_jobs) )        
        return self.Ztilde
          
    def M_projection(self,X_,Y,Z):
        self.attribute_means = self.computeAttributeMeans(Y, Z)
        if (self.attribute_means > Y.max()).any():
            self.attribute_means = np.clip(self.attribute_means,a_min=Y.min(),a_max=Y.max())
            # print(self.attribute_means.max())
            # raise ValueError("Att inconsistent")
        self.edge_means = self.computeEdgeMeans(Z)
        self.weight_means = self.computeWeightMeans( X_, Z)
        self.precompute_edge_divergences()
        self.communities_weights = Z.mean(axis=0)
        if (self.communities_weights == 0).any():
            raise ValueError ("ERROR Community is zero")
        # print("\n-----------------------------------------------------------\n",\
        #       "\nEDGE_MEANS: ",self.edge_means,
        #       "\nWeight_MEANS: ",self.weight_means,
        #       "\nAtt_MEANS: ",self.attribute_means)

    def logprob(self,X_,Y):
        H = pairwise_distances(Y,\
                               self.attribute_means,\
                                metric=self.attribute_divergence)
        log_prob_total = 0
        for node in range(self.N):
            divs = self.computeTotalDiv(node,X_,self.predicted_memberships,H)
            c = divs.min()
            divs -= c
            prob_i = self.communities_weights.dot(np.exp(-divs))
            log_prob_total += np.log(prob_i)
        return log_prob_total
    
    def stop_criterion(self,X,Y,Z_old,Z_new,old_log_prob,new_log_prob,iteration):
        # new_log_prob = self.logprob(X,Y)
        # np.allclose(Z_new,Z_old)
        if np.abs(old_log_prob - new_log_prob) < 0.1 or iteration >= self.n_iters:
            return True
        return False    
    
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