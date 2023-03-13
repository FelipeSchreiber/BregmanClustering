#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 16:59:19 2023

@author: maximilien
"""

import numpy as np
import scipy as sp
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from .divergences import *
from sklearn.metrics import adjusted_rand_score, accuracy_score
from scipy.spatial.distance import hamming
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.mixture import GaussianMixture
from sklearn.manifold import SpectralEmbedding
from tqdm import tqdm
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.preprocessing import normalize

def fromVectorToMembershipMatrice( z, n_clusters = 2 ):
    if len( set ( z ) ) > n_clusters:
        raise TypeError( 'There is a problem with the number of clusters' )
    n = len( z )
    Z = np.zeros( ( n, n_clusters ) )
    for i in range( n ):
        Z[ i, z[i] ] = 1
    return Z

def frommembershipMatriceToVector( Z ):
    n = Z.shape[0]
    z = np.zeros( n, dtype = int )
    for i in range( n ):
        z[ i ] = np.argwhere( Z[i,:] != 0 )[0][0]
    
    return z
    


class BregmanGraphPartitioning( BaseEstimator, ClusterMixin ):
    
    def __init__( self, n_clusters, divergence = kullbackLeibler_binaryMatrix, 
                 n_iters = 25, initializer="spectralClustering", init_iters=100 ):
        """
        Bregman Hard Clustering Algorithm
        Parameters
        ----------
        n_clusters : INT
            Number of clustes.
        divergence : function
            Pairwise divergence function. The default is euclidean.
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
            L[ k ] = kullbackLeibler_binaryMatrix( X, M )
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

    def __init__(self, n_clusters, divergence=euclidean, n_iters=1000, has_cov=False,
                 initializer="rand", init_iters=100, pretrainer=None):
        """
        Bregman Hard Clustering Algorithm
        Parameters
        ----------
        n_clusters : INT
            Number of clustes.
        divergence : function
            Pairwise divergence function. The default is euclidean.
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
            clus_sim = euclidean(init_vals, init_vals)
            np.fill_diagonal(clus_sim, np.inf)

            candidate = X[np.random.randint(X.shape[0])].reshape(1, -1)
            candidate_sims = euclidean(candidate, init_vals).flatten()
            closest_sim = candidate_sims.min()
            closest = candidate_sims.argmin()
            if closest_sim>clus_sim.min():
                replace_candidates_idx = np.array(np.unravel_index(clus_sim.argmin(), clus_sim.shape))
                replace_candidates = init_vals[replace_candidates_idx, :]

                closest_sim = euclidean(candidate, replace_candidates).flatten()
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
        dists = euclidean(X, self.params)
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
                 graph_divergence = kullbackLeibler_binaryMatrix, attribute_divergence = euclidean, 
                 initializer = 'nonrandom', 
                 graph_initializer = "spectralClustering", attribute_initializer = 'bregmanHardClustering', 
                 n_iters = 25, init_iters=100 ):
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
        

    def fit( self, X, Y ):
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
        
        self.initialize( X, Y )
        self.assignInitialLabels( X, Y )
        init_labels = self.predicted_memberships
        convergence = True
        iteration = 0
        while convergence:
            new_memberships = self.assignments( X, Y )

            self.attribute_means = self.computeAttributeMeans( Y, new_memberships )
            self.graph_means = self.computeGraphMeans( X, new_memberships )
            
            iteration += 1
            #if np.array_equal( new_memberships, self.predicted_memberships) or iteration >= self.n_iters:
            if accuracy_score( frommembershipMatriceToVector(new_memberships), frommembershipMatriceToVector(self.predicted_memberships) ) < 0.02 or iteration >= self.n_iters:
                convergence = False
                #print( accuracy_score( frommembershipMatriceToVector(new_memberships), frommembershipMatriceToVector(self.predicted_memberships) )  )
            self.predicted_memberships = new_memberships
        print( 'number of iterations : ', iteration)
        return self,init_labels
    
    def initialize( self, X, Y ):
        
        if self.attribute_initializer == 'bregmanHardClustering':
            model = BregmanHard( n_clusters = self.n_clusters, divergence = self.attribute_divergence, initializer="kmeans++" )
            model.fit( Y )
            self.memberships_from_attributes = fromVectorToMembershipMatrice( model.predict( Y ), n_clusters = self.n_clusters )
            self.attribute_means = self.computeAttributeMeans( Y, self.memberships_from_attributes )
        else:
            raise TypeError( 'The initializer provided for the attributes is not correct' )
            
        if self.graph_initializer == 'spectralClustering':
            self.memberships_from_graph = self.spectralClustering( X )
            self.graph_means = self.computeGraphMeans( X, self.memberships_from_graph )
        else:
            raise TypeError( 'The initializer provided for the graph is not correct' )
    
    
    def assignInitialLabels( self, X, Y ):
        
        if self.initializer == 'random':
            z =  np.random.randint( 0, self.n_clusters, size = X.shape[0] )
            self.predicted_memberships = fromVectorToMembershipMatrice( z, n_clusters = self.n_clusters )
        
        else:
            if (X<0).any():
                X = pairwise_kernels(X,metric='rbf')
            U = SpectralEmbedding(n_components=self.n_clusters,\
								affinity="precomputed")\
								.fit_transform(X)
		
            net_null_model = GaussianMixture(n_components=1).fit(U.copy())
            null_net = net_null_model.aic(U.copy())
            net_model = GaussianMixture(n_components=self.n_clusters).fit(U.copy())
            fitted_net = net_model.aic(U.copy())
            AIC_graph = fitted_net - null_net
            
            att_null_model = GaussianMixture(n_components=1).fit(Y.copy())
            null_attributes = att_null_model.aic(Y.copy())
            att_model = GaussianMixture(n_components=self.n_clusters).fit(Y.copy())
            fitted_attributes = att_model.aic(Y.copy())
            AIC_attribute = fitted_attributes - null_attributes
            
            n = self.memberships_from_attributes.shape[ 0 ]
            print(  'graph : ', self.graphChernoffDivergence( X, self.memberships_from_graph ) )
            print( 'attributes', self.attributeChernoffDivergence( Y, self.memberships_from_attributes ) / n )
            if AIC_graph < AIC_attribute:
                self.predicted_memberships = self.memberships_from_graph
                print( 'Initialisation chosen from the graph')
            else:
                self.predicted_memberships = self.memberships_from_attributes
                print( 'Initialisation chosen from the attributes' )

    
    def spectralClustering( self, X ):
        if (X<0).any():
            X = pairwise_kernels(X,metric='rbf')
        U = SpectralEmbedding(n_components=self.n_clusters,\
								affinity="precomputed")\
								.fit_transform(X)
        z_init = GaussianMixture(n_components=self.n_clusters).fit_predict(U)
        Z = fromVectorToMembershipMatrice( z_init, n_clusters = self.n_clusters )
        self.predicted_memberships = Z
        return self.predicted_memberships
    
    def computeAttributeMeans( self, Y, Z ):
        z = frommembershipMatriceToVector( Z )
        attribute_means = np.zeros( (self.n_clusters, Y.shape[1] )  )
        for k in range( self.n_clusters ):
            Y_k = Y[ z == k ]
            attribute_means[k] = np.mean( Y_k, axis = 0 )
        return attribute_means
    
    def computeGraphMeans( self, X, Z ):
        normalisation = np.linalg.pinv ( Z.T @ Z )
        return normalisation @ Z.T @ X @ Z @ normalisation
    
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
            
        if self.graphDistribution == 'bernoulli':
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
        graph_mean = self.computeGraphMeans( X, Z )
        attribute_mean = self.computeAttributeMeans( Y, Z )
        graphLikelihood = 1/2 * np.sum( self.graph_divergence( X, Z @ graph_mean @ Z.T ) )
        
        attributeLikelihood = 0
        for d in range( Y.shape[1] ):
            attributeLikelihood += np.sum( self.attribute_divergence( Y[:,d], Z @ attribute_mean[:,d] ) ) 

        return graphLikelihood + attributeLikelihood
    
    def likelihoodGraph( self, X, null_model = False ):
        Z = self.predicted_memberships
        if null_model:
            graph_mean = np.mean( X ) * np.ones( (self.n_clusters, self.n_clusters ) )
        else:
            graph_mean = self.graph_means
        return 1/2 * np.sum( self.graph_divergence( X, Z @ graph_mean @ Z.T ) )
    
    def likelihoodAttributes( self, Y, null_model = False ):
        Z = self.predicted_memberships
        if null_model:
            M = np.mean( Y ) * np.ones( ( self.n_clusters, Y.shape[1] ) )
        else:
            M = self.attribute_means
        res = 0
        for d in range( Y.shape[1] ):
            res += np.sum( self.attribute_divergence( Y[:,d], Z @ M[:,d] ) ) 
        return res
        #return np.sum( self.attribute_divergence( Y, Z @ M ) )    
    
    def assignments( self, X, Y ):
        z = np.zeros( X.shape[ 0 ], dtype = int )
        H = self.attribute_divergence( Y, self.attribute_means )
        for node in range( len( z ) ):
            z[ node ] = self.singleNodeAssignment( X, H, node )
        return fromVectorToMembershipMatrice( z, n_clusters = self.n_clusters )        
    
    def singleNodeAssignment( self, X, H, node ):
        L = np.zeros( self.n_clusters )
        for k in range( self.n_clusters ):
            Ztilde = self.predicted_memberships.copy()
            Ztilde[ node, : ] = 0
            Ztilde[ node, k ] = 1
            M = Ztilde @ self.graph_means @ Ztilde.T
            
            L[ k ] = self.graph_divergence( X[ node,: ], M[ node,: ] ) + H[ node, k ] #+ np.sum( self.attribute_divergence( Y[ node ], self.attribute_means[ k ] ) )
            #print( L[k] )
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

class SoftBregmanNodeAttributeGraphClustering( BaseEstimator, ClusterMixin ):
    def __init__( self, n_clusters, 
                 graph_divergence = kullbackLeibler_binaryMatrix, attribute_divergence = euclidean, 
                 initializer = 'nonrandom', 
                 graph_initializer = "spectralClustering", attribute_initializer = 'bregmanHardClustering', 
                 n_iters = 25, init_iters=100 ):
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
        

    def fit( self, X, Y ):
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
        
        self.initialize( X, Y )
        self.assignInitialLabels( X, Y )
        init_labels = self.predicted_memberships
        print(init_labels.shape,type(init_labels))
        self.attribute_means = self.computeAttributeMeans( Y, init_labels)
        self.graph_means = self.computeGraphMeans( X, init_labels)
        convergence = True
        iteration = 0
        while convergence:
            new_memberships = self.assignments( X, Y )
            print(new_memberships)
            self.attribute_means = self.computeAttributeMeans( Y, new_memberships )
            self.graph_means = self.computeGraphMeans( X, new_memberships )
            
            iteration += 1
            if np.allclose(new_memberships,self.predicted_memberships,rtol=0,atol=1e-03) or iteration >= self.n_iters:
                convergence = False
            self.predicted_memberships = new_memberships
            print(self.likelihood(X,Y,self.predicted_memberships))
        print( 'number of iterations : ', iteration)
        return self,init_labels
    
    def initialize( self, X, Y ):
        
        if self.attribute_initializer == 'bregmanHardClustering':
            model = BregmanHard( n_clusters = self.n_clusters, divergence = self.attribute_divergence, initializer="kmeans++" )
            model.fit( Y )
            self.memberships_from_attributes = fromVectorToMembershipMatrice( model.predict( Y ), n_clusters = self.n_clusters )
            self.attribute_means = self.computeAttributeMeans( Y, self.memberships_from_attributes )
        else:
            raise TypeError( 'The initializer provided for the attributes is not correct' )
            
        if self.graph_initializer == 'spectralClustering':
            self.memberships_from_graph = self.spectralClustering( X )
            self.graph_means = self.computeGraphMeans( X, self.memberships_from_graph )
        else:
            raise TypeError( 'The initializer provided for the graph is not correct' )
    
    
    def assignInitialLabels( self, X, Y ):
        
        if self.initializer == 'random':
            z =  np.random.randint( 0, self.n_clusters, size = X.shape[0] )
            self.predicted_memberships = fromVectorToMembershipMatrice( z, n_clusters = self.n_clusters )
        
        else:
            if (X<=0).any():
                X = pairwise_kernels(X,metric='rbf')
            U = SpectralEmbedding(n_components=self.n_clusters,\
								affinity="precomputed")\
								.fit_transform(X)
		
            net_null_model = GaussianMixture(n_components=1).fit(U)
            null_net = net_null_model.aic(U)
            net_model = GaussianMixture(n_components=self.n_clusters).fit(U)
            fitted_net = net_model.aic(U)
            AIC_graph = fitted_net - null_net
            
            att_null_model = GaussianMixture(n_components=1).fit(Y)
            null_attributes = att_null_model.aic(Y)
            att_model = GaussianMixture(n_components=self.n_clusters).fit(Y)
            fitted_attributes = att_model.aic(Y)
            AIC_attribute = fitted_attributes - null_attributes
            
            n = self.memberships_from_attributes.shape[ 0 ]
            print(  'graph : ', self.graphChernoffDivergence( X, self.memberships_from_graph ) )
            print( 'attributes', self.attributeChernoffDivergence( Y, self.memberships_from_attributes ) / n )
            if AIC_graph < AIC_attribute:
                self.predicted_memberships = self.memberships_from_graph
                print( 'Initialisation chosen from the graph')
            else:
                self.predicted_memberships = self.memberships_from_attributes
                print( 'Initialisation chosen from the attributes' )

    
    def spectralClustering( self, X ):
        sc = SpectralClustering( n_clusters = self.n_clusters, affinity = 'precomputed', assign_labels = 'kmeans' )
        z_init = sc.fit_predict( X )
        Z = fromVectorToMembershipMatrice( z_init, n_clusters = self.n_clusters )
        self.predicted_memberships = Z
        return self.predicted_memberships
    
    def computeAttributeMeans( self, Y, Z ):
        attribute_means = (Z.T@Y)/Z.sum(axis=0)[:,np.newaxis]
        return attribute_means
    
    def computeGraphMeans( self, X, Z ):
        N = X.shape[0]
        B = np.zeros((self.n_clusters,self.n_clusters))##SBM Matrix
        for l in range(self.n_clusters):
            for k in range(self.n_clusters):
                numerator = 0
                denominator = 0
                for i in range(N):
                    for j in range(N):
                        if j == i:
                            continue
                        product = Z[i,k]*Z[j,l]
                        denominator += product 
                        numerator += product*X[i,j]
                B[l,k] = numerator/denominator
        return B
    
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
            
        if self.graphDistribution == 'bernoulli':
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
        graph_mean = self.computeGraphMeans( X, Z )
        attribute_mean = self.computeAttributeMeans( Y, Z )
        graphLikelihood = 1/2 * np.sum( self.graph_divergence( X, Z @ graph_mean @ Z.T ) )
        
        attributeLikelihood = 0
        for d in range( Y.shape[1] ):
            attributeLikelihood += np.sum( self.attribute_divergence( Y[:,d], Z @ attribute_mean[:,d] ) ) 

        return graphLikelihood + attributeLikelihood
    
    def likelihoodGraph( self, X, null_model = False ):
        Z = self.predicted_memberships
        if null_model:
            graph_mean = np.mean( X ) * np.ones( (self.n_clusters, self.n_clusters ) )
        else:
            graph_mean = self.graph_means
        return 1/2 * np.sum( self.graph_divergence( X, Z @ graph_mean @ Z.T ) )
    
    def likelihoodAttributes( self, Y, null_model = False ):
        Z = self.predicted_memberships
        if null_model:
            M = np.mean( Y ) * np.ones( ( self.n_clusters, Y.shape[1] ) )
        else:
            M = self.attribute_means
        res = 0
        for d in range( Y.shape[1] ):
            res += np.sum( self.attribute_divergence( Y[:,d], Z @ M[:,d] ) ) 
        return res
        #return np.sum( self.attribute_divergence( Y, Z @ M ) )    
    
    def getNetDivMatrix(self,X,B,i,k):
        """
        This function computes the network divergence d(X_ij, B_kl), for every node j given that
        X_i belongs to community k and node j belongs to each community l.

        Returns matrix M (N x n_clusters) with all the divergences    
        """
        N = X.shape[0]
        M = np.zeros((N,self.n_clusters))
        for j in range(N):
            if j == i:
                M[j,:] = 0
                continue
            for l in range(self.n_clusters):
                M[j,l] = self.graph_divergence(X[i,j],B[k,l])
        return M

    def getNetDivMatrices(self,X,B):
        """
        This function computes the network divergence d(X_ij, B_kl), for every pairs of nodes and 
        for every k^2 combination of assignments between nodes.

        Returns a dictionary, such that entry [i][k] is the matrix M (N x n_clusters) 
        with all the divergences for node i belonging to community k   
        """
        matrices = {}
        N = X.shape[0]
        for i in range(N):
            matrices[i] = {}
            for k in range(self.n_clusters):
                matrices[i][k] = self.getNetDivMatrix(X,B,i,k)
        return matrices

    def update_assignments(self,X,Y,Z_old):
        N = X.shape[0]
        M = self.getNetDivMatrices(X,self.graph_means)
        H = self.attribute_divergence( Y, self.attribute_means)
        Z = np.zeros(H.shape)
        I = np.zeros(H.shape)
        priors = np.mean(Z_old,axis=0)
        for i in range(N):
            for k in range(self.n_clusters):
                # I[i,k] = np.multiply(Z_old,M[i][k]).sum()
                I[i,k] = M[i][k].sum()
        # I = normalize(I, axis=1, norm='l2')
        # H = normalize(H, axis=1, norm='l2')
        Z = np.multiply(priors[np.newaxis,:],\
			 	        np.exp(-I))
        return normalize(Z, axis=1, norm='l1')
    
    def assignments( self, X, Y ):
        iter__ = 0
        Z_old = self.predicted_memberships
        while True:
            iter__ += 1
            Z_new = self.update_assignments(X,Y,Z_old)
            if np.allclose(Z_new,Z_old,rtol=0,atol=1e-03) or iter__>10:
                break
            Z_old = Z_new
        return Z_new
    
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
