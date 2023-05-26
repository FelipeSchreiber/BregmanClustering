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
    n = Z.shape[0]
    z = np.zeros( n, dtype = int )
    for i in range( n ):
        z[ i ] = np.argwhere( Z[i,:] != 0 )[0][0]
    
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

class SoftBregmanNodeAttributeGraphClustering( BaseEstimator, ClusterMixin ):
    def __init__( self, n_clusters, 
                 edge_divergence = logistic_loss, attribute_divergence = euclidean_distance, 
                 initializer = 'chernoff', 
                 graph_initializer = "spectralClustering", attribute_initializer = 'GMM', 
                 n_iters = 25, init_iters=100,
                 normalize_=True, thresholding=True
                ):
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
        self.edge_divergence = edge_divergence
        self.attribute_divergence = attribute_divergence
        self.n_iters = n_iters
        self.initializer = initializer
        self.graph_initializer = graph_initializer
        self.attribute_initializer = attribute_initializer
        self.init_iters = init_iters
        self.scaler = MinMaxScaler()
        self.edgeDistribution = 'bernoulli'
        self.attributeDistribution = 'gaussian'
        self.normalize_ = normalize_
        self.thresholding = thresholding
        
    def spectralEmbedding( self, X ):
        if (X<0).any():
            X = pairwise_kernels(X,metric='rbf')
        U = SpectralEmbedding(n_components=self.n_clusters,\
								affinity="precomputed")\
								.fit_transform(X)
        return U
    
    def computeAttributeMeans( self, Y, Z ):
        attribute_means = np.dot(Z.T, Y)/(Z.sum(axis=0) + 10 * np.finfo(Z.dtype).eps)[:, np.newaxis]
        return attribute_means

    def computeGraphMeans(self,X,tau):
        graph_means = np.zeros((self.n_clusters,self.n_clusters))
        tau_sum = tau.sum(0)
        weights = np.tensordot(tau, tau, axes=((), ()))
        """
        weights[i,q,j,l] = tau[i,q]*tau[j,l]
        desired output:
        weights[q,l,i,j] = tau[i,q]*tau[j,l]
        """
        weights = np.transpose(weights,(1,3,0,2))
        for q in range(self.n_clusters):
            for l in range(self.n_clusters):
                graph_means[q,l]=np.sum(weights[q,l]*X)/np.sum(weights[q,l])
        #graph_means/=((tau_sum.reshape((-1, 1)) * tau_sum) - tau.T @ tau)
        np.nan_to_num(graph_means,copy=False)
        return graph_means 
    
    def likelihoodAttributes(self,Y,Z):
        M = self.computeAttributeMeans(Y,Z)
        total = np.sum( paired_distances(Y,Z@M) )
        return total

    def likelihoodGraph(self,X,Z):
        graph_mean = self.computeGMeans(X,Z)
        return 1/2 * np.sum( self.edge_divergence( X, Z @ graph_mean @ Z.T ) )
    
    def initialize( self, A, X, Y ):
        model = BregmanInitializer(self.n_clusters,initializer=self.initializer,
                                    edgeDistribution = self.edgeDistribution,
                                    attributeDistribution = self.attributeDistribution,
                                    weightDistribution = self.weightDistribution)
        if self.edge_index is None:
            self.edge_index = np.nonzero(A)
        model.initialize( X, Y , self.edge_index)
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
 
    def VE_step(self,X,Y,tau):
        """
        Inputs: 
        X: adjacency matrix
        Y: attributes matrix
        tau: membership matrix
        """
        N = X.shape[0]
        pi = tau.mean(0)
        """
        Compute divergences for every pair X[i,j], mu[k,l]
        """
        net_divergences_elementwise = pairwise_distances(X.reshape(-1,1),\
                                             self.graph_means.reshape(-1,1),\
                                             metric=self.edge_divergence)\
                                            .reshape((N,N,self.n_clusters,self.n_clusters))
        """
        net_divergences has shape N x N x K x K
        tau has shape N x K
        the result must be N x K
        result[i,k] = sum_j sum_l tau[j,l] * net_div[i,j,k,l]
        tensordot performs the multiplication and sum over specified axes.
        "j" appears at axes 0 for tau and at axes 1 for net_divergence
        "l" appears at axes 1 for tau and at axes 3 for net_divergence
        """
        net_divergence_total = np.tensordot(tau, net_divergences_elementwise, axes=[(0,1),(1,3)])
        #print(net_divergence_total)
        att_divergence_total = pairwise_distances(Y,self.attribute_means)
        if self.normalize_:
            #att_divergence_total = self.scaler.fit_transform(att_divergence_total)
            #net_divergence_total = self.scaler.fit_transform(net_divergence_total)
            net_divergence_total -= phi_kl(X).sum(axis=1)[:,np.newaxis]
            att_divergence_total -= phi_euclidean_distance( Y ).sum(axis=1)[:,np.newaxis]
        # print(att_divergence_total,net_divergence_total)
        temp = pi[np.newaxis,:]*np.exp(-net_divergence_total -att_divergence_total)
        if self.thresholding:
            max_ = np.argmax(temp,axis=1)
            tau = np.zeros((N,self.n_clusters))
            tau[np.arange(N),max_] = np.ones(N)
            return tau
        tau = normalize(temp,norm="l1",axis=1)
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
        old_ll = -np.inf
        self.indices_ones = list(X.nonzero())
        self.N = X.shape[0]
        if Z_init is None:
            self.initialize( X, Y )
            self.assignInitialLabels( X, Y )
        else:
            self.predicted_memberships = Z_init
        #init_labels = self.predicted_memberships
        self.attribute_means = self.computeAttributeMeans(Y,self.predicted_memberships)
        self.graph_means = self.computeGraphMeans(X,self.predicted_memberships)
        tau = self.predicted_memberships
        iter_ = 0 
        while True:
            print(iter_)
            new_tau = self.VE_step(X,Y,tau)
            self.attribute_means,self.graph_means = self.M_Step(X,Y,new_tau)
            if np.allclose(tau,new_tau) or iter_ > self.n_iters:
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
        return frommembershipMatriceToVector( self.predicted_memberships)

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
                 divergence_precomputed=True):
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

    def fit( self, A, X, Y, Z_init=None ):
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
        self.edge_index = np.nonzero(A)
        if Z_init is None:
            self.initialize( A, X, Y)
            self.assignInitialLabels( X, Y )
        else:
            self.predicted_memberships = Z_init
        #init_labels = self.predicted_memberships
        self.attribute_means = self.computeAttributeMeans(Y,self.predicted_memberships)
        self.edge_means = self.computeEdgeMeans(A,self.predicted_memberships)
        self.weight_means = self.computeWeightMeans(X,self.predicted_memberships)
        convergence = True
        iteration = 0
        while convergence:
            new_memberships = self.assignments( A, X, Y )

            self.attribute_means = self.computeAttributeMeans( Y, new_memberships )
            self.edge_means = self.computeEdgeMeans( A, new_memberships )
            self.weight_means = self.computeWeightMeans( X, new_memberships)
            
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
        model.initialize( X, Y , self.edge_index)
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
    
    def computeWeightMeans( self, X, Z ):
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
        edges_means = np.tensordot( weights,\
                                    X[self.edge_index[0],self.edge_index[1],:],\
                                    axes=[(2),(0)] )/(np.sum(weights,axis=-1)[:,:,np.newaxis])        
        return edges_means 
    
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
        node_indices = np.argwhere(self.edge_index[0] == node).flatten()
        v_indices_out = self.edge_index[1][node_indices]
        for q in range( self.n_clusters ):
            Ztilde = self.predicted_memberships.copy()
            Ztilde[ node, : ] = 0
            Ztilde[ node, q ] = 1
            z_t = np.argmax(Ztilde,axis=1)
            M = self.edge_means[np.repeat(q, self.N),z_t]
            E = self.weight_means
            """
            X has shape n x n x d
            E has shape k x k x d
            
            the edge divergence computes the difference between node i (from community q) edges and the means
            given node j belongs to community l:
            
            sum_j phi_edge(e_ij, E[q,l,:])  
            """
            att_div = H[node,q]
            graph_div = self.edge_divergence( A[node,:], M )
            edge_div = 0
            try:
                edge_div = np.sum( paired_distances(X[node,self.edge_index[1][node_indices],:],\
                                                    E[q,z_t[v_indices_out],:],\
                                                    metric=self.weight_divergence))
            except:
                print(np.isnan(X[node,self.edge_index[1][node_indices],:]).any(),\
                      np.isnan(E[q,z_t[v_indices_out],:]).any())
            L[ q ] = att_div + 0.5*(graph_div + edge_div)
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
                 n_iters = 25, init_iters=100,
                 reduce_by=None,
                 divergence_precomputed=True):
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
        self.communities_weights = np.ones(n_clusters)/n_clusters
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

    def initialize( self, A, X, Y ):
        model = BregmanInitializer(self.n_clusters,initializer=self.initializer,
                                    edgeDistribution = self.edgeDistribution,
                                    attributeDistribution = self.attributeDistribution,
                                    weightDistribution = self.weightDistribution)
        if self.edge_index is None:
            self.edge_index = np.nonzero(A)
        model.initialize( X, Y , self.edge_index)
        self.predicted_memberships = model.predicted_memberships
        self.memberships_from_graph = model.memberships_from_graph
        self.memberships_from_attributes = model.memberships_from_attributes
        self.graph_init = model.graph_init

    def assignInitialLabels( self, X, Y ):
        return self
    
    def computeTotalDiv(self,node,q,A,X,Z,H):
        z_t = np.argmax(Z,axis=1)
        E = self.weight_means
        ## select the v nodes reached by u
        node_indices_out = np.argwhere(self.edge_index[0] == node).flatten()
        v_indices_out = self.edge_index[1][node_indices_out]
        ## select the v nodes that reaches u
        node_indices_in = np.argwhere(self.edge_index[1] == node).flatten()
        v_indices_in = self.edge_index[0][node_indices_in]
        M = self.edge_means[np.repeat(q, self.N),z_t]
        att_div = H[node,q]
        graph_div = self.edge_divergence( A[node,:], M )
        edge_div = 0
        if len(v_indices_out)>0:
            edge_div += np.sum( paired_distances(X[node,v_indices_out,:],\
                                                 E[q,z_t[v_indices_out],:],metric=self.weight_divergence) )
        if len(v_indices_in)>0:
            edge_div += np.sum( paired_distances(X[v_indices_in,node,:],\
                                                 E[z_t[v_indices_in],q,:],metric=self.weight_divergence) )
        total = att_div + graph_div + edge_div
        print("Total div: ",total)
        return total

    def q_exp(self,x,q):
        return np.power(1 + (1-q)*x, 1/(1-q))
    
    def E_projection(self,A, X, Y):
        Ztilde = np.zeros( (self.N,self.n_clusters), dtype = float)
        H = pairwise_distances(Y,self.attribute_means,metric=self.attribute_divergence)
        for node in range(self.N):
            for q in range(self.n_clusters):
                total_div = self.computeTotalDiv(node,q,A,X,self.predicted_memberships,H)
                Ztilde[node,q] = self.communities_weights[q]*self.q_exp(-total_div,2)
        return normalize(Ztilde, axis=1, norm='l1')
            
    def M_projection(self,A,X,Y,Z):
        idx = np.argmax(Z, axis=-1)
        Z_threshold = np.zeros( Z.shape )
        Z_threshold[ np.arange(Z.shape[0]), idx] = 1
        self.attribute_means = self.computeAttributeMeans(Y, Z_threshold)
        self.edge_means = self.computeEdgeMeans( A, Z_threshold)
        self.weight_means = self.computeWeightMeans( X, Z_threshold)
        self.communities_weights = Z.mean(axis=0)

    def logprob(self,A,X,Y,Z):
        # self.M_projection(A,X,Y,Z)
        H = pairwise_distances(Y,self.attribute_means,metric=self.attribute_divergence)
        log_prob_total = 0
        for node in range(self.N):
            prob_i = 0 
            for q in range(self.n_clusters):
                total_div = self.computeTotalDiv(node,q,A,X,Z,H)
                prob_i += self.communities_weights[q]*np.exp(-total_div)
            log_prob_total += np.log(prob_i)
        return log_prob_total
    
    def stop_criterion(self,A,X,Y,Z_old,Z_new,iteration):
        old_log_prob = self.logprob(A,X,Y,Z_old)
        new_log_prob = self.logprob(A,X,Y,Z_new)
        if np.abs(old_log_prob - new_log_prob) < 0.1 or iteration >= self.n_iters:
            return True
        return False
    
    def fit( self, A, X, Y, Z_init=None ):
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
        self.edge_index = np.nonzero(A)
        self.node_indices = np.arange(self.N)
        if Z_init is None:
            self.initialize( A, X, Y)
            self.assignInitialLabels( X, Y )
        else:
            self.predicted_memberships = Z_init
        #init_labels = self.predicted_memberships
        self.M_projection(A,X,Y,self.predicted_memberships)
        convergence = False
        iteration = 0
        while not convergence:
            Z_new = self.E_projection(A, X, Y)
            self.M_projection(A,X,Y,Z_new)
            convergence = self.stop_criterion(A,X,Y,self.predicted_memberships,Z_new,iteration)
            self.predicted_memberships = Z_new
            iteration += 1 
        return self
    
    def computeAttributeMeans( self, Y, Z ):
        attribute_means = np.dot(Z.T, Y)/(Z.sum(axis=0) + 10 * np.finfo(Z.dtype).eps)[:, np.newaxis]
        return attribute_means
    
    def computeEdgeMeans( self, A, Z ):
        normalisation = np.linalg.pinv ( Z.T @ Z )
        return normalisation @ Z.T @ A @ Z @ normalisation
    
    def computeWeightMeans( self, X, Z ):
        weights = np.tensordot(Z, Z, axes=((), ()))
        """
        weights[i,q,j,l] = tau[i,q]*tau[j,l]
        desired output:
        weights[q,l,i,j] = tau[i,q]*tau[j,l]
        """
        weights = np.transpose(weights,(1,3,0,2))[:,:,self.edge_index[0],self.edge_index[1]]
        X = X[self.edge_index[0],self.edge_index[1],:]
        """
        X is a |E| x d tensor
        weights is a k x k x |E|
        desired output: 
        out[q,l,d] = sum_e X[e,d] * weights[q,l,e]
        """
        edges_means = np.tensordot( weights, X, axes=[(2),(0)] )/(np.sum(weights,axis=-1)[:,:,np.newaxis])
        return edges_means 
    
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
    
class BregmanNodeEdgeAttributeGraphClusteringVariational( BaseEstimator, ClusterMixin ):
    def __init__( self, n_clusters, 
                 edge_divergence = logistic_loss, attribute_divergence = euclidean_distance, 
                 initializer = 'chernoff', 
                 graph_initializer = "spectralClustering", attribute_initializer = 'GMM', 
                 n_iters = 25, init_iters=100,
                 normalize_=True, thresholding=True
                ):
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
        self.edge_divergence = edge_divergence
        self.attribute_divergence = attribute_divergence
        self.n_iters = n_iters
        self.initializer = initializer
        self.graph_initializer = graph_initializer
        self.attribute_initializer = attribute_initializer
        self.init_iters = init_iters
        self.scaler = MinMaxScaler()
        self.edgeDistribution = 'bernoulli'
        self.attributeDistribution = 'gaussian'
        self.normalize_ = normalize_
        self.thresholding = thresholding
        
    def spectralEmbedding( self, X ):
        if (X<0).any():
            X = pairwise_kernels(X,metric='rbf')
        U = SpectralEmbedding(n_components=self.n_clusters,\
								affinity="precomputed")\
								.fit_transform(X)
        return U
    
    def computeAttributeMeans( self, Y, Z ):
        attribute_means = np.dot(Z.T, Y)/(Z.sum(axis=0) + 10 * np.finfo(Z.dtype).eps)[:, np.newaxis]
        return attribute_means

    def computeGraphMeans(self,X,tau):
        graph_means = np.zeros((self.n_clusters,self.n_clusters))
        tau_sum = tau.sum(0)
        weights = np.tensordot(tau, tau, axes=((), ()))
        """
        weights[i,q,j,l] = tau[i,q]*tau[j,l]
        desired output:
        weights[q,l,i,j] = tau[i,q]*tau[j,l]
        """
        weights = np.transpose(weights,(1,3,0,2))
        for q in range(self.n_clusters):
            for l in range(self.n_clusters):
                graph_means[q,l]=np.sum(weights[q,l]*X)/np.sum(weights[q,l])
        #graph_means/=((tau_sum.reshape((-1, 1)) * tau_sum) - tau.T @ tau)
        np.nan_to_num(graph_means,copy=False)
        return graph_means 
    
    def likelihoodAttributes(self,Y,Z):
        M = self.computeAttributeMeans(Y,Z)
        total = np.sum( paired_distances(Y,Z@M) )
        return total

    def likelihoodGraph(self,X,Z):
        graph_mean = self.computeGMeans(X,Z)
        return 1/2 * np.sum( self.edge_divergence( X, Z @ graph_mean @ Z.T ) )
    
    def initialize( self, A, X, Y ):
        model = BregmanInitializer(self.n_clusters,initializer=self.initializer,
                                    edgeDistribution = self.edgeDistribution,
                                    attributeDistribution = self.attributeDistribution,
                                    weightDistribution = self.weightDistribution)
        if self.edge_index is None:
            self.edge_index = np.nonzero(A)
        model.initialize( X, Y , self.edge_index)
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
 
    def VE_step(self,X,Y,tau):
        """
        Inputs: 
        X: adjacency matrix
        Y: attributes matrix
        tau: membership matrix
        """
        N = X.shape[0]
        pi = tau.mean(0)
        """
        Compute divergences for every pair X[i,j], mu[k,l]
        """
        net_divergences_elementwise = pairwise_distances(X.reshape(-1,1),\
                                             self.graph_means.reshape(-1,1),\
                                             metric=self.edge_divergence)\
                                            .reshape((N,N,self.n_clusters,self.n_clusters))
        """
        net_divergences has shape N x N x K x K
        tau has shape N x K
        the result must be N x K
        result[i,k] = sum_j sum_l tau[j,l] * net_div[i,j,k,l]
        tensordot performs the multiplication and sum over specified axes.
        "j" appears at axes 0 for tau and at axes 1 for net_divergence
        "l" appears at axes 1 for tau and at axes 3 for net_divergence
        """
        net_divergence_total = np.tensordot(tau, net_divergences_elementwise, axes=[(0,1),(1,3)])
        #print(net_divergence_total)
        att_divergence_total = pairwise_distances(Y,self.attribute_means)
        if self.normalize_:
            #att_divergence_total = self.scaler.fit_transform(att_divergence_total)
            #net_divergence_total = self.scaler.fit_transform(net_divergence_total)
            net_divergence_total -= phi_kl(X).sum(axis=1)[:,np.newaxis]
            att_divergence_total -= phi_euclidean_distance( Y ).sum(axis=1)[:,np.newaxis]
        # print(att_divergence_total,net_divergence_total)
        temp = pi[np.newaxis,:]*np.exp(-net_divergence_total -att_divergence_total)
        if self.thresholding:
            max_ = np.argmax(temp,axis=1)
            tau = np.zeros((N,self.n_clusters))
            tau[np.arange(N),max_] = np.ones(N)
            return tau
        tau = normalize(temp,norm="l1",axis=1)
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
        old_ll = -np.inf
        self.indices_ones = list(X.nonzero())
        self.N = X.shape[0]
        if Z_init is None:
            self.initialize( X, Y )
            self.assignInitialLabels( X, Y )
        else:
            self.predicted_memberships = Z_init
        #init_labels = self.predicted_memberships
        self.attribute_means = self.computeAttributeMeans(Y,self.predicted_memberships)
        self.graph_means = self.computeGraphMeans(X,self.predicted_memberships)
        tau = self.predicted_memberships
        iter_ = 0 
        while True:
            print(iter_)
            new_tau = self.VE_step(X,Y,tau)
            self.attribute_means,self.graph_means = self.M_Step(X,Y,new_tau)
            if np.allclose(tau,new_tau) or iter_ > self.n_iters:
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
        return frommembershipMatriceToVector( self.predicted_memberships)