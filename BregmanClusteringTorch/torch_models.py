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
from BregmanInitializer.init_cluster import *
import networkx as nx
from sys import platform
import torch
from torch.nn.functional import one_hot
from torch_geometric.utils import index_to_mask,select,is_undirected

device = "cpu"
dtype = torch.FloatTensor
if platform == "win32":
    import torch_directml
    device = torch_directml.device(torch_directml.default_device())
    #device = torch_directml.device()
elif torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    device = "cuda"

from torch_geometric.utils import *
from torch import Tensor
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import warnings
warnings.filterwarnings("ignore")
    
class BregmanNodeEdgeAttributeGraphClusteringTorch( BaseEstimator, ClusterMixin ):
    def __init__( self, n_clusters, 
                 edgeDistribution = "bernoulli",
                 attributeDistribution = "gaussian",
                 weightDistribution = "gaussian",
                 initializer = 'chernoff', 
                 graph_initializer = "spectralClustering", attribute_initializer = 'GMM', 
                 n_iters = 25, init_iters=100,
                 reduce_by="sum",
                 divergence_precomputed=True,
                 use_random_init=False,
                 batch_size=32, 
                 strategy=None):
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
        if n_iters is not None:
            self.n_iters = n_iters
        else:
            self.n_iters = 30

        self.N = 0
        self.row_indices = torch.arange(2)
        self.initializer = initializer
        self.graph_initializer = graph_initializer
        self.attribute_initializer = attribute_initializer
        self.init_iters = init_iters
        self.edge_index = None 
        if reduce_by == "sum":
            self.reduce_by = torch.sum
        else:
            self.reduce_by = torch.mean
        ## Variable that stores which initialization was chosen
        self.graph_init = False
        self.edgeDistribution = edgeDistribution
        self.attributeDistribution = attributeDistribution
        self.weightDistribution = weightDistribution
        
        ## SET PHI
        self.edge_phi = make_phi_with_reduce(self.reduce_by, dist_to_phi_dict[self.edgeDistribution])
        self.weight_phi = make_phi_with_reduce(self.reduce_by, dist_to_phi_dict[self.weightDistribution])
        self.attribute_phi = make_phi_with_reduce(self.reduce_by, dist_to_phi_dict[self.attributeDistribution])

        if divergence_precomputed:
            ## SET DIVERGENCES precomputed

            ## inputs are vectors of length |V|, output is scalar
            self.edge_divergence = make_div_with_reduce(self.reduce_by,\
                                                        dist_to_divergence_dict[self.edgeDistribution])
            self.weight_divergence = make_div_with_reduce(self.reduce_by,\
                                                          dist_to_divergence_dict[self.weightDistribution])
            self.attribute_divergence = make_div_with_reduce(self.reduce_by,\
                                                             dist_to_divergence_dict[self.attributeDistribution])

        else:
            ##SET DIVERGENCES from definition: D_φ(X,Y) = φ(x) - φ(y) - <x - y, φ'(y)> 
            
            self.edge_divergence = make_breg_div(self.edge_phi)
            self.weight_divergence = make_breg_div(self.weight_phi)
            self.attribute_divergence = make_breg_div(self.attribute_phi)
    
        ## X is |E| x D, Y is |E| x D, output is scalar
        self.weight_divergence_reduced = make_pair_breg(
                                                    self.reduce_by,\
                                                    self.weight_divergence
                                                )

        ## X is n x m, y is k x m, output is n x k containing all the pairwise bregman divergences
        self.attribute_divergence_pairwise = make_pairwise_breg(self.attribute_divergence)
        self.precomputed_edge_div = torch.zeros((2,self.n_clusters,self.n_clusters))
        self.edge_means = torch.eye(self.n_clusters)
        ## This func takes the probability matrix of SBM and precompute the divergences,
        ## which output two other matrices of size KxK. The first is the divergence if 
        ## A_ij = 0, and the other if A_ij = 1
        # Input 2x1, K^2 x 1 -> 2xK^2 
        self.edge_div_pairwise = make_pairwise_breg(dist_to_divergence_dict[self.edgeDistribution])  
        self.use_random_init = use_random_init
        self.zero_and_one = torch.Tensor([0,1]).reshape(-1,1).to(device)
        self.batch_size = batch_size

    ## return true when fit proccess is finished
    def stop_criterion(self,old,new,iteration):
        correct = (old.argmin(dim=1) == new.argmin(dim=1)).float().sum()
        if correct < 0.02*self.N or iteration >= self.n_iters:
            return True
        return False
    
    def assignInitialLabels( self, X, Y ):
        ## For compatibility only
        pass
    
    ## This func takes the probability matrix of SBM and precompute the divergences,
    ## which output two other matrices of size KxK. The first is the divergence if 
    ## A_ij = 0, and the other if A_ij = 1
    # Input 2x1, K^2 x 1 -> 2xK^2 
    def precompute_edge_divergences(self):
        self.precomputed_edge_div = self.edge_div_pairwise(self.zero_and_one,\
                                             self.edge_means.reshape(-1,1))\
                                            .reshape(
                                                        (2,\
                                                        self.n_clusters,self.n_clusters
                                                        )
                                                    )    
    
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
        self.N = Y.shape[0]
        self.row_indices = torch.arange(self.N).to(device)
        self.edge_index = A.indices().long()
        self.constant_mul = 1 if is_undirected(self.edge_index) else 0.5
        if Z_init is None:
            e_ind = self.edge_index.detach().numpy()
            self.initialize(X.detach().numpy().reshape(len(e_ind[0]),1),Y.detach().numpy(), (e_ind[0,:],e_ind[1,:]))
        else:
            self.predicted_memberships = Z_init.type(dtype)
        
        ## send data to device
        A = A.type(dtype)
        X = X.type(dtype)
        Y = Y.type(dtype)
        X.requires_grad = True
        Y.requires_grad = True
        ## compute initial params
        self.attribute_means = self.computeAttributeMeans(Y,\
                                                          self.predicted_memberships).to(device)
        self.edge_means = self.computeEdgeMeans(A,\
                                                self.predicted_memberships).to(device)
        self.weight_means = self.computeWeightMeans(X,\
                                                    self.predicted_memberships).to(device)
        new_memberships = self.assignments( X, Y ).to(device)
        self.precompute_edge_divergences()
        convergence = True
        iteration = 0
        while convergence:
            self.edge_means = self.computeEdgeMeans(A,self.predicted_memberships)
            self.weight_means = self.computeWeightMeans(X,self.predicted_memberships)
            new_memberships = self.assignments( X, Y )
            self.precompute_edge_divergences()
            
            iteration += 1
            if self.stop_criterion(self.predicted_memberships,new_memberships,iteration):
                convergence = False
            self.predicted_memberships = new_memberships
        A = None
        X = None
        Y = None
        return self
    
    ## return a sample of size self.batch_size from the data
    def sample(self):
        return torch.randperm(self.N)[:self.batch_size]
    """
    TO DO: Implement partial fit
    """
    def mini_batch_fit( self, A, X, Y, Z_init=None):
        """
        Training step.
        Parameters
        ----------
        Y : ARRAY
            Input data matrix (n, m) of n samples and m features.
        X : ARRAY
            Input (E,d) tensor with edges. If a edge doesnt exist, is filled with NAN 
        A : ARRAY
            Input (n,n) matrix encoding the adjacency matrix
        Returns
        -------
        TYPE
            Trained model.
        """
        self.N = Y.shape[0]
        self.row_indices = torch.arange(self.N).to(device)
        self.edge_index = A.indices().long()
        self.constant_mul = 1 if is_undirected(self.edge_index) else 0.5
        if Z_init is None:
            e_ind = self.edge_index.detach().numpy()
            self.initialize(X.detach().numpy().reshape(len(e_ind[0]),1),Y.detach().numpy(), (e_ind[0,:],e_ind[1,:]))
        else:
            self.predicted_memberships = Z_init.type(dtype)
        
        ## stores the number of points in each cluster
        self.N_C = self.predicted_memberships.sum(axis=0)
        ## compute initial params
        self.attribute_means = self.computeAttributeMeans(Y,self.predicted_memberships).to(device)
        self.edge_means = self.computeEdgeMeans(A,self.predicted_memberships).to(device)
        self.weight_means = self.computeWeightMeans(X,self.predicted_memberships).to(device)
        new_memberships = self.assignments( X, Y ).to(device)
        self.precompute_edge_divergences()
        
        convergence = True
        iteration = 0
        while convergence:
            self.idx_sample = self.sample()
            
            ## send data to device
            A = A.type(dtype)
            edge_indices_in = torch.argwhere(torch.isin(self.edge_index[1],self.idx_sample)).flatten()
            edge_indices_out = torch.argwhere(torch.isin(self.edge_index[0],self.idx_sample)).flatten()
            edge_indices = torch.cat((edge_indices_in,edge_indices_out), 0)
            X_batch = X[edge_indices,:].type(dtype)
            Y_batch = Y[self.idx_sample,:].type(dtype)
            X_batch.requires_grad = True
            Y_batch.requires_grad = True
            ## Assign each data point in the mini-batch to
            #  the closest cluster. z_batch has shape (batch_size x m).
            z_batch = self.partial_assignments(X,Y)
            # update number of data for each cluster center
            self.N_C += z_batch.sum(0)
            #calculate learning rate for each cluster
            lr=1/self.N_C
            outer_prod = torch.outer(self.N_C, self.N_C)
            att_sums = z_batch.T@Y_batch
            M = z_batch.T@torch.sparse.mm(A,z_batch)
            lr_edge = 1/outer_prod  
            self.precompute_edge_divergences()
            
            iteration += 1
            if self.stop_criterion(self.predicted_memberships,new_memberships,iteration):
                convergence = False
            self.predicted_memberships = new_memberships
        A = None
        X = None
        Y = None
        return self
    
    def initialize( self, X, Y ,edge_index):
        model = BregmanInitializer(self.n_clusters,initializer=self.initializer,
                                    edgeDistribution = self.edgeDistribution,
                                    attributeDistribution = self.attributeDistribution,
                                    weightDistribution = self.weightDistribution)
        if self.edge_index is None:
            self.edge_index = edge_index
        Z_init = None
        if self.use_random_init == True:
            Z_init = one_hot(torch.random.randint(self.n_clusters,size=self.N)).to("cpu").numpy()
        model.initialize( X, Y , self.edge_index, Z_init=Z_init)
        self.predicted_memberships = torch.tensor(model.predicted_memberships).type(dtype)
        self.memberships_from_graph = model.memberships_from_graph
        self.memberships_from_attributes = model.memberships_from_attributes
        self.graph_init = model.graph_init

    def assignInitialLabels( self, X, Y ):
        return self
    
    """
    Inputs:
    Y is a N x m tensor of node attributes and 
    Z is a N x K node memberships
    """
    def computeAttributeMeans( self, Y, Z ):
        attribute_means = (Z.T@Y)/(Z.sum(dim=0) + 10 * torch.finfo(Z.dtype).eps)[:, None]
        return attribute_means
    
    """
    Inputs:
    A is a 2 x |E| sparse tensor enconding adjacency matrix and 
    Z is a N x K node memberships
    """
    def computeEdgeMeans( self, A, Z ):
        normalisation = torch.linalg.pinv(Z.T@Z)
        M = Z.T@torch.sparse.mm(A,Z)
        return normalisation @ M @ normalisation
    
    """
    Inputs:
    X is a |E| x d tensor with edge attributes, where |E| is the number of edges   
    Z is a N x K node memberships
    """  
    def computeWeightMeans( self, X, Z ):
        weight = torch.tensordot(Z, Z, dims=((), ()))
        """
        weights[i,q,j,l] = z[i,q]*z[j,l]
        desired output:
        weights[q,l,i,j] = z[i,q]*z[j,l]

        Apply permute to change indexes, and select only the weights for the existing edges
        """
        weight = weight.permute(1,3,0,2)[:,:,self.edge_index[0,:],self.edge_index[1,:]]
        """
        X is a |E| x d tensor
        weights is a k x k x |E|
        desired output: 
        out[q,l,d] = sum_e X[e,d] * weights[q,l,e]
        """
        weight_means = torch.tensordot( weight, X, dims=[(2,),(0,)] )/(torch.sum(weight,dim=-1)[:,:,None])
        # if (self.edge_means==0).any():
        #     null_model = X.mean(axis=0)
        #     undefined_idx = np.where(self.edge_means==0)
        #     weight_means[undefined_idx[0],undefined_idx[1],:] = null_model
        return weight_means
    
    def assignments( self, X, Y ):
        ## z must be in the same device as A,X,Y
        z = torch.zeros( (self.N,self.n_clusters)).to(device)
        H = self.attribute_divergence_pairwise(Y, self.attribute_means)
        for node in range( self.N ):
            k = self.singleNodeAssignment( X, H, node )
            z[node,k]=1
        return z

    def partial_assignments(self,X,Y):
        ## z must be in the same device as A,X,Y
        ## Y has shape (batch_size x m)
        batch_size = Y.shape[0]
        z = torch.zeros( (batch_size,self.n_clusters)).to(device)
        H = self.attribute_divergence_pairwise(Y, self.attribute_means)
        for i,node in enumerate(self.idx_sample):
            k = self.singleNodeAssignment( X, H, node )
            z[i,k]=1
        return z                   
    
    def singleNodeAssignment( self, X, H, node ):
        L = torch.zeros( self.n_clusters )
        edge_indices_in = torch.argwhere(self.edge_index[1] == node).flatten()
        v_idx_in = self.edge_index[0][edge_indices_in]
        
        edge_indices_out = torch.argwhere(self.edge_index[0] == node).flatten()
        v_idx_out = self.edge_index[1][edge_indices_out]
        
        mask_in = index_to_mask(v_idx_in)
        mask_out = index_to_mask(v_idx_out)
        
        v_idx_in_comp = torch.where(~mask_in)
        v_idx_out_comp = torch.where(~mask_out)

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
            weight_div = 0
            if E[q,z_t[v_idx_out],:].isnan().any():
                weight_div = torch.inf

            if len(v_idx_out) > 0:
                weight_div += torch.sum( self.weight_divergence(
                                            X[edge_indices_out,:],\
                                            E[q,z_t[v_idx_out],:]
                                            )
                                        )
            if E[z_t[v_idx_in],q,:].isnan().any():
                weight_div = torch.inf
            if len(v_idx_in) > 0:
                weight_div += torch.sum( self.weight_divergence(
                                                X[edge_indices_in,:],\
                                                E[z_t[v_idx_in],q,:]
                                                )
                                        )
            L[ q ] = att_div + (weight_div + edge_div)
        return torch.argmin( L )
    
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
        return torch.argmax(self.predicted_memberships,dim=1).to("cpu").numpy()

class BregmanNodeEdgeAttributeGraphClusteringTorchSoft( BaseEstimator, ClusterMixin ):
    def __init__( self, n_clusters, 
                 edgeDistribution = "bernoulli",
                 attributeDistribution = "gaussian",
                 weightDistribution = "gaussian",
                 initializer = 'chernoff', 
                 graph_initializer = "spectralClustering", attribute_initializer = 'GMM', 
                 n_iters = 100, init_iters=100,
                 reduce_by=None,
                 divergence_precomputed=True,
                 use_random_init=False,
                 strategy=0):
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
        self.node_indices = np.arange(self.N)
        if Z_init is None:
            self.initialize( A, X, Y)
            self.assignInitialLabels( X, Y )
        else:
            self.predicted_memberships = Z_init
        #init_labels = self.predicted_memberships
        self.M_projection(X,Y,self.predicted_memberships)
        convergence = False
        iteration = 0
        old_log_prob = np.inf
        while not convergence:
            Z_new = self.E_projection(X, Y)
            new_log_prob = self.logprob(X,Y)
            self.M_projection( X,Y,Z_new)
            convergence = self.stop_criterion(X,Y,\
                                              self.predicted_memberships,Z_new,\
                                                old_log_prob,new_log_prob,\
                                                iteration)
            self.predicted_memberships = Z_new
            old_log_prob = new_log_prob
            iteration += 1 
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
        model.initialize( X, Y , self.edge_index, Z_init=Z_init)
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
    
    def computeWeightMeans( self, X, Z):
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
    
    def computeTotalDiv(self,node,X,Z,H):
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
            weight_div = 0
            if len(v_idx_out) > 0:
                weight_div += np.sum( paired_distances(X[node,v_idx_out,:],\
                                                        E[q,z_t[v_idx_out],:],\
                                                        metric=self.weight_divergence))
            if len(v_idx_in) > 0:
                weight_div += np.sum( paired_distances(X[v_idx_in,node,:],\
                                                        E[z_t[v_idx_in],q,:],\
                                                        metric=self.weight_divergence))
            L[ q ] = att_div + weight_div + edge_div
        return L

    def q_exp(self,x,q):
        return np.power(1 + (1-q)*x, 1/(1-q))
    
    def E_projection(self, X, Y):
        Ztilde = np.zeros( (self.N,self.n_clusters), dtype = float)
        H = pairwise_distances(Y,self.attribute_means,metric=self.attribute_divergence)
        for node in range(self.N):
            Ztilde[node,:] = self.computeTotalDiv(node,X,self.predicted_memberships,H)
        c = Ztilde.max(axis=1)
        Ztilde -= c[:,None]
        Ztilde = self.communities_weights.reshape(1, -1)*np.exp(-Ztilde)
        return normalize(Ztilde, axis=1, norm='l1')
            
    def M_projection(self,X,Y,Z):
        Z_threshold = Z
        # idx = np.argmax(Z, axis=-1)
        # Z_threshold = np.zeros( Z.shape )
        # Z_threshold[ np.arange(Z.shape[0]), idx] = 1
        self.attribute_means = self.computeAttributeMeans(Y, Z_threshold)
        self.edge_means = self.computeEdgeMeans(Z_threshold)
        self.weight_means = self.computeWeightMeans( X, Z_threshold)
        self.precompute_edge_divergences()
        self.communities_weights = Z.mean(axis=0)
        # print("\n-----------------------------------------------------------\n",\
        #       "\nEDGE_MEANS: ",self.edge_means,
        #       "\nWeight_MEANS: ",self.weight_means,
        #       "\nAtt_MEANS: ",self.attribute_means)

    def logprob(self,X,Y):
        H = pairwise_distances(Y,\
                               self.attribute_means,\
                                metric=self.attribute_divergence)
        log_prob_total = 0
        for node in range(self.N):
            divs = self.computeTotalDiv(node,X,self.predicted_memberships,H)
            c = divs.max()
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
    
## GNN propriamente dita
class my_GCN(torch.nn.Module):
    def __init__(self,n_feat,n_clusters):
        super().__init__()
        self.conv1 = GCNConv(n_feat, 10)
        self.conv2 = GCNConv(10,n_clusters)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        #x = F.softmax(x,dim=-1)
        x = F.normalize(torch.exp(x),p=1,dim=-1)
        return x
        
class GNNBregmanClustering( BaseEstimator, ClusterMixin ):
    def __init__( self, n_clusters, 
                 graph_divergence = logistic_loss, attribute_divergence = euclidean_distance, 
                 initializer = 'chernoff', 
                 graph_initializer = "spectralClustering", attribute_initializer = 'GMM', 
                 n_iters = 25, init_iters=100,
                 normalize_=True, thresholding=True,
                 reduce_by = torch.sum,
                 epochs=100
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
        self.edge_divergence = graph_divergence
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
        self.epochs = epochs

    def make_model(self,n_feat):
        return my_GCN(n_feat,self.n_clusters).type(dtype)
    
    def loss_fn(self,X,Y,Z):
        W = self.get_dist_matrix(X,Y,Z)
        loss_ = torch.linalg.norm(Z@Z.T - W@W.T)
        return loss_
    
    def fit(self,G,Y,Z_init=None):
        X = nx.adjacency_matrix(G).todense()
        X[X!=0] = 1
        self.N = X.shape[0]
        if Z_init is None:
            model = BregmanNodeAttributeGraphClustering(n_clusters=self.n_clusters)
            model.initialize( X, Y )
            model.assignInitialLabels( X, Y )
            Z_graph = torch.tensor(model.memberships_from_graph).type(dtype)
            Z_att = torch.tensor(model.memberships_from_attributes).type(dtype)
            Y = torch.tensor(Y).type(dtype)
            X = torch.tensor(X).type(dtype)            
            self.attribute_means = self.computeAttributeMeans(Y,Z_att).to(device)
            self.graph_means = self.computeGraphMeans(X,Z_graph).to(device)  
            self.predicted_memberships = torch.tensor(model.predicted_memberships).type(dtype)
        else:
            self.predicted_memberships = torch.tensor(Z_init).type(dtype)
        model = self.model = self.make_model(Y.shape[1])
        X = torch.tensor(X,requires_grad=False).type(dtype)  ## Network data
        Y = torch.tensor(Y,requires_grad=False).type(dtype)  ## attributes
        edge_index = torch.nonzero(X) 
        #,edge_attr=X[edge_index[:,0],edge_index[:,1]]
        graph_data = Data(x=Y, edge_index=edge_index.T).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        total = 0
        #print(self.graph_means)
        graph_data.x = graph_data.x.type(dtype) 
        model.train()
        while total < self.epochs:
            optimizer.zero_grad()
            Z = model(graph_data)
            """if total%100==0 and total>0:
                self.attribute_means,self.graph_means = self.M_Step(X,Y,Z)
                print(self.graph_means)
            """
            loss = self.loss_fn(X,Y,Z)
            loss.backward()
            optimizer.step()
            total += 1
        self.predicted_memberships = Z
        return self
    
    def computeAttributeMeans( self, Y, Z ):
        attribute_means = (Z.T@Y)/(Z.sum(dim=0) + 10 * torch.finfo(Z.dtype).eps)[:, None]
        return attribute_means.detach()

    def computeGraphMeans(self,X,tau):
        #tau_sum = tau.sum(0)
        weights = torch.tensordot(tau, tau, dims=((), ()))
        """
        weights[i,q,j,l] = tau[i,q]*tau[j,l]
        desired output:
        weights[q,l,i,j] = tau[i,q]*tau[j,l]
        """
        weights = weights.permute(1,3,0,2)
        graph_means = (torch.sum(weights*X, dim=(2,3))/torch.sum(weights,dim=(2,3))).detach()
        torch.nan_to_num(graph_means,out=graph_means)
        return graph_means 
    
    def likelihoodAttributes(self,Y,Z):
        M = self.computeAttributeMeans(Y,Z).to(device)
        total = torch.sum( self.attribute_divergence(Y,Z@M) ).to(device)
        return total

    def likelihoodGraph(self,X,Z):
        graph_mean = self.computeGraphMeans(X,Z)
        return 1/2 * torch.sum( self.edge_divergence( X, Z @ graph_mean @ Z.T ) )
 
    def get_dist_matrix(self,X,Y,tau):
        """
        Inputs: 
        X: adjacency matrix
        Y: attributes matrix
        W: distance matrix
        """
        #N = X.shape[0]
        pi = tau.mean(0)
        """
        Compute net divergences for every pair X[i,j], mu[k,l]
        """
        net_divergences_elementwise = self.edge_divergence(X.reshape(-1,1)[:,None],\
                                             self.graph_means.reshape(-1,1)[None,:])\
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
        att_divergence_total = self.reduce_by(
                                                self.attribute_divergence(Y[:,None],\
                                                self.attribute_means[None,:]),\
                                                dim=-1
                                            )
        if self.normalize_:
            net_divergence_total = F.normalize(net_divergence_total, p=1, dim=-1)
            att_divergence_total = F.normalize(att_divergence_total, p=1, dim=-1)
        distance_matrix = F.normalize(torch.hstack([net_divergence_total,att_divergence_total]),p=1,dim=-1)
        return distance_matrix

    def M_Step(self,X,Y,tau):
        att_means = self.computeAttributeMeans(Y,tau).to(device)
        graph_means = self.computeGraphMeans(X,tau).to(device)
        return att_means,graph_means

    def predict(self, G, Y):
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
        return torch.argmax(self.predicted_memberships,dim=1).to("cpu").numpy()
