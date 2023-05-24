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
from torch_geometric.utils import index_to_mask,select

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

class BregmanEdgeClusteringTorchSparse( BaseEstimator, ClusterMixin ):
    def __init__( self, n_clusters, 
                 edgeDistribution = "bernoulli",
                 attributeDistribution = "gaussian",
                 weightDistribution = "gaussian",
                 initializer = 'chernoff', 
                 graph_initializer = "spectralClustering", 
                 attribute_initializer = 'GMM', 
                 n_iters = None, init_iters=100, 
                 reduce_by = "sum",
                 divergence_precomputed=True
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
        
        ## This func takes the probability matrix of SBM and precompute the divergences,
        ## which output two other matrices of size KxK. The first is the divergence if 
        ## A_ij = 1, and the other if A_ij = 0 
        self.edge_divergence_sparse = dist_to_divergence_dict[self.edgeDistribution]
        
    ## return true when fit proccess is finished
    def stop_criterion(self,old,new,iteration):
        correct = (old.argmin(dim=1) == new.argmin(dim=1)).float().sum()
        if correct < 0.02*self.N or iteration >= self.n_iters:
            return True
        return False
    
    """
    Input: X a numpy array |E| x d = 1 edge_attributes 
    Y a numpy array N x d of node attributes
    edge_index is a tuple (indices_i, indices_j)
    """
    def initialize( self, X, Y , edge_index):
        model = BregmanInitializer(self.n_clusters,initializer=self.initializer)
        model.initialize( X, Y , edge_index)  
        self.predicted_memberships = torch.tensor(model.predicted_memberships).type(dtype)
        self.memberships_from_graph = model.memberships_from_graph
        self.memberships_from_attributes = model.memberships_from_attributes
        self.graph_init = model.graph_init
    
    def assignInitialLabels( self, X, Y ):
        ## For compatibility only
        pass
    
    def fit( self, A, X, Y, Z_init=None ):
        """
        Training step.
        Parameters
        ----------
        Y : torch tensor
            Input data matrix (n, m) of n samples and m features.
        X : torch tensor
            Input (|E|,d) tensor with edge attributes, where |E| is the number of edges  
        A : torch tensor sparse
            Input (2,|E|) encoding the adjacency list
            The pair A[0,i], A[1,i] is the ith edge. 
            Let node u be A[0,i] and node v be A[1,i], then it encodes u->v
        Z_init: torch tensor 
            These are the initial labels
        Returns
        -------
        TYPE
            Trained model.
        """
        # A.requires_grad = True

        self.N = Y.shape[0]
        self.row_indices = torch.arange(self.N).to(device)
        self.edge_index = A.indices().long()
        self.constant_mul = 0.5 if is_undirected(self.edge_index) else 1
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
        self.attribute_means = self.computeAttributeMeans(Y,self.predicted_memberships).to(device)
        
        self.edge_means = self.computeEdgeMeans(A,self.predicted_memberships).to(device)
        self.ones = torch.ones((self.n_clusters,self.n_clusters)).to(device)
        self.zeros = torch.zeros((self.n_clusters,self.n_clusters)).to(device)
        self.ones_div = self.edge_divergence_sparse(self.ones,self.edge_means) 
        self.zeros_div = self.edge_divergence_sparse(self.zeros,self.edge_means)

        self.weight_means = self.computeWeightMeans(X,self.predicted_memberships).to(device)
        
        new_memberships = self.assignments( A, X, Y ).to(device)
        
        convergence = True
        iteration = 0
        while convergence:
            new_memberships = self.assignments( A, X, Y )

            self.attribute_means = self.computeAttributeMeans( Y, new_memberships )
            
            self.edge_means = self.computeEdgeMeans( A, new_memberships )
            self.ones_div = self.edge_divergence_sparse(self.ones,self.edge_means) 
            self.zeros_div = self.edge_divergence_sparse(self.zeros,self.edge_means)
            
            self.weight_means = self.computeWeightMeans( X, new_memberships)
            
            iteration += 1
            if self.stop_criterion(self.predicted_memberships,new_memberships,iteration):
                convergence = False
            self.predicted_memberships = new_memberships
        A = None
        X = None
        Y = None
        # self.attribute_means = self.graph_means = self.edge_means = new_memberships = None
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
        weights = torch.tensordot(Z, Z, dims=((), ()))
        """
        weights[i,q,j,l] = z[i,q]*z[j,l]
        desired output:
        weights[q,l,i,j] = z[i,q]*z[j,l]

        Apply permute to change indexes, and select only the weights for the existing edges
        """
        weights = weights.permute(1,3,0,2)[:,:,self.edge_index[0,:],self.edge_index[1,:]]
        """
        X is a |E| x d tensor
        weights is a k x k x |E|
        desired output: 
        out[q,l,d] = sum_e X[e,d] * weights[q,l,e]
        """
        weights_means = torch.tensordot( weights, X, dims=[(2,),(0,)] )/(torch.sum(weights,dim=-1)[:,:,None])
        return weights_means

    """
    Inputs:
    Y : torch tensor
        Input data matrix (n, m) of n samples and m features.
    X : torch tensor
        Input (|E|,d) tensor with edge attributes, where |E| is the number of edges  
    A : torch tensor sparse
        Input (2,|E|) encoding the adjacency list
        The pair A[0,i], A[1,i] is the ith edge. 
        Let node u be A[0,i] and node v be A[1,i], then it encodes u->v
    """   
    def assignments( self, A, X, Y ):
        ## z must be in the same device as A,X,Y
        z = torch.zeros( (self.N,self.n_clusters)).to(device)
        H = self.attribute_divergence_pairwise(Y, self.attribute_means)
        for node in range( self.N ):
            k = self.singleNodeAssignment( A, X, H, node )
            z[node,k]=1
        return z     
    
    def singleNodeAssignment( self, A, X, H, node ):
        L = torch.zeros( self.n_clusters )
        ## get all edges leaving node
        edge_indices_out = torch.argwhere(self.edge_index[0,:] == node).flatten()
        ## get the actual v nodes in u->v
        v_indices_out = self.edge_index[1,edge_indices_out]
        edge_indices_in = torch.argwhere(self.edge_index[1,:] == node).flatten()
        ## get the actual v nodes in v->u
        v_indices_in = self.edge_index[0,edge_indices_in]
        a_out_mask = index_to_mask(v_indices_out,size=self.N).to(device)
        a_in_mask = index_to_mask(v_indices_in,size=self.N).to(device)
        # a_out = torch.zeros(self.N,requires_grad=False).to(device)
        # a_out[v_indices_out] = 1
        # a_in = torch.zeros(self.N,requires_grad=False).to(device)
        # a_in[v_indices_in] = 1
        ## index_to_mask, select
        for q in range( self.n_clusters ):
            Ztilde = self.predicted_memberships
            Ztilde[ node, : ] = 0
            Ztilde[ node, q ] = 1
            z_t = torch.argmax(Ztilde,dim=1)
            # M_out = self.graph_means[torch.tensor([q]).expand(self.N),z_t]
            # M_out.requires_grad =True
            # M_in = self.graph_means[z_t,torch.tensor([q]).expand(self.N)]
            # M_in.requires_grad =True
            E = self.weight_means
            """
            X has shape |E| x d
            E has shape k x k x d
            
            the edge divergence computes the difference between node i (from community q) edges
            and the expected value given node j belongs to community l:
            sum_j phi_edge(e_ij, E[q,l,:])  
            """
            att_div = H[node,q]
            edge_div = self.reduce_by(
                            torch.hstack(
                                    (
                                    select(self.ones_div[torch.tensor([q]).expand(self.N),z_t],\
                                        a_in_mask,dim=0),
                                    select(self.zeros_div[torch.tensor([q]).expand(self.N),z_t],\
                                        ~a_in_mask,dim=0))
                            )
            )
                               
            weight_div=0
            if len(v_indices_out) > 0:
                weight_div += self.weight_divergence(X[edge_indices_out,:],\
                                                     E[q,z_t[v_indices_out],:])
            L[ q ] = att_div + (edge_div + weight_div) * 0.5
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


class SoftBregmanClusteringTorch( BaseEstimator, ClusterMixin ):
    def __init__( self, n_clusters, 
                 graph_divergence = logistic_loss, attribute_divergence = euclidean_distance, 
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

    """ 
    def spectralEmbedding( self, X ):
        if (X<0).any():
            X = rbf_kernel(X[:,None],X[None,:])
        L, V = torch.linalg.eig(X)
        U = V[:,-self.n_clusters:]
        return U """
    
    def computeAttributeMeans( self, Y, Z ):
        attribute_means = (Z.T@Y)/(Z.sum(dim=0) + 10 * torch.finfo(Z.dtype).eps)[:, None]
        return attribute_means

    def computeEdgeMeans(self,X,tau):
        #graph_means = torch.zeros((self.n_clusters,self.n_clusters))
        tau_sum = tau.sum(0)
        weights = torch.tensordot(tau, tau, dims=((), ()))
        """
        weights[i,q,j,l] = tau[i,q]*tau[j,l]
        desired output:
        weights[q,l,i,j] = tau[i,q]*tau[j,l]
        """
        weights = weights.permute(1,3,0,2)
        graph_means = torch.sum(weights*X, dim=(2,3))/torch.sum(weights,dim=(2,3))
        #for q in range(self.n_clusters):
        #    for l in range(self.n_clusters):
        #        graph_means[q,l]=torch.sum(weights[q,l]*X)/torch.sum(weights[q,l])
        #graph_means/=((tau_sum.reshape((-1, 1)) * tau_sum) - tau.T @ tau)
        torch.nan_to_num(graph_means,out=graph_means)
        return graph_means 
    
    def likelihoodAttributes(self,Y,Z):
        M = self.computeAttributeMeans(Y,Z).to(device)
        total = torch.sum( self.attribute_divergence(Y,Z@M) ).to(device)
        return total

    def likelihoodGraph(self,X,Z):
        graph_mean = self.computeEdgeMeans(X,Z)
        return 1/2 * torch.sum( self.edge_divergence( X, Z @ graph_mean @ Z.T ) )
 
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
            return tau.to(device)
        tau = temp/(temp.sum(dim=1)[:,None])
        return tau

    def M_Step(self,X,Y,tau):
        att_means = self.computeAttributeMeans(Y,tau).to(device)
        graph_means = self.computeGraphMeans(X,tau).to(device)
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
        self.row_indices = torch.arange(self.N).to(device)
        if Z_init is None:
            model = BregmanNodeAttributeGraphClustering(n_clusters=self.n_clusters)
            model.initialize( X, Y )
            model.assignInitialLabels( X, Y )  
            self.predicted_memberships = torch.tensor(model.predicted_memberships).type(dtype)
        else:
            self.predicted_memberships = torch.tensor(Z_init).type(dtype)
        #init_labels = self.predicted_memberships
        X = torch.tensor(X).type(dtype)
        Y = torch.tensor(Y).type(dtype)
        self.attribute_means = self.computeAttributeMeans(Y,self.predicted_memberships).to(device)
        self.graph_means = self.computeGraphMeans(X,self.predicted_memberships).to(device)
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
        return torch.argmax(self.predicted_memberships,dim=1).to("cpu").numpy()


class SoftBregmanClusteringTorchSparse( BaseEstimator, ClusterMixin ):
    def __init__( self, n_clusters, 
                 graph_divergence = logistic_loss, attribute_divergence = euclidean_distance, 
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

    def spectralEmbedding( self, X ):
        if (X<0).any():
            X = rbf_kernel(X[:,None],X[None,:])
        L, V = torch.linalg.eig(X)
        U = V[:,-self.n_clusters:]
        return U
    
    def sparse_dense_mul(self, s, d):
        i = s._indices()
        v = s._values()
        dv = d[i[0,:], i[1,:]]  # get values from relevant entries of dense matrix
        return torch.sparse.FloatTensor(i, v * dv, s.size())
        
    def computeAttributeMeans( self, Y, Z ):
        attribute_means = (Z.T@Y)/(Z.sum(dim=0) + 10 * torch.finfo(Z.dtype).eps)[:, None]
        return attribute_means

    def computeGraphMeans(self,X,tau):
        graph_means = torch.zeros((self.n_clusters,self.n_clusters))
        #tau_sum = tau.sum(0)
        #X = X.to_sparse()
        weights = torch.tensordot(tau, tau, dims=((), ()))
        """
        weights[i,q,j,l] = tau[i,q]*tau[j,l]
        desired output:
        weights[q,l,i,j] = tau[i,q]*tau[j,l]
        """
        weights = weights.permute(1,3,0,2)
        for q in range(self.n_clusters):
            for l in range(self.n_clusters):
                graph_means[q,l]=torch.sparse.sum(self.sparse_dense_mul(X,weights[q,l]))/torch.sum(weights[q,l])
        #graph_means/=((tau_sum.reshape((-1, 1)) * tau_sum) - tau.T @ tau)
        torch.nan_to_num(graph_means,out=graph_means)
        return graph_means 
    
    def likelihoodAttributes(self,Y,Z):
        M = self.computeAttributeMeans(Y,Z).to(device)
        total = torch.sum( self.attribute_divergence(Y,Z@M) ).to(device)
        return total

    def likelihoodGraph(self,X,Z):
        graph_mean = self.computeGraphMeans(X,Z).to(device)
        return 1/2 * torch.sum( self.edge_divergence( X.to_dense(), Z @ graph_mean @ Z.T ) )
 
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
        X = X.to_dense().type(dtype) 
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
            return tau.to(device)
        tau = temp/(temp.sum(dim=1)[:,None])
        return tau

    def M_Step(self,X,Y,tau):
        att_means =  self.computeAttributeMeans(Y,tau).to(device)
        graph_means = self.computeGraphMeans(X,tau).to(device)
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
        self.row_indices = torch.arange(self.N).to(device)
        if Z_init is None:
            model = BregmanNodeAttributeGraphClustering(n_clusters=self.n_clusters)
            model.initialize( X, Y )
            model.assignInitialLabels( X, Y )  
            self.predicted_memberships = torch.tensor(model.predicted_memberships).type(dtype)
        else:
            self.predicted_memberships = torch.tensor(Z_init).type(dtype)
        #init_labels = self.predicted_memberships
        X = torch.tensor(X).type(dtype)
        Y = torch.tensor(Y).type(dtype)
        X = X.to_sparse()
        self.attribute_means = self.computeAttributeMeans(Y,self.predicted_memberships).to(device)
        self.graph_means = self.computeGraphMeans(X,self.predicted_memberships).to(device)
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
        return torch.argmax(self.predicted_memberships,dim=1).to("cpu").numpy()

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
