import numpy as np
import scipy as sp
from .divergences import *
from sklearn.metrics.pairwise import pairwise_kernels, paired_distances, pairwise_distances
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.manifold import SpectralEmbedding
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import csr_matrix, csc_matrix, csr_array 
from sklearn.cluster import SpectralClustering
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClusterMixin
import igraph as ig
import leidenalg as la
import networkx as nx

def fromVectorToMembershipMatrice(z,k):
    z = z.reshape(-1, 1)
    ohe = OneHotEncoder(max_categories=k, sparse_output=False).fit(z)
    return ohe.transform(z)

def frommembershipMatriceToVector(Z):
    return Z.argmax(axis=1)

def fit_leiden(edge_index,E):
    G = ig.Graph(zip(edge_index[0].tolist(), edge_index[1].tolist()), 
                     edge_attrs={'weight': E.flatten().tolist()})
    partition = la.find_partition(G, la.ModularityVertexPartition)
    preds = np.array(partition.membership).reshape(-1, 1)
    print("OK")
    return preds

class BregmanInitializer():
    def __init__( self, n_clusters,initializer="AIC",\
                        edgeDistribution = "bernoulli",
                        attributeDistribution = "gaussian",
                        weightDistribution = "gaussian"):
        self.initializer = initializer
        self.n_clusters = n_clusters
        self.edgeDistribution = edgeDistribution
        self.attributeDistribution = attributeDistribution
        self.weightDistribution = weightDistribution
        self.edge_index = None

    def spectralEmbedding(self, X ):
        if (X<0).any():
            X = pairwise_kernels(X,metric='rbf')
        U = SpectralEmbedding(n_components=self.n_clusters,\
								affinity="precomputed")\
								.fit_transform(X)
        return U
        
    def AIC_initializer(self,X,Y):
        U = self.spectralEmbedding(X)
        net_null_model = GaussianMixture(n_components=1).fit(U)
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
            #print( 'Initialisation chosen from the graph')
        else:
            self.predicted_memberships = self.memberships_from_attributes
            self.graph_init = False
            #print( 'Initialisation chosen from the attributes' )
        return self
    
    def chernoff_initializer(self,edge_index,X,Y):
        n = Y.shape[0]
        if self.graphChernoffDivergence( edge_index, X, self.memberships_from_graph ) > \
                self.attributeChernoffDivergence( Y, self.memberships_from_attributes ) / n:
            self.predicted_memberships = self.memberships_from_graph
            self.graph_init = True
            print( 'Initialisation chosen from the graph')
        else:
            self.predicted_memberships = self.memberships_from_attributes
            self.graph_init = False
            print( 'Initialisation chosen from the attributes' )         
        return self

    def computeAttributeMeans( self, Y, Z ):
        attribute_means = np.dot(Z.T, Y)/(Z.sum(axis=0) + 10 * np.finfo(Z.dtype).eps)[:, np.newaxis]
        return attribute_means
    
    # def computeEdgeMeans( self, Z ):
    #     normalisation = np.linalg.pinv(Z.T@Z)
    #     return normalisation @ Z.T @ self.A @ Z @ normalisation

    def computeEdgeMeans(self,edge_index,tau):
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
        edge_means = weights[:,:,edge_index[0],edge_index[1]].sum(axis=-1)/\
            weights.sum(axis=(-1,-2))
        return edge_means 
    
    def computeWeightMeans( self, X, Z ):
        weights = np.tensordot(Z, Z, axes=((), ()))
        """
        weights[i,q,j,l] = tau[i,q]*tau[j,l]
        desired output:
        weights[q,l,i,j] = tau[i,q]*tau[j,l]
        """
        weights = np.transpose(weights,(1,3,0,2))[:,:,self.edge_index[0],self.edge_index[1]]
        """
        X is a |E| x d array
        weights is a k x k x |E|
        desired output: 
        out[q,l,d] = sum_e X[e,d] * weights[q,l,e]
        """
        edges_means = np.tensordot( weights, X, axes=[(2),(0)] )/(np.sum(weights,axis=-1)[:,:,np.newaxis])
        return edges_means 
    
    def J(self,θ_1,θ_2,t):
        ψ = dist_to_psi_dict_init[self.weightDistribution]
        return   t * ψ( θ_1 ) + (1-t) * ψ( θ_2 ) - ψ( t * θ_1 + (1-t)* θ_2 )
        
    def chernoffDivergence( self, a, b, c, t, graph_means, edge_means, distribution = 'bernoulli' ):
            p_ac = graph_means[a,c]
            p_bc = graph_means[b,c]
            θ_ac = edge_means[a,c]
            θ_bc = edge_means[b,c]
            if distribution.lower() == 'bernoulli':
                return (1-t) * p_ac + t * p_bc - (p_ac**t * p_bc**(1-t))*\
                    np.exp(-self.J(θ_ac,θ_bc,t))

    def make_renyi_div(self,pi,graph_means,edge_means,a,b):
            def renyi_div(t):
                total = 0
                for c in range(self.n_clusters):
                    total += pi[c] *self.chernoffDivergence( 
                                                            a, b, c, t,\
                                                            graph_means,\
                                                            edge_means
                                                        )
                return total
            return renyi_div

    def graphChernoffDivergence( self, edge_index, X, Z ):
        graph_means = self.computeEdgeMeans( edge_index, Z )
        edge_means = self.computeWeightMeans(X,Z)
        pi = Z.mean(axis=0)
            
        if self.edgeDistribution == 'bernoulli':
            res = 10000
            for a in range( self.n_clusters ):
                for b in range( a ):
                    renyi_div = self.make_renyi_div(pi,graph_means,edge_means,a,b)
                    minDiv = sp.optimize.minimize_scalar( renyi_div, bounds = (0,1), method ='bounded' )
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
    
    def assignInitialLabels( self , edge_index):
        if self.initializer == 'random':
            preds =  np.random.randint( 0, self.n_clusters, size = self.X.shape[0] )
            preds = preds.reshape(-1, 1)
            ohe = OneHotEncoder(max_categories=self.n_clusters, sparse_output=False).fit(preds)
            self.predicted_memberships = ohe.transform(preds)
        
        elif self.initializer == "AIC":
            self.AIC_initializer(self.sim_matrix,self.Y)
        
        ## Chernoff divergence
        elif self.initializer == "chernoff":
            self.chernoff_initializer(edge_index, self.X,self.Y)

    """
    X is N x N x 1 np.array or |E| x 1
    Y is N x d np.array
    edge_index is a tuple (indices_i, indices_j)
    """
    def initialize(self, X, Y, edge_index ,Z_init=None):
        self.N = X.shape[0]
        ## CASE X is |E| x d: do nothing
        # self.edge_index = edge_index
        sim_matrix = None
        ## CASE X is N x N x 1: pass to |E| x 1 
        if X.shape[0] == X.shape[1]:
            X = X[edge_index[0],edge_index[1],:]
        # else:   
        #     self.X = X

        preds = None
        if self.initializer == "AIC":
            U = self.spectralEmbedding(sim_matrix)
            model = GaussianMixture(n_components=self.n_clusters)
            preds = model.fit(U).predict(U).reshape(-1, 1)
            self.graph_model_init = model
        else:
            print("FIT LEIDEN")
            preds = fit_leiden(edge_index,X)
            self.graph_model_init = la
        
        ohe = OneHotEncoder(max_categories=self.n_clusters, sparse_output=False).fit(preds)
        self.memberships_from_graph = ohe.transform(preds)
        print("SHAPE: ",self.memberships_from_graph.shape)
        # self.sim_matrix = sim_matrix
        # self.Y = Y
        print("FIT GMM")
        model = GaussianMixture(n_components=self.n_clusters)
        preds = model.fit( Y ).predict( Y )
        preds = preds.reshape(-1, 1)
        ohe = OneHotEncoder(max_categories=self.n_clusters, sparse_output=False).fit(preds)
        self.memberships_from_attributes = ohe.transform(preds)
        self.attribute_model_init = model
        print("DONE \n")
        # self.A = csr_array((np.ones(self.edge_index[0].shape[0]),\
        #                      (self.edge_index[0],self.edge_index[1])),\
        #                      shape=(self.N, self.N)
        #                     )
        # self.assignInitialLabels()
        if self.initializer == 'random':
            preds =  np.random.randint( 0, self.n_clusters, size = self.N )
            preds = preds.reshape(-1, 1)
            ohe = OneHotEncoder(max_categories=self.n_clusters, sparse_output=False).fit(preds)
            self.predicted_memberships = ohe.transform(preds)
        
        elif self.initializer == "AIC":
            ## CASE X is N x N x 1: pass to |E| x 1 
            if X.shape[0] == X.shape[1]:
                sim_matrix = np.squeeze(X)
            else:   
                sim_matrix = csr_array((X.flatten(),\
                    (edge_index[0],edge_index[1])),\
                    shape=(self.N, self.N)
                )        
            self.AIC_initializer(sim_matrix,Y)
        
        ## Chernoff divergence
        elif self.initializer == "chernoff":
            self.chernoff_initializer(edge_index,X,Y)