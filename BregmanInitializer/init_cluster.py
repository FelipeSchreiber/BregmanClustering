import numpy as np
import scipy as sp
from .divergences import *
from sklearn.metrics.pairwise import pairwise_kernels, paired_distances, pairwise_distances
from sklearn.mixture import GaussianMixture
from sklearn.manifold import SpectralEmbedding
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import csr_matrix, csc_matrix 
from sklearn.cluster import SpectralClustering
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClusterMixin

def fromVectorToMembershipMatrice(z,k):
    z = z.reshape(-1, 1)
    ohe = OneHotEncoder(max_categories=k, sparse_output=False).fit(z)
    return ohe.transform(z)

def frommembershipMatriceToVector(Z):
    return Z.argmax(axis=1)

class BregmanGraphClustering( BaseEstimator, ClusterMixin ):
    def __init__( self, n_clusters, 
                 edgeDistribution = "bernoulli",
                 weightDistribution = "gaussian",
                 n_iters = 25, init_iters=100,
                 reduce_by=None,
                 divergence_precomputed=True):
        """
        Bregman Hard Clustering Algorithm for partitioning graph 
        Parameters
        ----------
        n_clusters : INT
            Number of clustes.
        edge_divergence, attribute_divergence : function
            Pairwise divergence function. The default is euclidean_distance.
        n_iters : INT, optional
            Number of clustering iterations. The default is 25.
        Returns
        -------
        None.
        """
        self.n_clusters = n_clusters
        self.n_iters = n_iters
        self.init_iters = init_iters
        ## Variable that stores which initialization was chosen
        self.graph_init = False
        self.edgeDistribution = edgeDistribution
        self.weightDistribution = weightDistribution
        self.edge_divergence = dist_to_divergence_dict_init[self.edgeDistribution]
        self.weight_divergence = dist_to_divergence_dict_init[self.weightDistribution]
        self.edge_index = None 

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
    
    def fit( self, A, X, Z_init=None ):
        """
        Training step.
        Parameters
        ----------
        X : ARRAY
            Input |E| x d matrix with edges attributes.  
        A : ARRAY
            Input (n,n) the adjacency matrix
        Returns
        -------
        TYPE
            Trained model.
        """
        self.N = A.shape[0]
        self.edge_index = np.nonzero(A)
        print("BGC>>>",A.shape,X.shape)
        if Z_init is None:
            SC = SpectralClustering(n_clusters=self.n_clusters,
            assign_labels='discretize',random_state=0).fit(A)
            preds = SC.labels_.reshape(-1, 1)
            ohe = OneHotEncoder(max_categories=self.n_clusters, sparse_output=False).fit(preds)
            self.predicted_memberships= ohe.transform(preds)
        else:
            self.predicted_memberships = Z_init
        print("Z.shape: ",self.predicted_memberships.shape)
        self.edge_means = self.computeEdgeMeans(A,self.predicted_memberships)
        self.weight_means = self.computeWeightMeans(A, X, self.predicted_memberships)
        self.precompute_edge_divergences()
        convergence = True
        iteration = 0
        while convergence:
            new_memberships = self.assignments( A, X)
            self.edge_means = self.computeEdgeMeans( A, new_memberships )
            self.weight_means = self.computeWeightMeans(A, X, new_memberships)
            self.precompute_edge_divergences()
            
            iteration += 1
            if accuracy_score( new_memberships,self.predicted_memberships) < 0.02\
                or iteration >= self.n_iters:
                convergence = False
            self.predicted_memberships = new_memberships
        return self
        
    def spectralEmbedding(self, X ):
        if (X<0).any():
            X = pairwise_kernels(X,metric='rbf')
        U = SpectralEmbedding(n_components=self.n_clusters,\
								affinity="precomputed")\
								.fit_transform(X)
        return U
    
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
        """
        X is a |E| x d tensor
        weights is a k x k x |E|
        desired output: 
        out[q,l,d] = sum_e X[e,d] * weights[q,l,e]
        """
        weight_means = np.tensordot( weights,\
                                    X,\
                                    axes=[(2),(0)] )/(np.sum(weights,axis=-1)[:,:,np.newaxis]) 
        
        if (self.edge_means==0).any():
            null_model = X.mean(axis=0)
            undefined_idx = np.where(self.edge_means==0)
            weight_means[undefined_idx[0],undefined_idx[1],:] = null_model
        return weight_means    

    def assignments( self, A, X):
        z = np.zeros( self.N, dtype = int )
        for node in range( len( z ) ):
            z[ node ] = self.singleNodeAssignment( A, X, node )
        return fromVectorToMembershipMatrice( z, self.n_clusters )        
    
    def singleNodeAssignment( self, A, X, node ):
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
            # M_out = self.edge_means[np.repeat(q, self.N),z_t]
            # M_in = self.edge_means[z_t,np.repeat(q, self.N)]
            E = self.weight_means
            """
            X has shape |E| x d
            E has shape k x k x d
            
            the edge divergence computes the difference between node i (from community q) edges and the means
            given node j belongs to community l:
            
            sum_j div_edge(e_ij, E[q,l,:])  
            """
            edge_div = self.precomputed_edge_div[1,z_t[v_idx_in],q].sum()\
                    + self.precomputed_edge_div[1,q,z_t[v_idx_out]].sum()\
                    + self.precomputed_edge_div[0,z_t[v_idx_in_comp],q].sum()\
                    + self.precomputed_edge_div[0,q,z_t[v_idx_out_comp]].sum()
            # edge_div = self.edge_divergence( A[node,:], M_out ).sum() \
            #             + self.edge_divergence( A[:,node], M_in ).sum()\
            #             - 2*self.edge_divergence(A[node,node],M_in[q])
            weight_div = 0
            if len(v_idx_out) > 0:
                weight_div += np.sum( paired_distances(X[edge_indices_out,:],\
                                                        E[q,z_t[v_idx_out],:],\
                                                        metric=self.weight_divergence))
            if len(v_idx_in) > 0:
                weight_div += np.sum( paired_distances(X[edge_indices_in,:],\
                                                        E[z_t[v_idx_in],q,:],\
                                                        metric=self.weight_divergence))
            L[ q ] = weight_div + edge_div
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
    
    def chernoff_initializer(self,X,Y):
        n = Y.shape[0]
        if self.graphChernoffDivergence( X, self.memberships_from_graph ) > \
                self.attributeChernoffDivergence( Y, self.memberships_from_attributes ) / n:
            self.predicted_memberships = self.memberships_from_graph
            self.graph_init = True
            #print( 'Initialisation chosen from the graph')
        else:
            self.predicted_memberships = self.memberships_from_attributes
            self.graph_init = False
            #print( 'Initialisation chosen from the attributes' )         
        return self

    def computeAttributeMeans( self, Y, Z ):
        attribute_means = np.dot(Z.T, Y)/(Z.sum(axis=0) + 10 * np.finfo(Z.dtype).eps)[:, np.newaxis]
        return attribute_means
    
    def computeEdgeMeans( self, A, Z ):
        normalisation = np.linalg.pinv(Z.T@Z)
        M = Z.T@A@Z
        return normalisation @ M @ normalisation
    
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

    def graphChernoffDivergence( self, X, Z ):
        graph_means = self.computeEdgeMeans( self.A , Z )
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
    
    def assignInitialLabels( self ):
        if self.initializer == 'random':
            preds =  np.random.randint( 0, self.n_clusters, size = self.X.shape[0] )
            preds = preds.reshape(-1, 1)
            ohe = OneHotEncoder(max_categories=self.n_clusters, sparse_output=False).fit(preds)
            self.predicted_memberships = ohe.transform(preds)
        
        elif self.initializer == "AIC":
            self.AIC_initializer(self.sim_matrix,self.Y)
        
        ## Chernoff divergence
        elif self.initializer == "chernoff":
            self.chernoff_initializer(self.X,self.Y)

    """
    X is N x N x 1 np.array or |E| x 1
    Y is N x d np.array
    edge_index is a tuple (indices_i, indices_j)
    """
    def initialize(self, X, Y , edge_index):
        self.N = Y.shape[0]
        A = None
        ## CASE X is |E| x d: do nothing
        self.edge_index = edge_index
        sim_matrix = None
        ## CASE X is N x N x 1: pass to |E| x 1 
        if X.shape[0] == X.shape[1]:
            self.X = X[self.edge_index[0],self.edge_index[1],:]
            sim_matrix = np.squeeze(X)
        else:           
            sim_matrix = np.zeros((self.N,self.N))
            sim_matrix[self.edge_index[0],self.edge_index[1]] = np.squeeze(X)
            self.X = X

        self.sim_matrix = sim_matrix
        self.A = csr_matrix((np.ones(self.edge_index[0].shape[0]),\
                             (self.edge_index[0],self.edge_index[1])),\
                             shape=(self.N, self.N)
                            )
        self.Y = Y
        model = GaussianMixture(n_components=self.n_clusters)
        preds = model.fit( Y ).predict( Y )
        preds = preds.reshape(-1, 1)
        ohe = OneHotEncoder(max_categories=self.n_clusters, sparse_output=False).fit(preds)
        self.memberships_from_attributes = ohe.transform(preds)
        self.attribute_model_init = model


        if self.initializer == "AIC":
            U = self.spectralEmbedding(sim_matrix)
            model = GaussianMixture(n_components=self.n_clusters)
            preds = model.fit(U).predict(U).reshape(-1, 1)
            self.graph_model_init = model
        else:
            print("K",self.n_clusters)
            model = BregmanGraphClustering(n_clusters=self.n_clusters,\
                                        edgeDistribution=self.edgeDistribution,\
                                        weightDistribution=self.weightDistribution
                                        )
            Z_init = fromVectorToMembershipMatrice(np.random.randint(self.n_clusters,size=self.N),
                                                                        self.n_clusters)
            preds = model.fit(self.A,self.X).predict(None, None).reshape(-1, 1)
            self.graph_model_init = model

        ohe = OneHotEncoder(max_categories=self.n_clusters, sparse_output=False).fit(preds)
        self.memberships_from_graph = ohe.transform(preds)
        
        self.assignInitialLabels()