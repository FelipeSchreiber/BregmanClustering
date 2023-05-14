import numpy as np
import scipy as sp
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.mixture import GaussianMixture
from sklearn.manifold import SpectralEmbedding
from sklearn.preprocessing import OneHotEncoder

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

    def computeAttributeMeans( self, Y, Z ):
        attribute_means = np.dot(Z.T, Y)/(Z.sum(axis=0) + 10 * np.finfo(Z.dtype).eps)[:, np.newaxis]
        return attribute_means
    
    def computeGraphMeans( self, A, Z ):
        normalisation = np.linalg.pinv ( Z.T @ Z )
        return normalisation @ Z.T @ A @ Z @ normalisation
    
    def chernoffDivergence( self, a, b, t, distribution = 'bernoulli' ):
        if distribution.lower() == 'bernoulli':
            return (1-t) * a + t *b - a**t * b**(1-t)
    
    def make_renyi_div(self,pi,graph_means,a,b):
        def renyi_div(t):
            total = 0
            for c in range(self.n_clusters):
                total += pi[c] *self.chernoffDivergence( 
                                                        graph_means[a,c],\
                                                        graph_means[b,c],\
                                                        t
                                                       )
            total = -(1-t) * total
            return total
        return renyi_div

    def graphChernoffDivergence( self, X, Z ):
        graph_means = self.computeGraphMeans( X , Z )
        pi = Z.mean(axis=0)
    
        if self.edgeDistribution == 'bernoulli':
            res = 10000
            for a in range( self.n_clusters ):
                for b in range( a ):
                    renyi_div = self.make_renyi_div(pi,graph_means,a,b)
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
    
    def assignInitialLabels( self, X, Y ):
        if self.initializer == 'random':
            preds =  np.random.randint( 0, self.n_clusters, size = X.shape[0] )
            preds = preds.reshape(-1, 1)
            ohe = OneHotEncoder(max_categories=self.n_clusters, sparse_output=False).fit(preds)
            self.predicted_memberships = ohe.transform(preds)
        
        elif self.initializer == "AIC":
            self.AIC_initializer(X,Y)
        
        ## Chernoff divergence
        elif self.initializer == "chernoff":
            self.chernoff_initializer(X,Y)

    def initialize(self, X, Y ):
        model = GaussianMixture(n_components=self.n_clusters)
        preds = model.fit( Y ).predict( Y )
        preds = preds.reshape(-1, 1)
        ohe = OneHotEncoder(max_categories=self.n_clusters, sparse_output=False).fit(preds)
        self.memberships_from_attributes = ohe.transform(preds)
        self.attribute_model_init = model

        U = self.spectralEmbedding(X)
        model = GaussianMixture(n_components=self.n_clusters)
        preds = model.fit(U).predict(U)
        preds = preds.reshape(-1, 1)
        ohe = OneHotEncoder(max_categories=self.n_clusters, sparse_output=False).fit(preds)
        self.memberships_from_graph = ohe.transform(preds)
        self.graph_model_init = model

        self.assignInitialLabels( X, Y )
      
# def interpolate_params(self,θ_1,θ_2,t):
#         return (1-t)*θ_1 + t*θ_2  
#     """
#     Computes Jensen-Bregman Divergence
#     """
# def J(self,θ_1,θ_2,t):
#         #ψ = dist_to_psi_dict[self.weightDistribution]
#         interpolated = self.interpolate_params(θ_1,θ_2)
#         return  (1-t)*self.weight_divergence(θ_1,interpolated) +\
#                  t*self.weight_divergence(θ_2,interpolated)
        
# def chernoffDivergence( self, a, b, c, t, graph_means, edge_means, distribution = 'bernoulli' ):
#         p_ac = graph_means[a,c]
#         p_bc = graph_means[b,c]
#         θ_ac = edge_means[a,c]
#         θ_bc = edge_means[b,c]
#         if distribution.lower() == 'bernoulli':
#             return (1-t) * p_ac + t * p_bc - (p_ac**t * p_bc**(1-t))*\
#                 torch.exp(-self.J(θ_ac,θ_bc,t))

# def make_renyi_div(self,pi,graph_means,edge_means,a,b):
#         def renyi_div(t):
#             total = 0
#             for c in range(self.n_clusters):
#                 total += pi[c] *self.chernoffDivergence( 
#                                                          a, b, c, t,\
#                                                          graph_means,\
#                                                          edge_means
#                                                        )
#             total = -(1-t) * total
#             return total
#         return renyi_div
    
# def graphChernoffDivergence( self, X, Z ):
#         graph_means = self.computeGraphMeans( X , Z )
#         edge_means = self.computeEdgeMeans(X,Z)
#         pi = Z.mean(dim=0)
            
#         if self.edgeDistribution == 'bernoulli':
#             res = 10000
#             for a in range( self.n_clusters ):
#                 for b in range( a ):
#                     div = self.make_renyi_div(pi,graph_means,edge_means,a,b)
#                     minDiv = minimize_constr( div, torch.tensor(0),bounds = {"lb":0,"ub":1})
#                     if - minDiv['fun'] < res:
#                         res = - minDiv['fun']
#         return res
    
# def attributeChernoffDivergence( self, Y, Z ):
#         res = 10000
#         attribute_means = self.computeAttributeMeans( Y, Z )
#         for a in range( self.n_clusters ):
#             for b in range( a ):
#                 div = lambda t : - t * (1-t)/2 * torch.linalg.norm(attribute_means[a] - attribute_means[b])
#                 minDiv = sp.optimize.minimize_scalar( div, bounds = (0,1), method ='bounded' )
#                 if - minDiv['fun'] < res:
#                     res = - minDiv['fun']

#         return res