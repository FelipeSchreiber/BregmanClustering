import numpy as np
import networkx as nx
from scipy.spatial.distance import pdist
from distributions import *

class BregmanBenchmark():
    def __init__(self,P,communities_sizes,min_,max_,dims=2,weight_variance=1,att_variance=1,\
                 weight_distribution="gamma",attributes_distribution="gaussian",radius=1):
        self.probability_matrix=P
        self.communities_sizes=communities_sizes
        ## min and max specifies the range of the weight distribution means in 1D
        self.min_ = min_
        self.max_ = max_
        self.weight_variance = weight_variance
        self.att_variance = att_variance
        self.n_clusters = P.shape[0]
        self.weight_distribution,self.get_w_params,self.get_w_param = distributions_dict[weight_distribution]
        self.att_distribution,self.get_att_params,self.get_att_param = distributions_dict[attributes_distribution]          
        self.dims = dims
        self.radius=radius
    
    def generate_WSBM(self):
        N = np.sum(self.communities_sizes)
        ## Generate binary connectivity matrix
        G = nx.stochastic_block_model(self.communities_sizes,self.probability_matrix,seed=42)
        A = nx.to_numpy_array(G)
        ## Draw the means of the weight distributions for each pair of community interaction
        means = np.linspace(self.min_, self.max_, num=int(self.n_clusters*(self.n_clusters+1)/2))
        ## Assume that each distribution has unit variance
        params = self.get_w_params(means,self.weight_variance,self.n_clusters)
        ## get weights
        X = np.zeros((N,N))
        for i in range(N):
            for j in range(i,N):
                if A[i,j] != 0:
                    q =  G.nodes[i]["block"]
                    l =  G.nodes[j]["block"]
                    p = params[q][l]
                    X[i,j] = X[j,i] = self.weight_distribution(*p)
        return X
    
    def generate_attributes(self):
        basis = []
        for i in range(self.n_clusters):
            basis.append((self.radius*np.cos(2*np.pi*i/self.n_clusters),\
                          self.radius*np.sin(2*np.pi*i/self.n_clusters)))
        
        N = np.sum(self.communities_sizes)
        Y = np.zeros((N,self.n_clusters))
        cumsum = np.cumsum(self.communities_sizes)
        cumsum = np.insert(cumsum,0,0)
        for q,clus_len in enumerate(self.communities_sizes):
            for l in range(len(basis[0])):
                p = self.get_att_param(basis[q][l],self.att_variance)
                Y[cumsum[q]:cumsum[q+1],l] = self.att_distribution(*p,size=clus_len)
        return Y
    
    def generate_benchmark_WSBM(self):
         X = self.generate_WSBM()
         Y = self.generate_attributes()
         return X,Y