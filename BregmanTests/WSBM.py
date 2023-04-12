import numpy as np
import networkx as nx
from BregmanTests.distributions import *

class BregmanBenchmark():
    def __init__(self,P,communities_sizes,min_=0,max_=1,dims=2,weight_variance=1,att_variance=1,\
                 weight_distribution="gamma",attributes_distribution="gaussian",radius=1,return_G=False):
        self.probability_matrix=P
        self.communities_sizes=communities_sizes
        ## min and max specifies the range of the weight distribution means in 1D
        self.min_ = min_
        self.max_ = max_
        self.weight_variance = weight_variance
        self.att_variance = att_variance
        self.n_clusters = P.shape[0]
        self.weight_distribution,f = distributions_dict[weight_distribution]
        self.get_w_params = make_weight_params(f)
        self.att_distribution,self.get_att_param = distributions_dict[attributes_distribution]          
        self.dims = dims
        self.radius=radius
        self.return_G = return_G
    
    def generate_WSBM(self):
        N = np.sum(self.communities_sizes)
        ## Generate binary connectivity matrix
        G = nx.stochastic_block_model(self.communities_sizes,self.probability_matrix,seed=42)
        ## Draw the means of the weight distributions for each pair of community interaction
        means = np.linspace(self.min_, self.max_, num=int(self.n_clusters*(self.n_clusters+1)/2))
        params = self.get_w_params(means,self.weight_variance,self.n_clusters)
        # ## get weights
        for e in G.edges:
            i,j = e
            q =  G.nodes[i]["block"]
            l =  G.nodes[j]["block"]
            p = params[q][l]
            G[i][j]['weight'] = self.weight_distribution(*p)
        return nx.to_numpy_array(G), G
    
    def get_unit_circle_coordinates(self):
        centers = []
        for i in range(self.n_clusters):
            centers.append([self.radius*np.cos(2*np.pi*i/self.n_clusters),\
                          self.radius*np.sin(2*np.pi*i/self.n_clusters)])
        return np.array(centers)
    
    def generate_attributes(self):
        centers = self.get_unit_circle_coordinates()
        if self.dims < centers.shape[1]:
            centers = centers[:,:self.dims]
        elif self.dims > centers.shape[1]:
            centers = np.hstack([centers,np.zeros((centers.shape[0],self.dims - centers.shape[1]))])
        N = np.sum(self.communities_sizes)
        Y = np.zeros((N,self.n_clusters))
        cumsum = np.cumsum(self.communities_sizes)
        cumsum = np.insert(cumsum,0,0)
        for q,clus_len in enumerate(self.communities_sizes):
            for l in range(len(centers[0])):
                p = self.get_att_param(centers[q][l],self.att_variance)
                Y[cumsum[q]:cumsum[q+1],l] = self.att_distribution(*p,size=clus_len)
        return Y
    
    def generate_benchmark_WSBM(self):
         X, G = self.generate_WSBM()
         Y = self.generate_attributes()
         labels_true = np.repeat(np.arange(self.n_clusters),self.communities_sizes)
         if self.return_G:
            for i in range(np.sum(self.communities_sizes)):
                G.nodes[i]["attr"] = Y[i,:]
            return X,Y,labels_true,G
         return X,Y,labels_true