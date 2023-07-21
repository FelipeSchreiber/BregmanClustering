import numpy as np
import networkx as nx
from BregmanTests.distributions import *
from BregmanClustering.models import BregmanNodeEdgeAttributeGraphClusteringEfficient as hardBreg
from BregmanClustering.models import BregmanNodeEdgeAttributeGraphClusteringSoft as softBreg
from BregmanClusteringTorch.torch_models import torchWrapper as torchBreg
from BregmanKernel.kernel_models import BregmanKernelClustering
from BregmanInitializer.init_cluster import *
from scipy.sparse import csr_matrix, csc_matrix, csr_array 
from BregmanInitializer.init_cluster import fit_leiden
from sklearn.metrics import adjusted_rand_score
from sklearn.datasets import make_classification
from torch_geometric.utils import to_networkx,to_dense_adj,from_networkx
from torch_geometric.data import Data
from torch_geometric.utils import *
from torch_geometric.datasets import Planetoid,WebKB
import pandas as pd
import torch 
import subprocess
from tqdm import tqdm
from .cfg import *
import os
from .utils import *
from copy import deepcopy
if (not other_algos_installed) and (not os.path.isfile("other_algos_installed.txt")):
    from .install_algorithms import main as install_env
    ## Optional: set repository for CRAN
    CRAN_repo = "https://cran.fiocruz.br/"
    install_env()
    with open("other_algos_installed.txt", 'w') as f:
        f.write('OK')
from CSBM.Python import functions as csbm
from itertools import product
from sklearn.cluster import KMeans,SpectralClustering
import igraph as ig
import leidenalg as la
from BregmanInitializer.init_cluster import frommembershipMatriceToVector, fromVectorToMembershipMatrice
from find_julia import find as find_jl

class BregmanBenchmark():
    def __init__(self,P=None,communities_sizes=None,min_=1,max_=10,\
                    dims=2,\
                    weight_variance=1,att_variance=1,\
                    attributes_distribution = "gaussian",\
                    edge_distribution = "bernoulli",\
                    weight_distribution = "exponential",\
                    radius=1,return_G=False,reduce_by="sum",\
                    att_centers = None, weight_centers = None, run_torch=False,\
                    divergence_precomputed=True, initializer="AIC",\
                    hard_clustering=True,preprocess=True):
        ## att_centers must have shape K x D, where K is the number
        #  of communities and D the number of dimensions.
        # If not specified, then the centers are taken from unit circle
        self.att_centers=att_centers
        ## weight_centers must have shape K(K+1)/2, where K is the number
        #  of communities
        # If not specified, then the weights are taken from linspace between [min_, max_]
        self.weight_centers=weight_centers
        self.probability_matrix=P
        self.reduce_by = reduce_by
        self.initializer = initializer
        self.communities_sizes=communities_sizes
        self.num_nodes = np.sum(self.communities_sizes)
        ## min and max specifies the range of the weight distribution means in 1D
        self.min_ = min_
        self.max_ = max_
        self.weight_variance = weight_variance
        self.att_variance = att_variance
        self.n_clusters = P.shape[0] if P is not None else None
        self.weight_distribution_name = weight_distribution
        self.weight_distribution,f = distributions_dict[weight_distribution]
        self.get_w_params = make_weight_params(f)
        self.attributes_distribution_name = attributes_distribution
        self.att_distribution,self.get_att_param = distributions_dict[attributes_distribution]          
        self.edge_distribution_name = edge_distribution
        self.dims = dims
        self.radius=radius
        self.return_G = return_G
        self.torch_model = run_torch
        self.divergence_precomputed = divergence_precomputed
        if run_torch:
            self.model_ = torchBreg
        elif hard_clustering:
            self.model_ = hardBreg
        else:
            self.model_ = softBreg
        self.preprocess = preprocess

    def generate_WSBM(self,complete_graph=False):
        N = np.sum(self.communities_sizes)
        ## Generate binary connectivity matrix
        G = None
        if complete_graph:
            G = nx.stochastic_block_model(self.communities_sizes,\
                                          np.ones(shape=(self.n_clusters,self.n_clusters)),\
                                          directed=True,seed=42)
        else:
            G = nx.stochastic_block_model(self.communities_sizes,self.probability_matrix,\
                                          directed=True,seed=42)
        ## Draw the means of the weight distributions for each pair of community interaction
        if self.weight_centers is None:
            self.weight_centers = np.zeros((self.n_clusters,self.n_clusters))
            self.weight_centers[np.triu_indices(self.n_clusters, k = 0)] = \
                np.linspace(self.min_, self.max_, num=int(self.n_clusters*(self.n_clusters+1)/2))
            self.weight_centers = self.weight_centers + self.weight_centers.T - np.diag(np.diag(self.weight_centers))
        params = self.get_w_params(self.weight_centers,self.weight_variance,self.n_clusters)
        print("\nSBM: ",self.probability_matrix)
        ## get weights
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
        print("\nDistribution Centers: ",centers)
        return np.array(centers)
    
    def generate_attributes(self):
        centers = None
        if self.att_centers is not None:
            centers = self.att_centers
        else: 
            centers = self.get_unit_circle_coordinates()
            if self.dims < centers.shape[1]:
                centers = centers[:,:self.dims]
            elif self.dims > centers.shape[1]:
                centers = np.hstack([centers,np.zeros((centers.shape[0],self.dims - centers.shape[1]))])
        N = np.sum(self.communities_sizes)
        Y = np.zeros((N,centers.shape[1]))
        cumsum = np.cumsum(self.communities_sizes)
        cumsum = np.insert(cumsum,0,0)
        for q,clus_len in enumerate(self.communities_sizes):
            for l in range(centers.shape[1]):
                p = self.get_att_param(centers[q][l],self.att_variance)
                Y[cumsum[q]:cumsum[q+1],l] = self.att_distribution(*p,size=clus_len)
        return Y
    
    ## Generate Benchmark where the edges follows a joint probability distribution
    #  given by bernoulli and weights from exponential familly 
    def generate_benchmark_joint(self):
         X, G = self.generate_WSBM()
         Y = self.generate_attributes()
         labels_true = np.repeat(np.arange(self.n_clusters),self.communities_sizes)
         if self.return_G:
            for i in range(np.sum(self.communities_sizes)):
                G.nodes[i]["x"] = Y[i,:].tolist()
            return X,Y,labels_true,G
         return X,Y,labels_true,None
    
    def generate_benchmark_dense(self):
        X, G = self.generate_WSBM(complete_graph=True)
        Y = self.generate_attributes()
        labels_true = np.repeat(np.arange(self.n_clusters),self.communities_sizes)
        if self.return_G:
            for i in range(np.sum(self.communities_sizes)):
                G.nodes[i]["x"] = Y[i,:].tolist()
            return X,Y,labels_true,G
        return X,Y,labels_true,None
    
    def gen_config_file(self,d_min=5,d_max=50,c_min=50,c_max=1000,t1=3,t2=2):
        cfg_data = f"""seed = "42"                   # RNG seed, use "" for no seeding
n = "{int(self.num_nodes)}"                   # number of vertices in graph
t1 = "{int(t1)}"                      # power-law exponent for degree distribution
d_min = "{int(d_min)}"                   # minimum degree ---- 2*log_2 (n)
d_max = "{int(d_max)}"                  # maximum degree ---- 100*log_2(n)
d_max_iter = "1000"           # maximum number of iterations for sampling degrees
t2 = "{int(t2)}"                      # power-law exponent for cluster size distribution
c_min = "{int(c_min)}"                  # minimum cluster size c_min = c_max, n/2 ... n/8
c_max = "{int(c_max)}"                # maximum cluster size
c_max_iter = "1000"           # maximum number of iterations for sampling cluster sizes
# Exactly one of xi and mu must be passed as Float64. Also if xi is provided islocal must be set to false or omitted.
xi = "0.2"                    # fraction of edges to fall in background graph
#mu = "0.2"                   # mixing parameter
islocal = "false"             # if "true" mixing parameter is restricted to local cluster, otherwise it is global
isCL = "false"                # if "false" use configuration model, if "true" use Chung-Lu
degreefile = "deg.dat"        # name of file do generate that contains vertex degrees
communitysizesfile = "cs.dat" # name of file do generate that contains community sizes
communityfile = "com.dat"     # name of file do generate that contains assignments of vertices to communities
networkfile = "edge.dat"      # name of file do generate that contains edges of the generated graph
nout = "100"                  # number of vertices in graph that are outliers; optional parameter
                              # if nout is passed and is not zero then we require islocal = "false",
                              # isCL = "false", and xi (not mu) must be passed
                              # if nout > 0 then it is recommended that xi > 0"""
        # cfg_data = '\n'.join(cfg_data.split("\n"))
        with open('my_config.toml', 'w') as f:
            f.write(cfg_data)
        
    def generate_ABCD_benchmark(self,d_min=5,d_max=50,c_min=50,c_max=1000,t1=3,t2=2,num_nodes=100):
        self.num_nodes = num_nodes
        self.gen_config_file(d_min=d_min,d_max=d_max,c_min=c_min,c_max=c_max,t1=t1,t2=t2)
        jl_path = find_jl()
        cur_path = os.getcwd()
        subprocess.call([f"{jl_path}",f"{path_to_ABCD_installer}"])
        subprocess.call([f"{jl_path}",f"{path_to_ABCD_sampler}",f"{cur_path}/my_config.toml"])
        # print(f"{jl_path}",f"{path_to_ABCD_sampler}","my_config.toml")
        #X = np.array(pd.read_csv('deg.dat',header=None)[0])
        edges = open('edge.dat', "r")
        G = nx.parse_edgelist(edges, nodetype=int)
        edges.close()
        com_df = pd.read_csv('com.dat',delimiter="\t",header=None)
        counts = com_df[1].value_counts()
        com_sizes = [counts[i] for i in range(1,com_df[1].max()+1)]
        self.communities_sizes = com_sizes
        self.n_clusters = len(com_sizes)
        mapping = {}
        sorted = com_df.sort_values(by=1)##sort by labels
        # sorted[2] = np.arange(df.shape[0])
        for i,node in enumerate(sorted[0]):
            mapping[node] = i
        G  = nx.relabel_nodes(G, mapping)
        Y = self.generate_attributes()
        labels_true = sorted[1].to_numpy() - 1
        return G,Y,labels_true
    
    def to_pyg_data(self,X,Y):
        X_sparse = torch.tensor(X).to_sparse()
        graph_data = Data(x=torch.tensor(Y),
                    edge_index=X_sparse.indices(),
                    edge_attr=X_sparse.values())
        return graph_data
    
    def run_test(self,n_average=10,cluster_sizes=[100],\
                 b=5,\
                 a_range=[ 5,7,9,11,13,15 ],\
                 r_range = [ 0,1,2,3,4,5 ],\
                 dense=False,\
                 binary=True,\
                 file_endings=".jpeg",\
                 n_iters=25):
        
        stats = {"varying":[],"a":[],"r":[],"ARI":[],"ARI_std":[],"algorithm":[]}
        n = np.sum(cluster_sizes)
        n_clusters = len(cluster_sizes)
        self.n_clusters = n_clusters
        pout = b * np.log( n ) / n
        for varying in ["attributes","graph"]:
        # varying = 'attributes'
        # #varying = 'graph'

            aris_attributes_mean = [ ]
            aris_graph_mean = [ ]
            aris_both_mean = [ ]
            aris_attSBM_mean = [ ]
            aris_IR_sLS_mean = [ ]
            # aris_IR_LS_mean = [ ]


            aris_attributes_std = [ ]
            aris_graph_std = [ ]
            aris_both_std = [ ]
            aris_attSBM_std = [ ]
            aris_IR_sLS_std = [ ]
            aris_IR_LS_std = [ ]
            # aris_both2_std = [ ]
            # aris_oracle_std = [ ]

            if varying == 'graph':
                loop = tqdm( range( len( a_range ) ) )
            else:
                loop = tqdm( range( len( r_range ) ) )

            for dummy in loop:
                if varying == 'graph':
                    a = a_range[ dummy ]
                    r = 1
                elif varying == 'attributes':
                    a = 8
                    r = r_range[ dummy ]

                pin = a * np.log( n ) / n
                p = (pin- pout) * np.eye( n_clusters ) + pout * np.ones( (n_clusters, n_clusters) )

                aris_attributes = [ ]
                aris_graph = [ ]
                aris_both = [ ]
                aris_attSBM = [ ]
                aris_IR_sLS  = [ ]
                # aris_IR_LS = [ ]
                # aris_oracle = [ ]
                
                path_ = path_to_data+f"a/{a}/r/{r}/"
                if not os.path.exists(path_):
                    os.makedirs(path_)

                total = 0
                for trial in range( n_average ):
                    self.probability_matrix=p
                    self.communities_sizes=cluster_sizes
                    self.att_variance = 1
                    self.weight_variance = 1
                    self.radius = r
                    self.return_G=False
                    ( X, Y, z_true, G) = self.generate_benchmark_joint()
                    
                    A = (X != 0).astype(int)
                    model = self.model_(n_clusters=n_clusters,\
                                        attributeDistribution=self.attributes_distribution_name,\
                                        edgeDistribution=self.edge_distribution_name,\
                                        weightDistribution=self.weight_distribution_name,\
                                        n_iters=n_iters,
                                        reduce_by=self.reduce_by,
                                        divergence_precomputed=self.divergence_precomputed,
                                        initializer=self.initializer)
                    if binary:
                        X = A
                    ## For comparison purposes, the initialization is the same for IR-sLS, IR-LS and ours    
                    model.initialize(A,X.reshape(n,n,-1),Y)
                    model.assignInitialLabels(A, Y)
                    z_init = deepcopy(model.predicted_memberships)
                    chernoff_init_graph = model.graph_init
                    chernoff_graph_labels = model.memberships_from_graph
                    chernoff_att_labels = model.memberships_from_attributes

                    #print("CHERNOFF SHAPE: ", chernoff_att_labels.shape, chernoff_graph_labels.shape) 
                    with open(f'{path_}att_{trial}.npy', 'wb') as g:
                        np.save(g, Y)
                    with open(f'{path_}net_{trial}.npy', 'wb') as g:
                        np.save(g, A)
                    with open(f'{path_}z_init_{trial}.npy', 'wb') as g:
                        np.save(g, csbm.convertZ(z_init)+1)

                    z_pred_both = model.fit(A,X.reshape(n,n,-1),Y).predict( X, Y )
                    z_pred_graph = chernoff_graph_labels
                    z_pred_attributes = chernoff_att_labels
                    
                    # this code is for initialization comparison
                    ### > Start
                    # if chernoff_init_graph == model.AIC_initializer(X,Y).graph_init:
                    #     total += 1
                    
                    # ## Warm start
                    # if model.graph_init:
                    #     model.fit( X, Y, chernoff_graph_labels)
                    # else:
                    #     model.fit(X, Y, chernoff_att_labels)
                    ### > end
                    
                    IR_sLS_pred = csbm.iter_csbm(X,Y,z_init,n_clusters)
                    # IR_LS_pred = iter_csbm2(X,Y,z_init,n_clusters)
                        
                    subprocess.call(["/usr/bin/Rscript","--vanilla",f"{base_path}/run_AttSBM.r",\
                                    f'{path_}att_{trial}.npy',\
                                    f'{path_}net_{trial}.npy',\
                                    f'{path_}z_init_{trial}.npy'])
                    attSBMPred = np.load("predict.npy")

                    aris_attributes.append( adjusted_rand_score( z_true, z_pred_attributes ) )
                    aris_graph.append( adjusted_rand_score( z_true, z_pred_graph ) )
                    aris_both.append( adjusted_rand_score( z_true, z_pred_both ) )
                    aris_attSBM.append( adjusted_rand_score( z_true, attSBMPred ) )
                    aris_IR_sLS.append( adjusted_rand_score( z_true, IR_sLS_pred ) )
                    #aris_IR_LS.append( adjusted_rand_score( z_true, IR_LS_pred ) )
                    
                    # if chernoff_init_graph:
                    #   z_pred_att_init = model.fit(A,A.reshape(n,n,1),Y,chernoff_att_labels).predict( X, Y )
                    #   ari_att_init = adjusted_rand_score( z_true, z_pred_att_init)
                    #   aris_oracle.append( max(aris_both[-1], ari_att_init))
                    # elif not chernoff_init_graph:
                    #   z_pred_graph_init = model.fit(A,A.reshape(n,n,1),Y,chernoff_graph_labels).predict( X, Y )
                    #   ari_graph_init = adjusted_rand_score( z_true, z_pred_graph_init)
                    #   aris_oracle.append( max(aris_both[-1], ari_graph_init))
                        
                aris_attributes_mean.append( np.mean( aris_attributes ) )
                aris_graph_mean.append( np.mean( aris_graph ) )
                aris_both_mean.append( np.mean( aris_both ) )
                aris_attSBM_mean.append( np.mean( aris_attSBM ) )
                aris_IR_sLS_mean.append( np.mean( aris_IR_sLS ) )
                #aris_IR_LS_mean.append( np.mean( aris_IR_LS ) )
                #aris_oracle_mean.append( np.mean( aris_oracle) )
                
                aris_attributes_std.append( np.std( aris_attributes ) )
                aris_graph_std.append( np.std( aris_graph ) )
                aris_both_std.append( np.std( aris_both ) )
                aris_attSBM_std.append( np.std( aris_attSBM ) )
                aris_IR_sLS_std.append( np.std( aris_IR_sLS ) )
                # aris_IR_LS_std.append( np.std( aris_IR_LS ) )
                # aris_oracle_std.append( np.std( aris_oracle) )
                
                stats["varying"].append(varying)
                stats["a"].append(a)
                stats["r"].append(r)
                stats["ARI"].append(aris_both_mean[-1])
                stats["ARI_std"].append(aris_both_std[-1])
                stats["algorithm"].append("ours")

                stats["varying"].append(varying)
                stats["a"].append(a)
                stats["r"].append(r)
                stats["ARI"].append(aris_IR_sLS_mean[-1])
                stats["ARI_std"].append(aris_IR_sLS_std[-1])
                stats["algorithm"].append("IR_sLS")

                stats["varying"].append(varying)
                stats["a"].append(a)
                stats["r"].append(r)
                stats["ARI"].append(aris_attSBM_mean[-1])
                stats["ARI_std"].append(aris_attSBM_std[-1])
                stats["algorithm"].append("attSBM")

                stats["varying"].append(varying)
                stats["a"].append(a)
                stats["r"].append(r)
                stats["ARI"].append(aris_graph_mean[-1])
                stats["ARI_std"].append(aris_graph_std[-1])
                stats["algorithm"].append("graph")

                stats["varying"].append(varying)
                stats["a"].append(a)
                stats["r"].append(r)
                stats["ARI"].append(aris_attributes_mean[-1])
                stats["ARI_std"].append(aris_attributes_std[-1])
                stats["algorithm"].append("attributes")
  
            curves = [ aris_attributes_mean, aris_graph_mean,\
                    aris_both_mean , aris_attSBM_mean, aris_IR_sLS_mean]

            curves_std = [ aris_attributes_std, aris_graph_std,\
                        aris_both_std , aris_attSBM_std, aris_IR_sLS_std]

            labels = [ 'attributes', 'graph', 'ours' , 'attSBM', 'IR_sLS']
            saveFig = True
            if varying == 'graph':    
                fileName = 'N_' + str(n) + '_K_' + str(n_clusters) + '_b_' + str(b) + '_r_' + str(r) +  '_nAverage' + str(n_average) + '.jpeg'
                plotting( a_range, curves, labels, curves_std = curves_std, xticks = a_range, xlabel = 'a', saveFig = saveFig, fileName = fileName )
                plt.close()
            elif varying == 'attributes':
                fileName = 'N_' + str(n) + '_K_' + str(n_clusters) + '_a_' + str(a) + '_b_' + str(b) +  '_nAverage_' + str(n_average) + '.jpeg'
                plotting( r_range, curves, labels, curves_std = curves_std, xticks = r_range, xlabel = 'r', saveFig = saveFig, fileName = fileName )
                plt.close()


    
    def run_2_1(self,n_average=10,cluster_sizes=100,\
                 b=5,\
                 a_range=[ 5,7,9,11,13,15 ],\
                 r_range = [ 0,1,2,3,4,5 ],\
                 dense=False,\
                 binary=True,\
                 n_iters=100):
        self.communities_sizes = cluster_sizes
        benchmark_instance = None
        if dense:
            benchmark_instance = self.generate_benchmark_dense
        else:
            benchmark_instance = self.generate_benchmark_joint

        n = np.sum(cluster_sizes)
        n_clusters = len(cluster_sizes)
        self.n_clusters = n_clusters
        pout = b * np.log( n ) / n
        stats = {"a":[],"r":[],"ARI":[]}
        aris_both_mean = [ ]
        aris_both_std = [ ]
        for a,r in tqdm(product(a_range,r_range)):
            pin = a * np.log( n ) / n
            p = (pin- pout) * np.eye( n_clusters ) + pout * np.ones( (n_clusters, n_clusters) )
            p[0,1] = 0
            self.probability_matrix = p
            self.radius = r
            aris_both = [ ]

            for _ in range( n_average ):
                ( X, Y, z_true, G) = benchmark_instance() 
                    
                A = (X != 0).astype(int)
                if binary:
                    X = A
                z_pred_both = None
                model = self.model_(n_clusters=n_clusters,\
                                        attributeDistribution=self.attributes_distribution_name,\
                                        edgeDistribution=self.edge_distribution_name,\
                                        weightDistribution=self.weight_distribution_name,\
                                        n_iters=n_iters,
                                        reduce_by=self.reduce_by,
                                        divergence_precomputed=self.divergence_precomputed,
                                        initializer=self.initializer)
                if self.torch_model:
                    z_pred_both = model.fit(A,X,Y).predict(None,None)
                else:
                    z_pred_both = model.fit(A,X.reshape(n,n,-1),Y).predict( X, Y)
                aris_both.append( adjusted_rand_score( z_true, z_pred_both ) )
                aris_both_mean.append( np.mean( aris_both ) )
                aris_both_std.append( np.std( aris_both ) )
                X = Y = z_true = G = None
            stats["a"].append(a)
            stats["r"].append(r)
            stats["ARI"].append(aris_both_mean[-1])
            # print("PREDS: ",model.predicted_memberships)        
        return stats

    def run_2_2(self,n_average=10,cluster_sizes=100,\
                 d_range=[ 0,1,2,3,4,5 ],\
                 mu_range = [ 0,1,2,3,4,5 ],\
                 dense=True,\
                 binary=False,\
                 n_iters=25,
                 a=5,
                 b=3):
        
        self.communities_sizes = cluster_sizes
        benchmark_instance = None
        if dense:
            benchmark_instance = self.generate_benchmark_dense
        else:
            benchmark_instance = self.generate_benchmark_joint

        n = np.sum(cluster_sizes)
        n_clusters = len(cluster_sizes)
        self.n_clusters = n_clusters

        pout = b * np.log( n ) / n 
        pin = a * np.log( n ) / n
        p = (pin- pout) * np.eye( n_clusters ) + pout * np.ones( (n_clusters, n_clusters) )
        self.probability_matrix = p
        
        stats = {"d":[],"mu":[],"ARI":[]}
        aris_both_mean = [ ]
        aris_both_std = [ ]
        for d,mu in tqdm(product(d_range,mu_range)):
            aris_both = [ ]
            self.dims=int(d*np.log( n ) / n)
            ### HERE ATT_CENTERS IS K x 1
            arr = self.att_centers.reshape(-1,1)
            self.att_centers = np.repeat(arr,d,axis=1)
            self.weight_centers = np.eye(self.n_clusters)*mu
            for _ in range( n_average ):
                ( X, Y, z_true, G) = benchmark_instance() 
                    
                A = (X != 0).astype(int)
                if binary:
                    X = A
                z_pred_both = None
                model = self.model_(n_clusters=n_clusters,\
                                        attributeDistribution=self.attributes_distribution_name,\
                                        edgeDistribution=self.edge_distribution_name,\
                                        weightDistribution=self.weight_distribution_name,\
                                        n_iters=n_iters,
                                        reduce_by=self.reduce_by,
                                        divergence_precomputed=self.divergence_precomputed,
                                        initializer=self.initializer)
                if self.torch_model:
                    graph_data = self.to_pyg_data(X,Y)
                    A = torch.tensor(A).to_sparse()
                    E = None
                    if graph_data.edge_attr is None:
                        E = torch.ones((graph_data.edge_index.shape[1],1))
                    else:
                        E = graph_data.edge_attr.reshape(-1,1)
                    z_pred_both = model.fit(A,E,graph_data.x).predict( E, graph_data.x )
                else:
                    print(A.shape,X.shape,Y.shape)
                    z_pred_both = model.fit(A,X.reshape(n,n,-1),Y).predict( X, Y )
                aris_both.append( adjusted_rand_score( z_true, z_pred_both ) )
                aris_both_mean.append( np.mean( aris_both ) )
                aris_both_std.append( np.std( aris_both ) )
            
            ##restore centers to original K x 1 shape
            self.att_centers = arr
            ## gather stats
            stats["d"].append(d)
            stats["mu"].append(mu)
            stats["ARI"].append(aris_both_mean[-1])
        
        return stats
    
    def run_2_3(self,n_average=10,cluster_sizes=100,\
                 d_range=[ 0,1,2,3,4,5 ],\
                 lambda_range = [ 0,1,2,3,4,5 ],\
                 a_range = [1,2,3],\
                 b = 5,\
                 dense=False,\
                 binary=False,\
                 n_iters=25):
        
        self.communities_sizes = cluster_sizes
        benchmark_instance = None
        if dense:
            benchmark_instance = self.generate_benchmark_dense
        else:
            benchmark_instance = self.generate_benchmark_joint

        n = np.sum(cluster_sizes)
        n_clusters = len(cluster_sizes)
        self.n_clusters = n_clusters
        stats = {"d":[],"lambda":[],"a":[],"ARI":[]}
        aris_both_mean = [ ]
        aris_both_std = [ ]
        pout = b * np.log( n ) / n
        for d,l,a in tqdm(product(d_range,lambda_range,a_range)):
            aris_both = [ ]
            self.dims=d
            ### HERE ATT_CENTERS IS K x 1
            arr = self.att_centers.reshape(-1,1)
            self.att_centers = np.repeat(arr,d,axis=1)
            self.weight_centers = np.eye(self.n_clusters)
            self.weight_centers[self.weight_centers == 0] = 1/l
            
            pin = a * np.log( n ) / n
            p = (pin- pout) * np.eye( n_clusters ) + pout * np.ones( (n_clusters, n_clusters) )
            self.probability_matrix = p
            for _ in range( n_average ):
                ( X, Y, z_true, G) = benchmark_instance() 
                    
                A = (X != 0).astype(int)
                if binary:
                    X = A
                z_pred_both = None
                model = self.model_(n_clusters=n_clusters,\
                                        attributeDistribution=self.attributes_distribution_name,\
                                        edgeDistribution=self.edge_distribution_name,\
                                        weightDistribution=self.weight_distribution_name,\
                                        n_iters=n_iters)
                if self.torch_model:
                    graph_data = self.to_pyg_data(X,Y)
                    A = torch.tensor(A).to_sparse()
                    E = None
                    if graph_data.edge_attr is None:
                        E = torch.ones((graph_data.edge_index.shape[1],1))
                    else:
                        E = graph_data.edge_attr.reshape(-1,1)
                    z_pred_both = model.fit(A,E,graph_data.x).predict( E, graph_data.x )
                else:
                    z_pred_both = model.fit(A,X.reshape(n,n,-1),Y).predict( X, Y )
                aris_both.append( adjusted_rand_score( z_true, z_pred_both ) )
                aris_both_mean.append( np.mean( aris_both ) )
                aris_both_std.append( np.std( aris_both ) )
            
            ##restore centers to original K x 1 shape
            self.att_centers = arr
            ## gather stats
            stats["d"].append(d)
            stats["lambda"].append(l)
            stats["a"].append(a)
            stats["ARI"].append(aris_both_mean[-1])
       
        return stats
    
    def run_2_4(self,n_average=10,cluster_sizes=[100],\
                 w_averages = [ 1, 2, 3, 4, 5],\
                 att_averages = [ 1, 2, 3, 4, 5],\
                 b = 5,\
                 dense=False,\
                 binary=False,\
                 n_iters=25):
        
        self.communities_sizes = cluster_sizes
        benchmark_instance = None
        if dense:
            benchmark_instance = self.generate_benchmark_dense
        else:
            benchmark_instance = self.generate_benchmark_joint

        n = np.sum(cluster_sizes)
        n_clusters = len(cluster_sizes)
        self.n_clusters = n_clusters
        stats = {"attributes_avg":[],"weights_avg":[],"ARI":[]}
        aris_both_mean = [ ]
        aris_both_std = [ ]
        pout = b * np.log( n ) / n
        for lw, la in tqdm(product(w_averages,att_averages)):
            aris_both = [ ]
            self.dims=1
            ### HERE ATT_CENTERS IS K x 1
            self.att_centers=np.array([1,la]).reshape(-1,1)
            self.weight_centers = np.eye(self.n_clusters)*lw
            self.weight_centers[self.weight_centers == 0] = 1
            
            pin = b * np.log( n ) / n
            # p = (pin- pout) * np.eye( n_clusters ) + pout * np.ones( (n_clusters, n_clusters) )
            p = np.ones((self.n_clusters,self.n_clusters))*pin
            self.probability_matrix = p
            for _ in range( n_average ):
                ( X, Y, z_true, G) = benchmark_instance() 
                    
                A = (X != 0).astype(int)
                if binary:
                    X = A
                z_pred_both = None
                model = self.model_(n_clusters=n_clusters,\
                                        attributeDistribution=self.attributes_distribution_name,\
                                        edgeDistribution=self.edge_distribution_name,\
                                        weightDistribution=self.weight_distribution_name,\
                                        n_iters=n_iters)
                if self.torch_model:
                    graph_data = self.to_pyg_data(X,Y)
                    A = torch.tensor(A).to_sparse()
                    E = None
                    if graph_data.edge_attr is None:
                        E = torch.ones((graph_data.edge_index.shape[1],1))
                    else:
                        E = graph_data.edge_attr.reshape(-1,1)
                    z_pred_both = model.fit(A,E,graph_data.x).predict( E, graph_data.x )
                else:
                    z_pred_both = model.fit(A,X.reshape(n,n,-1),Y).predict( X, Y )
                aris_both.append( adjusted_rand_score( z_true, z_pred_both ) )
                aris_both_mean.append( np.mean( aris_both ) )
                aris_both_std.append( np.std( aris_both ) )
            
            ## gather stats
            stats["weights_avg"].append(lw)
            stats["attributes_avg"].append(la)
            stats["ARI"].append(aris_both_mean[-1])
       
        return stats
    
    def run_2_5(self,n_average=10,cluster_sizes=[100],\
                 w_averages = [ 1, 2, 3, 4, 5],\
                 att_averages = [ 1, 2, 3, 4, 5],\
                 b = 5,\
                 dense=False,\
                 n_iters=25):
        
        self.communities_sizes = cluster_sizes
        benchmark_instance = None
        if dense:
            benchmark_instance = self.generate_benchmark_dense
        else:
            benchmark_instance = self.generate_benchmark_joint

        n = np.sum(cluster_sizes)
        n_clusters = len(cluster_sizes)
        self.n_clusters = n_clusters
        stats = {"attributes_avg":[],"weights_avg":[],"ARI":[], "varying":[], "ARI_std":[],"algorithm":[]}
        aris_both_mean = [ ]
        aris_both_std = [ ]
        pout = b * np.log( n ) / n

        for varying in ["graph","attributes"]:
            print(">>> ",varying)
            aris_attributes_mean = [ ]
            aris_graph_mean = [ ]
            aris_both_mean = [ ]
            aris_attSBM_mean = [ ]
            aris_IR_sLS_mean = [ ]
            aris_IR_LS_mean = [ ]
            aris_oracle_mean = [ ]

            aris_attributes_std = [ ]
            aris_graph_std = [ ]
            aris_both_std = [ ]
            aris_attSBM_std = [ ]
            aris_IR_sLS_std = [ ]
            aris_IR_LS_std = [ ]
            aris_oracle_std = [ ]

            if varying == 'graph':
                loop = tqdm( range( len( w_averages ) ) )
            else:
                loop = tqdm( range( len( att_averages ) ) )

            for dummy in loop:
                if varying == 'graph':
                    lw = w_averages[ dummy ]
                    la = 2
                elif varying == 'attributes':
                    lw = 2
                    la = att_averages[ dummy ]
                
                aris_attributes = [ ]
                aris_graph = [ ]
                aris_both = [ ]
                aris_attSBM = [ ]
                aris_IR_sLS  = [ ]
                aris_IR_LS = [ ]
                aris_oracle = [ ]
                
                path_ = path_to_data+f"lw/{lw}/la/{la}/"
                if not os.path.exists(path_):
                    os.makedirs(path_)
                
                self.dims=1
                ### HERE ATT_CENTERS IS K x 1
                self.att_centers=np.array([1,la]).reshape(-1,1)
                self.weight_centers = np.eye(self.n_clusters)*lw
                self.weight_centers[self.weight_centers == 0] = 1
                
                pin = b * np.log( n ) / n
                # p = (pin- pout) * np.eye( n_clusters ) + pout * np.ones( (n_clusters, n_clusters) )
                p = np.ones((self.n_clusters,self.n_clusters))*pin
                self.probability_matrix = p

                for trial in range( n_average ):
                    ( X, Y, z_true, G) = benchmark_instance() 
                        
                    A = (X != 0).astype(int)
                    model = self.model_(n_clusters=n_clusters,\
                                        attributeDistribution=self.attributes_distribution_name,\
                                        edgeDistribution=self.edge_distribution_name,\
                                        weightDistribution=self.weight_distribution_name,\
                                        n_iters=n_iters)
                    ## For comparison purposes, the initialization is the same for IR-sLS, IR-LS and ours    
                    model.initialize(A,X.reshape(n,n,-1),Y)
                    model.assignInitialLabels(A, Y)
                    z_init = deepcopy(model.predicted_memberships)
                    chernoff_init_graph = model.graph_init
                    chernoff_graph_labels = model.memberships_from_graph
                    chernoff_att_labels = model.memberships_from_attributes

                    with open(f'{path_}att_{trial}.npy', 'wb') as g:
                        np.save(g, Y)
                    with open(f'{path_}net_{trial}.npy', 'wb') as g:
                        np.save(g, X)
                    with open(f'{path_}z_init_{trial}.npy', 'wb') as g:
                        np.save(g, csbm.convertZ(z_init)+1)

                    z_pred_both = model.fit(A,X.reshape(n,n,-1),Y).predict( X, Y )
                    z_pred_graph = frommembershipMatriceToVector( chernoff_graph_labels )
                    z_pred_attributes = frommembershipMatriceToVector( chernoff_att_labels )
                    
                    IR_sLS_pred = csbm.iter_csbm(X,Y,z_init,n_clusters)
                    # IR_LS_pred = csbm.iter_csbm2(X,Y,z_init,n_clusters)
                        
                    subprocess.call(["/usr/bin/Rscript","--vanilla",f"{base_path}/run_AttSBM.r",\
                                    f'{path_}att_{trial}.npy',\
                                    f'{path_}net_{trial}.npy',\
                                    f'{path_}z_init_{trial}.npy'])
                    attSBMPred = np.load("predict.npy")

                    aris_attributes.append( adjusted_rand_score( z_true, z_pred_attributes ) )
                    aris_graph.append( adjusted_rand_score( z_true, z_pred_graph ) )
                    aris_both.append( adjusted_rand_score( z_true, z_pred_both ) )
                    aris_attSBM.append( adjusted_rand_score( z_true, attSBMPred ) )
                    aris_IR_sLS.append( adjusted_rand_score( z_true, IR_sLS_pred ) )
                    # aris_IR_LS.append( adjusted_rand_score( z_true, IR_LS_pred ) )
                    
                    # if chernoff_init_graph:
                    #     z_pred_att_init = model.fit(A,X.reshape(n,n,1),Y,chernoff_att_labels).predict( X, Y )
                    #     ari_att_init = adjusted_rand_score( z_true, z_pred_att_init)
                    #     aris_oracle.append( max(aris_both[-1], ari_att_init))
                    # elif not chernoff_init_graph:
                    #     z_pred_graph_init = model.fit(A,X.reshape(n,n,1),Y,chernoff_graph_labels).predict( X, Y )
                    #     ari_graph_init = adjusted_rand_score( z_true, z_pred_graph_init)
                    #     aris_oracle.append( max(aris_both[-1], ari_graph_init))
                
                ### End of trials for a pair (lw,la)        
                aris_attributes_mean.append( np.mean( aris_attributes ) )
                aris_graph_mean.append( np.mean( aris_graph ) )
                aris_both_mean.append( np.mean( aris_both ) )
                aris_attSBM_mean.append( np.mean( aris_attSBM ) )
                aris_IR_sLS_mean.append( np.mean( aris_IR_sLS ) )
                # aris_IR_LS_mean.append( np.mean( aris_IR_LS ) )
                #aris_oracle_mean.append( np.mean( aris_oracle) )
                
                aris_attributes_std.append( np.std( aris_attributes ) )
                aris_graph_std.append( np.std( aris_graph ) )
                aris_both_std.append( np.std( aris_both ) )
                aris_attSBM_std.append( np.std( aris_attSBM ) )
                aris_IR_sLS_std.append( np.std( aris_IR_sLS ) )
                # aris_IR_LS_std.append( np.std( aris_IR_LS ) )
                #aris_oracle_std.append( np.std( aris_oracle) )
                
                stats["varying"].append(varying)
                stats["weights_avg"].append(lw)
                stats["attributes_avg"].append(la)
                stats["ARI"].append(aris_both_mean[-1])
                stats["ARI_std"].append(aris_both_std[-1])
                stats["algorithm"].append("ours")

                stats["varying"].append(varying)
                stats["weights_avg"].append(lw)
                stats["attributes_avg"].append(la)
                stats["ARI"].append(aris_IR_sLS_mean[-1])
                stats["ARI_std"].append(aris_IR_sLS_std[-1])
                stats["algorithm"].append("IR_sLS")

                stats["varying"].append(varying)
                stats["weights_avg"].append(lw)
                stats["attributes_avg"].append(la)
                stats["ARI"].append(aris_attSBM_mean[-1])
                stats["ARI_std"].append(aris_attSBM_std[-1])
                stats["algorithm"].append("attSBM")

                stats["varying"].append(varying)
                stats["weights_avg"].append(lw)
                stats["attributes_avg"].append(la)
                stats["ARI"].append(aris_graph_mean[-1])
                stats["ARI_std"].append(aris_graph_std[-1])
                stats["algorithm"].append("graph")

                stats["varying"].append(varying)
                stats["weights_avg"].append(lw)
                stats["attributes_avg"].append(la)
                stats["ARI"].append(aris_attributes_mean[-1])
                stats["ARI_std"].append(aris_attributes_std[-1])
                stats["algorithm"].append("attributes")
                #stats["ARI_ORACLE"].append(aris_oracle_mean[-1])
            
            ## End of dummy loop
            curves = [ aris_attributes_mean, aris_graph_mean,\
                    aris_both_mean , aris_attSBM_mean, aris_IR_sLS_mean]

            curves_std = [ aris_attributes_std, aris_graph_std,\
                            aris_both_std ,aris_attSBM_std, aris_IR_sLS_std
                        ]

            labels = [ 'EM-GMM', 'SC', 'Algo1', 'attSBM','IR_sLS']
            saveFig = True
            if varying == 'graph':    
                fileName = 'N_' + str(n) + '_K_' + str(n_clusters) + '_att_' + str(la)  +  '_nAverage' + str(n_average) + '.jpeg'
                plotting( w_averages, curves, labels, curves_std = curves_std, xticks = w_averages, xlabel = 'weights_avg', saveFig = saveFig, fileName = fileName )
                plt.close()
            elif varying == 'attributes':
                fileName = 'N_' + str(n) + '_K_' + str(n_clusters) + '_w_' + str(lw) + '_nAverage_' + str(n_average) + '.jpeg'
                plotting( att_averages, curves, labels, curves_std = curves_std, xticks = att_averages, xlabel = 'attributes_avg', saveFig = saveFig, fileName = fileName )
                plt.close()
        
        return stats
    
    def run_ABCD_benchmark(self,n=100000,n_clusters = 4,n_iters=30):
        size = n//n_clusters
        self.num_nodes = n
        d_min = int(2*np.log2(n))
        d_max = int(100*np.log2(n))
        margin = int(0.01*size)
        G,Y,labels_true = self.generate_ABCD_benchmark(d_min=d_min,d_max=d_max,\
                                                       c_min=size-margin,\
                                                       c_max=size+margin,num_nodes=n)

        # A = nx.adjacency_matrix(G)
        A = nx.to_numpy_array(G,dtype=np.float16)
        E = None
        rows,cols = A.nonzero()
        E = A[rows,cols].reshape(-1,1)
        K = np.unique(labels_true).shape[0]   
        print(K)     
        both_soft = None
        model = BregmanInitializer(self.n_clusters,initializer=self.initializer,
                                    edgeDistribution = self.edgeDistribution,
                                    attributeDistribution = self.attributeDistribution,
                                    weightDistribution = self.weightDistribution)
        model.initialize( E, Y , (rows,cols))
        both_soft =  model.predicted_memberships
        n = A.shape[0]
        # Z_init = csr_array((np.ones(n),\
        #         (np.arange(n),labels_true)),\
        #         shape=(n, K)
        #     )   
        # Z_init = fromVectorToMembershipMatrice(labels_true,K)
        # print(Z_init.shape)
        # preds = fit_leiden((rows,cols),E,n)
        # both_soft = preds.flatten()  
        # both_soft = model_soft.fit(A,E,Y).predict( None, None )
        # SC5 = BregmanKernelClustering(K, 
        #         edgeSimilarity = "gaussian",
        #         weightDistribution = "gaussian",
        #         attributeDistribution = "gaussian",
        #         single_metric=True,
        #         n_components=int(2*np.log2(n)),use_nystrom=True)
        
        # SC5.fit(A,E,Y)
        # both_soft = SC5.labels_
        algo_names = ["soft"]
        y_preds = [both_soft]
        scores_all = get_metrics_all_preds(labels_true, y_preds, algo_names)
        return scores_all,algo_names
    
    def get_real_data(self):
        data_dir = "../../RealDataSets/"
        data_sets = ["CiteSeer","Cora"]
        data_sets2 = ["Cornell", "Texas", "Wisconsin"]
        data_names = []
        datas = []
        for data_set in data_sets:
            dataset = Planetoid(root=data_dir, name=data_set)
            data = dataset[0]
            datas.append(data)
            data_names.append(data_set)
        for data_set in data_sets2:
            dataset = WebKB(root=data_dir, name=data_set)
            data = dataset[0]
            datas.append(data)
            data_names.append(data_set)
        return datas,data_names
    
    def preprocess_real_data(self,data,reduction_method):
        E = None ##Edge data
        # A = nx.to_numpy_array(G_nx)
        edge_index, edge_attr, mask = remove_isolated_nodes(data.edge_index)
        A = to_dense_adj(edge_index).numpy()[0]
        n = A.shape[0]

        if data.edge_attr is None:
            E = A.reshape(n,n,1)
        else:
            E = data.edge_attr[mask].numpy()
        # print(A.shape,E.shape,attributes[mask].numpy().shape)
        z_true = data.y[mask].numpy()
        K = np.unique(z_true).shape[0] ##Number of clusters
        attributes = data.x[mask,:]
        if self.preprocess:
            attributes = torch.Tensor(preprocess(attributes.numpy(),z_true,method=reduction_method))
        
        return K,A,E,attributes.numpy(),z_true
    
    def real_data_single_run(self,K,A,E,Y,z_true,n_iters,\
                             edgeSimilarity,\
                             weightSimilarity,\
                             attributesSimilarity):
        H = np.hstack((A,A.T))
        
        if (H.sum(axis=1) == 0).any():
            raise ValueError("GOT disconnected node")
        
        edge_index = np.nonzero(A)
        # print(index[0].shape,E.shape)
        SC = SpectralClustering(n_clusters=K,\
                                assign_labels='discretize',random_state=0).fit(H)

        SC2 = BregmanKernelClustering(K, 
                edgeSimilarity = "jaccard",
                weightDistribution = weightSimilarity,
                attributeDistribution = attributesSimilarity,
                single_metric=True)
        
        SC2.fit(A,E,Y)

        # SC3 = BregmanKernelClustering(K, 
        #         edgeSimilarity = "hamming",
        #         weightDistribution = weightSimilarity,
        #         attributeDistribution = attributesSimilarity,
        #         single_metric=True)
        
        # SC3.fit(A,E,Y)

        SC4 = BregmanKernelClustering(K, 
                edgeSimilarity = "gaussian",
                weightDistribution = weightSimilarity,
                attributeDistribution = "gaussian",
                single_metric=True)
        
        SC4.fit(A,E,Y)

        SC5 = BregmanKernelClustering(K, 
                edgeSimilarity = "gaussian",
                weightDistribution = weightSimilarity,
                attributeDistribution = "gaussian",
                single_metric=False)
        
        SC5.fit(A,E,Y)

        both_soft = None
        model_soft = softBreg(n_clusters=K,\
                                        attributeDistribution=self.attributes_distribution_name,\
                                        edgeDistribution=self.edge_distribution_name,\
                                        weightDistribution=self.weight_distribution_name,\
                                        initializer=self.initializer,
                                        use_random_init=False,
                                        n_iters=n_iters
                            )
        
        #Z_init = fromVectorToMembershipMatrice(SC2.labels_,K)
        both_soft = model_soft.fit(A,E,Y).predict( None, None )

        both_hard = None
        model_hard = hardBreg(n_clusters=K,\
                                        attributeDistribution=self.attributes_distribution_name,\
                                        edgeDistribution=self.edge_distribution_name,\
                                        weightDistribution=self.weight_distribution_name,\
                                        initializer=self.initializer,
                                        use_random_init=False,
                                        n_iters=n_iters
                            )
        #Z_init = fromVectorToMembershipMatrice(SC2.labels_,K)
        both_hard = model_hard.fit(A,E,Y).predict( None, None )

        kmeans = KMeans(n_clusters=K, random_state=0, n_init="auto").fit(Y)
            
        leiden_labels_ = model_hard.memberships_from_graph

        y_preds = [
                both_hard,
                both_soft,
                model_hard.memberships_from_graph,
                model_hard.memberships_from_attributes,
                kmeans.labels_,
                leiden_labels_.flatten(),
                SC.labels_,
                SC2.labels_,
                SC4.labels_,
                SC5.labels_
            ]

        algo_names = [
                "both_hard",
                "both_soft",
                "net",
                "att",
                "kmeans",
                "leiden",
                "SC",
                "SC_jaccard",
                "SC_gaussian_1",
                "SC_gaussian_2"
            ]

        scores_all = get_metrics_all_preds(z_true, y_preds, algo_names)
        return scores_all,algo_names

    def run_real_data(self,n_iters=25,
                      reduction_method="KBest",\
                      plot_class_dist=True,\
                      n_runs=10,
                      edgeSimilarity="jaccard",
                      weightSimilarity="gaussian",
                      attributesSimilarity="hamming"):
        datas,data_names = self.get_real_data()
        scores_agg_datasets = {}

        dict_ = get_metrics_pred(np.random.randint(0,2,4),np.random.randint(0,2,4))
        metric_names = []
        for key in dict_:
            metric_names.append(key)
            scores_agg_datasets[key] = []        
            scores_agg_datasets[key+"_std"] = []
        print(metric_names)
        scores_agg_datasets["algorithm"] = []
        scores_agg_datasets["dataset"] = []

        for data,data_name in zip(datas,data_names):
            print("\nCURRENT DATASET: ",data_name)

            K,A,E,Y,z_true = self.preprocess_real_data(data,reduction_method)
            
            if plot_class_dist:
                plot_class_dist_(z_true,data_name)
            
            metrics_per_run = {}
            algo_names = None
            for metric in metric_names:
                metrics_per_run[metric] = np.zeros((10,n_runs))
            # print("INPUTS: ",A.shape,E.shape,Y.shape)
            
            for j in range(n_runs):
                scores_all,algo_names = self.real_data_single_run(K,A,E,Y,z_true,\
                                                                  n_iters,\
                                                                  edgeSimilarity,\
                                                                  weightSimilarity,
                                                                  attributesSimilarity)
                for metric_name in metric_names:
                    metrics_per_run[metric_name][:,j] = np.array(scores_all[metric_name])

            scores_agg_datasets["algorithm"].extend(algo_names)
            for metric in metric_names:
                scores_agg_datasets[metric].extend(list(metrics_per_run[metric].mean(axis=1)))
                scores_agg_datasets[metric+"_std"].extend(list(metrics_per_run[metric].std(axis=1)))
            scores_agg_datasets["dataset"].extend([data_name]*len(algo_names))
            
            A = None
            E = None
            Y = None
            z_true = None
            torch.cuda.empty_cache()
        return scores_agg_datasets