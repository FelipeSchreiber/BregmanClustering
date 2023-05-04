import numpy as np
import networkx as nx
from BregmanTests.distributions import *
from BregmanClustering.models import *
from BregmanClustering.models import BregmanNodeEdgeAttributeGraphClustering as edgeBreg
from BregmanClusteringTorch.torch_models import BregmanEdgeClusteringTorch as torchBreg
from BregmanClusteringTorch.torch_models import BregmanEdgeClusteringTorchSparse as sparseBreg
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, accuracy_score
from torch_geometric.utils import to_networkx,to_dense_adj
from torch_geometric.datasets import Planetoid,WebKB
import torch 
import subprocess
from tqdm import tqdm
from .cfg import *
import os
from .utils import *
from copy import deepcopy
if not other_algos_installed:
    from .install_algorithms import main as install_env
    ## Optional: set repository for CRAN
    CRAN_repo = "https://cran.fiocruz.br/"
    install_env()
from CSBM.Python import functions as csbm
from itertools import product

class BregmanBenchmark():
    def __init__(self,P=None,communities_sizes=None,min_=1,max_=10,\
                    dims=2,\
                    weight_variance=1,att_variance=1,\
                    attributes_distribution = "gaussian",\
                    edge_distribution = "bernoulli",\
                    weight_distribution = "exponential",\
                    radius=None,return_G=False,\
                    att_centers = None, weight_centers = None, run_gpu=False):
        ## att_centers must have shape K x D, where K is the number
        #  of communities and D the number of dimensions.
        # If not specified, then the centers are taken from unit circle
        self.att_centers=att_centers
        ## weight_centers must have shape K(K+1)/2, where K is the number
        #  of communities
        # If not specified, then the weights are taken from linspace between [min_, max_]
        self.weight_centers=weight_centers
        self.probability_matrix=P
        self.communities_sizes=communities_sizes
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
        self.sparse_model = run_gpu
        if run_gpu:
            #self.model_ = torchBreg
            self.model = sparseBreg
        else:
            self.model_ = edgeBreg
            
    def generate_WSBM(self,complete_graph=False):
        N = np.sum(self.communities_sizes)
        ## Generate binary connectivity matrix
        G = None
        if complete_graph:
            G = nx.stochastic_block_model(self.communities_sizes,np.ones(shape=(self.n_clusters,self.n_clusters)),seed=42)
        else:
            G = nx.stochastic_block_model(self.communities_sizes,self.probability_matrix,seed=42)
        ## Draw the means of the weight distributions for each pair of community interaction
        if self.weight_centers is None:
            self.weight_centers = np.zeros((self.n_clusters,self.n_clusters))
            self.weight_centers[np.triu_indices(self.n_clusters, k = 0)] = \
                np.linspace(self.min_, self.max_, num=int(self.n_clusters*(self.n_clusters+1)/2))
            self.weight_centers = self.weight_centers + self.weight_centers.T - np.diag(np.diag(self.weight_centers))
        params = self.get_w_params(self.weight_centers,self.weight_variance,self.n_clusters)
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
                G.nodes[i]["attr"] = Y[i,:]
            return X,Y,labels_true,G
         return X,Y,labels_true,None
    
    def generate_benchmark_dense(self):
        X, G = self.generate_WSBM(complete_graph=True)
        Y = self.generate_attributes()
        labels_true = np.repeat(np.arange(self.n_clusters),self.communities_sizes)
        if self.return_G:
            for i in range(np.sum(self.communities_sizes)):
                G.nodes[i]["attr"] = Y[i,:]
            return X,Y,labels_true,G
        return X,Y,labels_true,None
    
    def run_test(self,n_average=10,cluster_sizes=100,\
                 b=5,\
                 a_range=[ 5,7,9,11,13,15 ],\
                 r_range = [ 0,1,2,3,4,5 ],\
                 dense=False,\
                 file_endings=".jpeg"):
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
        stats = {"varying":[],"a":[],"r":[],"agreed":[],"ARI_chernoff":[],"ARI_AIC":[],"ARI_ORACLE":[]}
        for varying in ["attributes","graph"]:

            aris_attributes_mean = [ ]
            aris_graph_mean = [ ]
            aris_both_mean = [ ]
            aris_attSBM_mean = [ ]
            aris_IR_sLS_mean = [ ]
            aris_IR_LS_mean = [ ]
            aris_both2_mean = [ ]
            aris_oracle_mean = [ ]

            aris_attributes_std = [ ]
            aris_graph_std = [ ]
            aris_both_std = [ ]
            aris_attSBM_std = [ ]
            aris_IR_sLS_std = [ ]
            aris_IR_LS_std = [ ]
            aris_both2_std = [ ]
            aris_oracle_std = [ ]

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
                self.probability_matrix = p
                self.radius = r

                aris_attributes = [ ]
                aris_graph = [ ]
                aris_both = [ ]
                aris_attSBM = [ ]
                aris_IR_sLS  = [ ]
                aris_IR_LS = [ ]
                aris_both2 = [ ]
                aris_oracle = [ ]
                
                path_ = path_to_data+f"a/{a}/r/{r}/"
                if not os.path.exists(path_):
                    os.makedirs(path_)

                total = 0
                for trial in range( n_average ):
                    ( X, Y, z_true, G) = benchmark_instance() 
                    
                    A = (X != 0).astype(int)
                    """
                    TO DO: SET ONLY EDGE MODEL
                    """
                    """
                    model = BregmanNodeAttributeGraphClustering( n_clusters = n_clusters,\
                                                                    attributeDistribution=self.attributes_distribution_name,\
                                                                    edgeDistribution=self.weight_distribution_name,\
                                                                    initializer="AIC")
                    """
                    ## For comparison purposes, the initialization is the same for IR-sLS, IR-LS and ours    
                    model.initialize(X,Y)
                    model.assignInitialLabels(X, Y)
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

                    model.fit( X, Y )
                    z_pred_both = model.predict( X, Y )
                    z_pred_graph = frommembershipMatriceToVector( chernoff_graph_labels )
                    z_pred_attributes = frommembershipMatriceToVector( chernoff_att_labels )
                    
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

                    model2 = self.model_(n_clusters=n_clusters,\
                                    attributeDistribution=self.attributes_distribution_name,\
                                    edgeDistribution=self.edge_distribution_name,\
                                    weightDistribution=self.weight_distribution_name
                                    )
                    z_pred_both2 = model2.fit(A,X.reshape(n,n,1),Y,z_init).predict( X, Y )
                    
                    IR_sLS_pred = csbm.iter_csbm(X,Y,z_init,n_clusters)
                    IR_LS_pred = csbm.iter_csbm2(X,Y,z_init,n_clusters)
                        
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
                    aris_IR_LS.append( adjusted_rand_score( z_true, IR_LS_pred ) )
                    aris_both2.append( adjusted_rand_score( z_true, z_pred_both2 ))
                    
                    if chernoff_init_graph != model.AIC_initializer(X,Y).graph_init:
                        ## both initializations were done
                        aris_oracle.append( max(aris_both[-1],aris_both2[-1]))
                    elif chernoff_init_graph:
                        z_pred_att_init = model.fit(X,Y,chernoff_att_labels).predict(X,Y)
                        ari_att_init = adjusted_rand_score( z_true, z_pred_att_init)
                        aris_oracle.append( max(aris_both[-1], ari_att_init))
                    elif not chernoff_init_graph:
                        z_pred_graph_init = model.fit(X,Y,chernoff_graph_labels).predict(X,Y)
                        ari_graph_init = adjusted_rand_score( z_true, z_pred_graph_init)
                        aris_oracle.append( max(aris_both[-1], ari_graph_init))
                        
                aris_attributes_mean.append( np.mean( aris_attributes ) )
                aris_graph_mean.append( np.mean( aris_graph ) )
                aris_both_mean.append( np.mean( aris_both ) )
                aris_attSBM_mean.append( np.mean( aris_attSBM ) )
                aris_IR_sLS_mean.append( np.mean( aris_IR_sLS ) )
                aris_IR_LS_mean.append( np.mean( aris_IR_LS ) )
                aris_oracle_mean.append( np.mean( aris_oracle) )
                
                aris_attributes_std.append( np.std( aris_attributes ) )
                aris_graph_std.append( np.std( aris_graph ) )
                aris_both_std.append( np.std( aris_both ) )
                aris_attSBM_std.append( np.std( aris_attSBM ) )
                aris_IR_sLS_std.append( np.std( aris_IR_sLS ) )
                aris_IR_LS_std.append( np.std( aris_IR_LS ) )
                aris_oracle_std.append( np.std( aris_oracle) )
                
                stats["varying"].append(varying)
                stats["a"].append(a)
                stats["r"].append(r)
                stats["agreed"].append(total/n_average)
                stats["ARI_chernoff"].append(aris_both_mean[-1])
                stats["ARI_AIC"].append(aris_both2_mean[-1])
                stats["ARI_ORACLE"].append(aris_oracle_mean[-1])
                
            curves = [ aris_attributes_mean, aris_graph_mean,\
                    aris_both_mean , aris_attSBM_mean, aris_IR_sLS_mean,\
                    aris_IR_LS_mean, aris_both2_mean]

            curves_std = [ aris_attributes_std, aris_graph_std,\
                        aris_both_std , aris_attSBM_std, aris_IR_sLS_std,\
                        aris_IR_LS_std, aris_both2_std]

            labels = [ 'attributes', 'graph', 'both' , 'attSBM', 'IR_sLS', 'IR_LS', "edgeBreg"]
            saveFig = True
            if varying == 'graph':    
                fileName = 'N_' + str(n) + '_K_' + str(n_clusters) + '_b_' + str(b) + '_r_' + str(r) +  '_nAverage' + str(n_average) + file_endings
                plotting( a_range, curves, labels, curves_std = curves_std, xticks = a_range, xlabel = 'a', saveFig = saveFig, fileName = fileName )
                plt.close()
            elif varying == 'attributes':
                fileName = 'N_' + str(n) + '_K_' + str(n_clusters) + '_a_' + str(a) + '_b_' + str(b) +  '_nAverage_' + str(n_average) + file_endings
                plotting( r_range, curves, labels, curves_std = curves_std, xticks = r_range, xlabel = 'r', saveFig = saveFig, fileName = fileName )
                plt.close()
    
    def run_2_1(self,n_average=10,cluster_sizes=100,\
                 b=5,\
                 a_range=[ 5,7,9,11,13,15 ],\
                 r_range = [ 0,1,2,3,4,5 ],\
                 dense=False,\
                 plot_3d=False,\
                 binary=True):
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
            self.probability_matrix = p
            self.radius = r
            aris_both = [ ]

            for _ in range( n_average ):
                ( X, Y, z_true, G) = benchmark_instance() 
                    
                A = (X != 0).astype(int)
                if binary:
                    X = A
                X = X.reshape(n,n,1)
                print(X.shape)
                model = self.model_(n_clusters=n_clusters,
                                    attributeDistribution=self.attributes_distribution_name,
                                    edgeDistribution=self.edge_distribution_name,
                                    weightDistribution=self.weight_distribution_name
                                    )
                z_pred_both = model.fit(A,X,Y).predict( X, Y )
                aris_both.append( adjusted_rand_score( z_true, z_pred_both ) )
                """
                if model.AIC_initializer(X,Y).graph_init:
                    z_pred_att_init = model.fit(A,X.reshape(n,n,1),Y,chernoff_att_labels).predict( X, Y )
                    ari_att_init = adjusted_rand_score( z_true, z_pred_att_init)
                    aris_oracle.append( max(aris_both[-1], ari_att_init))
                else:
                    z_pred_graph_init =  model.fit(A,X.reshape(n,n,1),Y,chernoff_graph_labels).predict( X, Y )
                    ari_graph_init = adjusted_rand_score( z_true, z_pred_graph_init)
                    aris_oracle.append( max(aris_both[-1], ari_graph_init))
                """     
                aris_both_mean.append( np.mean( aris_both ) )
                aris_both_std.append( np.std( aris_both ) )

            stats["a"].append(a)
            stats["r"].append(r)
            stats["ARI"].append(aris_both_mean[-1])
            #stats["ARI_ORACLE"].append(aris_oracle_mean[-1])
        
        return stats

    def run_2_2(self,n_average=10,cluster_sizes=100,\
                 d_range=[ 0,1,2,3,4,5 ],\
                 mu_range = [ 0,1,2,3,4,5 ],\
                 dense=True,\
                 binary=False):
        
        self.communities_sizes = cluster_sizes
        benchmark_instance = None
        if dense:
            benchmark_instance = self.generate_benchmark_dense
        else:
            benchmark_instance = self.generate_benchmark_joint

        n = np.sum(cluster_sizes)
        n_clusters = len(cluster_sizes)
        self.n_clusters = n_clusters
        stats = {"d":[],"mu":[],"ARI":[]}
        aris_both_mean = [ ]
        aris_both_std = [ ]
        for d,mu in tqdm(product(d_range,mu_range)):
            aris_both = [ ]
            self.dims=d
            ### HERE ATT_CENTERS IS K x 1
            arr = self.att_centers.reshape(-1,1)
            self.att_centers = np.repeat(arr,d,axis=1)
            self.weight_centers = np.eye(self.n_clusters)*mu
            for _ in range( n_average ):
                ( X, Y, z_true, G) = benchmark_instance() 
                    
                A = (X != 0).astype(int)
                if binary:
                    X = A
                X = X.reshape(n,n,1)
                model = self.model_(n_clusters=n_clusters,\
                                    attributeDistribution=self.attributes_distribution_name,\
                                    edgeDistribution=self.edge_distribution_name,\
                                    weightDistribution=self.weight_distribution_name
                                    )
                z_pred_both = model.fit(A,X,Y).predict( X, Y )
                aris_both.append( adjusted_rand_score( z_true, z_pred_both ) )
                """
                if model.AIC_initializer(X,Y).graph_init:
                    z_pred_att_init = model.fit(A,X.reshape(n,n,1),Y,chernoff_att_labels).predict( X, Y )
                    ari_att_init = adjusted_rand_score( z_true, z_pred_att_init)
                    aris_oracle.append( max(aris_both[-1], ari_att_init))
                else:
                    z_pred_graph_init =  model.fit(A,X.reshape(n,n,1),Y,chernoff_graph_labels).predict( X, Y )
                    ari_graph_init = adjusted_rand_score( z_true, z_pred_graph_init)
                    aris_oracle.append( max(aris_both[-1], ari_graph_init))
                """     
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
                 binary=False):
        
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
                X = X.reshape(n,n,1)
                model = self.model_(n_clusters=n_clusters,\
                                    attributeDistribution=self.attributes_distribution_name,\
                                    edgeDistribution=self.edge_distribution_name,\
                                    weightDistribution=self.weight_distribution_name
                                    )
                z_pred_both = model.fit(A,X,Y).predict( X, Y )
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
    
    def get_real_data(self):
        data_dir = "../../RealDataSets/"
        data_sets = ["PubMed","Cora","CiteSeer"]
        datas = []
        for data_set in data_sets:
            dataset = Planetoid(root=data_dir, name=data_set)
            data = dataset[0]
            datas.append(data)
        data_sets2 = ["Cornell", "Texas", "Wisconsin"]
        for data_set in data_sets2:
            dataset = WebKB(root=data_dir, name=data_set)
            data = dataset[0]
            datas.append(data)
        data_sets.extend(data_sets2)
        return datas,data_sets
    
    def run_real_data(self,sparse=True):
        datas,data_names = self.get_real_data()
        scores = {}
        scores["dataset"] = []
        scores["ARI"] = []
        for data,data_name in zip(datas,data_names):
            attributes = data.x
            z_true = data.y.numpy()
            z_pred_both = None
            K = np.unique(z_true).shape[0]
            E = None
            if sparse:
                A = torch.tensor(to_dense_adj(data.edge_index).numpy()[0]).to_sparse()
                if data.edge_attr is None:
                    E = torch.ones((data.edge_index.shape[1],1))
                else:
                    E = data.edge_attr

                model = sparseBreg(n_clusters=K,\
                                        attributeDistribution=self.attributes_distribution_name,\
                                        edgeDistribution=self.edge_distribution_name,\
                                        weightDistribution=self.weight_distribution_name
                                    )
                
                z_pred_both = model.fit(A,E,attributes).predict( E, attributes )
            
            else:
                A = to_dense_adj(data.edge_index).numpy()[0]
                n = A.shape[0]
                if datas[0].edge_attr is None:
                    E = A.reshape(n,n,1)
                else:
                    E = datas[0].edge_attr.numpy()

                model = self.model_(n_clusters=K,\
                                        attributeDistribution=self.attributes_distribution_name,\
                                        edgeDistribution=self.edge_distribution_name,\
                                        weightDistribution=self.weight_distribution_name
                                        )
                
                z_pred_both = model.fit(A,E,attributes).predict( E, attributes )

            A = None
            E = None
            attributes = None
            scores["ARI"].append(adjusted_rand_score( z_true, z_pred_both ))
            scores["dataset"].append(data_name)
            z_pred_both = z_true = None
            torch.cuda.empty_cache()
        return scores