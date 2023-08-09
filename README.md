# BregmanClustering
To install simply type

```
pip install --upgrade --force-reinstall git+https://github.com/FelipeSchreiber/BregmanClustering.git
```
Usage example:
```python
##import model
from BregmanClustering.models import BregmanClusteringMemEfficient as hardBreg
from BregmanClustering.models import BregmanNodeEdgeAttributeGraphClusteringSoft as softBreg

""" Generate a benchmark with c=3 communities, each one with n=100 nodes, each one with attributes in R^d draw from specified distribution.
Edge weights can also be generate from a specified distribution, but only for 1-D. Requires a SBM probability matrix which describes the probability
connectivity between each community."""

from BregmanTests.benchmark import *
n_average = 5
n_clusters = 3
factor = 300
n = int(factor*n_clusters)
d = 1
sizes = [ n // n_clusters ]*np.ones( n_clusters, dtype = int )


attributes_distribution = "gaussian"
edge_distribution = "bernoulli"
weight_distribution = "exponential"
a_range = np.linspace(5,14,3)
r_range = np.linspace(0,0.7,3)*np.log(n)
P = np.array([[0.8, 0.2, 0.3],[0.2, 0.7, 0.4],[0.3, 0.4, 0.6]])
( X, Y, z_true, G) = BregmanBenchmark(P=P,communities_sizes=sizes,
                    att_variance=1,
                    attributes_distribution=attributes_distribution,
                    weight_variance=1,
                    weight_distribution=weight_distribution,
                    edge_distribution=edge_distribution,
                    run_torch=False,
                    initializer = 'chernoff',
                    hard_clustering=True, return_G=False).generate_benchmark_joint()

##X is the n x n weighted adjacency matrix. Converts to |E| x 1 matrix.
edge_index = X.nonzero()
E = X.reshape(n,n,-1)[edge_index[0],edge_index[1],:]

model = hardBreg(n_clusters=n_clusters
                                        attributeDistribution=attributes_distribution,\
                                        edgeDistribution=edge_distribution,\
                                        weightDistribution=weight_distribution,\
                                        n_iters=25,
                                        initializer="chernoff")

preds = model.fit(edge_index,E,Y).predict( None, None)
```
