# BregmanClustering
To install simply type

```
pip install --upgrade --force-reinstall git+https://github.com/FelipeSchreiber/BregmanClustering.git
```
Usage example:
```python
##import model. You can choose SoftBregmanNodeAttributeGraphClustering as well 
from BregmanClustering.models import BregmanNodeAttributeGraphClustering as bregClust
from BregmanClusteringTorch.torch_models import SoftBregmanClusteringTorch as torchBreg

##import benchmark
from BregmanClustering.WSBM import *

""" Generate a benchmark with c=3 communities, each one with n=100 nodes, each one with attributes in R^d draw from specified distribution.
Edge weights can also be generate from a specified distribution, but only for 1-D. Requires a SBM probability matrix which describes the probability
connectivity between each community."""

c = 3
d = 2
n = 100
P = np.array([[0.8, 0.2, 0.3],[0.2, 0.7, 0.4],[0.3, 0.4, 0.6]])
true_labels = [0]*n + [1]*n + [2]*n
X,Y= BregmanBenchmark(P,[n]*c,-10,10,dims=d,weight_variance=0.01,att_variance=0.1,\
                       weight_distribution="logistic",attributes_distribution="logistic").generate_benchmark_WSBM()
A = (X != 0).astype(int)

model = bregClust(n_clusters=c)
#model = torchBreg(n_clusters=c,thresholding=True,normalize_=True)
model.fit(A,Y)
labels = model.predict(A,Y)
```
