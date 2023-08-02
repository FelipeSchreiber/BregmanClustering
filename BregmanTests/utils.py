from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import subprocess
subprocess.call(["git","clone","https://github.com/MartijnGosgens/validation_indices"])
from validation_indices import NamedIndices
from BregmanInitializer.init_cluster import frommembershipMatriceToVector, fromVectorToMembershipMatrice
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import pairwise_kernels, rbf_kernel, euclidean_distances
from sklearn.manifold import SpectralEmbedding
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import hamming_loss
from scipy.spatial.distance import sokalsneath, cdist
from scipy import stats
import umap
import itertools
from sklearn.metrics import *
import seaborn as sns

SIZE_TITLE = 24
SIZE_LABELS = 24
SIZE_TICKS = 18
SIZE_LEGEND = 18

## taken from sklearn
def gen_even_slices(n, n_packs, *, n_samples=None):
    """Generator to create `n_packs` evenly spaced slices going up to `n`.

    If `n_packs` does not divide `n`, except for the first `n % n_packs`
    slices, remaining slices may contain fewer elements.

    Parameters
    ----------
    n : int
        Size of the sequence.
    n_packs : int
        Number of slices to generate.
    n_samples : int, default=None
        Number of samples. Pass `n_samples` when the slices are to be used for
        sparse matrix indexing; slicing off-the-end raises an exception, while
        it works for NumPy arrays.

    Yields
    ------
    `slice` representing a set of indices from 0 to n.

    See Also
    --------
    gen_batches: Generator to create slices containing batch_size elements
        from 0 to n.

    Examples
    --------
    >>> from sklearn.utils import gen_even_slices
    >>> list(gen_even_slices(10, 1))
    [slice(0, 10, None)]
    >>> list(gen_even_slices(10, 10))
    [slice(0, 1, None), slice(1, 2, None), ..., slice(9, 10, None)]
    >>> list(gen_even_slices(10, 5))
    [slice(0, 2, None), slice(2, 4, None), ..., slice(8, 10, None)]
    >>> list(gen_even_slices(10, 3))
    [slice(0, 4, None), slice(4, 7, None), slice(7, 10, None)]
    """
    start = 0
    if n_packs < 1:
        raise ValueError("gen_even_slices got n_packs=%s, must be >=1" % n_packs)
    for pack_num in range(n_packs):
        this_n = n // n_packs
        if pack_num < n % n_packs:
            this_n += 1
        if this_n > 0:
            end = start + this_n
            if n_samples is not None:
                end = min(n_samples, end)
            yield range(start, end)
            start = end

def make_riemannian_metric(N,n_features,gamma=None,att_dist_=None):
    if gamma is None:
        gamma = 1.0 / n_features
    
    if att_dist_ is None:
        def att_dist_func(row1,row2):
            d = euclidean_distances(row1.reshape(1, -1),row2.reshape(1, -1))
            return gamma*d
        att_dist_ = att_dist_func
    
    def riemannian_metric(row1,row2):
        net_d = hamming_loss(row1[:N],row2[:N])
        att_d = att_dist_(row1[N:],row2[N:])
        distance = np.exp(-(net_d + att_d)) 
        return distance
    
    return riemannian_metric

def plot_class_dist_(data,dataset_name):
    sns.countplot(x=data)
    plt.title(dataset_name+" class freq")
    plt.show()

def preprocess(X,Y,K=10,method="KBest"):
    X_train_fs = None
    if method == "KBest":
        fs = SelectKBest(score_func=chi2, k=K)
        fs.fit(X, Y)
        X_train_fs = fs.transform(X)
    else:
        X_train_fs = umap.UMAP(n_components=K,metric="hamming",random_state=42).fit_transform(X)
    return X_train_fs

def scatter_(dict_,x_name,y_name,z_name):
  fig, ax = plt.subplots()
  x,y = dict_[x_name],dict_[y_name]
  ax.scatter(x,y)
  for i, txt in enumerate(dict_[z_name]):
      ax.annotate("{:.2f}".format(txt), (x[i], y[i]))
  ax.set_xlabel(x_name)
  ax.set_ylabel(y_name)

def scatter_with_colorbar(dict_,x_name,y_name,z_name):
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111)
    x,y,z = dict_[x_name],dict_[y_name],dict_[z_name]
    ticks = np.linspace(np.min(z), np.max(z), 5, endpoint=True)
    C = ax.scatter(x=x,y=y,c=z,cmap="coolwarm")
    cb = fig.colorbar(C, ax=ax, fraction=0.02, pad=0.1, label=z_name,ticks=ticks)
    cb.set_label(label=z_name, size=SIZE_LEGEND)
    cb.ax.tick_params(labelsize=SIZE_TICKS)
    plt.xlabel( x_name, fontsize = SIZE_LABELS )
    plt.ylabel( y_name, fontsize = SIZE_LABELS )
    plt.xticks( fontsize = SIZE_TICKS )
    plt.yticks( fontsize = SIZE_TICKS )

def make_4d_plot(X,Y,Z,data,x_label="d",y_label="lambda",z_label="a",filename="contour.jpeg"):
    kw = {
            'vmin': data.min(),
            'vmax': data.max()
        }

    # Create a figure with 3D ax
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection='3d')

    # Set limits of the plot from coord limits
    xmin, xmax = X.min(), X.max()
    ymin, ymax = Y.min(), Y.max()
    zmin, zmax = Z.min(), Z.max()
    ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])

    # Plot contour surfaces
    """
    SCATTER PLOT
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection='3d')
    C = ax.scatter(xs=X,ys=Y,zs=Z,c=Data)
    fig.colorbar(C, ax=ax, fraction=0.02, pad=0.1, label='ARI')
    plt.show()
    """
    ### This is the X x Y plane for z=zmax 
    _ = ax.contourf(
            X[:, :, -1], Y[:, :, -1], data[:, :, -1],
            zdir='z', offset=zmax, **kw
        )
    
    ### This is the X x Y plane for z=zmin 
    _ = ax.contourf(
            X[:, :, 0], Y[:, :, 0], data[:, :, 0],
            zdir='z', offset=zmin, **kw
        )

    ### This is the X x Z plane for y=ymax 
    _ = ax.contourf(
            X[-1, :, :], data[-1, :, :], Z[-1, :, :],
            zdir='y', offset=ymax, **kw
        )

    ### This is the X x Z plane for y=ymin 
    _ = ax.contourf(
            X[0, :, :], data[0, :, :], Z[0, :, :],
            zdir='y', offset=ymin, **kw
        )

    ### This is the Y x Z plane for x=xmax 
    _ = ax.contourf(
            data[:, -1, :], Y[:, -1, :], Z[:, -1, :],
            zdir='x', offset=xmax, **kw
        )

    ### This is the Y x Z plane for x=xmin
    C = ax.contourf(
            data[:, 0, :], Y[:, 0, :], Z[:, 0, :],
            zdir='x', offset=xmin, **kw
        )

    # --
    # Plot edges
    edges_kw = dict(color='0.4', linewidth=1,zorder=-1e3)
    ax.plot([xmax, xmax], [ymin, ymax], zmin, **edges_kw)
    ax.plot([xmin, xmax], [ymin, ymin], zmin, **edges_kw)
    ax.plot([xmin, xmax], [ymin, ymin], [zmin, zmax], **edges_kw)

    # Set labels and zticks
    ax.set(
            xlabel=x_label,
            ylabel=y_label,
            zlabel=z_label
        )

    # Set zoom and angle view
    ax.view_init(30, 45, 0)
    ax.set_box_aspect(None, zoom=0.9)

    # Colorbar
    fig.colorbar(C, ax=ax, fraction=0.02, pad=0.1, label='ARI')

    plt.show()


def make_contour_plot(x,y,z,x_label="a",y_label="r",filename="contour.jpeg"):
    origin = 'lower'
    fig1, ax = plt.subplots(layout='constrained')
    CS = ax.contourf(x, y, z, cmap=plt.cm.bone,origin=origin)
    # Display z values on contour lines
    CS2 = ax.contour(CS, colors='r',origin=origin)
    # Make a colorbar for the ContourSet returned by the contourf call.
    cbar = fig1.colorbar(CS)
    cbar.ax.set_ylabel('ARI')
    # Add the contour line levels to the colorbar
    cbar.add_lines(CS2)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.savefig(filename)
    # Display the contour plot
    plt.show()

def plotting( x, curves, labels, xticks,
             curves_std = None,
             legendTitle = '', figTitle = '',
             xlabel = 'a', ylabel = 'ARI',
             saveFig = False, fileName = 'fig.jpeg'):
    
    if len( curves ) != len( labels ):
        raise TypeError( 'The number of labels is different from the number of curves' )
    
    if curves_std == None:
        for i in range( len( labels) ):
            plt.plot( x, curves[i], label = labels[i])
    else:
        for i in range( len( labels) ):
            plt.errorbar( x, curves[ i ], yerr = curves_std[ i ], linestyle = '-.', label = labels[ i ] )

    
    legend = plt.legend( title = legendTitle, loc=4,  fancybox=True, fontsize= SIZE_LEGEND,\
                        bbox_to_anchor=(1, 0., 0.5, 0.5))
    plt.setp( legend.get_title(),fontsize= SIZE_LEGEND )
    plt.xlabel( xlabel, fontsize = SIZE_LABELS )
    plt.ylabel( ylabel, fontsize = SIZE_LABELS )
    plt.xticks( xticks, fontsize = SIZE_TICKS )
    plt.yticks( fontsize = SIZE_TICKS )
    plt.title( figTitle, fontsize = SIZE_TITLE )
    if saveFig:
        plt.savefig( fileName, format = 'jpeg', bbox_inches = 'tight' )
    plt.show()

#Funcao que dada a matriz de probabilidades retornada pelo algoritmo e
#  o groundtruth computa a acurácia
def best_perm_of_func(y_true,y_pred,f=accuracy_score):
    if len(y_pred) != len(y_true):
        raise ValueError('x and y must be arrays of the same size')
    scores = []
    possible_combinations = list(itertools.permutations(np.unique(y_pred)))
    # permutations = []
    best_perm = None
    best_score = -np.inf
    i = 0
    for combination in possible_combinations:
        pred = np.array([combination[i] for i in y_pred])
        score = f(y_true,pred)
        if score > best_score:
            best_score = score
            best_perm = pred
        del pred
        del score
        i+=1
        print("Iteration: ",i)
    # permutations.append(pred)
    # id_max = np.argmax(scores)
    # score = scores[id_max]
    return best_score,best_perm

def get_metrics_pred(y_true,y_pred):
    # acc,y_best = best_perm_of_func(y_true,y_pred,f=accuracy_score)
    y_best = y_pred
    ari = adjusted_rand_score( y_true , y_best )
    nmi = normalized_mutual_info_score( y_true , y_best )
    ami = adjusted_mutual_info_score( y_true , y_best )
    sokalsneath_ = NamedIndices["S&S1"]
    pearson = NamedIndices["CC"]
    ses = sokalsneath_.score(y_true.tolist(),y_best.tolist())
    CC = pearson.score(y_true.tolist(), y_best.tolist())
    # f1 = f1_score( y_true , y_best , average='macro'),"ACC":acc,"F1":f1
    return {"NMI":nmi,"ARI":ari, "AMI":ami,"S&S":ses, "CC":CC}

def get_metrics_all_preds(y_true, y_preds, algo_names):
    results = {}
    metrics_ = get_metrics_pred(y_true,y_preds[0])
    for key in metrics_:
        results[key] = []
    results["algorithm"] = []

    for algo_name,y_pred in zip(algo_names,y_preds):
        results["algorithm"].append(algo_name)
        metrics_ = get_metrics_pred(y_true,y_pred)
        for key, value in metrics_.items():
            results[key].append(value)
    return results

def get_spectral_decomposition(A,k):
    if (A<0).any():
        A = pairwise_kernels(A,metric='rbf')
    U = SpectralEmbedding(n_components=k,affinity="precomputed").fit_transform(A)
    return U
def spectral(A,k):
    U = get_spectral_decomposition(A,k)
    return GaussianMixture(n_components=k).fit_predict(U.real)
