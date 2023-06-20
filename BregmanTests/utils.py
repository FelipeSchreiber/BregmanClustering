from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.manifold import SpectralEmbedding
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import umap

SIZE_TITLE = 24
SIZE_LABELS = 24
SIZE_TICKS = 18
SIZE_LEGEND = 18

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

def get_spectral_decomposition(A,k):
    if (A<0).any():
        A = pairwise_kernels(A,metric='rbf')
    U = SpectralEmbedding(n_components=k,affinity="precomputed").fit_transform(A)
    return U
def spectral(A,k):
    U = get_spectral_decomposition(A,k)
    return GaussianMixture(n_components=k).fit_predict(U.real)
