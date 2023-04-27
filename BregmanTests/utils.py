from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.manifold import SpectralEmbedding

SIZE_TITLE = 24
SIZE_LABELS = 24
SIZE_TICKS = 18
SIZE_LEGEND = 18

def make_contour_plot(x,y,z,filename="contour.jpeg",plot_3d=False):
    # Create contour lines or level curves using matplotlib.pyplot module
    if plot_3d:
    # Plot the 3D surface
        X,Y = np.meshgrid(x,y)

        ax = plt.figure().add_subplot(projection='3d')
        ax.plot_surface(X, Y, z, edgecolor='royalblue', lw=0.5, rstride=8, cstride=8,
                        alpha=0.3)

        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        z_min, z_max = np.min(z), np.max(z)
        # Plot projections of the contours for each dimension.  By choosing offsets
        # that match the appropriate axes limits, the projected contours will sit on
        # the 'walls' of the graph.
        ax.contour(x, y, z, zdir='z', offset=z_min-1, cmap='coolwarm')
        ax.contour(x, y, z, zdir='x', offset=x_min-1, cmap='coolwarm')
        ax.contour(x, y, z, zdir='y', offset=y_min-1, cmap='coolwarm')

        #xlim=(-40, 40), ylim=(-40, 40), zlim=(-100, 100),
        ax.set(xlim=(x_min -1, x_max), ylim=(y_min-1, y_max), zlim=(z_min-1,z_max),\
               xlabel='a', ylabel='r', zlabel='ARI')
    else:
        contours = plt.contour(x, y, z)
        # Display z values on contour lines
        plt.clabel(contours, inline=1, fontsize=10)
        plt.xlabel("a")
        plt.ylabel("r")
    plt.savefig(filename)
    # Display the contour plot
    #plt.show()

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

    
    legend = plt.legend( title = legendTitle, loc=4,  fancybox=True, fontsize= SIZE_LEGEND )
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