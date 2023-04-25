from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
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
        ax = plt.figure().add_subplot(projection='3d')
        ax.plot_surface(x, y, z, edgecolor='royalblue', lw=0.5, rstride=8, cstride=8,
                        alpha=0.3)

        # Plot projections of the contours for each dimension.  By choosing offsets
        # that match the appropriate axes limits, the projected contours will sit on
        # the 'walls' of the graph.
        ax.contour(x, y, z, zdir='z', offset=-100, cmap='coolwarm')
        ax.contour(x, y, z, zdir='x', offset=-40, cmap='coolwarm')
        ax.contour(x, y, z, zdir='y', offset=40, cmap='coolwarm')

        ax.set(xlim=(-40, 40), ylim=(-40, 40), zlim=(-100, 100),
        xlabel='a', ylabel='r', zlabel='ARI')
    else:
        contours = plt.contour(x, y, z)
        # Display z values on contour lines
        plt.clabel(contours, inline=1, fontsize=10)
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

def get_spectral_decomposition(A,k):
    if (A<0).any():
        A = pairwise_kernels(A,metric='rbf')
    U = SpectralEmbedding(n_components=k,affinity="precomputed").fit_transform(A)
    return U
def spectral(A,k):
    U = get_spectral_decomposition(A,k)
    return GaussianMixture(n_components=k).fit_predict(U.real)