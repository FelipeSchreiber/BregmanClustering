from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.manifold import SpectralEmbedding

SIZE_TITLE = 24
SIZE_LABELS = 24
SIZE_TICKS = 18
SIZE_LEGEND = 18

def make_4d_plot(x,y,z,data,filename="contour.jpeg"):
    kw = {
        'vmin': data.min(),
        'vmax': data.max(),
        'levels': np.linspace(data.min(), data.max(), 10),
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
    _ = ax.contourf(
        X[:, :, 0], Y[:, :, 0], data[:, :, 0],
        zdir='z', offset=0, **kw
    )
    ### This is the top of the box

    # _ = ax.contourf(
    #     X[:, :, -1], Y[:, :, -1], data[:, :, -1],
    #     zdir='z', offset=Z.max(), **kw
    # )
    _ = ax.contourf(
        X[0, :, :], data[0, :, :], Z[0, :, :],
        zdir='y', offset=0, **kw
    )
    C = ax.contourf(
        data[:, 0, :], Y[:, 0, :], Z[:, 0, :],
        zdir='x', offset=X.min(), **kw
    )
    # --

    # Plot edges
    edges_kw = dict(color='0.4', linewidth=1,zorder=-1e3)
    ax.plot([xmax, xmax], [ymin, ymax], zmin, **edges_kw)
    ax.plot([xmin, xmax], [ymin, ymin], zmin, **edges_kw)
    ax.plot([xmin, xmax], [ymin, ymin], [zmin, zmax], **edges_kw)

    # Set labels and zticks
    ax.set(
        xlabel='X [km]',
        ylabel='Y [km]',
        zlabel='Z [m]'
    )

    # Set zoom and angle view
    ax.view_init(30, 45, 0)
    ax.set_box_aspect(None, zoom=0.9)

    # Colorbar
    fig.colorbar(C, ax=ax, fraction=0.02, pad=0.1, label='ARI')

    plt.savefig(filename)
    # Show Figure
    plt.show()


def make_contour_plot(x,y,z,x_label="a",y_label="r",filename="contour.jpeg",plot_3d=False):
    # Create contour lines or level curves using matplotlib.pyplot module
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    z_min, z_max = np.min(z), np.max(z)
    if plot_3d:
    # Plot the 3D surface

        ax = plt.figure().add_subplot(projection='3d')
        ax.plot_surface(x, y, z, edgecolor='royalblue', lw=0.5, rstride=8, cstride=8,
                        alpha=0.3)

        # Plot projections of the contours for each dimension.  By choosing offsets
        # that match the appropriate axes limits, the projected contours will sit on
        # the 'walls' of the graph.
        ax.contour(x, y, z, zdir='z', offset=z_min, cmap='coolwarm')
        ax.contour(x, y, z, zdir='x', offset=x_min, cmap='coolwarm')
        ax.contour(x, y, z, zdir='y', offset=y_min, cmap='coolwarm')

        #xlim=(-40, 40), ylim=(-40, 40), zlim=(-100, 100),
        ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max), zlim=(z_min,z_max),\
               xlabel='a', ylabel='r', zlabel='ARI')
    else:
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
