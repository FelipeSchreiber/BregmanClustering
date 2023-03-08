#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 17:08:00 2023

@author: maximilien
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm


import models as models
import divergences as divergences
from sklearn.metrics import accuracy_score, adjusted_rand_score, adjusted_mutual_info_score


SIZE_TITLE = 24
SIZE_LABELS = 24
SIZE_TICKS = 18
SIZE_LEGEND = 18


"""
import warnings
warnings.filterwarnings("ignore" )
#Somehow some errors message sometimes arises such as 
#RuntimeWarning: divide by zero encountered in log
#But this shouldn't be a problem


n_average = 3

varying = 'attributes'

n = 450
n_clusters = 3
sizes = [ n // n_clusters ]*np.ones( n_clusters, dtype = int )


b = 4
pout = b * np.log( n ) / n

a_range = [ 5,7,9,11,13,15 ]
r_range = [ 0,1,2,3,4,5 ]

aris_attributes_mean = [ ]
aris_graph_mean = [ ]
aris_both_mean = [ ]

aris_attributes_std = [ ]
aris_graph_std = [ ]
aris_both_std = [ ]

if varying == 'graph':
    loop = tqdm( range( len( a_range ) ) )
else:
    loop = tqdm( range( len( r_range ) ) )

for dummy in loop:
    if varying == 'graph':
        a = a_range[ dummy ]
        r = 2
    elif varying == 'attributes':
        a = 8
        r = r_range[ dummy ]
    mu = [ [r], [-r] ]
    mu = unitRootCoordinates( d=3, r=r )
    
    pin = a * np.log( n ) / n
    p = (pin- pout) * np.eye( n_clusters ) + pout * np.ones( (n_clusters, n_clusters) )
    
    aris_attributes = [ ]
    aris_graph = [ ]
    aris_both = [ ]

    for trial in range( n_average ):
        ( X, Y, z_true ) = generateData( sizes, p, mu )
        
        model = models.BregmanNodeAttributeGraphClustering( n_clusters = n_clusters)
        model.fit( X, Y )
        z_pred_both = model.predict( X, Y )
        z_pred_graph = models.frommembershipMatriceToVector( model.memberships_from_graph )
        z_pred_attributes = models.frommembershipMatriceToVector( model.memberships_from_attributes )

        aris_attributes.append( adjusted_rand_score( z_true, z_pred_attributes ) )
        aris_graph.append( adjusted_rand_score( z_true, z_pred_graph ) )
        aris_both.append( adjusted_rand_score( z_true, z_pred_both ) )
        
    aris_attributes_mean.append( np.mean( aris_attributes ) )
    aris_graph_mean.append( np.mean( aris_graph ) )
    aris_both_mean.append( np.mean( aris_both ) )
    
    aris_attributes_std.append( np.std( aris_attributes ) )
    aris_graph_std.append( np.std( aris_graph ) )
    aris_both_std.append( np.std( aris_both ) )




curves = [ aris_attributes_mean, aris_graph_mean, aris_both_mean ]
curves_std = [ aris_attributes_std, aris_graph_std, aris_both_std ]

labels = [ 'attributes', 'graph', 'both' ]
saveFig = False
if varying == 'graph':    
    fileName = 'N_' + str(n) + '_K_' + str(n_clusters) + '_b_' + str(b) + '_r_' + str(r) +  '_nAverage' + str(n_average) + '.eps'
    plotting( a_range, curves, labels, curves_std = curves_std, xticks = a_range, xlabel = 'a', saveFig = saveFig, fileName = fileName )

elif varying == 'attributes':
    fileName = 'N_' + str(n) + '_K_' + str(n_clusters) + '_a_' + str(a) + '_b_' + str(b) +  '_nAverage_' + str(n_average) + '.eps'
    plotting( r_range, curves, labels, curves_std = curves_std, xticks = r_range, xlabel = 'r', saveFig = saveFig, fileName = fileName )


"""


def unitRootCoordinates( d = 2, r=1 ):
    coordinates = []
    for k in range( d ):
        coordinates.append( [ r*np.cos(2*k*np.pi / d ), r*np.sin( 2*k*np.pi / d ) ])
    return coordinates

def generateData( sizes, p, mu ):
    n = sum( sizes )
    n_clusters = len( sizes )
    d = len( mu[0] )
    
    labels_true = [ ]
    for k in range( n_clusters ):
        labels_true += [ k for i in range( sizes[ k ] ) ]
    labels_true = np.asarray( labels_true, dtype = int )

    G = nx.stochastic_block_model( sizes, p )
    X = nx.adjacency_matrix( G ).todense()
    
    Y = np.zeros( ( n,d ) )
    for i in range( n ):
        Y[i,:] = np.random.normal( loc = mu[ labels_true[i] ] )
    return np.asarray(X), Y, labels_true


def plotting( x, curves, labels, xticks,
             curves_std = None,
             legendTitle = '', figTitle = '',
             xlabel = 'a', ylabel = 'ARI',
             saveFig = False, fileName = 'fig.eps'):
    
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
        plt.savefig( fileName, format = 'eps', bbox_inches = 'tight' )


