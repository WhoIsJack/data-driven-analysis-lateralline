# -*- coding: utf-8 -*-
"""
Created on Sun Dec 03 16:52:25 2017

@author:   Jonas Hartmann @ Gilmour group @ EMBL Heidelberg

@descript: Some pre-configured matplotlib plots.

"""

#------------------------------------------------------------------------------

# IMPORTS

# External
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist, squareform
import networkx as nx


#------------------------------------------------------------------------------

# 3D Plots of Point Clouds

def point_cloud_3D(X, Y, Z, figsize=(12,12),
                   marker='.', c='b', cmap=None, s=20, alpha=None, 
                   title='', xlbl='', ylbl='', zlbl='', config=True,
                   init=True, fin=True, pre_fig=None, pre_ax=None):
    """Pre-configuration for 3D scatter plots (using Axes3D) with properly
    configured equal axes.
    
    Parameters
    ----------
    X, Y, Z : iterables of floats
        The scatter point coordinates.
    figsize : tuple of shape (s_x,s_y), optional, Default (12,12)
        Figure size, see kwargs of plt.figure().
    marker : string, optional, default '.'
        Scatter marker type, see args of plt.scatter().
    c : string, optional, default 'b'
        Scatter point color, see args of plt.scatter().
    cmap : string or cmap object, optional, Default None
        Scatter point colormap, see args of plt.scatter().
    s : int, optional, default 20
        Scatter point size, see args of plt.scatter().
    alpha : float, optional, default None
        Scatter point opacity, see args of plt.scatter().
        Note that changing this will prevent alpha from being used auto-
        matically to enhance the 3D effect of the plotting, so it should be
        left as None whenever possible.        
    title, xlbl, ylbl, zlbl : strings, optional, default ''
        Strings to label the plot and its three axes.
    config : bool, optional, default True
        If True, axes are configured to have equal aspect ratio. This should be
        set to False for adding single points to the plot.
    init : bool, optional, default True
        If True, a new figure is initialized. This should be set to False when
        a second set of points is added to a previously initialized figure. In
        this case, pre_fig and pre_ax must be passed (see below).
    fin : bool, optional, default True
        If True, plt.show() is called before returning. Otherwise, the mpl
        figure and axis instances are returned. Used for adding other things to
        the plot after this function.
    pre_fig, pre_ax : mpl figure and axis instances
        If init is false, a compatible (3D!) mpl figure and axis instance must
        be passed through these arguments. The new scatter plot is then added
        to this plot.    
    
    Returns
    -------
    fig, ax : mpl figure and axis instances
        If fin is False, plt.show() is not being called and instead the figure
        and axis instances are being returned. 
    """
    
    # Initialize (if necessary)
    if init:
        fig = plt.figure(figsize=figsize)
        ax  = fig.add_subplot(111, projection='3d')
        if config: ax.set_aspect('equal')
        
    else:
        fig = pre_fig
        ax  = pre_ax

    # Set panes white
    ax.axes.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.axes.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.axes.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    # Plot the points
    ax.scatter(X, Y, Z, 
               marker=marker, c=c, lw=0, s=s, cmap=cmap, alpha=alpha)

    # Some SO shenanigans to get the aspect ratios right in 3D
    if config:
        max_range = np.array([X.max()-X.min(), 
                              Y.max()-Y.min(), 
                              Z.max()-Z.min()]).max() / 2.0
        mid_x = (X.max()+X.min()) * 0.5
        mid_y = (Y.max()+Y.min()) * 0.5
        mid_z = (Z.max()+Z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Other cosmetics
    if title: ax.set_title(title)
    if xlbl: ax.set_xlabel(xlbl)
    if ylbl: ax.set_ylabel(ylbl)
    if zlbl: ax.set_zlabel(zlbl)
    
    # Show
    if fin:
        plt.show()
        return
    else:
        return fig, ax
        

#------------------------------------------------------------------------------

# Sort columns of a feature space based on a Monte Carlo optimization

def _similarity_argsort(fspace, lossfun='default', max_iter=2000,
                        norm_dists=True, is_dists=False, verbose=False):
    """Sort columns of a feature space based on similarity. (BETA)

    Uses the pairwise distances between columns (features in row (sample) space
    to compute a sort of columns that minimizes a loss function.

    The optimization is done with a very simplistic Monte Carlo approach, 
    introducing random changes in the order and accepting those that decrease 
    loss. This is probably as far from a safe and fast solution as one can 
    possibly get but it's simple and does the job.

    (BETA) Some functionalities of this function are untested!

    Parameters
    ----------
    fspace : ndarray of shape (N_samples, N_features)
        Feature space for which features (columns) are to be sorted.
        If is_dists==True, a pairwise distance matrix of shape (N_features,
        N_features) is expected instead.
    lossfun : 'default' or callable, optional, default 'default'
        If 'default', the function similarity_loss (see below) is used as loss
        function. In this case, the optimization sorts features based on their
        similarity. Instead, a callable can be passed for custom computation of
        the loss. The callable must accept two parameters, order and dists,
        where order is a 1D iterable reflecting the current order of the
        features (cmp. return variable `order` below) and dists is a pairwise
        distance matrix of shape (N_features, N_features). The callable must
        return the float loss, where a lower loss indicates a better sort.
    max_iter : int, optional, default 2000
        The optimization stops after evaluating `max_iter` number of random
        changes consecutively without any further reduction in loss.
    norm_dists : bool, optional, default True
        If True, the pairwise distances are normalized between 0 (min) and 1
        (max) before the optimization (recommended).
    is_dists : bool, optional, default False
        If True, the parameter `fspace` is expected to already be a distance
        matrix of shape (N_features, N_features). Allows pre-computing of
        distance measures.
    verbose : bool, optional, default False
        If True, more information is printed.

    Returns
    -------
    order : ndarray of shape (N_features)
        Indices to sort features/columns according to the optimization.
        Used just as like the output of np.argsort.
    """

    # Compute Euclidean pairwise distances
    if not is_dists:
        if verbose: print "Computing pairwise distances..."
        fdist = pdist(fspace.T)
        fdist = squareform(fdist)
    else:
        fdist = fspace

    # Normalize pairwise distances betwen 0 and 1
    if norm_dists:
        fdist = (fdist - fdist.min()) / (fdist.max() - fdist.min())

    # Default loss function based on rank similarity
    def similarity_loss(order, dists):
        loss = 0.0
        for i,io in enumerate(order):
            for j,jo in enumerate(order):
                loss += - dists[io,jo] * np.abs(i-j)
        return loss
    if lossfun == 'default':
        lossfun = similarity_loss

    # Initialize
    order = np.arange(fdist.shape[1]) # Initial order
    loss  = lossfun(order, fdist)     # Initial loss
    step  = 0

    # Iterate until replacements give no improvement within `max_iter` tries
    if verbose: print "Optimizing sort order..."
    while not step==max_iter:

        # Randomly pick two covariates
        pick1,pick2 = np.random.randint(0, order.size, size=2)

        # Swap their position in the order
        new_order = np.copy(order)
        new_order[pick1] = order[pick2]
        new_order[pick2] = order[pick1]

        # Compute the loss with this new order
        new_loss = lossfun(new_order, fdist)

        # If the loss is now smaller...
        if new_loss < loss:

            # Update the overall loss and order
            loss  = new_loss
            order = new_order
            step = 0

        # Otherwise, count toward stopping
        else:
            step += 1

    # return the final ordering
    if verbose: print "Optimization complete!"
    return order


#------------------------------------------------------------------------------

# FUNCTION TO GENERATE A NICE BI-GRAPH OF COVARIATES VS. PCs

def covar_pc_bigraph(covar_fspace_dists, threshold, 
                     covars_names, PC_names=None, height=1.0, 
                     show=True, verbose=False):
    """Generate a bigraph connecting one set of features (covariates) with
    another set of features (shape space PCs) based on correlations.
    
    The first feature set is sorted so as to minimize the cross-edges of the
    resulting bigraph. Correlations under the threshold are not displayed and
    edges are color-coded and size-coded based on directionality and strength
    of the correlation.
    
    Parameters
    ----------
    covar_fspace_dists : numpy array of shape (n_covars, n_PCs)
        Correlation (e.g. Pearson's r values) between the covariates and the 
        shape space PCs.
    threshold : float
        (Absolute) correlation values under this threshold are not displayed
        as edges.
    covar_names : list of strings of length n_covars
        Names for labeling the covariate nodes.
    PC_names : list of strings of length n_PCs, optional, default None
        Names for labeling the PC nodes. If None, no labels are plotted (but
        the nodes are numbered by default).
    height : float, optional, default 1.0
        Distance between the two rows of nodes.
    show : bool, optional, default True
        If True, the resulting bigraph is plotted. If False, the figure object
        is returned instead.
    verbose : bool, optional, default False
        If True, some information on internal processing is printed.
    
    Returns
    -------
    None or fig : matplotlib figure object
        If show is True, the resulting bigraph is displayed and None returned.
        Otherwise, the figure object is returned.
    """

    #--------------------------------------------------------------------------

    ### Sort the covariates to reduce cross-edges in the graph

    if verbose: print "Sorting covariates to reduce cross-edges..."

    # Special loss function for the graph case
    def graph_sort_loss(order, correl_measure):
        loss = 0.0
        for i,io in enumerate(order):
            for j in range(correl_measure.shape[0]):
                loss += correl_measure[j,io] * np.abs( (i/len(order)) - (j/correl_measure.shape[0]) )
        return loss

    # Finding the order with minimal loss
    graph_order = _similarity_argsort(np.abs(covar_fspace_dists).T,
                                      lossfun=graph_sort_loss, is_dists=True,
                                      max_iter=10000, verbose=verbose)

    # Apply the sort
    covar_fspace_dists = covar_fspace_dists[graph_order, :]


    #--------------------------------------------------------------------------

    ### Prepare and create the graph (2-level bipartite graph)

    if verbose: print "Preparing graph..."

    # Get node names
    covars_names = np.array(covars_names)[graph_order]
    fspace_names = np.array( ["PC"+str(n+1) for n
                              in range(covar_fspace_dists.shape[1])] )

    # Find edges
    edge_indices   = np.where(np.abs(covar_fspace_dists) > threshold)
    nodes1, nodes2 = edge_indices
    edges          = [ (n1,n2) for n1,n2 in zip(covars_names[nodes1],
                                                fspace_names[nodes2]) ]

    # Initialize
    B = nx.Graph()

    # Add the nodes
    B.add_nodes_from(covars_names, bipartite=0)
    B.add_nodes_from(fspace_names, bipartite=1)

    # Add the edges
    B.add_edges_from(edges)


    #--------------------------------------------------------------------------

    ### Create the plot

    if verbose: print "Generating figure...\n\n\n"

    # Prepare appropriate node positions
    pos = {}
    pos.update( { node : ((index+1)/len(covars_names), height) for index, node
                  in enumerate(covars_names) }
              )
    pos.update( { node : ((index+1)/len(fspace_names), 0) for index, node
                  in enumerate(fspace_names) }
              )

    # Prepare node colors
    node_colors = plt.cm.get_cmap(plt.cm.viridis)
    node_colors = [node_colors(i) for i in [0.3, 0.7]]

    # Initialize figure
    fig = plt.figure(figsize=(15,8))

    # Plot nodes
    nodes1 = nx.draw_networkx_nodes(B, pos, nodelist=list(covars_names),
                                    node_color="w", node_size=300)
    nodes2 = nx.draw_networkx_nodes(B, pos, nodelist=list(fspace_names),
                                    node_color="w", node_size=500)

    # Node cosmetics
    for nodes,node_color in zip([nodes1,nodes2], node_colors):
        nodes.set_linewidth(3)
        nodes.set_edgecolor(node_color)

    # Plot edges
    nx.draw_networkx_edges(B, pos, edgelist=edges,
                           width=np.abs(covar_fspace_dists)[edge_indices]*3.0,
                           edge_color=covar_fspace_dists[edge_indices],
                           edge_vmin=-1, edge_vmax=1, edge_cmap=plt.cm.RdBu)

    # Plot covariate labels
    covar_labels     = { l:l for l in covars_names }
    covar_labels_pos = { node : ((index+1)/len(covars_names), height+0.075)
                         for index, node in enumerate(covars_names) }
    labels1 = nx.draw_networkx_labels(B, covar_labels_pos, labels=covar_labels,
                                      font_size=11)
    for name in covars_names:
        labels1[name].set_rotation(45)
        labels1[name].set_ha('left')
        labels1[name].set_va('bottom')

    # Plot other labels
    fspace_labels = {l:n+1 for n,l in enumerate(fspace_names)}
    nx.draw_networkx_labels(B, pos, labels=fspace_labels, font_size=12)
    if PC_names is not None:
        PC_labels = {l:PC_name for l,PC_name in zip(fspace_names, PC_names)}
        PC_pos = {l:(p[0], p[1]-0.175) for l,p in pos.iteritems()}
        nx.draw_networkx_labels(B, PC_pos, labels=PC_labels, font_size=12)

    # Axis cosmetics
    plt.xlim([-1/len(covars_names), 1+4/len(covars_names)])
    plt.ylim([-1, 2])
    plt.axis('off')

    # Label the levels
    plt.text(0, 0, "shapeSpace\nPCs", fontsize=13, ha='center', va='center')

    # Done
    if show:
        plt.show()
        return
    else:
        return fig


#------------------------------------------------------------------------------



