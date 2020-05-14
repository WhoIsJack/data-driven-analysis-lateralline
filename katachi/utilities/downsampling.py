# -*- coding: utf-8 -*-
"""
Created on Wed Dec 06 11:23:47 2017

@author:    Jonas Hartmann @ Gilmour group @ EMBL Heidelberg

@descript:  Functions for downsampling of point clouds.
"""

#------------------------------------------------------------------------------

# IMPORTS

# External
from __future__ import division
from warnings import warn
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial import cKDTree

# Internal
from katachi.utilities.plotting import point_cloud_3D


#------------------------------------------------------------------------------

# FUNCTION: SUBSAMPLE RANDOMLY

def random_subsample(cloud, sample_size, replace=False):
    """Random downsampling to a specified sample size.

    Parameters
    ----------
    cloud : array of shape (n_points, n_dims)
        Point cloud to be subsampled.
    sample_size : int
        Number of points to sample from the cloud.
    replace: bool, optional, default False
        If True, the same point can be sampled multiple times.

    Returns
    -------
    cloud_subs : array of shape (sample_size, n_dims)
        Cloud of subsampled points.

    Warnings
    --------
    UserWarning : (code 1)
        If n_points is <= sample_size, no subsampling is performed and the
        unaltered point cloud is returned.
    """

    # Handle small point clouds
    if cloud.shape[0] <= sample_size:
        warn("(code 1) Point cloud is already <= desired sample size. " +
             "No subsampling is performed.")
        return cloud

    # Perform subsamping
    sample_indices = np.random.choice(np.arange(cloud.shape[0]),
                                      sample_size, replace=False)
    cloud_subs = cloud[sample_indices]

    # Return result
    return cloud_subs


#------------------------------------------------------------------------------

# FUNCTION: KMEANS REDUCTION OF POINT CLOUDS

def kmeans_subsample(cloud, sample_size, show=False):
    """Downsampling based on kmeans clustering.

    This essentially produces a smoothed downsampled version of the original
    point cloud distribution.

    Parameters
    ----------
    cloud : array of shape (n_points, n_dims)
        Point cloud to be subsampled.
    sample_size : int
        Number of points to sample from the cloud.
    show: bool, optional, default False
        If True, a 3D plot of the cluster centers and their clusters is shown.

    Returns
    -------
    kcenters : array of shape (sample_size, n_dims)
        Cloud of subsampled points (=the centers of the kmeans clusters).

    Warnings
    --------
    UserWarning : (code 1)
        If n_points is <= sample_size, no subsampling is performed and the
        unaltered point cloud is returned.
    """

    # Handle small point clouds
    if cloud.shape[0] <= sample_size:
        warn("(code 1) Point cloud is already <= desired sample size. " +
             "No subsampling is performed.")
        return cloud

    # Perform kmeans clustering
    kmeans_cluster = MiniBatchKMeans(n_clusters=sample_size, random_state=42,
                                     batch_size=500, init_size=3*sample_size)
    kmeans_cluster.fit(cloud)

    # Get labels and centroids
    klabels  = kmeans_cluster.labels_
    kcenters = kmeans_cluster.cluster_centers_

    # Show color-coded clusters
    if show:
        fig,ax = point_cloud_3D(cloud[:,2], cloud[:,1], cloud[:,0],
                                marker='.', c=klabels, cmap='viridis',
                                s=20, alpha=0.5, title='KMeans Reduction',
                                xlbl='x', ylbl='y', zlbl='z', fin=False)
        point_cloud_3D(kcenters[:,2], kcenters[:,1], kcenters[:,0],
                       marker='d', c=np.unique(klabels), cmap='viridis',
                       s=20, alpha=1.0, config=False, init=False,
                       pre_fig=fig, pre_ax=ax)

    # Return result
    return kcenters


#------------------------------------------------------------------------------

# FUNCTION: DENSITY-DEPENDENT DOWN-SAMPLING (DDDS)

def ddds(cloud, sample_size, presample=None, processes=10):
    """Density-dependent down-sampling to a sample size smaller or equal to a
    specified value.

    This approach down-samples a point cloud such that regions of higher
    density are less likely to be sampled from. The resulting cloud therefore
    has a more uniform density.

    The approach used here is directly inspired by SPADE [ref 1].

    Note that the number of points sampled by this approach scales with the
    size of the input cloud. A random subsampling step is included at the end
    of the process to bring the result down to the specified sample size.
    However, if the specified sample size is not small compared to the input
    cloud size, the down-sampling may result in fewer points, so the final size
    of the output cloud is not guaranteed to be the specified sample size.

    Parameters
    ----------
    cloud : array of shape (n_points, n_dims)
        Point cloud to be downsampled.
    sample_size : int
        Number of points to sample from the cloud. The resulting cloud will
        have at most this many points (but could have fewer).
    presample : int or None, optional, default None
        If an integer is passed, the input cloud is first randomly subsampled
        to size `presample` to make the determination of local densities less
        computationally costly. It is highly recommended to make use of this
        option for anything other than small clouds. A good starting value is
        often the same as sample_size.
    processes: int, optional, default 10
        Number of processes to use in multiprocessed neighbor computations.


    Returns
    -------
    cloud_ddds : array of shape (<=sample_size, n_dims)
        Cloud of subsampled points.

    Warnings
    --------
    UserWarning : (code 1)
        If n_points is <= sample_size, no subsampling is performed and the
        unaltered point cloud is returned.

    References
    ----------
    Ref 1 : Qiu et al., 2011. Nature Biotechnology
        "Extracting a cellular hierarchy from high-dimensional cytometry data
        with SPADE"

    """

    #--------------------------------------------------------------------------

    ### Prep

    # Handle small point clouds
    if cloud.shape[0] <= sample_size:
        warn("(code 1) Point cloud is already <= desired sample size. " +
             "No subsampling is performed.")
        return cloud


    #--------------------------------------------------------------------------

    ### Compute per-landmark local densities

    # Subsample randomly (for speed/memory efficiency)
    if presample is not None:
        cloud_presubs = random_subsample(cloud, presample)
    else:
        cloud_presubs = np.copy(cloud)

    # Compute distance of each subsampled point to the closest other point
    # Note: `k=2` is necessary since `k=1` is the point itself.
    tree = cKDTree(cloud)
    NN_dists = tree.query(cloud_presubs, k=2, n_jobs=processes)[0][:,1]

    # Get the size of the local neighborhood
    #    which is `alpha * median(smallest_distances)`,
    #    where a good value for alpha is 5 according to SPADE
    alpha = 5
    NN_size = alpha * np.median(NN_dists)

    # Get the local density (LD) of each landmark
    # ...which is the number of other landmarks in its local neighborhood
    LDs = tree.query_ball_point(cloud, NN_size, n_jobs=processes) # Get indices
    LDs = np.vectorize(len)(LDs)                                  # Count

    # Define the target density (TD)
    # Note: Good values according to SPADE: the 3rd or 5th percentile of LDs
    # Note: This effectively defines how strongly the data will be subsampled
    TD_percentile = 3
    TD = np.percentile(LDs, TD_percentile)


    #--------------------------------------------------------------------------

    ### Perform density-dependent subsampling

    # Create p(keep_lm) probability vector
    # Note: For each point i, the probability of keeping it is
    #        {         1   if LD_i < TD
    #        { TD / LD_i   otherwise
    p_keep = TD / LDs
    p_keep[LDs<TD] = 1

    # Randomly decide if a landmark should be kept according to p(keep_lm)
    rand = np.random.uniform(size=cloud.shape[0])
    keep = p_keep >= rand

    # Index the lms to be kept
    cloud_ddds = cloud[keep,:]


    #--------------------------------------------------------------------------

    ### Further random downsampling

    # Note: This ensures that the downsampled cloud does not grow with the
    #       input data and instead is of the specified sample_size or smaller.

    if cloud_ddds.shape[0] > sample_size:
        cloud_ddds = random_subsample(cloud_ddds, sample_size)

    #--------------------------------------------------------------------------

    ### Return result
    return cloud_ddds


#------------------------------------------------------------------------------




