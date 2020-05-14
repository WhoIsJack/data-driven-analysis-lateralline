# -*- coding: utf-8 -*-
"""
Created on Thu Dec 07 15:48:33 2017

@author:    Jonas Hartmann @ Gilmour group @ EMBL Heidelberg

@descript:  Functions for the extraction of various cluster features in the
            context of cluster-based embedding.
"""

#------------------------------------------------------------------------------

# IMPORTS

# External
from __future__ import division
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde


#------------------------------------------------------------------------------

# HELPER: BUILD A KDTREE OVER THE LANDMARKS OF A CELL

def build_kdtree(cloud):
    """Build a cKDTree over the landmarks of a cell's point cloud."""
    return cKDTree(cloud)


#------------------------------------------------------------------------------

# FEATURE: MANHATTEN DISTANCES FROM CLUSTER CENTER TO CENTROID OF kNNs

def feature_distsManhatten_kNN(kdtree, cloud, clust_centers):
    """Compute the "kNN-distsManh" feature.

    For each cluster, the k nearest neighbors are selected and their centroid
    is computed. The Manhatten distance of the cluster center and that centroid
    (in all dimensions) is the resulting feature. The distances are flattened
    and concatenated across clusters. k is determined as the number of points
    in the cloud divided by the number of cluster centers.

    Parameters
    ----------
    kdtree : [DASK] passed from `build_kdtree`
    cloud : (array) Point cloud array of shape (points, dimensions)
    clust_centers : (array) CBE cluster center array of shape
                    (n_clusters, dimensions)

    Returns
    -------
    feature vector : (array) 1D feature vector of shape (n_clusters*dimensions)
    """

    # Prep
    dists_knn = []

    # For each cluster center...
    for cc_idx,cc_vals in enumerate(clust_centers):

        # Get k nearest neighbors from cluster center
        knn_dist, knn_idx = kdtree.query(cc_vals, 
                                 k=int(cloud.shape[0]//len(clust_centers)),
                                 eps=0)

        # Get the centroid of the nearest neighbors
        knn_mean = np.mean(cloud[knn_idx, :], axis=0)

        # Get distances (ZYX) from kNN centroid to cluster center
        dists_knn.append(knn_mean - cc_vals)

    # Done
    return np.concatenate(dists_knn)


#------------------------------------------------------------------------------

# FEATURE: EUCLIDEAN DISTANCE FROM CLUSTER CENTER TO CENTROID OF kNNs

def feature_distEuclidean_kNN(dists_knn, dims):
    """Compute the "kNN-distEuclid" feature.

    For each cluster, the k nearest neighbors are selected and their centroid
    is computed. The Euclidean distance of the cluster center and that centroid
    is the resulting feature. The distances are concatenated across clusters.
    k is determined as the number of points in the cloud divided by the number 
    of cluster centers.

    Parameters
    ----------
    dists_knn : [DASK] passed from `feature_distsManhatten_kNN`
    dims : (int) dimensionality of the point cloud space (integer)

    Returns
    -------
    feature vector : (array) 1D feature vector of shape (n_clusters)
    """

    # Structure the dists_knn per cluster
    d = dists_knn.reshape((dists_knn.shape[0]//dims, dims))

    # Compute the Euclidean distance
    dist_knn = np.sqrt( np.sum(d**2, axis=1) )

    # Done
    return dist_knn


#------------------------------------------------------------------------------

# FEATURE: MANHATTEN DISTANCES FROM CLUSTER CENTER TO NEAREST NEIGHBOR

def feature_distsManhatten_NN(kdtree, cloud, clust_centers):
    """Compute the "NN-distsManh" feature.

    For each cluster, the Manhatten distance of the cluster center to its
    nearest neighbor is computed (in all dimensions). The distances are
    flattened and concatenated across clusters.

    Parameters
    ----------
    kdtree : [DASK] passed from `build_kdtree`
    cloud : (array) Point cloud array of shape (points, dimensions)
    clust_centers : (array) CBE cluster center array of shape
                    (n_clusters, dimensions)

    Returns
    -------
    feature vector : (array) 1D feature vector of shape (n_clusters*dimensions)
    """

    # Prep
    dists_nn = []

    # For each cluster center...
    for cc_idx,cc_vals in enumerate(clust_centers):

        # Get nearest neighbor from cluster center
        nn_dist,knn_idx = kdtree.query(cc_vals, k=1, eps=0)

        # Get the nearest neighbor point
        nn_pos = cloud[knn_idx, :]

        # Get distances (ZYX) from cluster center to NN
        dists_nn.append(nn_pos - cc_vals)

    # Done
    return np.concatenate(dists_nn)


#------------------------------------------------------------------------------

# FEATURE: EUCLIDEAN DISTANCE FROM CLUSTER CENTER TO NEAREST NEIGHBOR

def feature_distEuclidean_NN(dists_nn, dims):
    """Compute the "NN-distEuclid" feature.

    For each cluster, the Euclidean distance to its nearest neighbor is
    computed. The distances are concatenated across clusters.

    Parameters
    ----------
    dists_nn : [DASK] passed from `feature_distsManhatten_NN`
    dims : (int) dimensionality of the point cloud space (integer)

    Returns
    -------
    feature vector : (array) 1D feature vector of shape (n_clusters)
    """

    # Structure the dists_nn per cluster
    d = dists_nn.reshape((dists_nn.shape[0]//dims, dims))

    # Compute the Euclidean distance
    dist_nn = np.sqrt( np.sum(d**2, axis=1) )

    # Done
    return dist_nn


#------------------------------------------------------------------------------

# FEATURE: COUNT OF LANDMARKS IN THE VICINITY OF EACH CLUSTER

def feature_count_near(INPUT, cloud, clust_centers):
    """Compute the "count-near" feature.

    For each cluster, the number of landmarks within a neighborhood radius
    arround them is counted. The neighborhood radius is defined as the mean of
    all distances computed in `feature_distEuclidean_kNN`.

    Parameters
    ----------
    INPUT : [DASK] passed from `build_kdtree` and `feature_distEuclidean_kNN`
    cloud : (array) Point cloud array of shape (points, dimensions).
    clust_centers : (array) CBE cluster center array of shape
                    (n_clusters, dimensions)

    Returns
    -------
    feature vector : (array) 1D feature vector of shape (n_clusters)
    """

    # Unpack
    kdtree, dist_knn = INPUT

    # The vicinity is defined as the mean across kNN dists
    r = np.mean(dist_knn)

    # Prep
    counts_near = []

    # For each cluster center...
    for cc_idx,cc_vals in enumerate(clust_centers):

        # Find the number of landmarks in the vicinity
        counts_near.append(len(kdtree.query_ball_point(cc_vals, r)))

    # Done
    return np.array(counts_near)


#------------------------------------------------------------------------------

# FEATURE: COUNT OF LANDMARKS ASSIGNED TO EACH CLUSTER

def feature_count_assigned(clust_centers, c_labels):
    """Compute the "count-assigned" feature.

    For each cluster, the number of landmarks assigned to it in the original
    clustering is counted.

    Parameters
    ----------
    clust_centers : (array) CBE cluster center array of shape
                    (n_clusters, dimensions)
    c_labels : (array) Cluster labels for each point of the current cell.

    Returns
    -------
    feature vector : (array) 1D feature vector of shape (n_clusters)
    """

    # Prep
    counts_assigned = []

    # For each cluster center...
    for cc_idx,cc_vals in enumerate(clust_centers):

        # Find number of assigned landmarks
        counts_assigned.append(np.sum(c_labels==cc_idx))

    # Done
    return np.array(counts_assigned)


#------------------------------------------------------------------------------

# FEATURE: KDE ESTIMATED ON LANDMARKS AND SAMPLED AT EACH CLUSTER CENTER

def feature_kde(cloud, clust_centers, bw_method):
    """Compute the "kde" feature.

    A KDE is estimated on the cell's cloud and then sampled at every cluster
    center, meaning this feature is roughly equivalent to the local landmark
    density at the cluster centers.

    Parameters
    ----------
    cloud : (array) Point cloud array of shape (points, dimensions).
    clust_centers : (array) CBE cluster center array of shape
                    (n_clusters, dimensions)

    Returns
    -------
    feature vector : (array) 1D feature vector of shape (n_clusters)
    """

    # Infer KDE based on point cloud
    kernel = gaussian_kde(cloud.T, bw_method=bw_method)

    # Sample KDE at cluster centers
    feature_vector = kernel(clust_centers.T)

    # Done
    return feature_vector


#------------------------------------------------------------------------------

# HELPER: ASSEMBLE FEATURES OF A CELL INTO ONE VECTOR

def assemble_cell(features, feature_names):
    """Assemble different feature vectors for a cell into one feature vector.

    Assembles the different feature vectors extracted by different functions
    above into one feature vector. Also creates a header that indicates what
    feature is where in this feature vector.

    Parameters
    ----------
    features : [DASK] inherits from all requested feature extraction functions.
    feature_names : (list of strings) The name of each feature in the same
                    order as the results in features.

    Returns
    -------
    features : (array) 1D feature vector; concatenation of all individual
               feature vectors.
    header   : (list of strs) List of names for each feature in features. Has
               the same length as features.
    """

    # Assemble header
    header = []
    for feature,feature_name in zip(features, feature_names):
        header += [feature_name+"_%i" % i for i in range(feature.shape[0])]

    # Assemble features
    features = np.concatenate(features)

    # Return
    return (features, header)


#------------------------------------------------------------------------------

# HELPER: ASSEMBLE FEATURE VECTORS OF ALL CELLS INTO A FEATURE SPACE

def assemble_fspace(cell_assembly):
    """Assemble feature space from the individual feature vectors of all cells.

    Assembles a feature space of shape (n_cells, n_features) from the
    individual feature vectors of each cell.

    Parameters
    ----------
    cell_assembly : [DASK] inherits from `assemble_cell` of all cells. Contains
                    the input feature vector and the header.

    Returns
    -------
    fspace : (array) Complete feature space of shape (n_cells, n_features).
    header : (list of strs) List of names for each feature in fspace. Has the
             same length as the second axis of fspace (n_features).
    """

    features = [c[0] for c in cell_assembly]
    header   = cell_assembly[0][1]

    return np.array(features), header


#------------------------------------------------------------------------------



