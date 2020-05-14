# -*- coding: utf-8 -*-
"""
Created on Mon Dec 04 14:29:42 2017

@author:    Jonas Hartmann @ Gilmour group @ EMBL Heidelberg

@descript:  Create a feature space from a data set of arbitrary point clouds by
            Cluster-Based Embedding (CBE). Features for each point cloud are
            extracted in reference to a common set of cluster centers that are
            determined based on a downsampled overlay of the point clouds.
"""

#------------------------------------------------------------------------------

# IMPORTS

# External
from __future__ import division
import os, pickle
from warnings import warn
import numpy as np
np.random.seed(42)
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
#from sklearn import kernel_approximation
from scipy.spatial import cKDTree
from multiprocessing.pool import ThreadPool, cpu_count
import dask
from dask.diagnostics import ProgressBar
from dask.diagnostics import Profiler, ResourceProfiler, visualize

# Internal
import katachi.utilities.downsampling as ds
import katachi.utilities.features as fe


#------------------------------------------------------------------------------

# FUNCTION: NORMALIZE POINT CLOUD VOLUME

def vol_normalize(clouds, verbose=False):
    """Normalize the volume of a set of point clouds.

    Normalizes the volume of a point cloud by dividing the magnitude of each
    point's vector by the sum of all magnitudes. This is done vectorized for a
    series of point clouds of the same shape.

    Parameters
    ----------
    clouds : numpy float array of shape (cloud, points, dimensions)
        Each cloud will be volume-normalized independently. A single cloud can
        also be passed as an array of shape (points, dimensions).
    verbose : bool, optional, default False
        If True, more information is printed.

    Returns
    -------
    clouds_norm : numpy float array of same shape as input
        The normalized point cloud(s).
    """

    # Add axis in case this is a single cloud
    single_cloud=False
    if clouds.ndim == 2:
        clouds = clouds[np.newaxis,:,:]
        single_cloud=True

    # Get distances from centroid
    dists = np.sqrt(  clouds[:,:,0]**2.0
                    + clouds[:,:,1]**2.0
                    + clouds[:,:,2]**2.0)

    # Get sum distance
    sums_dists = np.sum(dists, axis=1)

    # Divide each vector component by the total length
    clouds_norm = clouds / sums_dists[:,np.newaxis,np.newaxis]

    # Check that the sum is now 1
    if verbose:
        print "  Normalized vector test (must be 1.0):",
        print np.sum(np.sqrt(  clouds_norm[:,:,0]**2.0
                             + clouds_norm[:,:,1]**2.0
                             + clouds_norm[:,:,2]**2.0)) / clouds.shape[0]

    # Return result
    if single_cloud:
        return clouds_norm[0,:,:]
    else:
        return clouds_norm


#------------------------------------------------------------------------------

# FUNCTION: TRANSFORM POINT CLOUD TO PAIRWISE DISTANCE (PD) SPACE

def pd_transform(cloud, percentiles=None):
    """Transform a point cloud to its pairwise distance (PD) space.

    Expresses each point in a point cloud through its pairwise distance to all
    other points. If desired, these distances are then subsampled (by selecting
    percentiles, see below) to reduce the dimensionality of the resulting
    space.

    Note that the resulting space is invariant to rigit spatial transformations
    in the original space.

    Note that reconstruction of the original space is non-trivial. There is no
    guarantee that the positions of all points can be reconstructed if the PD
    space is reduced by percentile selection. However, a large number of points
    should usually be placeable if >=4 dimensions are kept.

    Parameters
    ----------
    cloud : numpy float array of shape (points, dimensions)
        Cloud to be transformed.
    percentiles : int or None, optional, default None
        If an integer is passed, the PD space is reduced to that number of
        dimensions. This is done by selecting a corresponding number of
        distance percentiles in a linear range between the 10th to the 90th
        percentile (inclusive).

    Returns
    -------
    distances : numpy float array of shape (points, distances)
        PD-transformed cloud. If percentiles is None, the shape of the output
        will be (n_points, n_points*n_points), otherwise the shape will be
        (n_points, percentiles).
    """

    # Construct KDTree
    kdTree = cKDTree(cloud)

    # Query tree to get all pairwise neighbors
    # Note: Distances are automatically ordered along axis 0.
    distances, indices = kdTree.query(cloud, k=cloud.shape[0])

    # Reduce dimensionality by only selecting some distances (percentile-based)
    if percentiles is not None:
        distances = np.percentile(distances,
                                  np.linspace(10, 90, percentiles),
                                  axis=1).T

    # Return result
    return distances


#------------------------------------------------------------------------------

# FUNCTION: EXTRACT FEATURES BY CLUSTER-BASED EMBEDDING

def cbe(fpaths_lm, downsample, clustering, features,
        normalize_vol=False, presample=None,
        cfor=None, standardize='default',
        custom_feature_funcs=None, bw_method=None,
        dask_graph_path=None, processes=None, profiling=False,
        suffix_out='default', save_metadata=True,
        save_presampled=False, save_cfor=False,
        verbose=False, legacy=False):
    """Create a feature space from a set of point clouds by cluster-based
    embedding (CBE).

    This includes the following steps:
        1. Loading a set of point clouds
        2. Normalizing point clouds by volume (optional)
        3. Down-sampling of each point cloud individually (optional)
            - Available options are random, kmeans or custom downsampling
        4. Making point clouds invariant to spatial transformation (optional)
            - Also called the "Cell Frame Of Reference" (CFOR)
            - There are currently 2 ways of accomplishing this
                - Transform to pairwise distance space (PD)
                - Transform to PCA space (PCA) [DEPRECATED]
            - It is also possible to pass a custom transform function.
        5. Merging point clouds
        6. Downsampling of merged point clouds (optional but recommended!)
            - Reduces computational cost/scaling of subsequent step
            - Options are density-dep., kmeans, random or custom downsampling
        7. Extracting cluster centers as common reference points
            - Options are kmeans, dbscan and custom clustering
        8. Extracting "cluster features" relative to the reference points
            - Done with dask for effecient chaining of operations
            - Multiple feature options available, see below
        9. Saving the resulting feature space as well as intermediate results

    Cluster features that can be extracted:
        - "kNN-distsManh"  : Manhatten distance in all dimensions of each
                             cluster to the mean point of its k nearest
                             neighbor landmarks.
        - "kNN-distEuclid" : Euclidean distance of each cluster to the mean
                             point of its k nearest neighbor landmarks.
        - "NN-distsManh"   : Manhatten distance in all dimensions of each
                             cluster to the nearest neighboring landmark.
        - "NN-distEuclid"  : Euclidean distance of each cluster to the nearest
                             neighboring landmark.
        - "count-near"     : Number of landmarks near to the cluster, where
                             'near' is the mean distance of the k nearest
                             neighbor landmarks of the cluster.
        - "count-assigned" : Number of landmarks assigned to the cluster during
                             the clustering itself.
        - "kde"            : KDE estimated from cell landmarks sampled for each
                             cluster center.
        - custom features  : See custom_feature_funcs in parameters.

    Feature computations are in part dependent on each other. To make this both
    efficient and readable/elegant, dask is used for chaining the feature
    extraction steps appropriately.

    At the end, features are concatenated into a single array of shape
    (cells, features) and then saved for each input stack separately.

    Parameters
    ----------
    fpaths_lm : single string or list of strings
        A path or list of paths (either local from cwd or global) to npy files
        containing cellular landmarks as generated by
        `katachi.tools.assign_landmarks` or `...find_TFOR`.
    downsample : tuple (algorithm, output_size) or None
        A tuple specifying the algorithm to use for downsampling of the merged
        point cloud prior to cluster extraction. Available algorithms are
        "ddds" (density-dependent downsampling), "kmeans" (perform kmeans and
        use cluster centers as new points) or "random". If "default" is passed,
        "ddds" is used.
        Example: ("ddds", 200000).
        Alternatively, if instead of a string denoting the algorithm a callable
        is passed, that callable is used for downsampling.
        The call signature is
        `all_lms_ds = downsample[0](all_lms, downsample)`
        where all_lms is an array of shape (all_landmarks, dimensions) holding
        all input landmarks merged into one point cloud. Since the `downsample`
        tuple itself is passed, additional arguments can be specified in
        additional elements of that tuple. all_lms_ds must be an array of shape
        (output_size, dimensions).
        If None, no downsampling is performed. This is not recommended for
        inputs of relevant sizes (total landmarks > 20000).
        WARNING: downsampling (especially by ddds) can be very expensive for
        large numbers of cells. In those cases, it is recommended to first run
        a representative subsets of the cells and then use the resulting CBE
        clusters to extract features for the entire dataset (using the
        `previous` setting in the `clustering` argument).
    clustering : tuple (algorithm, n_clusters)
        A tuple specifying the algorithm to use for computing the clusters to
        use in cluster-based feature extraction. Available algorithms are
        "kmeans" or "dbscan". If "default" is passed, "kmeans" is used.
        Example: ('kmeans', 10)
        Alternatively, one may pass a tuple `('previous', clustering_object)`,
        where `clustering_object` is a previously fitted clustering instance
        similar to an instantiated and fitted sklearn.cluster.KMeans object. It
        must have the attribute `cluster_centers_`, which is an array of shape
        (clusters, dimensions) and the method `predict`, which given an array
        of shape `(all_landmarks, dimensions)` will return cluster labels for
        each landmark. Clustering objects from previous runs are stored in
        the metadata under the key `"clustobj-"+identifier`.
        Alternatively, if instead of a string denoting the algorithm a callable
        is passed, that callable is used for clustering.
        The call signature is
        `clust_labels, clust_centers = clustering[0](all_lms, clustering)`
        where all_lms is an array of shape (all_landmarks, dimensions) holding
        all input landmarks merged into one point cloud (and downsampled in the
        previous step). Since the `clustering` tuple itself is passed,
        additional arguments can be specified in additional elements of that
        tuple. `clust_labels` must be a 1D integer array assigning each input
        landmark to a corresponding cluster center. `clust_centers` must be an
        array of shape (clusters, dimensions) and contain the coordinates of
        the cluster centers. The first axis must be ordered such that the
        integers in `clust_labels` index it correctly. The number of clusters
        must match n_clusters.
    features : list of strings
        List containing any number of cluster features to be extracted. The
        strings noted in the explanation above are allowed. If custom feature
        extraction functions are passed (see below), their names must also be
        included in this list.
        Example: ["kNN-distEuclid", "count-near"]
    normalize_vol : bool, optional, default False
        If True, the volume of each input point cloud is normalized by dividing
        each landmark vector magnitude by the sum of all magnitudes.
    presample : tuple (algorithm, output_size) or None, optional, default None
        If not None, the algorithm specified is used to downsample each input
        cloud individually to output_size points. Available algorithms are
        "kmeans" (perform kmeans and use cluster centers as new points) or
        "random".
        Example: ('random', 50)
        Alternatively, if instead of a string denoting the algorithm a callable
        is passed, that callable is used for downsampling.
        The call signature is
        ```for cell in range(lms.shape[0]):
               lms_ds[cell,:,:] = presample[0](lms[cell,:,:], presample)```
        where lms is an array of shape (cells, landmarks, dimensions) holding
        the set of input point clouds. Since the `presample` tuple itself is
        passed, additional arguments can be specified in additional elements of
        that tuple. lms_ds must be an array of shape
        (cells, output_size, dimensions).
        If None, no presampling is performed.
    cfor : tuple (algorithm, dimensions) or None, optional, default None
        A tuple specifying the algorithm to use for recasting the landmarks in
        a space that is invariant to spatial transformations. There are two
        options available: "PD" (pairwise distance transform) and "PCA"
        (per-cell PCA and transform).
        For "PD", the total complement of pairwise distances between all points
        is computed and then subsampled to `dimensions` by selecting a
        corresponding number of distance percentiles in a linear range between
        the 10th to the 90th percentile (inclusive).
        For "PCA", the number of dimensions in the resulting space is equal to
        the number of dimensions of the input (should be 3). The `dimensions`
        part of the argument is ignored (but it must still be suplied!).
        If "default" is passed, "PD" is used.
        Example 1: ('PD', 6)
        Example 2: ('default', 6)  # defaults to 'PD'
        Example 3: ('PCA', 3)
        Alternatively, if a callable is passed instead of a stringm that
        callable is used for downsampling.
        The call signature is
        ```for cell in range(lms.shape[0]):
               lms_cfor[cell,:,:] = cfor[0](lms[cell,:,:], cfor)```
        where lms is an array of shape (cells, landmarks, dimensions) holding
        the set of input point clouds. Since the `cfor` tuple itself is passed,
        additional arguments can be specified in additional elements of that
        tuple. lms_ds must be an array of shape
        (cells, output_size, dimensions).
        If None, no transformation is performed; cells are left in the original
        3D space.
    standardize : bool or 'default', optional, default 'default'
        If True, the point cloud dimensions of the merged CFOR point cloud are
        standardised to zero mean and unit variance. This is also propagated
        to the individual clouds used for feature extraction and for saving
        in case the CFOR is being saved.
        If 'default', standardization is performed only if cfor is set to "PD".
        If False, no standardization is performed.
    custom_feature_funcs : list of tuples or None, optional, default None
        List used to specify one or more custom feature extraction functions.
        Each custom function is specified through a tuple in the list that
        is structured as such:
            `(feature_name, extraction_func, parent_names, other_params)`
        where `feature_name` is the name of the feature as it appears in the
        `features` argument, `extraction_func` is a callable, `parent_names`
        is a lsit of parent feature names (as they appear in the `features`
        argument) used as input to `extraction_func`, and `other_params` is a
        list of other parameters for `extraction_func`.
        The call signature is
        ```dask_graph[custom_func[0]+"_%i" % c] =
               (feature_name, [parent+"_%i" % c for parent in parent_names],
                other_params, lms[c,:,:], clust_centers, clust_labels[c]) ```
        within the dask graph, where `c` is the index of a cell.
        The callable must therefore accept a list of parent features (can be
        an empty list), a list of other parameters (can alos be empty), the
        (preprocessed) landmarks of the given cell, the cluster centers and
        the cluster labels of the given cell.
        It must return a 1D array of float values; the feature vector for the
        current cell `c`.
    bw_method : str, scalar, callable or None, optional, default None
        The method used to calculate the estimator bandwidth for the gaussian
        kde when computing the "kde" feature. This can be ‘scott’, ‘silverman’,
        a scalar constant or a callable. If a scalar, this will be used
        directly as `kde.factor`. If a callable, it should take a gaussian_kde
        instance as only parameter and return a scalar. If None (default),
        ‘scott’ is used. This is ignored if "kde" is not in `features`.
        < Modified from `scipy.stats.gaussian_kde` doc string. >
    dask_graph_path : string or None, optional, default None
        If a path (including a file ending matching a known image format, such
        as '.png') is specified as a string, a dask graph image is created that
        summarizes the feature extraction pipeline for the first 3 cells.
        Note: If the resulting graph contains multiple separate graphs, the
        only relevant graph is the one leading into `fspace` as an end result.
    processes : int or None, optional, default None
        Number of processes to use in multiprocessed and dask-controlled
        operations. If None, a number equal to half the available PCUs is used.
        If `1` (one), no multiprocessing is performed and `dask.get` is used
        instead of `dask.threaded.get`.
    profiling : bool, optional, default False
        If True, dask resource profiling is performed and visualized after the
        pipeline run is finished. This may generate a `profile.html` file in
        the working directory [bug in dask].
    suffix_out : 'default' or dict, optional, default 'default'
        If 'default', the ouput is saved using '_PRES', '_CFOR', '_DS', and
        '_CBE' as suffices for the presampled landmarks (if `presample` is not
        None), for the CFOR-transformed landmarks (if `cfor` is not None), for
        overlayed downsampling (if `downsample` is not None)(note that this is
        not saved explicitly but is part of the suffix for the CBE-embedded
        feature space), and for the CBE-embedded feature space, respectively.
        The suffices are chained as appropriate. If a dict is passed, each of
        these suffices can be specified manually using the keys 'PRES', 'CFOR',
        'DS', 'CBE' and 'META'.
        The suffix specified in 'META' is added to all relevant metadata
        dictionary keys. For any suffices not specified in the suffix_out dict,
        the 'default' suffix is used.
    save_metadata : bool, optional, default True
        If True, cluster samples, cluster labels and a feature header are saved
        to the metadata of each input stack as appropriate.
    save_presampled : bool, optional, default False
        If True, the result of the presampling step is saved with the suffix
        "PRES" for later use.
    save_cfor : bool, optional, default False
        If True, the result of the cfor step is saved with the suffix "CFOR"
        for later use.
    verbose : bool, optional, default False
        If True, more information is printed.
    legacy : bool, optional, default False
        If True (and standardize is also set to True), the feature extraction
        is not performed in standardized space. Instead, the cluster centroids
        are transformed back to the un-standardized space.
        Triggers a deprecation warning.
    """

    #--------------------------------------------------------------------------

    ### Load data

    if verbose: print "Loading data..."

    # Handle cases of single paths
    if type(fpaths_lm)==str:
        fpaths_lm = [fpaths_lm]
    if len(fpaths_lm)==1:
        warn("fpaths_lm specifies only a single path. Usually, multiple paths"+
             " are specified so that many samples can be overlayed for"+
             " feature extraction!")

    # Import the landmark data
    # Note: The order of fpaths_lm is maintained and an index array is created!
    lms     = []
    lms_idx = []
    for idx, fpath_lm in enumerate(fpaths_lm):
        try:
            lms_in = np.load(fpath_lm)
            lms.append(lms_in)
            lms_idx += [idx for i in range(lms_in.shape[0])]
        except:
            print "Attempting to load landmark data from "+str(fpath_lm),
            print "failed with this error:"
            raise
    lms_idx = np.array(lms_idx, dtype=np.int)
    lms     = np.concatenate(lms)
    if verbose: print "Total input data shape:", lms.shape

    # Check if downsampling is specified
    if downsample is None:
        warn("It is highly recommended to use downsampling (unless the data "+
             "set is very small)!")

    # Handle processes being None
    if processes is None:
        processes = cpu_count() // 2

    # Handle standardize being default
    if standardize=='default':
        standardize=False
        if cfor[0]=='PD':
            standardize=True

    # Handle legacy mode
    if legacy:
        warn("Running in LEGACY mode! This is DEPRECATED!", DeprecationWarning)


    #--------------------------------------------------------------------------

    ### Normalize volume [per cell]

    if normalize_vol:
        if verbose: print "Normalizing volumes..."
        lms = vol_normalize(lms, verbose=verbose)


    #--------------------------------------------------------------------------

    ### Individual downsampling (presampling) [per cell]

    if presample is not None:
        if verbose: print "Presampling..."

        # Prep
        lms_ps = np.zeros((lms.shape[0], presample[1], lms.shape[2]))

        # Random subsampling
        if presample[0] == 'random':
            for cell in range(lms.shape[0]):
                lms_ps[cell,:,:] = ds.random_subsample(lms[cell,:,:],
                                                       presample[1])

        # Kmeans-based downsampling
        elif presample[0] == 'kmeans':
            for cell in range(lms.shape[0]):
                lms_ps[cell,:,:] = ds.kmeans_subsample(lms[cell,:,:],
                                                       presample[1])

        # Custom downsampling function
        elif callable(presample[0]):
            for cell in range(lms.shape[0]):
                lms_ps[cell,:,:] = presample[0](lms[cell,:,:],
                                                presample)

        # Handle other cases
        else:
            raise ValueError("Invalid presampling method: "+str(presample[0]))

        # Assign the downsampled data back
        lms = lms_ps


    #--------------------------------------------------------------------------

    ### Transform to "Cell Frame Of Reference" (CFOR) [per cell]

    if cfor is not None:
        if verbose: print "Transforming to CFOR..."

        # Prep
        lms_cfor = np.zeros((lms.shape[0], lms.shape[1], cfor[1]))

        # Pairwise distance transform
        if cfor[0] == 'PD' or cfor[0] == 'default':
            for cell in range(lms.shape[0]):
                lms_cfor[cell,:,:] = pd_transform(lms[cell,:,:],
                                                  percentiles=cfor[1])

        # PCA transform
        elif cfor[0] == 'PCA':
            for cell in range(lms.shape[0]):
                lms_cfor[cell,:,:] = PCA().fit_transform(lms[cell,:,:])

        ## RBF transform by Nystroem embedding
        ## REMOVED: This does not create matched dimensions and thus cannot be
        ##          used for this purpose.
        #if cfor[0] == 'RBF':
        #    for cell in range(lms.shape[0]):
        #        Ny = kernel_approximation.Nystroem(kernel='rbf',
        #                                           gamma=1/lms.shape[1],
        #                                           n_components=cfor[1],
        #                                           random_state=42)
        #        lms_cfor[cell,:,:] = Ny.fit_transform(lms[cell,:,:])

        # Custom CFOR transform
        elif callable(cfor[0]):
            for cell in range(lms.shape[0]):
                 lms_cfor[cell,:,:] = cfor[0](lms[cell,:,:], cfor)

        # Handle other cases
        else:
            raise ValueError("Invalid CFOR method: "+str(cfor[0]))

        # Assign the CFOR data back
        lms = lms_cfor


    #--------------------------------------------------------------------------

    ### Collective downsampling (all cells overlayed) [altogether]
    #   Note: This is done to improve cluster retrieval and to make it more
    #         efficient. It does not affect the feature extraction afterwards.

    # Flatten cells of all samples together
    all_lms = lms.reshape((lms.shape[0]*lms.shape[1], lms.shape[2]))

    # For CFOR-PD: standardize the dimensions
    if standardize and not legacy:

        # Standardize pooled landmarks
        cloud_means = all_lms.mean(axis=0)
        cloud_stds  = all_lms.std(axis=0)
        all_lms = (all_lms - cloud_means) / cloud_stds

        # Overwrite unpooled landmarks for feature extraction in standard space
        lms = all_lms.reshape((lms.shape[0], lms.shape[1], lms.shape[2]))

    # Downsampling
    if downsample is not None and clustering[0] != 'previous':
        if verbose: print "Downsampling merged cloud..."

        # Default is density dependent downsampling
        if downsample[0]=='default' or downsample[0]=='ddds':
            all_lms_ds = ds.ddds(all_lms, downsample[1],
                                 presample=downsample[1],
                                 processes=processes)

        # Alternative: kmeans downsampling
        elif downsample[0]=='kmeans':
            all_lms_ds = ds.kmeans_subsample(all_lms, downsample[1])

        # Alternative: random downsampling
        elif downsample[0]=='random':
            all_lms_ds = ds.random_subsample(all_lms, downsample[1])

        # Custom downsampling
        elif callable(downsample[0]):
            all_lms_ds = downsample[0](all_lms, downsample)

        # Handle other cases
        else:
            raise ValueError("Invalid downsampling method: " +
                             str(downsample[0]))

    # No downsampling
    else:
        all_lms_ds = all_lms

    # LEGACY: Standardization after downsampling and without overwriting the
    #         unpooled landmarks!
    if legacy and standardize:
        cloud_means = all_lms_ds.mean(axis=0)
        cloud_stds  = all_lms_ds.std(axis=0)
        all_lms_ds = (all_lms_ds - cloud_means) / cloud_stds


    #--------------------------------------------------------------------------

    ### Find reference points by clustering [altogether]

    if verbose: print "Clustering to find reference points..."

    # Default: kmeans clustering
    if clustering[0] == 'default' or clustering[0] == 'kmeans':

        # Perform clustering
        my_clust = MiniBatchKMeans(n_clusters=clustering[1], random_state=42)
        my_clust.fit(all_lms_ds)

        # Get labels and centroids
        clust_labels  = my_clust.labels_
        clust_centers = my_clust.cluster_centers_

        # Predict labels for whole data set (if downsampled)
        if downsample is not None:
            clust_labels = my_clust.predict(all_lms)

    # To be added: DBSCAN
    elif clustering[0] == 'dbscan':
        raise NotImplementedError("And likely never will be...")

    # Using a given (already fitted) clustering object
    elif clustering[0] == 'previous':
        my_clust = clustering[1]
        clust_centers = my_clust.cluster_centers_
        clust_labels  = my_clust.predict(all_lms)

    # Custom alternative
    elif callable(clustering[0]):
        clust_labels, clust_centers = clustering[0](all_lms, clustering)

    # Handle other cases
    else:
        raise ValueError("Invalid clustering method: "+str(clustering[0]))

    # LEGACY: Back-transform of centroids to un-standardized space
    #         In legacy, feature extraction was done on the un-standardized
    #         space, using the back-transformed centroids
    if legacy and standardize:
        clust_centers = clust_centers * cloud_stds + cloud_means

    # Unpool cluster labels
    clust_labels = clust_labels.reshape((lms.shape[0], lms.shape[1]))


    #--------------------------------------------------------------------------

    ### Extract features relative to reference points [per cell]

    if verbose: print "Extracting cluster features..."

    # Init dask graph
    dask_graph = dict()

    # For each cell...
    for c in range(lms.shape[0]):

        # Node to compute kdtree
        dask_graph["kdtree_%i" % c] = (fe.build_kdtree, lms[c,:,:])

        # Nodes for the features
        dask_graph["kNN-distsManh_%i" % c]  = (fe.feature_distsManhatten_kNN,
                                               "kdtree_%i" % c, lms[c,:,:],
                                               clust_centers)

        dask_graph["kNN-distEuclid_%i" % c] = (fe.feature_distEuclidean_kNN,
                                               "kNN-distsManh_%i" % c,
                                               lms.shape[2])

        dask_graph["NN-distsManh_%i" % c]   = (fe.feature_distsManhatten_NN,
                                               "kdtree_%i" % c, lms[c,:,:],
                                               clust_centers)

        dask_graph["NN-distEuclid_%i" % c]  = (fe.feature_distEuclidean_NN,
                                               "NN-distsManh_%i" % c,
                                               lms.shape[2])

        dask_graph["count-near_%i" % c]     = (fe.feature_count_near,
                                               ["kdtree_%i" % c,
                                                "kNN-distEuclid_%i" % c],
                                               lms[c,:,:], clust_centers)

        dask_graph["count-assigned_%i" % c] = (fe.feature_count_assigned,
                                               clust_centers, clust_labels[c])

        dask_graph["kde_%i" % c] = (fe.feature_kde, lms[c,:,:], clust_centers,
                                    bw_method)

        # Nodes for custom feature extraction functions
        if custom_feature_funcs is not None:
            for custom_func in custom_feature_funcs:
                custom_parents = [parent+"_%i" % c
                                  for parent in custom_func[2]]
                dask_graph[custom_func[0]+"_%i" % c] = (custom_func[1],
                                                        custom_parents,
                                                        custom_func[3],
                                                        lms[c,:,:],
                                                        clust_centers,
                                                        clust_labels[c])

        # Node to collect requested features for a cell
        dask_graph["fvector_%i" % c] = (fe.assemble_cell,
                                        [f+"_%i" % c for f in features],
                                        features)

        # Render example graph for first 3 cells
        if c==2 and dask_graph_path is not None:
            from dask.dot import dot_graph
            dask_graph["fspace"] = (fe.assemble_fspace,
                                    ["fvector_%i" % c for c in range(3)])
            dot_graph(dask_graph, filename=dask_graph_path)

    # Final node to put per-cell features into a feature space
    dask_graph["fspace"] = (fe.assemble_fspace,
                            ["fvector_%i" % c for c in range(lms.shape[0])])

    # Run without multiprocessing
    if processes == 1:
        with ProgressBar(dt=1):
            fspace, fheader = dask.get(dask_graph, 'fspace')

    # Run with multiprocessing
    else:

        # Set number of threads
        dask.set_options(pool=ThreadPool(processes))

        # Run the pipeline (no profiling)
        if not profiling:
            with ProgressBar(dt=1):
                fspace, fheader = dask.threaded.get(dask_graph, 'fspace')

        # Run the pipeline (with resource profiling)
        if profiling:
            with ProgressBar(dt=1):
                with Profiler() as prof, ResourceProfiler(dt=0.1) as rprof:
                    fspace, fheader = dask.threaded.get(dask_graph, 'fspace')
                visualize([prof,rprof], save=False)


    #--------------------------------------------------------------------------

    ### Save [per stack], report and return

    if verbose: print "Saving result..."

    # For each stack...
    for sample_idx, sample_fpath in enumerate(fpaths_lm):

        # Prepare suffix
        suffix = ""

        # Save individually downsampled landmark distributions if desired
        if presample is not None and save_presampled:
            if suffix_out=='default' or 'PRES' not in suffix_out.keys():
                suffix = suffix + "_PRES"
            else:
                suffix = suffix + suffix_out['PRES']
            np.save(sample_fpath[:-4]+suffix,
                    lms_ps[lms_idx==sample_idx,:,:])

        # Save CFOR if desired
        if cfor is not None and save_cfor:
            if suffix_out=='default' or 'CFOR' not in suffix_out.keys():
                suffix = suffix + "_CFOR"
            else:
                suffix = suffix + suffix_out['CFOR']
            np.save(sample_fpath[:-4]+suffix,
                    lms[lms_idx==sample_idx,:,:])

        # Include downsampling in suffix
        if downsample is not None:
            if suffix_out=='default' or 'DS' not in suffix_out.keys():
                suffix = suffix + '_DS'
            else:
                suffix = suffix + suffix_out['DS']

        # Save shape space
        if suffix_out=='default' or 'CBE' not in suffix_out.keys():
            suffix = suffix + "_CBE"
        else:
            suffix = suffix + suffix_out['CBE']
        np.save(sample_fpath[:-4]+suffix,
                fspace[lms_idx==sample_idx, :])

        # Save new metadata
        if save_metadata:

            # Construct metadata path
            dirpath, fname = os.path.split(sample_fpath)
            fpath_meta = os.path.join(dirpath,
                                      fname[:10]+"_stack_metadata.pkl")

            # Open metadata
            with open(fpath_meta, 'rb') as metafile:
                meta_dict = pickle.load(metafile)

            # Prepare metadata suffix
            if suffix_out=='default' or 'META' not in suffix_out.keys():
                if suffix[0] == '_':
                    m_suffix = suffix[1:]
                else:
                    m_suffix = suffix
            else:
                if suffix[0] == '_':
                    m_suffix = suffix[1:] + suffix_out['META']
                else:
                    m_suffix = suffix + suffix_out['META']

            # Slightly awkward addition of TFOR tag
            if 'TFOR' in fpaths_lm[0]:
                m_suffix = 'TFOR_' + m_suffix

            # Add new metadata
            meta_dict["clustobj-"+m_suffix] = my_clust
            meta_dict["clusters-"+m_suffix] = clust_centers
            meta_dict["labels-"+m_suffix]   = clust_labels[lms_idx==sample_idx]
            meta_dict["features-"+m_suffix] = fheader

            # Write metadata
            with open(fpath_meta, 'wb') as metafile:
                pickle.dump(meta_dict, metafile, pickle.HIGHEST_PROTOCOL)

    # Report and return
    if verbose: print "Processing complete!"
    return


#------------------------------------------------------------------------------



