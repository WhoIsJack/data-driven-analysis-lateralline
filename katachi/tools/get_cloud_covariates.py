# -*- coding: utf-8 -*-
"""
Created on Wed Jan 03 11:54:58 2018

@author:    Jonas Hartmann @ Gilmour group @ EMBL Heidelberg

@descript:  Functions to extract covariates from 3D point clouds representing
            fluorescence distributions within cells.
"""

#------------------------------------------------------------------------------

# IMPORTS

# External
from __future__ import division
import pickle
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform

# Internal
from katachi.utilities.hierarchical_data import HierarchicalData
from katachi.utilities.sympy_roundness import compute_roundness_deviation


#------------------------------------------------------------------------------

# FUNCTION: EXTRACT PER-SAMPLE COVARIATES FROM TFOR POINT CLOUD

def get_pcl_covars_sample(fpath_tfor, fpath_meta,
                          tfor_lms=None, metadata=None, covars=None,
                          verbose=False):
    """Extract per-sample covariates from TFOR point cloud.

    Currently extracts the following covariates:
        - covars.pcl.sample.extents
        - covars.pcl.sample.aspects

    To be implemented:
        - covars.img.sample.segments.* [see TODO in code]
        - ...?

    Parameters
    ----------
    fpath_tfor : string
        The path (either local from cwd or global) to a npy file containing
        landmarks for the cells of a prim as extracted in
        katachi.tools.extract_landmarks and transformed to the tissue FOR with
        katachi.tools.find_TFOR.
    fpath_meta : string
        The path (either local from cwd or global) to a pkl file containing
        metadata, including the TFOR centroid positions as computed in
        katachi.tools.find_TFOR.
    tfor_lms : numpy array of shape (cells, lms, dims), optional, default None
        A numpy array containing the landmarks for the cells of a prim to be
        used instead of the file at fpath_tfor. If this is passed, fpath_tfor
        is ignored.
    metadata : metadata dictionary, optional, default None
        A dictionary of metadata to be used instead of the file at fpath_meta.
        If this is passed, fpath_meta is ignored.
    covars : HierarchicalData instance, optional, default None
        An instance of katachi.utilities.hierarchical_data.HierarchicalData to
        which the determined covariates will be added. If this is None, a new
        instance is created.
    verbose : bool, optional, default False
        If True, more information is printed.

    Returns
    -------
    covars : HierarchicalData instance
        An instance of katachi.utilities.hierarchical_data.HierarchicalData
        containing the generated covariate measurements in addition to any
        covariates that it the parameter `covars` already contained if it was
        passed.
    """

    #--------------------------------------------------------------------------

    ### Load data

    # Load TFOR landmark data
    if tfor_lms is None:

        # Report
        if verbose: print "Loading TFOR landmark data..."

        # Try loading the landmark data
        try:
            tfor_lms = np.load(fpath_tfor)
        except:
            print "Attempting to load TFOR LM data failed with this error:"
            raise

    # Load metadata
    if metadata is None:

        # Report
        if verbose: print "Loading metadata..."

        # Try loading the landmark data
        try:
            with open(fpath_meta, 'rb') as infile:
                metadata = pickle.load(infile)
        except:
            print "Attempting to load metadata failed with this error:"
            raise

    # Assemble entire prim point cloud
    p  = tfor_lms.shape
    pr = (p[1], p[0], p[2])
    prim_lms = tfor_lms.reshape(pr) + metadata["centroids_TFOR"]
    prim_lms = prim_lms.reshape((p[0]*p[1], p[2]))


    #--------------------------------------------------------------------------

    ### Prepare covariate class

    # Ignore if class is given
    if covars is None:

        # Instantiate empty hierarchical class
        covars = HierarchicalData()


    #--------------------------------------------------------------------------

    ### Extract TFOR prim covariates

    if verbose: print "Extracting TFOR prim covariates..."

    # Maximum extents in ZYX [covars.pcl.sample.extents]
    extents = np.max(prim_lms, axis=0) - np.min(prim_lms, axis=0)
    covars.pcl.sample.extents = extents

    # Aspect ratios of maximum extents (ZY, ZY, YX) [covars.pcl.sample.aspects]
    aspects = np.zeros(3)
    aspects[0] = extents[0] / extents[1]
    aspects[1] = extents[0] / extents[2]
    aspects[2] = extents[1] / extents[2]
    covars.pcl.sample.aspects = aspects

    # TODO: Max extents along segments of the prim
    #  It would be possible to get more information about the prim's shape by
    #  computing something like the convex hull or a more basic (advanced?)
    #  profile based on marching over segments along the main axes, computing
    #  something like a topographical map.
    #  However, this is low priority for the moment.
    pass


    #--------------------------------------------------------------------------

    ### Report and return results

    if verbose: print "Processing complete!"
    return covars


#------------------------------------------------------------------------------

# FUNCTION: EXTRACT PER-CELL COVARIATES AT THE TISSUE LEVEL FROM TFOR CLOUDS

def get_pcl_covars_tissue(fpath_tfor, fpath_meta,
                          tfor_lms=None, metadata=None, covars=None,
                          verbose=False):
    """Extract tissue-level covariates from TFOR point cloud.

    NOTE: This function is currently only here for completeness sake. The only
          covariate being extracted comes directly from the metadata and the
          parameters referring to TFOR landmarks are ignored.

    Currently extracts the following covariates:
        - covars.pcl.tissue.centroids

    To be implemented:
        - ...?

    Parameters
    ----------
    fpath_tfor : string
        The path (either local from cwd or global) to a npy file containing
        landmarks for the cells of a prim as extracted in
        katachi.tools.extract_landmarks and transformed to the tissue FOR with
        katachi.tools.find_TFOR.
        NOTE: This is currently being ignored because it is not needed!
    fpath_meta : string
        The path (either local from cwd or global) to a pkl file containing
        metadata, including the TFOR centroid positions as computed in
        katachi.tools.find_TFOR.
    tfor_lms : numpy array of shape (cells, lms, dims), optional, default None
        A numpy array containing the landmarks for the cells of a prim to be
        used instead of the file at fpath_tfor. If this is passed, fpath_tfor
        is ignored.
        NOTE: This is currently being ignored because it is not needed!
    metadata : metadata dictionary, optional, default None
        A dictionary of metadata to be used instead of the file at fpath_meta.
        If this is passed, fpath_meta is ignored.
    covars : HierarchicalData instance, optional, default None
        An instance of katachi.utilities.hierarchical_data.HierarchicalData to
        which the determined covariates will be added. If this is None, a new
        instance is created.
    verbose : bool, optional, default False
        If True, more information is printed.

    Returns
    -------
    covars : HierarchicalData instance
        An instance of katachi.utilities.hierarchical_data.HierarchicalData
        containing the generated covariate measurements in addition to any
        covariates that it the parameter `covars` already contained if it was
        passed.
    """

    #--------------------------------------------------------------------------

    ### Load data

    ## Load TFOR landmark data
    ## NOTE: THIS IS CURRENTLY UNUSED AND THEREFORE COMMENTED OUT!
    #if tfor_lms is None and False:
    #
    #    # Report
    #    if verbose: print "Loading TFOR landmark data..."
    #
    #    # Try loading the landmark data
    #    try:
    #        tfor_lms = np.load(fpath_tfor)
    #    except:
    #        print "Attempting to load TFOR LM data failed with this error:"
    #        raise

    # Load metadata
    if metadata is None:

        # Report
        if verbose: print "Loading metadata..."

        # Try loading the landmark data
        try:
            with open(fpath_meta, 'rb') as infile:
                metadata = pickle.load(infile)
        except:
            print "Attempting to load metadata failed with this error:"
            raise


    #--------------------------------------------------------------------------

    ### Prepare covariate class

    # Ignore if class is given
    if covars is None:

        # Instantiate empty hierarchical class
        covars = HierarchicalData()


    #--------------------------------------------------------------------------

    ### Get TFOR tissue covariates

    if verbose: print "Extracting TFOR tissue covariates..."

    # Centroid position in TFOR [covars.pcl.tissue.centroids]
    # Note: This is simply taken from the meta data but is added to the
    #       covariates in this fashion for the sake of completeness!
    covars.pcl.tissue.centroids = metadata["centroids_TFOR"]


    #--------------------------------------------------------------------------

    ### Report and return results

    if verbose: print "Processing complete!"
    return covars


#------------------------------------------------------------------------------

# FUNCTION: EXTRACT PER-CELL COVARIATES AT THE CELL LEVEL FROM CLOUDS

def get_pcl_covars_cell(fpath_lms, channel_name, M=5, no_cfor=False,
                        fpath_lms_cfor=None, lms_cfor=None, lms=None,
                        covars=None, verbose=False):
    """Extract per-cell cell-scale covariates from cellular point clouds.

    Currently extracts the following covariates:
        - covars.pcl.cell.<channel_name>.extents
        - covars.pcl.cell.<channel_name>.aspects
        - covars.pcl.cell.<channel_name>.extents_pca
        - covars.pcl.cell.<channel_name>.aspects_pca
        - covars.pcl.cell.<channel_name>.sphericity [deprecated]
        - covars.pcl.cell.<channel_name>.symmetry [deprecated]
        - covars.pcl.cell.<channel_name>.sphericity_deviation
        - covars.pcl.cell.<channel_name>.roundness_deviation
        - covars.pcl.cell.<channel_name>.distp_coords
        - covars.pcl.cell.<channel_name>.distp_dist
        - covars.pcl.cell.<channel_name>.distp_angles
        - covars.pcl.cell.<channel_name>.nn_dists_mean
        - covars.pcl.cell.<channel_name>.nn_dists_std
        - covars.pcl.cell.<channel_name>.all_dists_mean
        - covars.pcl.cell.<channel_name>.all_dists_std
        - covars.pcl.cell.<channel_name>.cen_dists_mean
        - covars.pcl.cell.<channel_name>.cen_dists_std
        - covars.pcl.cell.<channel_name>.moments
        - covars.pcl.cell.<channel_name>.moments_cfor
        - covars.pcl.cell.<channel_name>.eccentricity

    To be implemented:
        - ...?

    Parameters
    ----------
    fpath_lms : string
        The path (either local from cwd or global) to a npy file containing
        landmarks for the cells of a prim as extracted in
        katachi.tools.extract_landmarks. Ideally, the cells were also
        transformed to the tissue FOR with katachi.tools.find_TFOR, but this
        is not absolutely essential for the covariates extracted here.
    channel_name : string
        The intensity channel name, i.e. the name to be used as attribute in
        the hiararchical covars class for accessing measurements derived from
        the given intensity stack.
    M : int, optional, default 5
        Highest-level moments to extract from point cloud. The moments array
        will have length (M+1)*(M+2)*(M+3)/6-1.
    no_cfor : bool, optional, default False
        If True, the CFOR-based moments will not be computed and no CFOR data
        is required at any point.
    fpath_lms_cfor : None or string, optional, default None
        The path (either local from cwd or global) to a npy file containing
        CFOR-transformed as produced by `katachi.tools.perform_CBE`. Required
        for computing the CFOR-based moments.
        Either fpath_lms_cfor or lms_cfor must be specified unless no_cfor is
        set to True.
    lms_cfor : numpy array of shape (cells, lms, dims), optional, default None
        A numpy array containing the CFOR-transformed landmarks for the cells
        of a sample, to be used instead of the file at fpath_lms_cfor. If this
        is passed, fpath_lms_cfor is ignored.
        Either lms_cfor or fpath_lms_cfor must be specified unless no_cfor is
        set to True.
    lms : numpy array of shape (cells, lms, dims), optional, default None
        A numpy array containing the landmarks for the cells of a prim to be
        used instead of the file at fpath_lms. If this is passed, fpath_lms
        is ignored.
    covars : HierarchicalData instance, optional, default None
        An instance of katachi.utilities.hierarchical_data.HierarchicalData to
        which the determined covariates will be added. If this is None, a new
        instance is created.
    verbose : bool, optional, default False
        If True, more information is printed.

    Returns
    -------
    covars : HierarchicalData instance
        An instance of katachi.utilities.hierarchical_data.HierarchicalData
        containing the generated covariate measurements in addition to any
        covariates that it the parameter `covars` already contained if it was
        passed.
    """

    #--------------------------------------------------------------------------

    ### Load data

    # Load TFOR landmark data
    if lms is None:

        # Report
        if verbose: print "Loading TFOR landmark data..."

        # Try loading the landmark data
        try:
            lms = np.load(fpath_lms)
        except:
            print "Attempting to load landmark data failed with this error:"
            raise

    # Load CFOR landmark data
    if not no_cfor and lms_cfor is None:

        # Check if a path is given
        if fpath_lms_cfor is None:
            raise IOError("`no_cfor` is False but no `fpath_lms_cfor` has " +
                          "been provided!")

        # Report
        if verbose: print "Loading CFOR landmark data..."

        # Try loading the CFOR data
        try:
            lms_cfor = np.load(fpath_lms_cfor)
        except:
            print "Attempting to load CFOR data failed with this error:"
            raise


    #--------------------------------------------------------------------------

    ### Prepare covariate class

    # Ignore if class is given
    if covars is None:

        # Instantiate empty hierarchical class
        covars = HierarchicalData()


    #--------------------------------------------------------------------------

    ### Extract extent-based covariates

    if verbose: print "Extracting extent-based cell covariates..."

    # Maximum extents in ZYX (cell height, width, length)
    extents = np.max(lms, axis=1) - np.min(lms, axis=1)
    covars.pcl.cell._gad(channel_name).extents = extents

    # Aspect ratios of maximum extents (ZY, ZY, YX)
    aspects = np.zeros((lms.shape[0], 3))
    aspects[:,0] = extents[:,0] / extents[:,1]
    aspects[:,1] = extents[:,0] / extents[:,2]
    aspects[:,2] = extents[:,1] / extents[:,2]
    covars.pcl.cell._gad(channel_name).aspects = aspects

    # Maximum extents along major axes
    lms_pca = np.empty_like(lms)
    for i in range(lms.shape[0]):
        cell_pca = PCA()
        lms_pca[i,...] = cell_pca.fit_transform(lms[i,...])
    extents_pca = np.max(lms_pca, axis=1) - np.min(lms_pca, axis=1)
    covars.pcl.cell._gad(channel_name).extents_pca = extents_pca

    # Aspect ratios of extents along major axes (PC 1-2, 1-3, 2-3)
    aspects_pca = np.zeros((lms.shape[0], 3))
    aspects_pca[:,0] = extents_pca[:,0] / extents_pca[:,1]
    aspects_pca[:,1] = extents_pca[:,0] / extents_pca[:,2]
    aspects_pca[:,2] = extents_pca[:,1] / extents_pca[:,2]
    covars.pcl.cell._gad(channel_name).aspects_pca = aspects_pca


    #--------------------------------------------------------------------------

    ### Extract sphericity/roundness-based covariates

    if verbose: print "Extracting symmetry-based cell covariates..."

    # Sphericity [DEPRECATED]
    # Mean deviation from mean sphere around centroid, normalized, inverted
    # [DEPRECATED: This old implementation doesn't quite normalize the measure
    #              correctly, although in practice it doesn't make much of a
    #              difference!]
    lms_mag = np.sqrt(np.sum(lms**2.0, axis=2))
    lms_mag_normed = lms_mag / np.expand_dims(lms_mag.mean(axis=1), 1)
    sphericity = 1 / (1 + np.mean(np.abs(lms_mag_normed-1.0), axis=1))
    covars.pcl.cell._gad(channel_name).sphericity = sphericity

    # Symmetry [DEPRECATED]
    # Distance of center of mass from geometrical center, normalized, inverted
    # [DEPRECATED: This measure doesn't seem to encode meaningful/sensible
    #              information and the same intention is better handled by the
    #              eccentricity measures.]
    geom_cen = np.mean([np.max(lms, axis=1), np.min(lms, axis=1)], axis=0)
    mass_cen = np.mean(lms, axis=1)
    g_m_dist = np.sqrt(np.sum((geom_cen-mass_cen)**2.0, axis=1))
    g_m_dist_normed = g_m_dist / lms_mag.mean(axis=1)
    symmetry = 1 / (1 + g_m_dist_normed)
    covars.pcl.cell._gad(channel_name).symmetry = symmetry

    # Sphericity
    # Mean deviation from mean sphere around centroid, normalized, inverted
    lms_mag = np.sqrt(np.sum(lms**2.0, axis=2))
    lms_mag_factor = lms_mag / np.expand_dims(lms_mag.mean(axis=1), 1)
    sphericity = 1 - np.mean(np.abs(lms_mag_factor-1.0), axis=1)
    covars.pcl.cell._gad(channel_name).sphericity_deviation = sphericity

    # Roundness
    # Mean deviation from circumscribed ellipsoid, normalized, inverted
    roundness = compute_roundness_deviation(lms_pca, aligned=True,
                                            semi_axes=extents_pca/2.0)
    covars.pcl.cell._gad(channel_name).roundness_deviation = roundness


    #--------------------------------------------------------------------------

    ### Extract covariates based on the most distal point

    if verbose: print "Extracting distal-most point cell covariates..."

    # Most distal point: coordinates
    dp_index  = np.argmax(lms_mag, axis=1)
    dp_coords = lms[np.arange(lms.shape[0]), dp_index, :]
    covars.pcl.cell._gad(channel_name).distp_coords = dp_coords

    # Most distal point: distance from centroid
    dp_dist = np.sqrt(np.sum(dp_coords**2.0, axis=1))
    covars.pcl.cell._gad(channel_name).distp_dist = dp_dist

    # Most distal point: orientation angles (in planes ZY, ZX, YX)
    # Function for getting the orientation angles
    def compute_orientation(v2d):
        """Computes angles of a set of 2D vectors to the reference vector (0,1)
        IN  -- v2d : np arr of shape (N,2); array of N 2D vectors
        OUT --   s : np arr of shape (N); resulting angles
        """

        # Prepare reference axis
        ref = np.array([0,1])

        # Compute angles (trust me...)
        c = np.dot(v2d, ref) / np.linalg.norm(v2d,axis=1) / np.linalg.norm(ref)
        d = np.rad2deg(np.arccos(np.clip(c, -1, 1)))

        # Add sign back
        s = d * np.sign(v2d[:,0])

        # Return result
        return s

    # Function calls to get the different orientation angles
    dp_angles = np.zeros((lms.shape[0],3))
    dp_angles[:,0] = compute_orientation(dp_coords[:,(0,1)])
    dp_angles[:,1] = compute_orientation(dp_coords[:,(0,2)])
    dp_angles[:,2] = compute_orientation(dp_coords[:,(1,2)])
    covars.pcl.cell._gad(channel_name).distp_angles = dp_angles


    #--------------------------------------------------------------------------

    ### Extract distance-based covariates

    if verbose: print "Extracting distance-based cell covariates..."

    # Prep
    nn_dists_mean  = np.empty(lms.shape[0])
    nn_dists_std   = np.empty(lms.shape[0])
    all_dists_mean = np.empty(lms.shape[0])
    all_dists_std  = np.empty(lms.shape[0])

    # For each cell...
    for i in range(lms.shape[0]):

        # Find all pairwise distances among landmarks
        dists = squareform(pdist(lms[i,...]))

        # Nearest neighbor distances
        dists_eyemasked = np.ma.array(dists, mask=np.eye(dists.shape[0]))
        nn_dists = np.min(dists_eyemasked, axis=1)
        nn_dists_mean[i] = nn_dists.mean()
        nn_dists_std[i]  = nn_dists.std()

        # All neighbors distances
        all_dists_mean[i] = dists.mean()
        all_dists_std[i]  = dists.std()

    # Distances from centroid (magnitudes)
    cen_dists_mean = lms_mag.mean(axis=1)
    cen_dists_std  = lms_mag.std(axis=1)

    # Add to covar object
    covars.pcl.cell._gad(channel_name).nn_dists_mean  = nn_dists_mean
    covars.pcl.cell._gad(channel_name).nn_dists_std   = nn_dists_std
    covars.pcl.cell._gad(channel_name).all_dists_mean = all_dists_mean
    covars.pcl.cell._gad(channel_name).all_dists_std  = all_dists_std
    covars.pcl.cell._gad(channel_name).cen_dists_mean = cen_dists_mean
    covars.pcl.cell._gad(channel_name).cen_dists_std  = cen_dists_std


    #--------------------------------------------------------------------------

    ### Moments

    if verbose: print "Extracting cell cloud moments..."

    # Function to compute moments from a 3D point cloud
    def compute_moments(cloud, M=5):
        """Computes moments from a 3D point cloud up to Mth order.

        Computes the 1st raw moments, the 2nd central moments and the 3rd to
        Mth normalized moments of a 3D point cloud. Moments are returned as a
        1d array of length (M+1)*(M+2)*(M+3)/6-1. Alongside, an array of shape
        ((M+1)*(M+2)*(M+3)/6-1, 3) is returned that indicates the order of the
        corresponding moment.

        IN  -- cloud : numpy array of shape (N,3);
                       point cloud of N points in 3 dims
        IN  --     M : int (default 5);
                       highest-order moment to compute, must be in range(1,21)
        OUT --   mmt : numpy array of shape ((M+1)*(M+2)*(M+3)/6-1,);
                       computed moments
        OUT --   onm : numpy array of shape ((M+1)*(M+2)*(M+3)/6-1, 3);
                       order numbers of moments
        """

        # Check if M is reasonable
        if M < 1:
            raise ValueError("M too small.")
        if M > 20:
            raise ValueError("M too large.")

        # Compute 1st raw moments
        mmt = np.mean(cloud, axis=0).tolist()
        onm = [[1,0,0], [0,1,0], [0,0,1]]

        # Return if M==1
        if M==1:
            return np.array(mmt), np.array(onm)

        # Centralize cloud
        cloud_cen = cloud - mmt

        # Compute 2nd centralized moments
        for i in range(3):
            for j in range(3-i):
                k = 2-i-j
                mmt.append( np.mean(  cloud_cen[:,0]**i
                                    * cloud_cen[:,1]**j
                                    * cloud_cen[:,2]**k) )
                onm.append( [i,j,k] )

        # Return if M==2
        if M==2:
            return np.array(mmt), np.array(onm)

        # Compute 3rd to Mth normalized moments
        for m in range(3, M+1):
            for i in range(m+1):
                for j in range(m+1-i):
                    k = m-i-j
                    cen = np.mean(  cloud_cen[:,0]**i
                                  * cloud_cen[:,1]**j
                                  * cloud_cen[:,2]**k)
                    nrm = (  np.std(cloud_cen[:,0])**i
                           * np.std(cloud_cen[:,1])**j
                           * np.std(cloud_cen[:,2])**k)
                    mmt.append( cen / nrm )
                    onm.append( [i,j,k] )

        # Return
        return np.array(mmt), np.array(onm)

    # Compute the moments for each cell
    moments = np.zeros(( lms.shape[0], int( (M+1)*(M+2)*(M+3)/6-1 )))
    for i in range(lms.shape[0]):
        moments[i] = compute_moments(lms[i,...], M=M)[0]
    covars.pcl.cell._gad(channel_name).moments = moments

    # Compute the moments for each cell in the Cell Frame Of Reference (CFOR)
    if not no_cfor:
        moments_cfor = np.zeros(( lms_cfor.shape[0], int( (M+1)*(M+2)*(M+3)/6-1 )))
        for i in range(lms_cfor.shape[0]):
            moments_cfor[i] = compute_moments(lms_cfor[i,...], M=M)[0]
        covars.pcl.cell._gad(channel_name).moments_cfor = moments_cfor


    #--------------------------------------------------------------------------

    ### Eccentricity
    # XXX: WARNING -- Calculating this based on PCA is not 100% correct, but in
    #                 practice it seems like a nice simple way of doing it and
    #                 it works well!

    if verbose: print "Extracting cell eccentricities..."

    # Function calculating eccentricity from principal semi-axes
    def compute_eccentricity(semi_maj, semi_min):
        """Compute eccentricity from extents of a fitted ellipse's semi-axes
        IN  -- semi_maj, semi_min : float; extents of semi-axes
        OUT -- eccentricity -- float; computed eccentricity
        """
        return np.sqrt(1 - (semi_min**2 / semi_maj**2))

    # For each cell...
    eccentricity = np.empty((lms.shape[0], 3))
    for i in range(lms.shape[0]):

        # Get principal axes by PCA
        # [Already done previously]
        #pca = PCA()
        #lms_pca = pca.fit_transform(lms[i,...])

        # Get semi-axis extents from PCA
        semi_axes = (  np.max(lms_pca[i,...], axis=0)
                     - np.min(lms_pca[i,...], axis=0) ) / 2
        semi_axes = np.sort(semi_axes)[::-1]

        # Calculate the eccentricities (ZY, ZX, YX)
        eccentricity[i,:] = [compute_eccentricity(semi_axes[0], semi_axes[1]),
                             compute_eccentricity(semi_axes[0], semi_axes[2]),
                             compute_eccentricity(semi_axes[1], semi_axes[2])]

    # Add to covariates
    covars.pcl.cell._gad(channel_name).eccentricity = eccentricity


    #--------------------------------------------------------------------------

    ### Report and return results

    if verbose: print "Processing complete!"

    return covars


#------------------------------------------------------------------------------




