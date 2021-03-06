# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 16:10:21 2017

@author:    Jonas Hartmann @ Gilmour group @ EMBL Heidelberg

@descript:  Dask pipeline for feature embedding by point cloud sampling and
            cluster-based embedding from single-cell segmentations.
"""

#------------------------------------------------------------------------------

# IMPORTS

# External
from __future__ import division
import os
from warnings import warn
from multiprocessing.pool import ThreadPool, cpu_count
import dask
from functools import partial
from dask.diagnostics import ProgressBar
from dask.diagnostics import Profiler, ResourceProfiler, visualize

# Internal
from katachi.tools.assign_landmarks import assign_landmarks
from katachi.tools.find_TFOR import transform_to_TFOR
from katachi.tools.perform_CBE import cbe


#------------------------------------------------------------------------------

# FUNCTION: RUN FEATURE EXTRACTION FOR A DIRECTORY / CHANNEL

def feature_extraction(dirpath, suffix_seg, suffix_int,
                       num_LMs, downsample, clustering, features,
                       recurse=False, select_IDs='all',
                       assign_landmarks_kwargs='default',
                       compute_TFOR=True,
                       transform_to_TFOR_kwargs='default',
                       perform_CBE_TFOR_kwargs='default',
                       compute_CFOR=True,
                       perform_CBE_CFOR_kwargs='default',
                       processes=None, dask_graph_path=None,
                       profiling=False, verbose=False):
    """Extract latent features from fluorescence distributions of single-cell
    segmentations by point cloud sampling and cluster-based embedding.

    This is a dask pipeline that applies point-cloud sampling from
    `katachi.tools.assign_landmars`, transformation to the TFOR (optional)
    from `katachi.tools.find_TFOR` and cluster-based embedding (either on TFOR
    data or by constructing a CFOR, or both) from `katachi.tools.perform_CBE`
    to a dataset of single-cell segmentations that has been generated by
    `katachi.pipelines.segmentation` or an equivalent approach.

    WARNING: Not all options provided by this pipeline have been extensively
    tested. Use with prudence!

    Parameters
    ----------
    dirpath : string
        The path (either local from cwd or global) to the directory with the
        input data to be processed.
    suffix_seg : string
        File suffix that identifies target segmentation files as produced by
        `katachi.pipelines.segmentation`. This will usually be "seg.tif" but
        could contain more information to distinguish different segmentations.
    suffix_int : string
        File suffix that identifies target intensity files matching the shape
        of the target segmentation files. Each retrieved segmentation file must
        have a matching intensity file.
    num_LMs : int
        The number of landmarks to extract for each cell.
    downsample : tuple (algorithm, output_size) or None
        A tuple specifying the algorithm to use for downsampling of the merged
        point cloud prior to cluster extraction.
        See `katachi.tools.perform_CBE` for more information.
    clustering : tuple (algorithm, n_clusters)
        A tuple specifying the algorithm to use for computing the clusters to
        use in cluster-based feature extraction.
        See `katachi.tools.perform_CBE` for more information.
        Special case: both elements of clustering (i.e. `algorithm` and
        `n_clusters`) may themselves be tuples. In this case, their first and
        second elements will be used in CBE on TFOR and CFOR, respectively.
    features : list of strings
        List containing any number of cluster features to be extracted.
        See `katachi.tools.perform_CBE` for more information.
    recurse : bool, optional, default False
        If True, files are searched recursively in the subdirs of fpath.
    select_IDs : 'all' or list of strings, optional, default 'all'
        If 'all' (default), all detected input files (i.e. all samples) are
        used. Instead, a list of strings containing IDs (as assigned by
        `katachi.tools.initialize`) can be passed, in which case only samples
        whose IDs are in the list are used. If there are IDs in the list for
        which no matching files were found, a warning is shown.
    assign_landmarks_kwargs : dict or 'default', optional, default 'default'
        Dictionary specifying kwargs for assign_landmarks function.
        See `katachi.tools.assign_landmarks.assign_landmarks` for information
        about available options.
        See section "Prepare kwargs for landmark assignment" in this function
        for information on default settings.
    compute_TFOR : bool, optional, default True
        If True, the prim frame of reference is computed and CBE is performed
        on the TFOR landmark data.
        At least one of compute_TFOR or compute_CFOR must be set to True.
    transform_to_TFOR_kwargs : dict or 'default', optional, default 'default'
        Dictionary specifying kwargs for transform_to_TFOR function.
        See `katachi.tools.find_TFOR.transform_to_TFOR` for information
        about available options.
        See section "Prepare kwargs for transformation to TFOR" in this
        function for information on default settings.
    perform_CBE_TFOR_kwargs : dict or 'default', optional, default 'default'
        Dictionary specifying kwargs for cbe function applied to TFOR.
        See `katachi.tools.perform_CBE.cbe` for information about available
        options.
        See section "Prepare kwargs for CBE on TFOR" in this function for
        information on default settings.
    compute_CFOR : bool, optional, default True
        If True, the cell frame of reference is computed and CBE is performed
        on the CFOR landmark data.
        At least one of compute_TFOR or compute_CFOR must be set to True.
    perform_CBE_CFOR_kwargs : dict or 'default', optional, default 'default'
        Dictionary specifying kwargs for cbe function applied to CFOR.
        See `katachi.tools.perform_CBE.cbe` for information about available
        options.
        See section "Prepare kwargs for CBE on CFOR" in this function for
        information on default settings.
    processes : int or None, optional
        Number of processes dask may use for parallel processing. If None, half
        of the available CPUs are used. If set to 1, the entire code is run
        sequentially (but dask is still required for CBE!).
    dask_graph_path : string or None, optional, default None
        If a path (including a file ending matching a known image format, such
        as '.png') is specified as a string, a dask graph image is created that
        shows the constructed dask pipeline.
        Note: The resulting graph may get very large if many samples are used
        at the same time.
    profiling: bool, optional, default False
        If True, dask resource profiling is performed and visualized after the
        pipeline run is finished. This may generate a `profile.html` file in
        the working directory [bug in dask].
    verbose : bool, optional, default False
        If True, more information is printed.
    """

    #--------------------------------------------------------------------------

    ### Get a list of files to run

    # Function to select pairs of files (seg, dir) and create paths
    def prepare_fpaths(dirpath, fnames):

        # Find segmentation files
        seg_names = [fname for fname in fnames 
                     if fname.endswith(suffix_seg+".tif")]
        
        # Exclude files not in select_IDs
        if not select_IDs == 'all':
            seg_names = [fname for fname in seg_names
                         if any([fname.startswith(ID) for ID in select_IDs])]

        # Get IDs
        seg_IDs = [fname[:10] for fname in seg_names]

        # Get matching intensity files
        int_names = []
        for ID in seg_IDs:
            int_name = [fname for fname in fnames
                        if fname.startswith(ID)
                        and fname.endswith(suffix_int+".tif")]
            try:
                int_names.append(int_name[0])
            except IndexError:
                raise IOError("Could not find matching intensity file for "+
                              "segmentation file with ID "+ID)

        # Create path
        seg_paths = [os.path.join(dirpath, name) for name in seg_names]
        int_paths = [os.path.join(dirpath, name) for name in int_names]

        # Return results
        return [(seg_paths[i], int_paths[i]) for i in range(len(seg_paths))]

    # Remove .tif if it was specified with the suffix
    if suffix_seg.endswith(".tif"): suffix_seg = suffix_seg[:-4]
    if suffix_int.endswith(".tif"): suffix_int = suffix_int[:-4]

    # Run for single dir
    if not recurse:
        fnames = os.listdir(dirpath)
        fpaths = prepare_fpaths(dirpath, fnames)

    # Run for multiple subdirs
    if recurse:
        fpaths = []
        for dpath, _, fnames in os.walk(dirpath):
            fpaths += prepare_fpaths(dpath, fnames)

    # Test if all samples in select_IDs are present
    if not select_IDs == 'all':
        fpaths_IDs = [os.path.split(fp[0])[1][:10] for fp in fpaths]
        orphan_IDs = [ID for ID in select_IDs if ID not in fpaths_IDs]
        if any(orphan_IDs):
            warn("No matching files found for some of the IDs in select_IDs: "+
                 ", ".join(orphan_IDs))

    # Check
    if len(fpaths) == 0:
        raise IOError("No matching files found in target directory.")

    # Handle processes
    if processes is None:
        processes = cpu_count() // 2

    # More checks
    if not compute_TFOR and not compute_CFOR:
        raise IOError("At least one of compute_TFOR or compute_CFOR must be "+
                      "set to True.")

    # Report
    if verbose:
        print "Detected", len(fpaths), "target file pairs."


    #--------------------------------------------------------------------------

    ### Prepare kwargs for landmark assignment

    # Default kwargs for landmark assignment
    la_kwargs = dict()
    la_kwargs['save_centroids']       = True
    la_kwargs['fpath_out']            = None
    la_kwargs['show_cells']           = None
    la_kwargs['verbose']              = False
    la_kwargs['global_prep_func']     = None
    la_kwargs['global_prep_params']   = None
    la_kwargs['local_prep_func']      = None
    la_kwargs['local_prep_params']    = None
    la_kwargs['landmark_func']        = 'default'
    la_kwargs['landmark_func_params'] = None

    # User-specified kwargs for landmark assignment
    if assign_landmarks_kwargs != 'default':
        for kw in assign_landmarks_kwargs.keys():
            la_kwargs[kw] = assign_landmarks_kwargs[kw]

    # Safety check
    if la_kwargs['fpath_out'] is not None:
        raise IOError("`assign_landmarks_kwargs['fpath_out']` must be set to "+
                      "`None`, otherwise files will overwrite each other.")


    #--------------------------------------------------------------------------

    ### Prepare kwargs for TFOR transformation

    # Default kwargs for transformation to TFOR
    TFOR_kwargs = dict()
    TFOR_kwargs['n_points'] = 3000
    TFOR_kwargs['verbose']  = False
    TFOR_kwargs['show']     = False

    # User-specified kwargs for TFOR
    if transform_to_TFOR_kwargs != 'default':
        for kw in transform_to_TFOR_kwargs.keys():
            TFOR_kwargs[kw] = transform_to_TFOR_kwargs[kw]

    # Safety check
    if not compute_TFOR and transform_to_TFOR_kwargs is not 'default':
        warn("Non-default kwargs were passed for transformation to TFOR but "+
             "compute_TFOR is set to False!")


    #--------------------------------------------------------------------------

    ### Prepare args for CBE

    # Handle differing clustering inputs for TFOR and CFOR
    if type(clustering[0])==tuple:
        clustering_TFOR = (clustering[0][0], clustering[1][0])
        clustering_cfor = (clustering[0][1], clustering[1][1])
    else:
        clustering_TFOR = clustering_cfor = clustering


    #--------------------------------------------------------------------------

    ### Prepare kwargs for CBE on TFOR

    # Default kwargs for CBE
    cbe_TFOR_kwargs = dict()
    cbe_TFOR_kwargs['normalize_vol']        = None
    cbe_TFOR_kwargs['presample']            = None
    cbe_TFOR_kwargs['cfor']                 = None
    cbe_TFOR_kwargs['standardize']          = False
    cbe_TFOR_kwargs['custom_feature_funcs'] = None
    cbe_TFOR_kwargs['dask_graph_path']      = None
    cbe_TFOR_kwargs['processes']            = processes
    cbe_TFOR_kwargs['profiling']            = False
    cbe_TFOR_kwargs['suffix_out']           = {'META':suffix_int}
    cbe_TFOR_kwargs['save_metadata']        = True
    cbe_TFOR_kwargs['save_presampled']      = False
    cbe_TFOR_kwargs['save_cfor']            = False
    cbe_TFOR_kwargs['verbose']              = False

    # User-specified kwargs for CBE
    if perform_CBE_TFOR_kwargs != 'default':
        for kw in perform_CBE_TFOR_kwargs.keys():
            cbe_TFOR_kwargs[kw] = perform_CBE_TFOR_kwargs[kw]


    #--------------------------------------------------------------------------

    ### Prepare kwargs for CBE on CFOR

    # Default kwargs for CBE
    cbe_cfor_kwargs = dict()
    cbe_cfor_kwargs['normalize_vol']        = True
    cbe_cfor_kwargs['presample']            = None
    cbe_cfor_kwargs['cfor']                 = ('PD', 3)
    cbe_cfor_kwargs['standardize']          = True
    cbe_cfor_kwargs['custom_feature_funcs'] = None
    cbe_cfor_kwargs['dask_graph_path']      = None
    cbe_cfor_kwargs['processes']            = processes
    cbe_cfor_kwargs['profiling']            = False
    cbe_cfor_kwargs['suffix_out']           = {'META':suffix_int}
    cbe_cfor_kwargs['save_metadata']        = True
    cbe_cfor_kwargs['save_presampled']      = False
    cbe_cfor_kwargs['save_cfor']            = True
    cbe_cfor_kwargs['verbose']              = False

    # User-specified kwargs for CBE
    if perform_CBE_CFOR_kwargs != 'default':
        for kw in perform_CBE_CFOR_kwargs.keys():
            cbe_cfor_kwargs[kw] = perform_CBE_CFOR_kwargs[kw]


    #--------------------------------------------------------------------------

    ### If desired: run sequentially

    if processes == 1:

        if verbose: print "Processing target file pairs sequentially..."

        # Landmark extraction
        if verbose: print "--Assigning landmarks..."
        fpaths_lm = []
        for seg_path, int_path in fpaths:
            assign_landmarks(seg_path, int_path, num_LMs,
                             **la_kwargs)
            fpaths_lm.append( (seg_path, int_path[:-4]+"_LMs.npy") )

        # Computing the TFOR and performing CBE on TFOR
        if compute_TFOR:

            # Run the transformation to TFOR
            if verbose: print "--Transforming to TFOR..."
            fpaths_TFOR = []
            for seg_path, lm_path in fpaths_lm:
                transform_to_TFOR(seg_path, lm_path, **TFOR_kwargs)
                fpaths_TFOR.append( lm_path[:-4]+"_TFOR.npy" )

            # Performing CBE on TFOR
            if verbose: print "--Performing CBE on TFOR..."
            cbe(fpaths_TFOR, downsample, clustering_TFOR, features,
                **cbe_TFOR_kwargs)

        # Performing CBE on CFOR
        if compute_CFOR:
            if verbose: print "--Performing CBE on CFOR..."
            lm_paths = [fpath[1] for fpath in fpaths_lm]
            cbe(lm_paths, downsample, clustering_cfor, features,
                **cbe_cfor_kwargs)

        # Done
        if verbose: print "Processing complete!"
        return


    #--------------------------------------------------------------------------

    ### Prepare dask dict

    dask_graph = dict()

    # For each input...
    fpaths_lm   = []
    fpaths_TFOR = []
    for idx,fpath in enumerate(fpaths):

        # Landmark extraction nodes
        seg_path, int_path = fpath
        asgn_lms = partial(assign_landmarks, **la_kwargs)
        dask_graph["asgn_lms_%i" % idx] = (asgn_lms, seg_path, int_path,
                                           num_LMs)
        lm_path = int_path[:-4]+"_LMs.npy"
        fpaths_lm.append(lm_path)

        # Transform to TFOR
        if compute_TFOR:

            # Transform to TFOR
            tf2TFOR = partial(transform_to_TFOR, **TFOR_kwargs)
            tf2TFOR_await = lambda _, s, lmp  : tf2TFOR(s, lmp)
            dask_graph["tf2TFOR_%i" % idx] = (tf2TFOR_await,
                                              "asgn_lms_%i" % idx,
                                              seg_path, lm_path)
            fpaths_TFOR.append( lm_path[:-4]+"_TFOR.npy" )

    # Perform CBE on TFOR
    if compute_TFOR:
        cbe_TFOR = partial(cbe, **cbe_TFOR_kwargs)
        cbe_TFOR_await = lambda _, lmp, ds, cl, fe : cbe_TFOR(lmp, ds, cl, fe)
        dask_graph["CBE_TFOR"] = (cbe_TFOR_await,
                                  ["tf2TFOR_%i" % idx
                                   for idx in range(len(fpaths))],
                                  fpaths_TFOR, downsample, clustering_TFOR,
                                  features)

    # Perform CBE on CFOR
    if compute_CFOR:

        cbe_cfor = partial(cbe, **cbe_cfor_kwargs)
        cbe_cfor_await = lambda _, lmp, ds, cl, fe : cbe_cfor(lmp, ds, cl, fe)

        # Don't parallelize CBEs; wait for TFOR-CBE to finish
        if compute_TFOR:
            dask_graph["CBE_CFOR"] = (cbe_cfor_await, "CBE_TFOR",
                                      fpaths_lm, downsample, clustering_cfor,
                                      features)
        else:
            dask_graph["CBE_CFOR"] = (cbe_cfor_await,
                                      ["asgn_lms_%i" % idx
                                       for idx in range(len(fpaths))],
                                      fpaths_lm, downsample, clustering_cfor,
                                      features)

    # Create dask graph
    if dask_graph_path is not None:
        from dask.dot import dot_graph
        dot_graph(dask_graph, filename=dask_graph_path)


    #--------------------------------------------------------------------------

    ### Run in parallel (with dask)

    # Report
    if verbose: print "Processing target file pairs in parallel..."

    # Set number of threads
    dask.set_options(pool=ThreadPool(processes))

    # Run the pipeline (no profiling)
    if not profiling:
        if compute_CFOR:
            with ProgressBar(dt=1):
                dask.threaded.get(dask_graph, 'CBE_CFOR')
        else:
            with ProgressBar(dt=1):
                dask.threaded.get(dask_graph, 'CBE_TFOR')

    # Run the pipeline (with resource profiling)
    if profiling:
        if compute_CFOR:
            with ProgressBar(dt=1):
                with Profiler() as prof, ResourceProfiler(dt=0.1) as rprof:
                    dask.threaded.get(dask_graph, 'CBE_CFOR')
                visualize([prof,rprof], save=False)
        else:
            with ProgressBar(dt=1):
                with Profiler() as prof, ResourceProfiler(dt=0.1) as rprof:
                    dask.threaded.get(dask_graph, 'CBE_TFOR')
                visualize([prof,rprof], save=False)

    # Report and return
    if verbose: print "Processing complete!"
    return


#------------------------------------------------------------------------------



