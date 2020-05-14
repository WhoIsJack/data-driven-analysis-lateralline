# -*- coding: utf-8 -*-
"""
Created on Mon Jan 08 10:51:06 2018

@author:    Jonas Hartmann @ Gilmour group @ EMBL Heidelberg

@descript:  Dask pipeline to extract engineered features from 3D single-cell
            segmentations.
            
@note:      Originally, the "engineered features" were referred to as 
            "covariates". There are still various instances in the code
            that use this terminology.
"""

#------------------------------------------------------------------------------

# IMPORTS

# External
from __future__ import division
import os, pickle
from warnings import warn
import numpy as np
from tifffile import imread
from multiprocessing.pool import ThreadPool, cpu_count
import dask
from dask.diagnostics import ProgressBar
from dask.diagnostics import Profiler, ResourceProfiler, visualize

# Internal
import katachi.tools.get_image_covariates as gic
import katachi.tools.get_cloud_covariates as gcc
from katachi.utilities.hierarchical_data import HierarchicalData


#------------------------------------------------------------------------------

# FUNCTION: EXTRACT ALL ENGINEERED FEATURES

def feature_engineering(dirpath, channels, IDs=None,
                        recurse=False, overwrite_previous=False,
                        seg_channel="",
                        no_lms=False, no_tfor=False, no_cfor=False,
                        mem_d=3, M=8,
                        save_baselines=True, processes=None,
                        dask_graph_path=None, profiling=False, verbose=False):
    """Extract a series of measurements from segmented images and point clouds.

    This is a dask pipeline that runs the covariate extraction functions in
    `katachi.tools.get_image_covariates` & `katachi.tools.get_cloud_covariates`
    on datasets that have been initialized, segmented and feature-extracted
    using other katachi pipelines.

    WARNING: The approach used here has been developed for the Zebrafish
    posterior lateral line primordium. It is likely not readily applicable to
    other tissues!

    Parameters
    ----------
    dirpath : string
        The path (either local from cwd or global) to the directory with the
        input data to be processed.
    channels : list
        A list of channels from which to extract channel-specific covariates.
        For each channel, a tif file must be present that ends on
        `channel+".tif"` and a .npy file must be present that ends either on
        `channel+"_LMs_TFOR.npy"` (recommended) or on `channel+"_LMs.npy"`.
        The channels will be used as class attributes in the output object and
        therefore must not contain characters incompatible with this use.
    IDs : list of strings or None, optional, default None
        If a list of strings (IDs) is given, only samples within dirpath that
        match this ID will be processed.
    recurse : bool, optional, default False
        If True, files are searched recursively in the subdirs of dirpath.
    overwrite_previous : bool, optional, default False
        If True and a covariate file already exists for a given sample, that
        file will be deleted and a completely new file will be written in its
        place. If False and a covariate file already exists for a given sample,
        the new covariates will be added to it if they have a different name.
        For covariates with identical names, the new will overwrite the old.
    seg_channel : str or "", optional, default ""
        If for some reason the target directories are expected to contain more
        than one file that ends on "_seg.tif", seg_channel can be specified to
        identify the correct target file, which will have the form
        `<basename> + seg_channel + "_seg.tif"`.
        Note that having multiple segmentation files in one target directory is
        deprecated in general.
    no_lms : bool, optional, default False
        If True, it is expected that no landmark data is available. In this
        case, only image covariates are computed.
    no_tfor : bool, optional, default False
        If True, it is expected that no TFOR landmark data is available. In
        this case, untransformed landmarks are loaded and covariates depending
        on TFOR covariates are not computed (specifically pcl_covars_sample and
        pcl_covars_tissue).
    no_cfor : bool, optional, default False
        If True, the CFOR-based moments and baseline will not be computed and
        no CFOR data is required at any point.
    mem_d : int, optional, default 3
        Estimated diameter (in pixels) of the membrane region in the shell of a
        single cell. Used for extraction of intensity-based covariates.
    M : int, optional, default 8
        Highest-level moments to extract from point cloud. The moments array
        constructed will have shape (M+1,M+1,M+1).
    save_baselines : bool, optional, default True
        Whether to save the flattened moments arrays as feature space baselines
        in the form (N_cells, N_features), where N_features is length (M+1)**3.
        If True, two files are created for each channel, one for the base
        moments (usually TFOR, unless no_tfor is set to True or no TFOR data is
        available) and one for the PD-transformed (rotationally invariant) and
        volume-normalized cells, suffixed "_baseline.npy" and
        "_volnormPDbaseline.npy", respectively.
    processes : int or None, optional
        Number of processes dask may use for parallel processing. If None, half
        of the available CPUs are used. If set to 1, the entire code is run
        sequentially (dask is not used).
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

    if verbose: print "Retrieving matching datasets..."

    # Function to select suitable datasets and create paths
    def prepare_fpaths(fpath, fnames):
        
        # Keep only those in specified IDs
        if IDs is not None:
            fnames = [fname for fname in fnames
                      if any([fname.startswith(ID) for ID in IDs])]
        
        # Find the metadata file
        meta_file = None
        for fname in fnames:
            if fname.endswith("_stack_metadata.pkl"):
                meta_file = fname
                meta_path = os.path.join(fpath, meta_file)

        # Quit if no metadata file is found
        if meta_file is None:
            return None

        # Find segmentation file
        seg_file = [fname for fname in fnames
                    if fname.endswith(seg_channel+"_seg.tif")]

        # Handle failure cases
        if len(seg_file)==0:
            return None
        if len(seg_file)>1:
            raise IOError("More than one segmentation file (*_seg.tif) found "+
                          "in directory "+fpath+". Use seg_channel kwarg to "+
                          "specify which file to use.")
        else:
            seg_file = seg_file[0]
            seg_path = os.path.join(fpath, seg_file)

        # Find TFOR segmentation landmarks
        tfor_path = []
        if not no_tfor and not no_lms:

            # Search for the file
            tfor_file = [fname for fname in fnames
                         if fname.endswith(seg_channel+"_seg_LMs_TFOR.npy")]

            # Give up if nothing is found
            if len(tfor_file)==0:
                return None

            # Else keep the result
            tfor_file = tfor_file[0]
            tfor_path = os.path.join(fpath, tfor_file)

        # Find channel landmark files
        lm_paths = []
        if not no_lms:
            for channel in channels:

                # Search for TFOR landmarks
                if not no_tfor:
                    lm_file = [fname for fname in fnames
                               if fname.endswith(channel+"_LMs_TFOR.npy")]
                else:
                    lm_file = []

                # Search for non-TFOR landmarks
                if len(lm_file)==0:
                    lm_file = [fname for fname in fnames
                               if fname.endswith(channel+"_LMs.npy")]
                    if not no_tfor:
                        warn("No TFOR landmarks found for channel " + channel +
                             ". " + "Using standard landmarks.")

                # Give up if nothing is found
                if not lm_file:
                    return None

                # Else keep the result
                lm_file = lm_file[0]
                lm_path = os.path.join(fpath, lm_file)
                lm_paths.append(lm_path)

        # Find CFOR-transformed channel landmark files
        cfor_paths = []
        if not no_cfor and not no_lms:
            for channel in channels:

                # Get CFOR landmark paths
                cfor_file = [fname for fname in fnames
                             if channel in fname
                             and fname.endswith('CFOR.npy')][0]
                cfor_path = os.path.join(fpath, cfor_file)
                cfor_paths.append(cfor_path)

        # Find image files
        img_paths = []
        for channel in channels:

            # Search for image files
            img_file = [fname for fname in fnames
                        if fname.endswith(channel+".tif")]

            # Give up if nothing is found
            if not img_file:
                return None

            # Else keep the result
            img_file = img_file[0]
            img_path = os.path.join(fpath, img_file)
            img_paths.append(img_path)

        # Return the paths
        return {"meta_path"  : meta_path,
                "seg_path"   : seg_path,
                "tfor_path"  : tfor_path,
                "lm_paths"   : lm_paths,
                "img_paths"  : img_paths,
                "cfor_paths" : cfor_paths}

    # Run for single dir
    if not recurse:
        fnames = os.listdir(dirpath)
        all_paths = [prepare_fpaths(dirpath, fnames)]
        if all_paths is None:
            raise IOError("The specified path does not contain the required "+
                          "files (and recurse=False).")

    # Run for multiple subdirs
    if recurse:
        all_paths = []
        for dpath, _, fnames in os.walk(dirpath):
            fpaths = prepare_fpaths(dpath, fnames)
            if fpaths is not None:
                all_paths.append(fpaths)
        if not all_paths:
            raise IOError("Could not find any data directories containing "+
                          "all required files.")

    # Report
    if verbose: print "-- Retrieved", len(all_paths), "matching data sets."


    #--------------------------------------------------------------------------

    ### If desired: run sequentially (does not use dask/multiprocessing)

    if processes == 1:

        if verbose: print "Processing target files sequentially..."

        # For each dataset...
        for paths in all_paths:

            # Load previously generated covariates file (if available)
            has_previous = False
            if not overwrite_previous:
                mroot, mfile = os.path.split(paths["meta_path"])
                prevfpath = os.path.join(mroot, mfile[:10]+"_covariates.pkl")
                if os.path.isfile(prevfpath):
                    with open(prevfpath, 'rb') as prevfile:
                        covars = pickle.load(prevfile)
                    has_previous = True

            # Load data
            img_seg  = imread(paths["seg_path"])
            if not no_lms and not no_tfor:
                tfor_lms = np.load(paths["tfor_path"])
            with open(paths["meta_path"], 'rb') as metafile:
                meta_dict = pickle.load(metafile)

            # Extract image covariates
            covars = gic.get_img_covars_sample("_", img_seg=img_seg,
                                       covars=covars if has_previous else None)
            covars = gic.get_img_covars_tissue("_", img_seg=img_seg,
                                               covars=covars)
            covars = gic.get_img_covars_cell_seg("_", '_',
                                                 img_seg=img_seg,
                                                 metadata=meta_dict,
                                                 covars=covars)
            for c, channel in enumerate(channels):
                covars = gic.get_img_covars_cell_int("_",
                                                     paths["img_paths"][c],
                                                     channel, mem_d,
                                                     img_seg=img_seg,
                                                     covars=covars)

            # Extract point cloud covariates
            if not no_tfor and not no_lms:
                covars = gcc.get_pcl_covars_sample("_", "_",
                                                   tfor_lms=tfor_lms,
                                                   metadata=meta_dict,
                                                   covars=covars)
                covars = gcc.get_pcl_covars_tissue("_", "_",
                                                   tfor_lms=tfor_lms,
                                                   metadata=meta_dict,
                                                   covars=covars)
            if not no_lms:
                for c, channel in enumerate(channels):
                    covars = gcc.get_pcl_covars_cell(paths["lm_paths"][c],
                                         channel, M=M, no_cfor=no_cfor,
                                         fpath_lms_cfor=paths["cfor_paths"][c],
                                         covars=covars)

                # Saving the moments as a baseline feature space
                if save_baselines:

                    # Prep base path
                    bp = paths["lm_paths"][c][:-4]

                    # Save TFOR baseline
                    m = covars.pcl.cell._gad(channel).moments
                    np.save(bp+"_baseline.npy", m)

                    # Save CFOR baseline
                    if not no_cfor:
                        m = covars.pcl.cell._gad(channel).moments_cfor
                        np.save(bp+"_CFORbaseline.npy", m)

            # Saving the extracted covariates
            mroot, mfile = os.path.split(paths["meta_path"])
            outfpath = os.path.join(mroot, mfile[:10]+"_covariates.pkl")
            with open(outfpath, 'wb') as outfile:
                pickle.dump(covars, outfile, pickle.HIGHEST_PROTOCOL)

        # Report and return
        if verbose: print "Processing complete!"
        return


    #--------------------------------------------------------------------------

    ### Prepare dask dict
    # Note: This is slightly suboptimal because some datasets have to be
    #       reloaded multiple times. However, it seems difficult to solve this
    #       in a way that permits carrying them over.

    if verbose: print "Processing target files in parallel..."

    dask_graph = dict()

    # For each dataset...
    for idx, paths in enumerate(all_paths):

        # Getting previous covariates: function
        def get_previous_covariates(prevfpath):
            with open(prevfpath, 'rb') as prevfile:
                covars = pickle.load(prevfile)
            return covars

        # Get previous covars (if existing and desired)
        has_previous = False
        if not overwrite_previous:
            mroot, mfile = os.path.split(paths["meta_path"])
            prevfpath = os.path.join(mroot, mfile[:10]+"_covariates.pkl")
            if os.path.isfile(prevfpath):
                dask_graph['prev_covars_%i' % idx] = (get_previous_covariates,
                                                      prevfpath)
                has_previous = True

        # Extract image covariates
        dask_graph["img_sample_%i" % idx] = (gic.get_img_covars_sample,
                                             paths["seg_path"])
        dask_graph["img_tissue_%i" % idx] = (gic.get_img_covars_tissue,
                                             paths["seg_path"])
        dask_graph["img_cell_seg_%i" % idx] = (gic.get_img_covars_cell_seg,
                                               paths["seg_path"],
                                               paths["meta_path"])
        for c, channel in enumerate(channels):
            dask_graph["img_cell_int_%s_%i" % (channel, idx)] = (
                                                   gic.get_img_covars_cell_int,
                                                   paths["seg_path"],
                                                   paths["img_paths"][c],
                                                   channel, mem_d)

        # Extract point cloud covariates
        if not no_tfor and not no_lms:
            dask_graph["pcl_sample_%i" % idx] = (gcc.get_pcl_covars_sample,
                                                 paths["tfor_path"],
                                                 paths["meta_path"])
            dask_graph["pcl_tissue_%i" % idx] = (gcc.get_pcl_covars_tissue,
                                                 paths["tfor_path"],
                                                 paths["meta_path"])
        if not no_lms:
            for c, channel in enumerate(channels):
                dask_graph["pcl_cell_%s_%i" % (channel, idx)] = (
                                                       gcc.get_pcl_covars_cell,
                                                       paths["lm_paths"][c],
                                                       channel, M, no_cfor,
                                                       paths["cfor_paths"][c])

                # Saving the moments as a baseline feature space
                if save_baselines:

                    # Baseline saving function
                    def save_baseline(covars, channel, basepath, no_cfor):

                        # Save TFOR baseline
                        m = covars.pcl.cell._gad(channel).moments
                        np.save(basepath+"_baseline.npy", m)

                        # Save CFOR baseline
                        if not no_cfor:
                            m = covars.pcl.cell._gad(channel).moments_cfor
                            np.save(basepath+"_CFORbaseline.npy", m)

                        # Forward result
                        return covars

                    # Add to graph
                    basepath = paths["lm_paths"][c][:-4]
                    dask_graph["pcl_cell_blsave_%s_%i" % (channel, idx)] = (
                                                save_baseline,
                                                "pcl_cell_%s_%i" % (channel,
                                                                    idx),
                                                channel, basepath, no_cfor)

        # Merging the extracted covariates: function
        def merge_covariates(covars_list):
            covars = covars_list[0]
            for cv in covars_list[1:]:
                covars._merge(cv)
            return covars

        # Merging the extracted covariates: input name list construction
        covars_list  = ["img_sample_%i" % idx, "img_tissue_%i" % idx,
                        "img_cell_seg_%i" % idx]
        covars_list += ["img_cell_int_%s_%i" % (channel, idx)
                        for channel in channels]
        if not no_tfor and not no_lms:
            covars_list += ["pcl_sample_%i" % idx, "pcl_tissue_%i" % idx]
        if save_baselines and not no_lms:
            covars_list += ["pcl_cell_blsave_%s_%i" % (channel, idx)
                            for channel in channels]
        elif not no_lms:
            covars_list += ["pcl_cell_%s_%i" % (channel, idx)
                            for channel in channels]
        if has_previous:
            covars_list += ['prev_covars_%i' % idx]


        # Merging the extracted covariates: dask call
        dask_graph["merge_results_%i" % idx] = (merge_covariates, covars_list)

        # Saving the extracted covariates
        def save_covariates(covars, outfpath):
            with open(outfpath, 'wb') as outfile:
                pickle.dump(covars, outfile, pickle.HIGHEST_PROTOCOL)
        mroot, mfile = os.path.split(paths["meta_path"])
        outfpath = os.path.join(mroot, mfile[:10]+"_covariates.pkl")
        dask_graph["save_results_%i" % idx] = (save_covariates,
                                               "merge_results_%i" % idx,
                                               outfpath)

    # Collecting the results
    dask_graph['done'] = (lambda x : "done",
                          ["save_results_%i" % idx
                           for idx in range(len(all_paths))])

    # Saving the graph visualization
    if dask_graph_path is not None:
        from dask.dot import dot_graph
        dot_graph(dask_graph, filename=dask_graph_path)


    #--------------------------------------------------------------------------

    ### Run in parallel (with dask)

    # If necessary: choose number of threads (half of available cores)
    if processes is None:
        processes = cpu_count() // 2

    # Set number of threads
    dask.set_options(pool=ThreadPool(processes))

    # Run the pipeline (no profiling)
    if not profiling:
        with ProgressBar(dt=1):
            dask.threaded.get(dask_graph, 'done')

    # Run the pipeline (with resource profiling)
    if profiling:
        with ProgressBar(dt=1):
            with Profiler() as prof, ResourceProfiler(dt=0.1) as rprof:
                dask.threaded.get(dask_graph, 'done')
            visualize([prof,rprof], save=False)

    # Report and return
    if verbose: print "Processing complete!"
    return


#------------------------------------------------------------------------------



