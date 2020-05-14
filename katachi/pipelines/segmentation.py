# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 22:46:31 2017

@author:    Jonas Hartmann @ Gilmour group @ EMBL Heidelberg

@descript:  Dask pipeline for 3D single-cell segmentation of single cells from
            a membrane-labeled tissue.

            Steps included:
                - Linear unmixing (optional)
                - 3D single-cell segmentation

@usage:     Developed and tested for segmentation on membranous Lyn:EGFP in
            high-quality 3D stacks stacks of the zebrafish lateral line
            primordium acquired at the Zeiss LSM880.
"""

#------------------------------------------------------------------------------

# IMPORTS

# External
from __future__ import division
import os
from warnings import warn, catch_warnings, simplefilter
from multiprocessing.pool import ThreadPool, cpu_count
import dask
from dask.diagnostics import ProgressBar
from dask.diagnostics import Profiler, ResourceProfiler, visualize

# Internal
from katachi.tools.linearly_unmix import unmix_linear, unmix_linear_legacy
from katachi.tools.segment import segment_3D, segment_3D_legacy


#------------------------------------------------------------------------------

# FUNCTION: RUN FULL SEGMENTATION FOR A DIRECTORY

def full_segmentation(dirpath, channel, IDs=None, lin_unmix=False, 
                      recurse=False, ignore_old=True, fname_prefix=None,
                      processes=None, subprocesses=1,
                      profiling=False, verbose=False,
                      unmix_params=(0.0, 1.0, 20),
                      segment_params={'median_size'     : 3,
                                      'gaussian_sigma'  : 3,
                                      'max_offset'      : 10,
                                      'offset_step'     : 1,
                                      'clean_small'     : 1000,
                                      'clean_big'       : 1000000,
                                      'expansion_sigma' : 3} ,
                      use_legacy_unmix=False,
                      use_legacy_seg=False):
    """Segment single cells from 3D stacks of membrane-labeled tissues.

    This is a dask pipeline that applies linear unmixing (optional) from
    `katachi.tools.linearly_unmix` and 3D single-cell segmentation from
    `katachi.tools.segment` to a dataset that has previously been initialized
    using `katachi.pipelines.initialization`.

    WARNING: The approach used here has been developed for the Zebrafish
    posterior lateral line primordium. It is likely not readily applicable to
    other tissues!

    Parameters
    ----------
    dirpath : string
        The path (either local from cwd or global) to the directory with the
        input data to be processed.
    channel : string
        The channel to be used for segmentation.
    IDs : list of strings or None, optional, default None
        If None, all files found within dirpath that have the `channel` suffix
        are processed. If a list of strings (IDs) is given, only files with one
        of the given IDs as prefix are processed.
    lin_unmix : string or False, optional, default False
        If a string is given, linear unmixing will be performed, otherwise not.
        The string must be the channel designation of the 'contaminant'.
    recurse : bool, optional, default False
        If True, files are searched recursively in the subdirs of fpath.
    ignore_old : bool, optional, default True
        If True, files that already have a matching segmentation in the same
        directory will be ignored.
    fname_prefix : str or None, optional
        If not None, only file names that start with the given string are used.
    processes : int or None, optional
        Number of processes dask may use for parallel processing. If None, half
        of the available CPUs are used. If set to 1, the entire code is run
        sequentially (dask is not used).
    subprocesses : int, optional, default 1
        Number of processes that can be spawned for multiprocessing during
        linear unmixing. IMPORTANT: Note that the total number of processes
        running can reach up to `processes * subprocesses`! The default (1)
        runs sequentially (no multiprocessing code is used).
    profiling: bool, optional, default False
        If True, dask resource profiling is performed and visualized after the
        pipeline run is finished. This may generate a `profile.html` file in
        the working directory [bug in dask].
    verbose : bool, optional, default False
        If True, more information is printed.
    unmix_params: tuple, optional, default (0.0, 1.0, 20)
        Parameters for linear unmixing. For the default approach, it is simply
        the `a_range` tuple, which designates the values of `a` to be scanned
        in the form `(start, stop, n_steps)`. For more information, see
        `katachi.tools.linearly_unmix.unmix_linear`.
        For the legacy approach, the `unmix_params` tuple instead contains
        `(a_range, thresh)`. For more information, see
        `katachi.tools.linearly_unmix.unmix_linear_legacy`.
    segment_params : dict, optional
        Dict specifying parameters for segmentation. For more information see
        `katachi.tools.segment.segment_3D`.
    use_legacy_unmix : bool, optional, default False
        If True, the old parametric approach is used instead of the new one.
        Note that this requires adjustment of the unmix_params.
        Running in this mode triggers a DeprecationWarning.
    use_legacy_seg : bool, optional, default False
        If True, the old segmentation pipeline is used instead of the new one.
        Note that this requires adjustment of the segment_params.
        Running in this mode triggers a DeprecationWarning.
    """

    #--------------------------------------------------------------------------

    ### Get a list of files to run

    # Function to select file names and create paths
    def prepare_fpaths(fpath, fnames):

        # Select correct channel
        channel_fnames = [fname for fname in fnames
                          if fname.endswith(channel+".tif")]

        # Ignore files that have already been segmented
        if ignore_old:
            lu = ""
            if lin_unmix: lu = "_linUnmix"
            channel_fnames = [fname for fname in channel_fnames
                              if fname[:-4]+lu+"_seg.tif" not in fnames]
            
        # Ignore channels that don't match any of the given IDs
        if IDs is not None:
            channel_fnames = [fname for fname in channel_fnames if
                              any([fname.startswith(ID) for ID in IDs])]

        # Ignore channels with the wrong prefix
        if fname_prefix:
            channel_fnames = [fname for fname in channel_fnames
                              if fname.startswith(fname_prefix)]
            
        # Create full paths
        fpaths = [os.path.join(fpath, fname) for fname in channel_fnames]

        # Return results
        return fpaths

    # Clean channel if specified with file ending
    if channel.endswith(".tif"):
        channel = channel[:-4]

    # Run for single dir
    if not recurse:
        fnames = os.listdir(dirpath)
        fpaths = prepare_fpaths(dirpath, fnames)

    # Run for multiple subdirs
    if recurse:
        fpaths = []
        for dpath, _, fnames in os.walk(dirpath):
            fpaths += prepare_fpaths(dpath, fnames)

    # Check
    if len(fpaths) == 0 and ignore_old:
        with catch_warnings():
            simplefilter("always")
            warn("No matching files found in target directory! Doing nothing!"+
                 " Could be that all matching files have already been"+
                 " processed and are ignored now because `ignore_old=True`!")
        return
    elif len(fpaths) == 0:
        raise IOError("No matching files found in target directory.")

    # Check for linear unmixing files
    if lin_unmix:

        for fpath in fpaths:
            if not os.path.isfile(fpath.replace(channel, lin_unmix)):
                raise IOError("File(s) for the contaminant channel '" +
                              lin_unmix + "' for linear unmixing not found. ")

    # Warn about use of deprecated approaches
    if use_legacy_unmix:
        warn("Using legacy linear unmixing is deprecated!", DeprecationWarning)
    if use_legacy_seg:
        warn("Using legacy segmentation is deprecated!", DeprecationWarning)

    # Report
    if verbose:
        print "-- Detected", len(fpaths), "target files."


    #--------------------------------------------------------------------------

    ### If desired: run sequentially (does not use dask/multiprocessing)

    if processes == 1:

        if verbose: print "Processing target files sequentially..."

        if lin_unmix:
            if verbose: print "--Unmixing..."
            for fi,fpath in enumerate(fpaths):
                fpath_conta = fpath.replace(channel, lin_unmix)
                if not use_legacy_unmix:
                    unmix_linear(fpath, fpath_conta, subprocesses,
                                 unmix_params)
                else:
                    unmix_linear_legacy(fpath, fpath_conta, subprocesses,
                                        unmix_params[0], unmix_params[1])
                fpaths[fi] = fpath[:-4] + "_linUnmix.tif"

        if verbose: print "--Segmenting..."
        for fpath in fpaths:
            if not use_legacy_seg:
                segment_3D(fpath, params=segment_params)
            else:
                segment_3D_legacy(fpath, params=segment_params)

        if verbose: print "Processing complete!"
        return


    #--------------------------------------------------------------------------

    ### Prepare dask dict

    if verbose: print "Processing target files in parallel..."

    dask_graph = dict()

    # With linear unmixing
    if lin_unmix:

        # Wrapper to enable waiting for unmixing before segmentation is started
        def await_linear_unmixing(fpath, fpath_conta, subprocesses,
                                  unmix_params):
            if not use_legacy_unmix:
                unmix_linear(fpath, fpath_conta, subprocesses, unmix_params)
            else:
                unmix_linear_legacy(fpath, fpath_conta, subprocesses,
                                    unmix_params[0], unmix_params[1])
            return fpath[:-4] + "_linUnmix.tif"

        # Unmixing
        for fi, fpath in enumerate(fpaths):
            fpath_conta = fpath.replace(channel, lin_unmix)
            dask_graph["unmix_%i" % fi] = (await_linear_unmixing,
                                           fpath, fpath_conta, subprocesses,
                                           unmix_params)
            fpaths[fi] = fpath[:-4] + "_linUnmix.tif"

        # Segmentation
        for fi, fpath in enumerate(fpaths):
            if not use_legacy_seg:
                dask_graph["segment_%i" % fi] = (segment_3D, "unmix_%i" % fi,
                                                 False, segment_params)
            else:
                dask_graph["segment_%i" % fi] = (segment_3D_legacy,
                                                 "unmix_%i" % fi,
                                                 False, segment_params)

    # Without linear unmixing
    else:
        for fi, fpath in enumerate(fpaths):
            dask_graph["segment_%i" % fi] = (segment_3D, fpath,
                                             False, segment_params)

    # Collecting the results
    for fpath in fpaths:
        dask_graph['done'] = (lambda x : "done",
                              ["segment_%i" % fi for fi in range(len(fpaths))])


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



