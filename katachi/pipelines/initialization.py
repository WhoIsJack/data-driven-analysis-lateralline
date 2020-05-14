# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 22:25:36 2017

@author:    Jonas Hartmann @ Gilmour group @ EMBL Heidelberg

@descript:  Dask pipeline to initialize the data structure of a directory of 
            new image stacks. 
"""

#------------------------------------------------------------------------------

# IMPORTS

# External
from __future__ import division
import os, pickle
from multiprocessing.pool import ThreadPool, cpu_count
import dask
from dask.diagnostics import ProgressBar
from dask.diagnostics import Profiler, ResourceProfiler, visualize

# Internal
from katachi.tools.initialize import initialize_stack


#------------------------------------------------------------------------------

# FUNCTION: INITIALIZE FILES IN DIRECTORY

def initialize_dir(dirpath, idpath, meta_dict, recurse=False, IDR_data=False,
                   IDR_IDs=None, ignore_old=True, fname_prefix=None, 
                   fname_suffix=None, processes=None, profiling=False, 
                   verbose=False):
    """Intialize the data structure for a directory of new image stacks.
    
    This is a dask pipeline that applies the function `initialize_stack` from
    `katachi.tools.initialize` to an entire directory.
    
    See `katachi.tools.initialize.initialize_stack` for more information.
    
    Parameters
    ----------
    dirpath : string
        The path (either local from cwd or global) to the directory with the
        input data to be processed.
    idpath : string or None
        Path of the text file containing previously generated IDs.
        Necessary to ensure that newly generated IDs are unique.
    meta_dict : dict 
        A dictionary containing the initial (user-defined) metadata for the
        stack. See Notes below for the keys that must be included.
    recurse : bool, optional, default False
        If True, files are searched recursively in the subdirs of fpath.
        This is ignored if `IDR_data` is True, as recursing through subfolders
        is not supported on IDR data.
    IDR_data : bool, optional, default False
        If True, the data is expected to already be grouped into subdirectories
        named according to already assigned IDs, as this is how the data was
        deposited on the IDR database.
    IDR_IDs : list of IDs or None, optional, default None
        If IDR_data is True, a list of IDs can be passed to specify a subset of 
        samples for which this pipeline is to be run. 
    ignore_old : bool, optional, default True
        If True, files that already have a known ID listed in the ID file will
        be ignored. This is not supported for IDR data, so if IDR_data is True
        and ignore_old is True, an error is raised.
    fname_prefix : str or None, optional
        If not None, only file names that start with the given string are used.
    fname_suffix : str or None, optional
        If not None, only file names that end with the given string (or with
        the given string + .tif) are used.
    processes : int or None, optional
        Number of processes dask may use for parallel processing. If None, half
        of the available CPUs are used. If set to 1, the entire code is run
        sequentially (dask is not used).
    profiling: bool, optional, default False
        If True, dask resource profiling is performed and visualized after the
        pipeline run is finished. This may generate a `profile.html` file in
        the working directory [bug in dask]. 
    verbose : bool, optional, default False
        If True, more information is printed.
    
    Notes
    -----
    The meta_dict dictionary must contain the following entries:
    - 'channels'   : A list of strings naming the channels in order. Must not 
                     contain characters that cannot be used in file names.
    - 'resolution' : A list of floats denoting the voxel size of the input
                     stack in order ZYX.
    It may optionally contain other entries as well.
    """


    #--------------------------------------------------------------------------
    
    ### Get a list of files to run
    
    if verbose: print "Detecting target files..."
    
    # Function to select file names and create paths
    def get_fnames_ready(fnames, fpath, known_ids=None):
        
        fnames = fnames[:]
        
        fnames = [fname for fname in fnames if fname.endswith(".tif")]
        
        if ignore_old:
            fnames = [fname for fname in fnames
                      if not any([fname.startswith(ID) for ID in known_ids])] 
        
        if fname_prefix:
            fnames = [fname for fname in fnames 
                      if fname.startswith(fname_prefix)]
        if fname_suffix:
            fnames = [fname for fname in fnames
                      if fname.endswith(fname_suffix+".tif")
                      or fname.endswith(fname_suffix)]    
        
        fpaths = [os.path.join(fpath, fname) for fname in fnames]
        
        return fpaths
    
    # If this is run on IDR data, most of the work is already done!
    if IDR_data:
        
        # Handle inputs
        if ignore_old:
            raise IOError("`ignore_old` is not supported for IDR data. Be "+
                          "careful when running this so as to avoid over"+
                          "writing important metadata. Aborting for now; set "+
                          "`ignore_old` to False to prevent this error.")
        if IDR_IDs is None:
            IDR_IDs = [ID for ID in os.listdir(dirpath) if os.path.isdir(ID)
                       and len(ID)==10]
            
        # Write the metadata files; all else is already done
        if verbose: print "Creating metadata files for IDR data..."
        for ID in IDR_IDs:
            meta_path = os.path.join(dirpath, ID, ID+'_stack_metadata.pkl')
            with open(meta_path, 'wb') as outfile:
                pickle.dump(meta_dict, outfile, pickle.HIGHEST_PROTOCOL)  
        if verbose: print "Processing complete!"
        return
    
    # If needed, load previously generated IDs (to exclude those files)
    if ignore_old:
        try:
            with open(idpath,"r") as infile:
                known_ids = [line.strip() for line in infile.readlines()]
        except:
            print("Attempting to load existing IDs from id_file failed " +
                  "with this error:")
            raise
    else:
        known_ids = None
    
    # Run for single dir
    if not recurse:
        fnames = os.listdir(dirpath)
        fpaths = get_fnames_ready(fnames, dirpath, known_ids=known_ids)
    
    # Run for multiple subdirs
    if recurse:
        fpaths = []
        for dpath, _, fnames in os.walk(dirpath):
            fpaths += get_fnames_ready(fnames, dpath, known_ids)
    
    # Check
    if len(fpaths) == 0:
        raise IOError("No matching files found in target directory.")

    # Report
    if verbose:
        print "-- Detected", len(fpaths), "target files."


    #--------------------------------------------------------------------------

    ### If desired: run sequentially (does not use dask)

    if processes == 1:
        if verbose: print "Processing target files sequentially..."
        for fpath in fpaths:
            initialize_stack(fpath, idpath, meta_dict, verbose=False)
        if verbose: print "Processing complete!"
        return


    #--------------------------------------------------------------------------
    
    ### Prepare dask dict
    
    if verbose: print "Processing target files in parallel..."
    
    dask_graph = dict()
    for i, fpath in enumerate(fpaths):
        dask_graph["initialize_%i" % i] = (initialize_stack, 
                                           fpath, idpath, meta_dict, False)
    dask_graph['done'] = (lambda x : "done", 
                          ["initialize_%i" % i for i in range(len(fpaths))])


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



