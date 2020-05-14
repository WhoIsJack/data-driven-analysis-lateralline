# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 17:26:25 2018

@author:    Jonas Hartmann @ Gilmour group @ EMBL Heidelberg

@descript:  Dask pipeline for construction of a mapped feature space atlas from
            a reference channel and a secondary channel.

@note:      Dask is used simply for consistency in this case; it has no real
            practical use for this particular pipeline.
"""

#------------------------------------------------------------------------------

# IMPORTS

# External
from __future__ import division
import os
from multiprocessing.pool import ThreadPool, cpu_count
import dask
from dask.diagnostics import ProgressBar
from dask.diagnostics import Profiler, ResourceProfiler, visualize

# Internal
from katachi.tools.predict_atlas import predict_atlas


#------------------------------------------------------------------------------

# FUNCTION: RUN ATLAS PREDICTION FOR TWO GIVEN CHANNELS

def atlas_construction(train_dirpath, predict_dirpath,
                       ref_channel, sec_channel,
                       recurse=False, ignore_self=True, ignore_old=False,
                       train_IDs=None, predict_IDs=None,
                       processes=None, profiling=False, verbose=False,
                       outlier_removal_ref = None,
                       outlier_removal_sec = None,
                       outlier_removal_cov = None,
                       covariates_to_use   = None,
                       regressor = 'MO-SVR',
                       outlier_params_ref = {},
                       outlier_params_sec = {},
                       outlier_params_cov = {},
                       regressor_params   = { 'kernel'  : 'rbf'},
                       atlas_params       = { 'zscore_X'       : False,
                                              'zscore_y'       : False,
                                              'pca_X'          : False,
                                              'pca_y'          : False,
                                              'rezscore_X'     : False,
                                              'rezscore_y'     : False,
                                              'subselect_X'    : None,
                                              'subselect_y'    : None,
                                              'add_covariates' : None }):
    """Predict a secondary channel's feature space based on a reference channel
    through regression fitted on appropriate training data.

    This is a dask pipeline that applies atlas prediction from
    `katachi.tools.predict_atlas` to feature space datasets constructed with
    `katachi.pipelines.feature_extraction`.

    Parameters
    ----------
    train_dirpath : string
        The path (either local from cwd or global) to the directory with the
        training data on which the model will be fitted.
    predict_dirpath : string
        The path (either local from cwd or global) to the directory with the
        reference dataset for which a prediction will be constructed.
    ref_channel : string or list of strings
        The channel to be used as reference (usually the shape space),
        including the full processing suffix, or a list of multiple such
        suffices.
        Example: 'seg_LMs_TFOR_SUBS_CBE'
    sec_channel : string or list of strings
        The channel for which a prediction is made, including the full
        processing suffix, or a list of multiple such suffices.
        Example: 'NLStdTomato_LMs_TFOR_SUBS_CBE'
    recurse : bool, optional, default False
        If True, files are searched recursively in the subdirs of fpath.
    ignore_self : bool, optional, default True
        If True, predictions are not run for primordia used in training.
    ignore_old : bool, optional, default False
        If True, files that already have a matching prediction in the same
        directory will be ignored.
        WARNING: This feature has not been implemented for this pipeline!
    train_IDs : list of strings or None
        If None, all matching files in `train_dirpath` are used for training.
        If a list of strings (IDs), only the samples matching the IDs are used.
    predict_IDs : list of strings or None
        If None, all matching files in `predict_dirpath` are used for
        prediction. If a list of strings (IDs), only the samples matching the 
        IDs are used.
    processes : int or None, optional, default None
        Number of processes available for use during multi-processed model
        fitting and prediction. Works for 'MO-SVR' and 'MT-Lasso' regressors.
        WARNING: The 'MLP' regressor also performs multi-processing but does
        not seem to support an n_jobs argument...
        If None, half of the available CPUs are used.
        If set to 1, the code is run without use of dask.
    profiling: bool, optional, default False
        If True, dask resource profiling is performed and visualized after the
        pipeline run is finished. This may generate a `profile.html` file in
        the working directory [bug in dask].
    verbose : bool, optional, default False
        If True, more information is printed.
    outlier_removal_ref : string or None, default None
        If None, no outlier removal is done on reference feature spaces.
        Otherwise this must be a string denoting the method for outlier removal
        (one of `absolute_thresh`, `percentile_thresh`,
        `merged_percentile_thresh` or `isolation_forest`). Note that outlier
        removal is only done on training data, not on prediction data.
        See katachi.utilities.outlier_removal.RemoveOutliers for more info.
    outlier_removal_sec : string or None, default None
        If None, no outlier removal is done on the target feature spaces.
        Otherwise this must be a string denoting the method for outlier removal
        (see outlier_removal_ref above).
    outlier_removal_cov : string or None, optional, default None
        If None, no outlier removal is done based on covariate information.
        Otherwise this must be a string denoting the method for outlier removal
        (see outlier_removal_ref above).
    covariates_to_use : string, list of strings or None, optional, default None
        A string denoting the selection tree to select a covariate to be used
        for outlier detection from the HierarchicalData covariate object. Can
        also be a list of multiple such strings, in which case the covariates
        are merged into an fspace. The specified covariates must each be single
        numeric columns.
    regressor : string or sklearn regressor instance, optional, default 'MO-SVR'
        Regressor to be used in the atlas pipeline for prediction. If a string,
        must be one of 'MO-SVR', 'MT-ENetCV', 'MT-Lasso' or 'MLP'. See doc 
        string of katachi.tools.predict_atlas.predict_atlas for more info.
    outlier_params_ref : dict, optional, default {}
        kwarg dictionary for the chosen outlier removal method to be applied
        to the reference feature space.
        See katachi.utilities.outlier_removal.RemoveOutliers for more info.
    outlier_params_sec : dict, optional, {}
        kwarg dictionary for the chosen outlier removal method to be applied
        to the target feature space.
    outlier_params_cov : dict, optional, default {}
        kwarg dictionary for the chosen outlier removal method to be applied
        to the covariates. There default is to fall back to the defaults of
        katachi.utilities.outlier_removal.RemoveOutliers.
    regressor_params : dict, optional, default is a standard RBF MO-SVR
        dictionary for the chosen regressor's instantiation. See doc string of
        katachi.tools.predict_atlas.predict_atlas function for more info.
    pipeline_params : dict, optional, default are default settings
        kwarg dictionary for AtlasPipeline instantiation. See doc string of
        katachi.tools.predict_atlas.predict_atlas function for more info.
    """

    #--------------------------------------------------------------------------

    ### Construct lists of files to include

    if verbose: print "Retrieving matching datasets..."
    
    def prepare_fpaths(fpath, fnames, channel, IDs):
        
        # Select only files matching the IDs
        if IDs is not None:
            fnames = [fname for fname in fnames
                      if any([fname.startswith(ID) for ID in IDs])]
        
        # Select correct file names
        channel_fnames = [fname for fname in fnames
                          if any([fname.endswith(c+".npy") for c in channel])]

        # Ignore files that already have predictions
        if ignore_old:
            raise NotImplementedError("This would be annoying to implement "+
                                      "and likely won't ever be needed.")

        # Create full paths
        fpaths = [os.path.join(fpath, fname) for fname in channel_fnames]

        # Return results
        return fpaths

    # Handle single channel suffices
    if type(ref_channel) == str:
        ref_channel = [ref_channel]
    if type(sec_channel) == str:
        sec_channel = [sec_channel]

    # Clean channel if specified with file ending
    ref_channel = [rc[:-4] if rc.endswith(".npy") else rc
                   for rc in ref_channel]
    sec_channel = [sc[:-4] if sc.endswith(".npy") else sc
                   for sc in sec_channel]

    # Run for single dir
    if not recurse:

        # Get training data
        fnames = os.listdir(train_dirpath)
        fpaths_ref_train = prepare_fpaths(train_dirpath, fnames, 
                                          ref_channel, train_IDs)
        fpaths_sec_train = prepare_fpaths(train_dirpath, fnames, 
                                          sec_channel, train_IDs)

        # Get prediction data
        fnames = os.listdir(predict_dirpath)
        fpaths_ref_predict = prepare_fpaths(predict_dirpath, fnames,
                                            ref_channel, predict_IDs)

    # Run for multiple subdirs
    if recurse:

        # Get training data
        fpaths_ref_train = []
        fpaths_sec_train = []
        for dpath, _, fnames in os.walk(train_dirpath):
            fpaths_ref_train += prepare_fpaths(dpath, fnames, 
                                               ref_channel, train_IDs)
            fpaths_sec_train += prepare_fpaths(dpath, fnames, 
                                               sec_channel, train_IDs)

        # Get prediction data
        fpaths_ref_predict = []
        for dpath, _, fnames in os.walk(predict_dirpath):
            fpaths_ref_predict += prepare_fpaths(dpath, fnames, 
                                                 ref_channel, predict_IDs)

    # Remove training data from prediction data
    if ignore_self:
        fpaths_ref_predict = [f for f in fpaths_ref_predict
                              if not f in fpaths_ref_train]

    # Check
    if len(fpaths_ref_train) == 0:
        raise IOError("No reference files found in training directory.")
    if len(fpaths_sec_train) == 0:
        raise IOError("No secondary files found in training directory.")
    if len(fpaths_ref_predict) == 0:
        raise IOError("No reference files found in prediction directory.")
    if not len(fpaths_ref_train) == len(fpaths_sec_train):
        raise IOError("Found unequal number of reference and secondary" +
                      " files in the traning directory." +
                      " Ref files: "+str(len(fpaths_ref_train)) +
                      " Sec files: "+str(len(fpaths_sec_train)) )

    # Report
    if verbose:
        print "-- Detected", len(fpaths_ref_train), "training file pairs."
        print "-- Detected", len(fpaths_ref_predict), "prediction files."


    #--------------------------------------------------------------------------

    ### If desired: run 'sequentially' (without dask)

    if processes == 1:

        if verbose: print "Running pipeline without multiprocessing/dask..."

        predict_atlas(fpaths_ref_train, fpaths_sec_train,
                      fpaths_ref_predict,
                      outlier_removal_ref=outlier_removal_ref,
                      outlier_removal_sec=outlier_removal_sec,
                      outlier_removal_cov=outlier_removal_cov,
                      covariates_to_use=covariates_to_use,
                      regressor=regressor, n_jobs=1,
                      save_predictions=True, save_pipeline=True,
                      verbose=False,
                      outlier_options_ref=outlier_params_ref,
                      outlier_options_sec=outlier_params_sec,
                      outlier_options_cov=outlier_params_cov,
                      regressor_options=regressor_params,
                      pipeline_options=atlas_params)

        if verbose: print "Processing complete!"
        return


    #--------------------------------------------------------------------------

    ### Prepare dask dict

    if verbose: print "Running pipeline with multiprocessing/dask..."

    # If necessary: choose number of threads (half of available cores)
    if processes is None:
        processes = cpu_count() // 2

    # Create the dask 'graph'
    dask_graph = {'done' : (predict_atlas, fpaths_ref_train, fpaths_sec_train,
                            fpaths_ref_predict, outlier_removal_ref,
                            outlier_removal_sec, outlier_removal_cov,
                            covariates_to_use, regressor, processes, True, True,
                            False, outlier_params_ref, outlier_params_sec,
                            outlier_params_cov, regressor_params, atlas_params)
                  }


    #--------------------------------------------------------------------------

    ### Run with dask

    # Run the pipeline (no profiling)
    if not profiling:
        with ProgressBar(dt=1):
            dask.get(dask_graph, 'done')

    # Run the pipeline (with resource profiling)
    if profiling:
        with ProgressBar(dt=1):
            with Profiler() as prof, ResourceProfiler(dt=0.1) as rprof:
                dask.get(dask_graph, 'done')
            visualize([prof,rprof], save=False)

    # Report and return
    if verbose: print "Processing complete!"
    return


#------------------------------------------------------------------------------



