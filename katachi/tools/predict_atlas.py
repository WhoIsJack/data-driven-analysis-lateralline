# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 16:17:14 2018

@author:   Jonas Hartmann @ Gilmour group @ EMBL Heidelberg

@descript: Training and prediction of secondary channel feature space positions
           of cells, based on a reference channel feature space (usually the
           shape space).
"""

#------------------------------------------------------------------------------

# IMPORTS

# External
from __future__ import division
import os, pickle, re
import numpy as np
from sklearn import preprocessing, decomposition
from sklearn import svm, linear_model, neural_network, multioutput, base

# Internal
from katachi.utilities.outlier_removal import RemoveOutliers


#------------------------------------------------------------------------------

### SKLEARN-LIKE PIPELINE: FEATURE SPACE MAPPING REGRESSOR WITH PREPROCESSING

class AtlasPipeline(base.BaseEstimator, base.RegressorMixin):
    """Pipeline for multivariate-multivariable regression for the prediction of
    a secondary channel feature space (e.g. nuclei) based on a reference
    feature space (usually the shape space). Includes a number of preprocessing
    options and handles similar to an sklearn estimator.

    WARNING: This may not work as a proper sklearn estimator!
             Beware of unexpected behaviors when using it that way!
    """

    #--------------------------------------------------------------------------

    def __init__(self, regressor,
                 zscore_X=False, zscore_y=False,
                 pca_X=False, pca_y=False,
                 rezscore_X=False, rezscore_y=False,
                 subselect_X=None, subselect_y=None,
                 add_covariates=None, verbose=False):
        """Initialize instance of AtlasPipeline.

        Parameters
        ----------
        regressor : sklearn regressor instance
            An instance of an sklearn(-like) multivariate multivariable
            regressor. Must support a method fit(X, y), where both X and y are
            2d arrays of shape (n_samples, n_features), where n_samples is
            identical for X and y, and the method predict(X) where X is a 2d
            array of shape (n_samples, n_features), where n_features is
            identifal to n_features in fit(X, y).
        zscore_X : bool, optional, default True
            Standardize features of X to zero mean and unit variance before
            fitting and prediction. The mean and standard deviation determined
            during fitting are considered parameters of the model and are used
            again for standardization during prediction.
        zscore_y : bool, optional, default False
            Standardize features of y to zero mean and unit variance before
            fitting and prediction. The mean and standard deviation determined
            during fitting are considered parameters of the model and are used
            again for standardization during prediction.
        pca_X : bool, optional, default True
            Transform features of X by PCA before fitting and prediction. The
            fitted PCA parameters are considered parameters of the model and
            are used again for PCA transformation during prediction.
        pca_y : bool, optional, default True
            Transform features of X by PCA before fitting and prediction. The
            fitted PCA parameters are considered parameters of the model and
            are used again for PCA transformation during prediction.
        rezscore_X : bool, optional, default False
            Standardize features back to zero mean and unit variance after the
            PCA. This is only applied if pca_X is True. The mean and standard
            deviation determined during fitting are considered parameters of
            the model and are used again for re-standardization after PCA
            during prediction.
        rezscore_y : bool, optional, default False
            Standardize features back to zero mean and unit variance after the
            PCA. This is only applied if pca_y is True. The mean and standard
            deviation determined during fitting are considered parameters of
            the model and are used again for re-standardization after PCA
            during prediction.
        subselect_X : int, index array, or None, optional, default None
            If an integer, only the first n features in subselect_X are used,
            that is X is set to X[:, :subselect_X]. This is useful to reject
            low-relevance features after a PCA.
            If an index array, only those features of X designated by the array
            are used, that is X is set to X[:, subselect_X]
            If None (default), all features of X are used.
        subselect_y : int, index array, or None, optional, default None
            If an integer, only the first n features in subselect_X are used,
            that is y is set to y[:, :subselect_y]. This is useful to reject
            low-relevance features after a PCA.
            If an index array, only those features of y designated by the array
            are used, that is y is set to y[:, subselect_y]
            If None (default), all features of X are used.
        add_covariates : list of strings, optional, default None
            List of strings denoting covariates in the covariate dictionary.
            These designated covariates are added to the source feature space
            (X) prior to preprocessing and fitting/prediction.
            WARNING: This feature is not implemented yet!
        verbose : bool, optional, default False
            If True, more information is printed.
        """

        self.regressor = regressor
        self.zscore_X = zscore_X
        self.zscore_y = zscore_y
        self.pca_X = pca_X
        self.pca_y = pca_y
        self.rezscore_X = rezscore_X
        self.rezscore_y = rezscore_y
        self.subselect_X = subselect_X
        self.subselect_y = subselect_y
        self.add_covariates = add_covariates
        self.verbose = verbose


    #--------------------------------------------------------------------------

    def fit(self, X, y):
        """Fit the Atlas Regressor model according to given training data.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Source feature space.
        y : array of shape (n_samples, n_target_features)
            Target feature space. n_samples must be the same as in X.

        Returns
        -------
        self

        """

        # Report
        if self.verbose:
            print "Preprocessing feature spaces..."

        # Standardization of source fspace
        if self.zscore_X:
            self.scaler_X = preprocessing.StandardScaler()
            self.scaler_X.fit(X)
            X = self.scaler_X.transform(X)

        # Standardization of target fspace
        if self.zscore_y:
            self.scaler_y = preprocessing.StandardScaler()
            self.scaler_y.fit(y)
            y = self.scaler_y.transform(y)

        # PCA of source fspace
        if self.pca_X:
            self.pcaobj_X = decomposition.PCA()
            self.pcaobj_X.fit(X)
            X = self.pcaobj_X.transform(X)

        # PCA of target fspace
        if self.pca_y:
            self.pcaobj_y = decomposition.PCA()
            self.pcaobj_y.fit(y)
            y = self.pcaobj_y.transform(y)

        # Restandardization of source fspace
        if self.pca_X and self.rezscore_X:
            self.rescaler_X = preprocessing.StandardScaler()
            self.rescaler_X.fit(X)
            X = self.rescaler_X.transform(X)

        # Restandardization of target fspace
        if self.pca_y and self.rezscore_y:
            self.rescaler_y = preprocessing.StandardScaler()
            self.rescaler_y.fit(y)
            y = self.rescaler_y.transform(y)

        # Subselect dimensions of source fspace
        if self.subselect_X is not None:

            # Subselect up to a given number of dimensions
            if type(self.subselect_X)==int:
                X = X[:, :self.subselect_X]

            # Subselect a set of dimensions
            else:
                X = X[:, self.subselect_X]

        # Subselect dimensions of target fspace
        if self.subselect_y is not None:

            # Subselect up to a given number of dimensions
            if type(self.subselect_y)==int:
                y = y[:, :self.subselect_y]

            # Subselect a set of dimensions
            else:
                y = y[:, self.subselect_y]

        # Add covariate information to source fspace
        if self.add_covariates is not None:
            raise NotImplementedError("Adding covariates wasn't much use "+
                                      "in early tests, so it's currently not "+
                                      "implemented.")

        # Report
        if self.verbose:
            print "Preprocessing complete:"
            print "  N:", X.shape[0], "samples"
            print "  X:", X.shape[1], "dimensions"
            print "  y:", y.shape[1], "dimensions"
            print "Fitting regressor..."

        # Perform the fit
        self.regressor.fit(X, y)

        # Report
        if self.verbose:
            print "Fitting complete."

        # Return self
        return self


    #--------------------------------------------------------------------------

    def predict(self, X):
        """Perform regression on samples in X.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Samples on which regression is to be performed. n_features must be
            the same as n_features in X during fitting.

        Returns
        -------
        y_pred: array of shape (n_samples, n_target_features)
            Predicted target feature space for the samples in X.
        """

        # Report
        if self.verbose:
            print "Preprocessing feature space..."

        # Standardization of source fspace
        if self.zscore_X:
            X = self.scaler_X.transform(X)

        # PCA of source fspace
        if self.pca_X:
            X = self.pcaobj_X.transform(X)

        # Restandardization of source fspace
        if self.pca_X and self.rezscore_X:
            X = self.rescaler_X.transform(X)

        # Subselect dimensions of source fspace
        if self.subselect_X is not None:

            # Subselect up to a given number of dimensions
            if type(self.subselect_X)==int:
                X = X[:, :self.subselect_X]

            # Subselect a set of dimensions
            else:
                X = X[:, self.subselect_X]

        # Add covariate information to source fspace
        if self.add_covariates is not None:
            raise NotImplementedError("Adding covariates wasn't much use "+
                                      "in early tests, so it's currently not "+
                                      "implemented.")

        # Report
        if self.verbose:
            print "Preprocessing complete:"
            print "  N:", X.shape[0], "samples"
            print "  X:", X.shape[1], "dimensions"
            print "Predicting..."

        # Perform the fit
        y_pred = self.regressor.predict(X)

        # Report
        if self.verbose:
            print "Prediction complete."

        # Return result
        return y_pred


    #--------------------------------------------------------------------------

    def preprocess_y(self, y):
        """Perform preprocessing on y. This is required to make test y data
        compatible with predicted y data in order to assess the result.

        WARNING: Since this processes test data in a way that is dependent on
                 the fitted model, the test data is no longer truly separate
                 from the training data. Validation done in this way thus only
                 validates the core regressor itself, not the entire pipeline!

        Parameters
        ----------
        y : array of shape (n_samples, n_target_features)
            Target feature space of the test data. n_samples must be the same
            as in the test data X, i.e. the X data used during prediction.

        Returns
        ------
        y : array of shape (n_samples, n_target_features)
            Target feature space preprocessed the same way as the target space
            is preprocessed during AtlasPipeline.fit.
        """

        # Report
        if self.verbose:
            print "Preprocessing feature space..."

        # Standardization of source fspace
        if self.zscore_y:
            y = self.scaler_y.transform(y)

        # PCA of source fspace
        if self.pca_y:
            y = self.pcaobj_y.transform(y)

        # Restandardization of source fspace
        if self.pca_y and self.rezscore_y:
            y = self.rescaler_y.transform(y)

        # Subselect dimensions of source fspace
        if self.subselect_y is not None:

            # Subselect up to a given number of dimensions
            if type(self.subselect_y)==int:
                y = y[:, :self.subselect_y]

            # Subselect a set of dimensions
            else:
                y = y[:, self.subselect_y]

        # Add covariate information to source fspace
        if self.add_covariates is not None:
            raise NotImplementedError("Adding covariates wasn't much use "+
                                      "in early tests, so it's currently not "+
                                      "implemented.")

        # Report
        if self.verbose:
            print "Preprocessing complete:"
            print "  N:", y.shape[0], "samples"
            print "  y:", y.shape[1], "dimensions"
            print "Returning result."

        # Return result
        return y


#------------------------------------------------------------------------------

### CALLING FUNCTION

def predict_atlas(fpaths_refspace_train, fpaths_secspace_train,
                  fpaths_refspace_predict,
                  outlier_removal_ref = None,
                  outlier_removal_sec = None,
                  outlier_removal_cov = None,
                  covariates_to_use   = None,
                  regressor='MO-SVR', n_jobs=1,
                  save_predictions=False, save_pipeline=False, 
                  verbose=False,
                  outlier_options_ref = {},
                  outlier_options_sec = {},
                  outlier_options_cov = {},
                  regressor_options = { 'kernel'  : 'rbf'},
                  pipeline_options  = { 'zscore_X'          : False,
                                        'zscore_y'          : False,
                                        'pca_X'             : False,
                                        'pca_y'             : False,
                                        'rezscore_X'        : False,
                                        'rezscore_y'        : False,
                                        'subselect_X'       : None,
                                        'subselect_y'       : None,
                                        'add_covariates'    : None } ):
    """Predict a secondary channel feature space by fitting an atlas regression
    model on paired "secondary channel - reference channel" training data and
    then performing regression on "reference channel"-only test data.

    Input data is retrieved from files specified in lists of file paths and the
    predicted output data is written to the corresponding paths, appropriately
    named and tagged as 'PREDICTED'.

    The channel names for the predicted channels are added to the metadata
    channels index (also tagged as 'PREDICTED') and the full atlas regression
    objects are also added to the metadata.

    Parameters
    ----------
    fpaths_refspace_train : single string or list of strings
        A path or list of paths (either local from cwd or global) to npy files
        containing training feature space data for the reference channel used
        as the basis of prediction (usually the shape space).
    fpaths_secspace_train : single string or list of strings
        A path or list of paths (either local from cwd or global) to npy files
        containing training feature space data for the secondary channel that
        is to be the target of the regression.
    fpaths_refspace_predict : single string or list of strings
        A path or list of paths (either local from cwd or global) to npy files
        containing prediction feature space data for the reference channel
        based on which the target secondary channel will be predicted
    outlier_removal_ref : string or None, optional, default None
        If None, no outlier removal is done on the reference feature space.
        Otherwise this must be a string denoting the method for outlier removal
        (one of `absolute_thresh`, `percentile_thresh`,
        `merged_percentile_thresh` or `isolation_forest`). Note that outlier
        removal is only done on training data, not on prediction data.
        See katachi.utilities.outlier_removal.RemoveOutliers for more info.
    outlier_removal_sec : string or None, optional, default None 
        If None, no outlier removal is done on the target feature space.
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
        If a string, must be one of 'MO-SVR', 'MT-ENetCV', 'MT-Lasso', 'MLP'. 
        In the first case a multioutput SVR is used for regression, in the 
        second a Multi-Task Elastic Net with Cross Validation, in the third a 
        Multi-Task Lasso linear regression, and in the fourth a Multi-Layer 
        Perceptron. If an sklearn(-like) regressor instance is passed, it 
        must be a multivariate-multivariable regressor that supports the fit 
        and predict methods.
    n_jobs : int, optional, default 1
        Number of processes available for use during multi-processed model
        fitting and prediction. Works for 'MO-SVR', 'MT-ENetCV' and 'MT-Lasso' 
        regressors.
        WARNING: The 'MLP' regressor also performs multi-processing but does
        not seem to support an n_jobs argument.
    save_predictions : bool, optional, default False
        If True, the predictions are saved in the corresponding paths and the
        metadata is updated.
    save_pipeline : bool, optional, default False
        If True, the atlas pipeline object is saved in the corresponding paths
        as a separate file with the name `<prim_ID>_atlas_pipeline.pkl`.
    verbose : bool, optional, default False
        If True, more information is printed.
    outlier_options_ref : dict, optional, default {}
        kwarg dictionary for the chosen outlier removal method to be applied
        to the reference feature space.
        See katachi.utilities.outlier_removal.RemoveOutliers for more info.
    outlier_options_sec : dict, optional, default {}
        kwarg dictionary for the chosen outlier removal method to be applied
        to the target feature space.
        See katachi.utilities.outlier_removal.RemoveOutliers for more info.
    outlier_options_cov : dict, optional, default {}
        kwarg dictionary for the chosen outlier removal method to be applied
        to the covariates. There default is to fall back to the defaults of
        katachi.utilities.outlier_removal.RemoveOutliers.
    regressor_options : dict, optional, default is a standard RBF MO-SVR
        kwarg dictionary for the chosen regressor's instantiation.
        See the chosen regressor's doc string for more information.
    pipeline_options : dict, optional, default is no additional processing
        kwarg dictionary for AtlasPipeline instantiation.
        See the AtlasPipeline doc string for more information.

    Returns
    -------
    secspace_predict : array of shape (n_predict_samples, n_secspace_features)
        Predicted secondary channel feature space.
    refspace_predict_idx : array of shape (n_predict_samples)
        Index array mapping rows (cells) of secspace_predict to paths (prims)
        in fpaths_refspace_predict.
    atlas_pipeline : predict_atlas.AtlasPipeline instance
        Fitted instance of the regressor pipeline.
    """

    #--------------------------------------------------------------------------

    ### Load data

    if verbose: print "\n# Loading data..."

    # Handle cases of single paths for training data
    if type(fpaths_secspace_train)==str and type(fpaths_refspace_train)==str:
        fpaths_secspace_train = [fpaths_secspace_train]
        fpaths_refspace_train = [fpaths_refspace_train]
    elif (   type(fpaths_secspace_train)==str
          or type(fpaths_refspace_train)==str
          or len(fpaths_secspace_train)!=len(fpaths_refspace_train) ):
        raise IOError("Different number of secondary and reference space "+
                      "input file paths specified.")

    # Handle cases of single paths for prediction data
    if type(fpaths_refspace_predict)==str:
        fpaths_refspace_predict = [fpaths_refspace_predict]

    # Load training data
    secspace_train = []
    refspace_train = []
    for secpath, refpath in zip(fpaths_secspace_train, fpaths_refspace_train):
        secspace_train.append(np.load(secpath))
        refspace_train.append(np.load(refpath))
    secspace_train = np.concatenate(secspace_train, axis=0)
    refspace_train = np.concatenate(refspace_train, axis=0)

    # Check that everything is fine
    if not secspace_train.shape[0] == refspace_train.shape[0]:
        raise IOError("Secondary and reference space do not have the same "+
                      "number of cells.")

    # Load prediction data
    refspace_predict = []
    refspace_predict_idx = []
    for idx, refpath in enumerate(fpaths_refspace_predict):
        refspace_predict.append(np.load(refpath))
        refspace_predict_idx.append([idx for v
                                     in range(refspace_predict[-1].shape[0])])
    refspace_predict     = np.concatenate(refspace_predict, axis=0)
    refspace_predict_idx = np.concatenate(refspace_predict_idx, axis=0)

    # Check that everything is fine
    if not refspace_train.shape[1] == refspace_predict.shape[1]:
        raise IOError("Reference feature spaces for training and prediction "+
                      "do not have the same number of features!")

    # Handle covariate loading
    if outlier_removal_cov is not None:

        # Sanity checks
        if covariates_to_use is None:
            raise IOError("When outlier_removal_cov is not None, covariates "+
                          "to use for determining outliers must be specified "+
                          "in covariates_to_use!")

        # Handle single covariates
        if type(covariates_to_use) == str:
            covariates_to_use = [covariates_to_use]

        # Load covariates
        covars = []
        for refpath in fpaths_refspace_train:

            # Create covarpath
            revdir, reffile = os.path.split(refpath)
            covpath = os.path.join(revdir, reffile[:10]+'_covariates.pkl')

            # Load covar file
            with open(covpath, 'rb') as covfile:
                covtree = pickle.load(covfile)

            # Get relevant covariates
            covs2use = []
            for c2u in covariates_to_use:
                covs2use.append( np.expand_dims(covtree._gad(c2u), -1) )
            covs2use = np.concatenate(covs2use, axis=1)

            # Add to other samples
            covars.append( covs2use )

        # Concatenate
        covars = np.concatenate(covars)



    #--------------------------------------------------------------------------

    ### Prepare regressor

    # Report
    if verbose: print "\n# Preparing regressor..."


    # Multi-Output Support Vector Regression with RBF Kernel
    if regressor=='MO-SVR':
        svr = svm.SVR(**regressor_options)
        regressor = multioutput.MultiOutputRegressor(svr, n_jobs=n_jobs)
        
    # Multi-task Elastic Net Regression with Cross Validation
    elif regressor=='MT-ENetCV':
        regressor = linear_model.MultiTaskElasticNetCV(
            random_state=42, 
            n_jobs=n_jobs)
        
    # Multivariate-Multivariable Linear Regression by Multi-Task Lasso
    elif regressor=='MT-Lasso':
        regressor = linear_model.MultiTaskLassoCV(random_state=42,
                                                  n_jobs=n_jobs,
                                                  **regressor_options)

    # Multi-Layer Perceptron Regressor
    elif regressor=='MLP':
        regressor = neural_network.MLPRegressor(random_state=42,
                                                **regressor_options)

    # Other regressor strings
    elif type(regressor)==str:
        raise ValueError('Regressor not recognized.')

    # Regressor object given as argument
    else:

        # Check if object has fit method
        fit_attr = getattr(regressor, "fit", False)
        if not callable(fit_attr):
            raise ValueError("Regressor object has no 'fit' method.")

        # Check if object has predict method
        predict_attr = getattr(regressor, "predict", False)
        if not callable(predict_attr):
            raise ValueError("Regressor object has no 'predict' method.")


    #--------------------------------------------------------------------------

    ### Remove outliers from training data

    # Find and remove outliers based on covariate values
    if outlier_removal_cov is not None:

        # Report
        if verbose:
            print "\n# Removing outliers based on covariates..."
            print "Started with %i," % refspace_train.shape[0],

        # Find and remove outliers
        orem_cov = RemoveOutliers(outlier_removal_cov, **outlier_options_cov)
        orem_cov.fit(covars)
        covars, (refspace_train, secspace_train) = orem_cov.transform(
                                                            covars,
                                                            [refspace_train,
                                                             secspace_train] )

        # Report
        if verbose:
            print "removed %i, kept %i samples" % (orem_cov.X_removed_,
                                                   refspace_train.shape[0])

    # Find and remove outliers based on reference space
    if outlier_removal_ref is not None:

        # Report
        if verbose:
            print "\n# Removing reference outliers..."
            print "Started with %i," % refspace_train.shape[0],

        # Find and remove outliers
        orem_ref = RemoveOutliers(outlier_removal_ref,
                                  **outlier_options_ref)
        orem_ref.fit(refspace_train)
        refspace_train, secspace_train = orem_ref.transform(refspace_train,
                                                            secspace_train)

        # Report
        if verbose:
            print "removed %i, kept %i samples" % (orem_ref.X_removed_,
                                                   refspace_train.shape[0])

    # Find and remove outliers based on secondary space
    if outlier_removal_sec is not None:

        # Report
        if verbose:
            print "\n# Removing target outliers..."
            print "Started with %i," % refspace_train.shape[0],

        # Find and remove outliers
        orem_sec = RemoveOutliers(outlier_removal_sec,
                                  **outlier_options_sec)
        orem_sec.fit(secspace_train)
        secspace_train, refspace_train = orem_sec.transform(secspace_train,
                                                            refspace_train)

        # Report
        if verbose:
            print "removed %i, kept %i samples" % (orem_sec.X_removed_,
                                                   refspace_train.shape[0])


    #--------------------------------------------------------------------------

    ### Fit and predict

    # Construct pipeline
    atlas_pipeline = AtlasPipeline(regressor, verbose=verbose,
                                   **pipeline_options)

    # Fit
    if verbose: print "\n# Fitting..."
    atlas_pipeline.fit(refspace_train, secspace_train)

    # Predict
    if verbose: print "\n# Predicting..."
    secspace_predict = atlas_pipeline.predict(refspace_predict)


    #--------------------------------------------------------------------------

    ### Update the metadata

    if save_predictions:

        if verbose: print "\n# Saving metadata..."

        # For each path...
        for idx, refpath in enumerate(fpaths_refspace_predict):

            # Load metadata file
            refdir, reffname = os.path.split(refpath)
            prim_ID  = reffname[:10]
            metapath = os.path.join(refdir, prim_ID+"_stack_metadata.pkl")
            with open(metapath, "rb") as metafile:
                metadict = pickle.load(metafile)

            # Construct channel designation
            pattern = re.compile("8bit_(.+?(?=_))")
            secpath = fpaths_secspace_train[0]
            channel = re.search(pattern, secpath).group(1) + "_PREDICTED"
            
            # Add channel to metadata
            if not channel in metadict["channels"]:
                metadict["channels"].append(channel)

            # Save metadata
            with open(metapath, "wb") as outfile:
                pickle.dump(metadict, outfile,
                            protocol=pickle.HIGHEST_PROTOCOL)
    
    
    #--------------------------------------------------------------------------
                
    ### Save fitted atlas pipeline as separate metadata file
    
    if save_pipeline:
        
        if verbose: print "\n# Saving pipeline..."
            
        # For each path...
        for idx, refpath in enumerate(fpaths_refspace_predict):
            
            # Load atlas metadata file if it exists
            refdir, reffname = os.path.split(refpath)
            prim_ID   = reffname[:10]
            atlaspath = os.path.join(refdir, prim_ID+"_atlas_pipeline.pkl")
            if os.path.isfile(atlaspath):
                with open(atlaspath, "rb") as atlasfile:
                    atlasdict = pickle.load(atlasfile)
            else:
                atlasdict = {}
                
            # Construct designation
            pattern = re.compile("8bit_(.+?(?=\.))")
            secpath = fpaths_secspace_train[0]
            atlasname = re.search(pattern, secpath).group(1) + "_ATLASPIP"
            
            # Add pipeline to dict
            atlasdict[atlasname] = atlas_pipeline

            # Save atlas dict
            with open(atlaspath, "wb") as outfile:
                pickle.dump(atlasdict, outfile,
                            protocol=pickle.HIGHEST_PROTOCOL)
    
                
    #--------------------------------------------------------------------------

    ### Save the predictions

    if save_predictions:

        if verbose: print "\n# Saving predictions..."

        # For each path...
        for idx, refpath in enumerate(fpaths_refspace_predict):

            # Construct outpath
            to_replace = refpath[refpath.index("8bit_")+5:]
            secpath    = fpaths_secspace_train[0]
            replace_by = secpath[secpath.index("8bit_")+5:]
            replace_by = replace_by[:-4] + "_PREDICTED.npy"
            outpath    = refpath.replace(to_replace, replace_by)

            # Write file
            np.save(outpath, secspace_predict[refspace_predict_idx==idx])


    #--------------------------------------------------------------------------

    ### Return results

    # Report
    if verbose: print "\nDone!"

    # Return
    return secspace_predict, refspace_predict_idx, atlas_pipeline


#------------------------------------------------------------------------------



