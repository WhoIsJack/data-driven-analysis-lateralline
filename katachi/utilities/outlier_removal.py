# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 16:13:20 2018

@author:    Jonas Hartmann @ Gilmour group @ EMBL Heidelberg

@descript:  Sklearn-like class for removal of outliers.
"""

#------------------------------------------------------------------------------

# IMPORTS

# External general
from __future__ import division
import numpy as np

# External specific
from sklearn.ensemble import IsolationForest


#------------------------------------------------------------------------------

### SKLEARN-LIKE TRANSFORM: OUTLIER REMOVAL

class RemoveOutliers(object):
    """Identifies and removes outliers from feature spaces based on one or
    more approaches/criteria selected from the following:

    - Value of any feature above or below a given threshold value
    - Value of any feature above or below a given percentile
    - Sum of absolute standardized feature values above a given percentile
    - Classified as outlier by sklearn.ensemble.IsolationForest

    Implemented as an sklearn-like transformer with these methods:
        .fit(X) -- determines the threshold on X.
        .transform(X[,y]) -- removes outliers from X [and y].
        .fit_transform(X[,y]) -- combines fit and transform.

    WARNING: Sklearn does not fully support transform operations on y.
             This class therefore cannot be used in sklearn pipelines.
    """

    def __init__(self, method, bounds='upper', threshold=None, percentile=95,
                 isoforest_params='default'):
        """Initialize RemoveOutliers instance.

        Parameters
        ----------
        method : string
            One of the following strings denoting the method to use:
                'absolute_thresh'
                    Remove any sample that has a feature value above/below a
                    given absolute threshold. The threshold can be a single
                    value for all features or a distinct value per feature.
                'percentile_thresh'
                    Remove any sample that has a feature value above/below a
                    given percentile threshold (determined based on training
                    data). The percentile threshold is computed for each
                    feature individually.
                'merged_percentile_thresh'
                    Remove any sample that has overall comparably extreme
                    values. Specifically, remove samples for which the sum
                    of the absolute standardized feature values is above a
                    given percentile threshold (determined based on training
                    data).
                'isolation_forest'
                    Use sklearn.ensemble.IsolationForest to determine and
                    remove outliers.
        bounds : string, optional, default 'upper'
            Relevant if method is 'absolute_thresh' or 'percentile_thresh';
            ignored otherwise.
            Bounds can be either 'upper' or 'lower', designating whether values
            above or below the threshold should be considered outliers. By
            default ('upper'), any samples with values above the threshold are
            removed as outliers.
        threshold : numeric or array of shape (N_features) or None, optional, default None
            Relevant if method is 'absolute_thresh'; ignored otherwise.
            Specifies either a single threshold for all features or an array of
            individual thresholds for each feature.
        percentile : int in range [1,99], optional, default 95
            Relevant if method is 'percentile_thresh' or
            'merged_percentile_thresh'; ignored otherwise.
            Percentile to compute the threshold.
        isoforest_params : dict or 'default', optional, default 'default'
            Parameters for instantiation of sklearn.ensemble.IsolationForest
            class. If 'default', sklearn's defaults are used. Otherwise must be
            a dictionary of keyword arguments.
        """

        # Bind parameters
        self.method           = method
        self.bounds           = bounds
        self.threshold        = threshold
        self.percentile       = percentile
        self.isoforest_params = isoforest_params

        # Check that method exists
        if method not in ['absolute_thresh', 'percentile_thresh',
                          'merged_percentile_thresh', 'isolation_forest']:
            raise NotImplementedError("Method not implemented: " + method)

        # Check that bounds argument makes sense (where relevant)
        if method in ['absolute_thresh', 'percentile_thresh']:
            if bounds not in ['upper', 'lower']:
                raise ValueError("bounds must be 'upper' or 'lower' for " +
                                 "method: " + method)

        # Check that threshold is defined (where relevant)
        if method=='absolute_thresh' and self.threshold is None:
            raise ValueError("threshold must not be None if method is " +
                             "absolute_thresh!")


    def fit(self, X):
        """Determine threshold for outlier detection.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Feature space used for threshold prediction.

        Returns
        -------
        self
        """

        # Absolute threshold
        if self.method == 'absolute_thresh':
            self.threshold_ = self.threshold

        # Percentile threshold
        elif self.method == 'percentile_thresh':
            percentile = np.percentile(X, self.percentile, axis=0)
            self.threshold_ = percentile

        # Merged percentile threshold
        elif self.method == 'merged_percentile_thresh':
            self.X_mean_ = X.mean(axis=0)
            self.X_std_  = X.std(axis=0)
            X_merged = (X - self.X_mean_) / self.X_std_
            X_merged = np.sum(np.abs(X_merged), axis=1)
            percentile = np.percentile(X_merged, self.percentile)
            self.threshold_ = percentile

        # Isolation forest
        elif self.method == 'isolation_forest':
            if self.isoforest_params == 'default':
                self.isoforest_ = IsolationForest()
            else:
                self.isoforest_ = IsolationForest(**self.isoforest_params)
            self.isoforest_.fit(X)

        # Done
        return self


    def transform(self, X, y=None):
        """Apply thresholds determined in fit to remove outliers from given
        feature space.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Outliers will be removed from this array.
        y : array of shape (n_samples, m_features) or list of such arrays, optional, default None
            Outliers determined based on X will also be removed from this array
            or form each array in an iterable of arrays.

        Returns
        -------
        Xt : array of shape (n_samples-n_outliers, n_features)
            Same as X but without outliers.
        yt : array of shape (n_samples-n_outliers, m_features) or list of such arrays
            Same as y but without outliers. Only returned if y is not None.
        """

        # Create inlier mask for absolute or percentile threshold
        if self.method in ['absolute_thresh', 'percentile_thresh']:
            if self.bounds == 'upper':
                outlier_mask = X > self.threshold_
            if self.bounds == 'lower':
                outlier_mask = X < self.threshold_
            self.is_inlier_ = ~ outlier_mask.max(axis=1)

        # Create inlier mask for merged percentile threshold
        elif self.method == 'merged_percentile_thresh':
            X_merged = X_merged = (X - self.X_mean_) / self.X_std_
            X_merged = np.sum(np.abs(X_merged), axis=1)
            outlier_mask = X_merged > self.threshold_
            self.is_inlier_ = ~ outlier_mask

        # Create inlier mask for isolation forest
        elif self.method == 'isolation_forest':
            is_inlier = self.isoforest_.predict(X)
            self.is_inlier_ = is_inlier == 1

        # Remove outliers from X
        Xt = X[self.is_inlier_, :]
        self.X_removed_ = X.shape[0] - Xt.shape[0]

        # Remove outliers from y
        if y is not None:

            if type(y) == list:
                yt = []
                for y_i in y:
                    yt.append(y_i[self.is_inlier_, :])

            else:
                yt = y[self.is_inlier_, :]

        # Return results
        if y is None:
            return Xt
        else:
            return Xt, yt


    def fit_transform(self, X, y=None):
        """Calls fit(X, y) followed by transform(X, y).
        See the respective methods for more information.
        """

        return self.fit(X).transform(X, y)


#------------------------------------------------------------------------------



