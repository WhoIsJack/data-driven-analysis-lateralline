# -*- coding: utf-8 -*-
"""
Created on Fri Jun 07 16:17:08 2019

@author:    Jonas Hartmann @ Gilmour group @ EMBL Heidelberg

@descript:  Functions to aid data wrangling in the point cloud.
"""

#------------------------------------------------------------------------------

# IMPORTS

# External general
from __future__ import division
import numpy as np


#------------------------------------------------------------------------------

### Perform Gaussian smoothing on values in arbitrary point cloud

def pcl_gaussian_smooth(dists, vals, sg_percentile=1.7):
    """Perform a Gaussian smooth on values defined on the points in a point
    cloud with an arbitrary number of dimensions.
    
    Parameters
    ----------
    dists : ndarray of shape (n_pts, n_pts)
        Squareform of all pairwise distances between points of the point
        cloud, as computed e.g. by scipy.spatial.distance.pdist.
    vals : ndarray of shape (n_pts, n_dims)
        Values to be smoothed defined for each point. The different dimensions
        are smoothed independently.
    sg_percentile : numeric, default 1.7
        To determine the sigma (sg) of the Gaussian, the n-th percentile of 
        all distances in dists is used. This value specifies that n.
    """

    # Generate Gaussian function for smoothing
    def gaussian_factory(mu, sg):
        gaussian = lambda x : 1 / (sg*np.sqrt(2.0*np.pi)) * np.exp(-1/2*((x-mu)/sg)**2.0)
        return gaussian
    gaussian_func = gaussian_factory(0.0, np.percentile(dists, sg_percentile))

    # Smoothen the distances
    gaud = gaussian_func(dists)

    # Use smoothened distances to smoothen values
    smooth = np.empty_like(vals)
    for dim in range(vals.shape[1]):
        smooth[:,dim] = np.sum(gaud*vals[:,dim], axis=1) / np.sum(gaud, axis=1)
    
    # Done
    return smooth


#------------------------------------------------------------------------------



