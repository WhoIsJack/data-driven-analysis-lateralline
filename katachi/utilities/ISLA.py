# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 11:11:57 2017

@author:    Jonas Hartmann @ Gilmour group @ EMBL Heidelberg

@descript:  Intensity-biased Stochastic Landmark Assignment (ISLA)
    
            Generate a sparse point cloud representation of an input image by
            stochastically distributing a set number of landmarks based on the
            image's pixel intensities, which are treated as a multinomial
            distribution from which landmark coordinates are sampled. 
            The higher the intensity of a pixel/voxel, the higher the chance of 
            a landmark being assigned to it.            
"""

#------------------------------------------------------------------------------

# IMPORTS

# External
from __future__ import division
import numpy as np


#------------------------------------------------------------------------------

# FUNCTION: INTENSITY-BIASED STOCHASTIC LANDMARK ASSIGNMENT (ISLA)
    
def isla(img, n_hits, replace=True, seed=None):
    """Intensity-biased Stochastic Landmark Assignment (ISLA)
    
    Generates a sparse point cloud representation of an image by treating pixel
    intensities as a multinomial distribution and randomly sampling a set
    number of landmarks from it.
    
    Parameters
    ----------
    img : nD array (dtype int or float)
        N-dimensional array with values representing image intensities.
    n_hits : int
        Number of landmarks to be distributed.
    replace : bool, optional, default False
        If True, the same pixel can be sampled multiple times. Otherwise, this
        is prevented.
    seed : int, optional
        A seed for the random number generator.
        
    Returns
    -------
    hits : 2D array (dtype float)
        An array of shape (landmarks, dim) containing the coordinates of the
        sampled landmaris. `dim` corresponds to the number of dimensions in the
        input image and has the same order.
    """

    ### Prep

    # Seed random number generator
    if seed:
        np.random.seed(seed)

    # Normalize to a total intensity of 1
    rdy = img.astype(np.float) / np.sum(img)


    ### Generate landmarks
    
    # Create a flat index array referencing the pixels
    idx_arr = np.arange(rdy.flatten().shape[0])
    
    # Draw from the distribution
    hits_arr = np.random.choice(idx_arr,
                                size=n_hits,
                                replace=replace,
                                p=rdy.flatten())
    
    # Unravel flat index hits array
    hits = np.array(np.unravel_index(hits_arr, np.shape(rdy))).T

    # Return result
    return hits.astype(np.float)


#------------------------------------------------------------------------------



