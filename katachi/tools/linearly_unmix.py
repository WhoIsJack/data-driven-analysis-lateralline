# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 23:18:07 2017

@author:    Jonas Hartmann @ Gilmour group @ EMBL Heidelberg

@descript:  Simple approach to remove bleed-through from one channel to another
            by linear unmixing.

@usage:     Developed and tested for removal of nuclear NLS-tdTomato bleed-
            through into membranous Lyn:EGFP in 3D stacks of the zebrafish
            lateral line primordium acquired at the Zeiss LSM880.
"""


#------------------------------------------------------------------------------

# IMPORTS

# External
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread, imsave
import multiprocessing.pool
from multiprocessing.pool import cpu_count


#------------------------------------------------------------------------------

### FUNCTION: COMPUTE CORRECTED CORR COEF BETWEEN CLEAN & CONTAMINANT IMAGE

def _get_corrected_img_corrcoef(params):
    """Compute corrected correlation coefficient between an unmixed image
    (`clean`) and its contaminant.

    The correction refers to the idea that `clean` should never be reduced
    below its natural background by the unmixing, which is why such low
    values are inverted and therefore contribute to the correlation again.
    See function `linear_unmixing` for more information.

    Parameters
    ----------
    params : tuple (a, dirty, conta)
        Tuple containing inputs. `dirty` and `conta` are numpy images, `a` is
        a factor. See function `linear_unmixing` for more information.

    Returns
    -------
    corr : float
        Corrected correlation coefficient between the flattened images.
    """

    # Unpack params
    a, dirty, conta = params

    # Flatten
    dirty = dirty.flatten()
    conta = conta.flatten()

    # Compute the clean image
    clean = dirty - a * conta

    # Subtract mean background and invert negative values
    # This ensures that the absolute minimum image correlations
    # reflects the ideal value of `a`.
    clean = np.abs( clean - clean.mean() )

    # Compute correlation coefficient
    corr = np.corrcoef(clean,conta)[0,1]

    # Return result
    return corr


#------------------------------------------------------------------------------

# FUNCTION: PERFORM LINEAR UNMIXING

def unmix_linear(fpath_dirty, fpath_conta, processes=None,
                 a_range=(0.0, 1.0, 20),
                 accept_arrays=False, save_result=True, return_result=False,
                 show=False, verbose=False):
    """Clean a 'dirty stack' by removing bleed-through from a 'contaminant
    stack' using simple linear unmixing.

    The linear unmixing approach used here computes the following:

    CLEAN = DIRTY - a * CONTAMINANT

    where `a` is determined by minimizing CORR over a range of possible `a`,
    with CORR defined as follows:

    CORR  = image_correlation( CONTAMINANT, | CLEAN - mean(CLEAN) | )

    Note that using the absolute value of the mean-background subtracted CLEAN
    images ensures that high values of `a` are punished since overly unmixed
    regions start correlating with the CONTAMINANT again. An intuitive way of
    looking at this is to argue that the unmixing should not reduce the signal
    of the DIRTY channel below its normal background (here approximated by the
    mean), hence such overreductions are punished.

    Parameters
    ----------
    fpath_dirty : string
        The path (either local from cwd or global) to the 'dirty' stack that
        should be cleaned. The stack should be a single channel and time point.
    fpath_conta : string
        The path (either local from cwd or global) to the 'contaminant' stack
        that should be cleaned. Must have the same shape as the 'dirty' stack.
    processes : int or None, optional
        Number of processes that may be used for multiprocessing. If None, half
        of the available CPUs are used. If set to 1, the entire code is run
        sequentially (no multiprocessing code is used).
    a_range : tuple (start, stop, n_steps), optional, default (0.0, 1.0, 20)
        Range of values of `a` to be tested. Must start with 0.0 and end with
        a positive float (stop). n_steps is the number of regular steps tested
        between 0.0 and stop (see np.linespace).
    accept_arrays : bool, optional, default False
        If True, fpath_dirty and fpath_conta are expected to be already loaded
        image arrays instead of paths.
    save_result : bool, optional, default True
        If True, the unmixed image will be saved as a tif file with the suffix
        `_linUnmix.tif`.
    return_result : bool, optional, default False
        If True, the unmixed image and the selected alpha are returned.
    show : bool, optional, default False
        If True, a plot is displayed showing the image correlations as a
        function of `a` and indicating the selected value of `a`.
    verbose : bool, optional, default False
        If True, more information is printed.
    """

    #--------------------------------------------------------------------------

    ### Load the files and prepare the data

    if not accept_arrays:

        if verbose: print "Loading stacks..."

        # Add .tif to filenames if necessary
        if not fpath_dirty.endswith('.tif'):
            fpath_dirty = fpath_dirty + '.tif'
        if not fpath_conta.endswith('.tif'):
            fpath_conta = fpath_conta + '.tif'

        # Try loading the dirty channel
        try:
            img_dirty = imread(fpath_dirty)
        except:
            print "Attempting to load dirty stack failed with this error:"
            raise

        # Try loading the contaminant channel
        try:
            img_conta = imread(fpath_conta)
        except:
            print "Attempting to load contaminant stack failed with this error:"
            raise

    # If the input was provided as arrays already
    else:
        img_dirty = fpath_dirty
        img_conta = fpath_conta

    # Prepare range of `a` to be tested
    a_arr = np.linspace(*a_range)


    #--------------------------------------------------------------------------

    ### Compute correlations across a range of possible `a` (serial)

    if processes == 1:

        if verbose: print "Computing correlations sequentially..."

        corrs = []
        for a in a_arr:
            corrs.append(_get_corrected_img_corrcoef((a, img_dirty, img_conta)))
        corrs = np.array(corrs)


    #--------------------------------------------------------------------------

    ### Compute correlations across a range of possible `a` (parallel)

    else:

        if verbose: print "Computing correlations in parallel..."

        # If necessary: choose number of processes (half of available cores)
        if processes is None:
            processes = cpu_count() // 2

        # Prepare for multiprocessing
        my_pool = multiprocessing.Pool(processes=processes)
        param_list = [(a, img_dirty, img_conta) for a in a_arr]

        # Execute function on the input range
        corrs = my_pool.map(_get_corrected_img_corrcoef, param_list)

        # Clean up
        my_pool.close()
        my_pool.join()
        corrs = np.array(corrs)


    #--------------------------------------------------------------------------

    ### Get optimal unmixing factor based on minimum cross correlation

    if verbose: print "Detecting optimal factor and creating clean image..."

    # Get factor
    target_a = a_arr[np.argmin(corrs)]

    # Plot
    if show:

        fig, ax1 = plt.subplots()
        ax1.plot(a_arr, corrs, c='b')
        ax1.set_ylabel('Image corr coeff', color='b')
        ax1.tick_params('y', colors='b')

        ax2 = ax1.twinx()
        ax2.plot(a_arr, corrs / corrs[0], c='g')
        ax2.set_ylabel('Relative image corrr coeff', color='g')
        ax2.tick_params('y', colors='g')

        ax1.vlines(target_a, corrs.min(), corrs.max(),
                   color="r", label='target value of `a`')
        ax1.legend(frameon=False)

        ax1.set_xlabel('Factor `a`')

        plt.show()


    #--------------------------------------------------------------------------

    ### Generate the unmixed image

    img_clean = img_dirty - target_a * img_conta
    img_clean[img_clean<0] = 0
    img_clean = img_clean.astype(np.uint8)


    #--------------------------------------------------------------------------

    ### Write the result and return

    if save_result:
        if verbose: print "Saving result..."
        imsave(fpath_dirty[:-4]+'_linUnmix.tif', img_clean, bigtiff=True)

    if verbose: print "Processing complete!"

    if return_result:
        return img_clean, target_a
    else:
        return


#------------------------------------------------------------------------------

### FUNCTION: COMPUTE CORRELATION COEFFICIENT BETWEEN TWO IMAGES

def _get_img_corrcoef_legacy(params):
    """Compute corrected correlation coefficient between an unmixed image
    (`clean`) and its contaminant.

    Parameters
    ----------
    params : tuple (a, dirty, conta)
        Tuple containing inputs. `dirty` and `conta` are numpy images, `a` is
        a factor. See function `linear_unmixing)_legacy` for more information.

    Returns
    -------
    corr : float
        Correlation coefficient between the flattened images.
    """

    # Unpack params
    a, dirty, conta = params

    # Flatten
    dirty = dirty.flatten()
    conta = conta.flatten()

    # Compute the clean image
    clean = dirty - a * conta

    # Remove negative values and return to uint8
    clean[clean<0] = 0
    clean = clean.astype(np.uint8)

    # Compute correlation coefficient
    corr = np.corrcoef(clean,conta)[0,1]

    # Return result
    return corr


#------------------------------------------------------------------------------

# FUNCTION: PERFORM LINEAR UNMIXING

def unmix_linear_legacy(fpath_dirty, fpath_conta, processes=None,
                        a_range=(0.0, 1.0, 40), thresh=0.7,
                        accept_arrays=False, save_result=True,
                        return_result=False, show=False, verbose=False):
    """Clean a 'dirty stack' by removing bleed-through from a 'contaminant
    stack' using simple linear unmixing.

    The linear unmixing approach used here computes the following:

    CLEAN = DIRTY - a * CONTAMINANT

    where `a` is determined by computing the image correlation of CLEAN and
    CONTAMINANT over a range of different possible values of `a` and selecting
    the lowest value of `a` that reduces the correlation to below a fraction of
    the original correlation given by `thresh`. `thresh` depends on how much
    true (non-bleed-through) correlation is expected between the images.

    Parameters
    ----------
    fpath_dirty : string
        The path (either local from cwd or global) to the 'dirty' stack that
        should be cleaned. The stack should be a single channel and time point.
    fpath_conta : string
        The path (either local from cwd or global) to the 'contaminant' stack
        that should be cleaned. Must have the same shape as the 'dirty' stack.
    processes : int or None, optional
        Number of processes that may be used for multiprocessing. If None, half
        of the available CPUs are used. If set to 1, the entire code is run
        sequentially (no multiprocessing code is used).
    a_range : tuple (0.0, stop, n_steps), optional, default (0.0, 2.0, 20)
        Range of values of `a` to be tested. Must start with 0.0 and end with
        a positive float (stop). n_steps is the number of regular steps tested
        between 0.0 and stop (see np.linespace).
    thresh : float, optional, default 0.6
        Threshold for the fraction of the original correlation that may remain
        for a value of `a` to be accepted. `thresh` relates to the desired
        correlation of the unmixed (reduced) image as:
        `thresh = reduced_corr / original_corr`
    accept_arrays : bool, optional, default False
        If True, fpath_dirty and fpath_conta are expected to be already loaded
        image arrays instead of paths.
    save_result : bool, optional, default True
        If True, the unmixed image will be saved as a tif file with the suffix
        `_linUnmix.tif`.
    return_result : bool, optional, default False
        If True, the unmixed image and the selected alpha are returned.
    show : bool, optional, default False
        If True, a plot is displayed showing the image correlations as a
        function of `a` and indicating the selected value of `a`.
    verbose : bool, optional, default False
        If True, more information is printed.


    Notes
    -----
    The default values chosen for a_range and thresh are based on the unmixing
    of nuclear NLS-tdTomato bleed-through into membranous Lyn:EGFP in 3D stacks
    of the zebrafish lateral line primordium acquired at the Zeiss LSM880.
    """

    #--------------------------------------------------------------------------

    ### Load the files and prepare the data

    if not accept_arrays:

        if verbose: print "Loading stacks..."

        # Add .tif to filenames if necessary
        if not fpath_dirty.endswith('.tif'):
            fpath_dirty = fpath_dirty + '.tif'
        if not fpath_conta.endswith('.tif'):
            fpath_conta = fpath_conta + '.tif'

        # Try loading the dirty channel
        try:
            img_dirty = imread(fpath_dirty)
        except:
            print "Attempting to load dirty stack failed with this error:"
            raise

        # Try loading the contaminant channel
        try:
            img_conta = imread(fpath_conta)
        except:
            print "Attempting to load contaminant stack failed with this error:"
            raise

    # If the input was provided as arrays already
    else:
        img_dirty = fpath_dirty
        img_conta = fpath_conta

    # Prepare range of `a` to be tested
    a_arr = np.linspace(*a_range)


    #--------------------------------------------------------------------------

    ### Compute correlations across a range of possible `a` (serial)
    # TODO [ENH]: Check and stop when `thresh` is reached.

    if processes == 1:

        if verbose: print "Computing correlations sequentially..."

        corrs = []
        for a in a_arr:
            corrs.append(_get_img_corrcoef_legacy((a, img_dirty, img_conta)))
        corrs = np.array(corrs)


    #--------------------------------------------------------------------------

    ### Compute correlations across a range of possible `a` (parallel)

    else:

        if verbose: print "Computing correlations in parallel..."

        # If necessary: choose number of processes (half of available cores)
        if processes is None:
            processes = cpu_count() // 2

        # Prepare for multiprocessing
        my_pool = multiprocessing.Pool(processes=processes)
        param_list = [(a, img_dirty, img_conta) for a in a_arr]

        # Execute function on the input range
        corrs = my_pool.map(_get_img_corrcoef_legacy, param_list)

        # Clean up
        my_pool.close()
        my_pool.join()
        corrs = np.array(corrs)


    #--------------------------------------------------------------------------

    ### Set threshold at T-fold reduction of correlation

    if verbose: print "Setting threshold and creating clean image..."

    reduced_corrs = corrs / corrs[0]
    target_a = a_arr[np.where(reduced_corrs <= thresh)[0][0]]

    # Plot
    if show:

        fig, ax1 = plt.subplots()
        ax1.plot(a_arr, corrs, c='b')
        ax1.set_ylabel('Image corr coeff', color='b')
        ax1.tick_params('y', colors='b')

        ax2 = ax1.twinx()
        ax2.plot(a_arr, corrs / corrs[0], c='g')
        ax2.set_ylabel('Relative image corrr coeff', color='g')
        ax2.tick_params('y', colors='g')

        ax1.vlines(target_a, corrs.min(), corrs.max(),
                   color="r", label='target value of `a`')
        ax1.legend(frameon=False)

        ax1.set_xlabel('Factor `a`')

        plt.show()


    #--------------------------------------------------------------------------

    ### Generate the unmixed image

    img_clean = img_dirty - target_a * img_conta
    img_clean[img_clean<0] = 0
    img_clean = img_clean.astype(np.uint8)


    #--------------------------------------------------------------------------

    ### Write the result and return

    if save_result:
        if verbose: print "Saving result..."
        imsave(fpath_dirty[:-4]+'_linUnmix.tif', img_clean, bigtiff=True)

    if verbose: print "Processing complete!"

    if return_result:
        return img_clean, target_a
    else:
        return


#------------------------------------------------------------------------------



