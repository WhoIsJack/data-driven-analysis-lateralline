# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 17:12:41 2017

@author:    Jonas Hartmann @ Gilmour group @ EMBL Heidelberg

@descript:  3D cell segmentation pipeline based on automated thresholding and
            watershed expansion along with some smoothing/filtering.

            There are two implementations:
                - segment_3D_legacy
                  An older version that works very well only for extremely
                  high-quality images acquired at the LSM880 in AiryFAST mode.
                - segment_3D
                  A more robust newer version that works also for lower quality
                  images but is slightly more sensitive to background objects
                  such as skin cells, some of which may erroneously be included
                  in some cases (although this is rare problem).

            Below are step by step overviews of both pipelines. Each step is
            implemented in a separate function that can easily be exchanged.

            segment_3D:
                - Preprocessing (smoothing)
                    - Median smooth
                    - Gaussian smooth
                - Membrane masking (thresholding)
                    - Max Histogram & labeled-object-count thresholding
                - Removing artefacts
                    - Size-based filtering
                - Object expansion
                    - Watershed on Gaussian smooth
                - Postprocessing
                    - Setting bg to 0
                    - Removing disconnected cells
                    - Continuous numbering

            segment_3D_legacy:
                - Preprocessing (smoothing)
                    - Median smooth
                - Membrane masking (thresholding)
                    - Object-count thresholding
                - Removing artefacts (intermediate clean-up)
                    - Removing disconnected components
                    - Size-based filtering
                - Object expansion
                    - Watershed on Gaussian smooth
                - Postprocessing
                    - Setting bg to 0
                    - Removing disconnected cells
                    - Continuous numbering

@usage:     Developed and tested for segmentation on membranous Lyn:EGFP in
            high-quality 3D stacks stacks of the zebrafish lateral line
            primordium acquired at the Zeiss LSM880.
"""

#------------------------------------------------------------------------------

# IMPORTS

# External
from __future__ import division
from warnings import warn
import numpy as np
import scipy.ndimage as ndi
from tifffile import imread, imsave
from skimage.morphology import watershed


#------------------------------------------------------------------------------

# FUNCTION: PREPROCESSING BY MEDIAN SMOOTHING

def _median_smooth(img, size=3):
    """Preprocessing function: median smoothing

    Performs median smoothing on `img` to remove shot noise. The kernel used
    is `np.ones(size)`.

    Default for `size` is `3`, based on segmentation of the lateral line.
    """

    return ndi.median_filter(img, size=size)


#------------------------------------------------------------------------------

# FUNCTION: PREPROCESSING BY GAUSSIAN SMOOTHING

def _gaussian_smooth(img, sigma=3):
    """Preprocessing function: gaussian smoothing

    Performs gaussian smoothing on `img` to even out noise.

    Default for `sigma` is `3`, based on segmentation of the lateral line.
    """

    return ndi.gaussian_filter(img, sigma)


#------------------------------------------------------------------------------

# FUNCTION: BACKGROUND SUBTRACTION

def _background_sub(img, method='mean', bg=None):
    """Preprocessing function: background subtraction

    Subtract a background through one of multiple methods:
        - If `method` is 'mean', the mean of the image is subtracted.
        - If `method` is 'median', the median of the image is subtracted.
        - If `method` is 'adaptive', another image of the same shape as `img`
          has to be passed as `bg`, which is then subtracted from `img`.

    Pixels that would be negative following the subtraction are set to 0.

    The default method is 'mean'.
    """

    # Input checks
    if method=='adaptive' and bg is None:
        raise ValueError("`bg` must not be None if `method` is 'adaptive'.")
    if method=='adaptive' and bg.shape!=img.shape:
        raise ValueError("`img` and `bg` must have the same shape.")

    # Mean subtraction
    if method=='mean':
        img_sub = img - np.mean(img).astype(np.uint8)
        img_sub[img < np.mean(img)] = 0

    # Median subtraction
    if method=='median':
        img_sub = img - np.median(img).astype(np.uint8)
        img_sub[img < np.median(img)] = 0

    # Adaptive subtraction
    if method=='adaptive':
        img_sub = img - bg
        img_sub[img < bg] = 0

    # Return results
    return img_sub


#------------------------------------------------------------------------------

# FUNCTION: AUTOMATED THRESHOLDING BY OBJECT-COUNT THRESHOLDING

def _object_count_threshold(img, thresh_stepsize=2, thresh_lower=4.0,
                            verbose=False):
    """Membrane masking function: object-count thresholding

    Masks membranes by thresholding. The threshold is automatically detected by
    scanning through a range of thresholds (`range(0,255,thresh_stepsize)`)
    and counting the connected component objects resulting for the different
    thresholds.

    The object count usually first increases a bit and then sharply decreases
    before evening out or increasing again. The ideal threshold lies on the
    lower end of the sharp decrease. Here, it is selected by enforcing that it
    must be after the peak and that the counts must have decreased to below
    some fraction of the peak count (`count <= peak_count / thresh_lower`).

    Defaults for `stepsize` and `lower` are `2` and `4.0`, respectively, based
    on segmentation of the lateral line.
    """

    # Prep
    thresholds = np.arange(1, 255, thresh_stepsize)
    counts     = np.zeros_like(thresholds)

    # Run threshold series
    # NOTE: Since 880 images need very low thresholds, this approach is faster
    #       than the multiprocessed evaluation of all thresholds!
    for index,threshold in enumerate(thresholds):

        # Just in case: Raise error if you fail to find the target by the end
        if index == len(thresholds)-1:
            raise Exception("Could not detect object-count threshold!")

        # Apply current threshold, count objects
        img_thresh = img>=threshold
        counts[index] = ndi.label(img_thresh)[1]

        # Check if the current threshold is the right one. If so, break.
        # The checks are:
        #  1) Is the index of the maximum before the current index?
        #  2) Is the current value below the max value divided by thresh_lower?
        if ( np.argmax(counts) < index and
             counts[index] < (np.max(counts) / thresh_lower) ):

            if verbose:
                print "-- Accepted:", threshold, counts[index]

            break

        else:

            if verbose:
                print "-- Rejected:", threshold, counts[index]

    # Return resulting thresholded image
    return img_thresh


#------------------------------------------------------------------------------

# FUNCTION: IMPROVED AUTOMATED THRESHOLDING (MAX HIST & LABELED-OBJECT-COUNT)

def _improved_threshold(img,
                        max_offset=10, offset_step=1,
                        sizelim_small=1000, sizelim_big=1000000,
                        verbose=False):
    """Improved automated thresholding for membrane masking.

    The threshold is automatically detected as the peak of the histogram plus
    an offset. The offset is determined by scanning through a small range of
    values (from 0 to `max_offset` in steps of `offset_step`) and counting the
    number of labeled objects in the expected size range that would result from
    applying a given threshold. The maximum number of labeled objects is the
    target (as many cells unmasked as possible with as few holes as possible).

    Defaults for `max_offset` and `offset_step` are `10` and `1`, respectively,
    based on segmentation of the lateral line.
    """

    # Get base threshold (histogram argmax)
    base_threshold = np.argmax(np.bincount(img.flatten()))

    # Scan through offsets
    counts = []
    offsets = range(0, max_offset, offset_step)
    for offset in offsets:

        # Apply threshold
        img_thresh = img > base_threshold + offset

        # Get number of labeled elements within acceptable size range
        _, sizes = np.unique(ndi.label(~img_thresh)[0], return_counts=True)
        count = sizes[np.logical_and(sizes >= sizelim_small,
                                     sizes < sizelim_big)].size

        # Keep count
        counts.append(count)

    # Compute and apply final threshold
    threshold = base_threshold + offsets[np.argmax(counts)]
    img_thresh = img > threshold

    # Report
    if verbose: print "-- Threshold:", base_threshold, '+', offsets[np.argmax(counts)]

    # Return resulting thresholded image
    return img_thresh


#------------------------------------------------------------------------------

# FUNCTION: LABELING CELLS AND REMOVING ARTEFACTS OF MEMBRANE THRESHOLDING

def _clean_membranes(img_thresh, sizelim_small=1000, sizelim_big=1000000,
                     verbose=False, fill_holes=True):
    """Intermediate clean-up: cleaning the membrane thresholding

    Removes disconnected components, labels the cell bodies (the connected
    components of the inverted image), removes very small labeled objects and
    marks very large labeled objects as background.

    The default size thresholds are based on segmentation of the lateral line.

    WARNING: If the target object is not connected to the image boundary
    anywhere, it will be removed completely!
    """

    # Clean membrane segmentation by removing disconnected objects
    if fill_holes:
        img_thresh = ~ndi.binary_fill_holes(~img_thresh)

    # Label
    img_lbl, cell_num = ndi.label(~img_thresh)

    # Get object sizes and discard small fragments
    if verbose: print '-- Objects before filter:', cell_num
    cell_IDs, cell_sizes = np.unique(img_lbl, return_counts=True)
    keep = cell_IDs[cell_sizes >= sizelim_small]
    bg_objects = cell_IDs[cell_sizes >= sizelim_big]
    img_lbl[np.in1d(img_lbl.flatten(), keep, invert=True).reshape(img_lbl.shape)] = 0   # Nice...
    if verbose: print '-- Objects after filter:', len(np.unique(img_lbl))-1

    # Return results
    return img_lbl, bg_objects


#------------------------------------------------------------------------------

# FUNCTION: EXPANDING LABELS BY WATERSHED ON SMOOTHED IMAGE

def _watershed_on_smoothed(img, lbl, sigma=3):
    """Expand labels: watershed on smoothed image

    Expand labels `lbl` by performing watershed on image `img` following
    Gaussian smoothing with `sigma`. The default value of `sigma` is `3` and
    was chosen based on segmentation of the lateral line.
    """

    # Smoothen the image for the watershed (to get smooth cell surfaces)
    # XXX: FLAG -- IMPROVEMENT: This helps a bit but is not perfect...
    img = ndi.gaussian_filter(img, sigma=sigma)

    # Expand with watershed to partition remaining space (formerly membranes)
    lbl = watershed(img, lbl)

    # Return result
    return lbl


#------------------------------------------------------------------------------

# FUNCTION: POSTPROCESSING (VARIOUS)

def _postprocessing(img_lbl, bg_objects, verbose=False):
    """Various postprocessing stuff on a labeled image (`img_lbl`).

    Set background objects to 0 (based on `bg_objects` produced by function
    `_clean_membranes`. Remove cells disconnected from the tissue (assuming the
    tissue is the largest overall object). Make the numbering of the labels
    continuous again.
    """

    # Set background objects to 0 (determined by size original threshold)
    img_lbl[np.in1d(img_lbl.flatten(), bg_objects).reshape(img_lbl.shape)] = 0

    # Remove cells that are disconnected from the prim
    chunk_lbl    = ndi.label(img_lbl!=0)[0]              # Identify chunks
    chunk_counts = np.bincount(chunk_lbl.flatten())[1:]  # Get sizes
    img_lbl[chunk_lbl!=np.argmax(chunk_counts)+1] = 0    # Keep largest chunk
    if verbose:
        print "-- Objects discarded as disconnected from tissue:",
        print chunk_counts.shape[0]-1

    # Make the numbering continuous
    img_lbl_new = np.zeros_like(img_lbl)
    for new_id,old_id in enumerate(np.unique(img_lbl)):
        img_lbl_new[img_lbl==old_id] = new_id
    img_lbl = img_lbl_new

    # return result
    return img_lbl


#------------------------------------------------------------------------------

# FUNCTION: AUTOMATED 3D SINGLE-CELL SEGMENTATION (FULL PIPELINE)

def segment_3D_legacy(fpath, verbose=False,
                      params={'median_size'     : 3,
                              'thresh_stepsize' : 2,
                              'thresh_lower'    : 4.0,
                              'clean_small'     : 1000,
                              'clean_big'       : 1000000,
                              'expansion_sigma' : 3} ):
    """Single-cell segmentation of high-quality 3D membrane-labeled stacks.

    WARNING: This function has been replaced by a new and better version!

    Produces a tif stack of identical shape as the input with the pixels
    belonging to each cell labeled with unique integers (background is 0).

    Note that this pipeline is put together from individual functions which can
    be exchanged very easily to adapt the pipeline to different samples or
    different types of images.

    WARNING: This has been implemented to segment high-quality 3D stacks of the
    Zebrafish posterior lateral line primordium labeled by cldnB::Lyn:EGFP and
    acquired at the Zeiss LSM880. A number of steps will NOT work as intended
    with other types of tissues or lower-quality images!

    Parameters
    ----------
    fpath : string
        The path (either local from cwd or global) to the input tif file of a
        membrane-labeled 3D image stack.
    params : dict, optional
        Dictionary containing additional parameters for the various individual
        processing steps. The default parameters are based on segmentation of
        the lateral line primordium. For details, see the docstrings of the
        respective functions.
    verbose : bool, optional, default False
        If True, more information is printed.
    """

    # Deprecation warning
    warn("This segmentation function has been replaced by a new and better "+
         "version. It is only here for legacy support.", DeprecationWarning)


    #--------------------------------------------------------------------------

    ### Load data

    if verbose: print "Loading stack..."

    # Try loading the stack
    try:
        img_raw = imread(fpath)
    except:
        print "Attempting to load stack failed with this error:"
        raise

    # Check dimensionality
    if not img_raw.ndim == 3:
        raise IOError("Expected a 3D stack, got "+str(img_raw.ndim)+"D.")


    #--------------------------------------------------------------------------

    ### Preprocessing

    if verbose: print "Preprocessing..."

    # Small median filter to clean up noise
    img_rdy = _median_smooth(img_raw, params['median_size'])


    #--------------------------------------------------------------------------

    ### Thresholding

    if verbose: print "Searching threshold..."

    # Automated thresholding by object-count threshold detection
    img_thresh = _object_count_threshold(img_rdy,
                                         params['thresh_stepsize'],
                                         params['thresh_lower'],
                                         verbose=verbose)


    #--------------------------------------------------------------------------

    ### Intermediate clean-up step

    if verbose: print "Cleaning & filtering binarized image..."

    # Removing disconnected components and size-based filtering
    img_lbl, bg_objects = _clean_membranes(img_thresh,
                                           sizelim_small=params['clean_small'],
                                           sizelim_big=params['clean_big'],
                                           verbose=verbose)


    #--------------------------------------------------------------------------

    ### Watershed expansion

    if verbose: print "Smoothing & watershedding..."

    # Watershed on Gaussian-smoothed image
    img_lbl = _watershed_on_smoothed(img_rdy, img_lbl,
                                     sigma=params['expansion_sigma'])


    #--------------------------------------------------------------------------

    ### Postprocessing

    if verbose: print "Postprocessing..."

    # Set bg objects to 0, remove disconnected objects, continuous labeling
    img_lbl = _postprocessing(img_lbl, bg_objects, verbose=verbose)


    #--------------------------------------------------------------------------

    ### Write the result and return

    if verbose: print "Saving result..."

    imsave(fpath[:-4]+'_seg.tif', img_lbl.astype(np.uint16), bigtiff=True)

    if verbose: print "Processing complete!"

    return


#------------------------------------------------------------------------------

# FUNCTION: IMPROVED AUTOMATED 3D SINGLE-CELL SEGMENTATION (FULL PIPELINE)

def segment_3D(fpath, verbose=False,
               params={'median_size'     : 3,
                       'gaussian_sigma'  : 3,
                       'do_bgsub'        : False,
                       'bgsub_method'    : 'mean',
                       'bgsub_sigma'     : 10,
                       'max_offset'      : 10,
                       'offset_step'     : 1,
                       'clean_small'     : 1000,
                       'clean_big'       : 1000000,
                       'expansion_sigma' : 3} ):
    """Improved single-cell segmentation of 3D membrane-labeled stacks.

    Produces a tif stack of identical shape as the input with the pixels
    belonging to each cell labeled with unique integers (background is 0).

    Note that this pipeline is put together from individual functions which can
    be exchanged very easily to adapt the pipeline to different samples or
    different types of images.

    WARNING: This has been implemented to segment high-quality 3D stacks of the
    Zebrafish posterior lateral line primordium labeled by cldnB::Lyn:EGFP and
    acquired at the Zeiss LSM880. A number of steps will NOT work as intended
    with other types of tissues or lower-quality images!

    Parameters
    ----------
    fpath : string
        The path (either local from cwd or global) to the input tif file of a
        membrane-labeled 3D image stack.
    params : dict, optional
        Dictionary containing additional parameters for the various individual
        processing steps. The default parameters are based on segmentation of
        the lateral line primordium. For details, see the docstrings of the
        respective functions.
    verbose : bool, optional, default False
        If True, more information is printed.
    """

    #--------------------------------------------------------------------------

    ### Load data

    if verbose: print "Loading stack..."

    # Try loading the stack
    try:
        img_raw = imread(fpath)
    except:
        print "Attempting to load stack failed with this error:"
        raise

    # Check dimensionality
    if not img_raw.ndim == 3:
        raise IOError("Expected a 3D stack, got "+str(img_raw.ndim)+"D.")


    #--------------------------------------------------------------------------

    ### Preprocessing

    if verbose: print "Preprocessing..."

    # Small median filter to clean up noise
    img_rdy = _median_smooth(img_raw, params['median_size'])

    # Gaussian filter to even out variations
    img_smooth = _gaussian_smooth(img_rdy, params['gaussian_sigma'])

    # Background subtraction
    if 'do_bgsub' in params.keys() and params['do_bgsub']:

        # Non-adaptive bgsub
        if params['bgsub_method'] in ['mean','median']:
            img_smooth = _background_sub(img_smooth,
                                         method=params['bgsub_method'])

        # Adaptive bgsub
        elif params['bgsub_method'] == 'adaptive':
            bg = _gaussian_smooth(img_smooth, params['bgsub_sigma'])
            img_smooth = _background_sub(img_smooth,
                                         method=params['bgsub_method'], bg=bg)

        # Error handling
        else:
            raise ValueError("bgsub_method not recognized.")


    #--------------------------------------------------------------------------

    ### Thresholding

    if verbose: print "Searching threshold..."

    # Automated thresholding by object-count threshold detection
    img_thresh = _improved_threshold(img_smooth,
                                     max_offset=params['max_offset'],
                                     offset_step=params['offset_step'],
                                     sizelim_small=params['clean_small'],
                                     sizelim_big=params['clean_big'],
                                     verbose=verbose)


    #--------------------------------------------------------------------------

    ### Intermediate clean-up step

    if verbose: print "Cleaning & filtering binarized image..."

    # Removing disconnected components and size-based filtering
    img_lbl, bg_objects = _clean_membranes(img_thresh, fill_holes=False,
                                           sizelim_small=params['clean_small'],
                                           sizelim_big=params['clean_big'],
                                           verbose=verbose)


    #--------------------------------------------------------------------------

    ### Watershed expansion

    if verbose: print "Smoothing & watershedding..."

    # Watershed on Gaussian-smoothed image
    img_lbl = _watershed_on_smoothed(img_rdy, img_lbl,
                                     sigma=params['expansion_sigma'])


    #--------------------------------------------------------------------------

    ### Postprocessing

    if verbose: print "Postprocessing..."

    # Set bg objects to 0, remove disconnected objects, continuous labeling
    img_lbl = _postprocessing(img_lbl, bg_objects, verbose=verbose)


    #--------------------------------------------------------------------------

    ### Write the result and return

    if verbose: print "Saving result..."

    imsave(fpath[:-4]+'_seg.tif', img_lbl.astype(np.uint16), bigtiff=True)

    if verbose: print "Processing complete!"

    return


#------------------------------------------------------------------------------



