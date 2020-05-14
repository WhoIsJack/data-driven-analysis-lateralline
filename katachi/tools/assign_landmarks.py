# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 11:04:07 2017

@author:    Jonas Hartmann @ Gilmour group @ EMBL Heidelberg

@descript:  Generates a sparse 3D point cloud representation for each cell in
            a segmented stack based on the intensity distribution of a second
            stack within the cell's segmentation region.

"""

#------------------------------------------------------------------------------

# IMPORTS

# External
from __future__ import division
import os, pickle
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from tifffile import imread

# Internal
from katachi.utilities.ISLA import isla


#------------------------------------------------------------------------------

# FUNCTION: ASSIGN LANDMARKS TO EACH CELL OF A SEGMENTATION

def assign_landmarks(fpath_seg, fpath_int, num_LMs, save_centroids=False,
                     fpath_out=None, show_cells=None, verbose=False,
                     global_prep_func=None, global_prep_params=None,
                     local_prep_func=None, local_prep_params=None,
                     landmark_func='default', landmark_func_params=None):
    """Generate a sparse 3D point cloud representation of a segmented cell.

    For each segmented object (cell) specified by a labeled segmentation stack,
    a point cloud is generated that represents the intensity distribution of a
    second stack in that cell's region.

    Custom preprocessing functions can be applied both to the global image and
    locally to each cell crop-out. The function used for determining landmark
    coordinates may also be specified by the user. By default, no preprocessing
    is done and landmarks are extracted using ISLA (for more information, see
    `katachi.utilities.ISLA`).

    The results are written as a .npy file containing an array of shape (N,M,3)
    where N is the number of cells, M the number of landmarks per cell and 3
    are the three spatial dimensions of the input stack. The origin of the
    coordinate system is set to the centroid of the cell's segmentation region
    and the axes are scaled according to the specified pixel resolution.

    Parameters
    ----------
    fpath_seg : string
        The path (either local from cwd or global) to a tif file containing
        the single-cell segmentation stack. A point cloud will be generated for
        each labeled object in this segmentation.
    fpath_int : string
        The path (either local from cwd or global) to a tif file containing
        intensity values. This stack must have the same shape as the fpath_seg
        stack. The point clouds generated for each cell in the segmentation
        represent the intensity distribution of this stack within that cell.
    num_LMs : int
        The number of landmarks to extract for each cell.
    save_centroids : bool, optional, default False
        If True, the centroid coordinates (in the image space) of each labeled
        object will be added to the stack_metadata file (with key "centroids").
    fpath_out : string, optional, default None
        A path (either local from cwd or global) specifying a file to which the
        results should be written. If None, fpath_int is used but with the
        suffix `_LMs.npy` instead of the original suffix `.tif`.
    show_cells : int, optional, default None
        If an int `i` is given, a slice from each cell up to `i` is displayed
        with an overlay of the extracted landmarks. Once `i` cells have been
        processed, the rest of the cells is processed without displaying.
    verbose : bool, optional, default False
        If True, more information is printed.
    global_prep_func : callable, optional, default None
        A user-defined function for custom preprocessing. This function will
        be applied to the entire intensity stack prior to landmark extraction.
        The callable must accept the intensity stack and the segmentation stack
        as the first two arguments and it may accept a third argument that
        packages additional parameters (see just below). The function must
        return another valid intensity stack that is of the same shape as the
        original.
    global_prep_params : any type, optional, default None
        Third argument (after intensity stack and segmentation stack) to the
        global_prep_func (see just above). This could be anything, but will
        usually be a list of additional parameters for the function.
    local_prep_func : callable, optional, default None
        A user-defined function for custom preprocessing. This function will
        be applied to the bounding box crop of each cell just prior to landmark
        extraction. The callable must accept the crops of the intensity stack
        and the segmentation stack as well as the cell index as the first three
        arguments. It may accept a fourth argument that packages additional
        parameters (see just below). The function must return another valid
        cropped intensity stack that is of the same shape as the original.
    local_prep_params : any type, optional, default None
        4th argument (after intensity crop, segmentation crop and cell index)
        to the local_prep_func (see just above). This could be anything, but
        will usually be a list of additional parameters for the function.
    landmark_func : callable or 'default', optional, default 'default'
        Function used to extract landmark coordinates from the bounding box
        crop of each cell after the region around the cell (outside the cell's
        segmentation region) have been set to zero. The function must accept
        the (masked) intensity crop and num_LMs as its first two arguments and
        may accept a third argument that packages additional parameters (see
        just below). The function must return an array of shape (M,3), where M
        is the number of landmarks and 3 are the three spatial dimensions of
        the input cell.
    landmark_func_params : any type, optional, default None
        Third argument (after masked intensity crop and num_LMs) to the
        landmark_func (see just above). This could be anything, but will
        usually be a list of additional parameters for the function.
    """

    #--------------------------------------------------------------------------

    ### Load data

    if verbose: print "Loading stacks..."

    # Try loading the segmentation stack
    try:
        img_seg = imread(fpath_seg)
    except:
        print "Attempting to load segmentation stack failed with this error:"
        raise

    # Check dimensionality
    if not img_seg.ndim == 3:
        raise IOError("Expected a 3D segmentation stack, got " +
                      str(img_seg.ndim) + "D instead.")

    # Try loading the intensity stack
    try:
        img_int = imread(fpath_int)
    except:
        print "Attempting to load intensity stack failed with this error:"
        raise

    # Check dimensionality
    if not img_int.ndim == 3:
        raise IOError("Expected a 3D intensity stack, got " +
                      str(img_int.ndim) + "D instead.")

    # Double-check that show_cells is an integer
    if show_cells is not None and type(show_cells) != int:
        raise IOError("Argument show_cells expects int or None, got " +
                      str(type(show_cells)))

    # Load the metadata and get the resolution from it
    try:
        dirpath, fname = os.path.split(fpath_int)
        fpath_meta = os.path.join(dirpath, fname[:10]+"_stack_metadata.pkl")
        with open(fpath_meta, 'rb') as metafile:
            meta_dict = pickle.load(metafile)
            res = meta_dict['resolution']
    except:
        print "Getting resolution from metadata failed with this error:"
        raise


    #--------------------------------------------------------------------------

    ### Apply a user-defined global preprocessing function

    if global_prep_func is not None:
        if verbose: print "Applying global preprocessing function..."
        if global_prep_params:
            img_int = global_prep_func(img_int, img_seg, global_prep_params)
        else:
            img_int = global_prep_func(img_int, img_seg)


    #--------------------------------------------------------------------------

    ### Run landmark extraction for each cell

    if verbose: print "Performing landmark assignments..."

    # Get bounding boxes
    bboxes = ndi.find_objects(img_seg)

    # Prep
    cell_indices = np.unique(img_seg)[1:]
    landmarks    = np.zeros((cell_indices.size, num_LMs, 3))
    if save_centroids: centroids = np.zeros((cell_indices.size, 3))

    # For each cell...
    for c, cell_idx in enumerate(cell_indices):

        # Crop the cell
        cell_int = np.copy(img_int[bboxes[c][0], bboxes[c][1], bboxes[c][2]])
        cell_seg =         img_seg[bboxes[c][0], bboxes[c][1], bboxes[c][2]]

        # Apply a user-defined local preprocessing function
        if local_prep_func is not None:
            if local_prep_params:
                cell_int = local_prep_func(cell_int, cell_seg, cell_idx,
                                           local_prep_params)
            else:
                cell_int = local_prep_func(cell_int, cell_seg, cell_idx)

        # Mask the signal
        cell_int[cell_seg!=cell_idx] = 0

        # Assign landmarks
        if landmark_func == 'default':
            landmarks[c,:,:] = isla(cell_int, num_LMs, seed=42)
        elif landmark_func_params is None:
            landmarks[c,:,:] = landmark_func(cell_int, num_LMs)
        else:
            landmarks[c,:,:] = landmark_func(cell_int, num_LMs,
                                             params=landmark_func_params)

        # Show the first few cells
        if show_cells is not None:

            plt.imshow(cell_int[cell_int.shape[0]//2,:,:],
                       interpolation='none', cmap='gray')
            lms_show = landmarks[c,landmarks[c,:,0]==cell_int.shape[0]//2,:]
            plt.scatter(lms_show[:,2], lms_show[:,1],
                        c='red', edgecolor='', marker='s', s=5)
            plt.title("Cell "+str(c+1))
            plt.axis('off')
            plt.show()

            if c >= show_cells-1:
                show_cells = None

        # Center origin on segmentation object's centroid
        # Note: `centroid==center_of_mass` for uniformly dense bodies.
        centroid = np.array(ndi.center_of_mass(cell_seg==cell_idx))
        landmarks[c,:,:] = landmarks[c,:,:] - centroid

        # Keep the centroids (relative to image space) as metadata
        if save_centroids:
            centroids[c,:] = centroid + np.array( [bboxes[c][d].start
                                                   for d in range(3)] )

    # Scale the axes to the stack's pixel resolution
    landmarks = landmarks * np.array(res)
    if save_centroids: centroids = centroids * np.array(res)


    #--------------------------------------------------------------------------

    ### Save, report and return

    if verbose: print "Saving result..."

    # Save the landmarks
    if fpath_out is None:
        np.save(fpath_int[:-4]+"_LMs", landmarks)
    else:
        np.save(fpath_out, landmarks)

    # Save the centroids to the metadata
    if save_centroids:
        dirpath, fname = os.path.split(fpath_int)
        fpath_meta = os.path.join(dirpath, fname[:10]+"_stack_metadata.pkl")
        with open(fpath_meta, 'rb') as metafile:
            meta_dict = pickle.load(metafile)
        meta_dict["centroids"] = centroids
        with open(fpath_meta, 'wb') as metafile:
            pickle.dump(meta_dict, metafile, pickle.HIGHEST_PROTOCOL)

    # Report and return
    if verbose: print "Processing complete!"
    return


#------------------------------------------------------------------------------



