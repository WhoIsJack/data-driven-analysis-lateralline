# -*- coding: utf-8 -*-
"""
Created on Wed Jan 03 11:04:25 2018

@author:    Jonas Hartmann @ Gilmour group @ EMBL Heidelberg

@descript:  Functions to extract covariates from the 3D image of each cell in a
            segmented stack based either on the segmentation mask itself or on
            the intensity distribution in a second channel.
"""

#------------------------------------------------------------------------------

# IMPORTS

# External, general
from __future__ import division
import pickle
import numpy as np
import scipy.ndimage as ndi
from tifffile import imread

# External, specific
from skimage.measure import marching_cubes_lewiner, mesh_surface_area
from sklearn.decomposition import PCA

# Internal
from katachi.utilities.hierarchical_data import HierarchicalData


#------------------------------------------------------------------------------

# FUNCTION: EXTRACT PER-SAMPLE COVARIATES FROM SEGMENTATION

def get_img_covars_sample(fpath_seg, img_seg=None, covars=None,
                          verbose=False):
    """Extract per-sample covariates from segmentation mask.

    Currently extracts the following covariates:
        - covars.img.sample.cellnum
        - covars.img.sample.volume

    To be implemented:
        - covars.img.sample.rosettes.* [see TODO in code]
        - ...?

    Parameters
    ----------
    fpath_seg : string
        The path (either local from cwd or global) to a tif file containing
        a 3D single-cell segmentation stack.
    img_seg : 3D numpy array, optional, default None
        3D single-cell segmentation stack to be used instead of loading the
        stack from the file at fpath_seg. If this is passed, fpath_seg is
        ignored.
    covars : HierarchicalData instance, optional, default None
        An instance of katachi.utilities.hierarchical_data.HierarchicalData to
        which the determined covariates will be added. If this is None, a new
        instance is created.
    verbose : bool, optional, default False
        If True, more information is printed.

    Returns
    -------
    covars : HierarchicalData instance
        An instance of katachi.utilities.hierarchical_data.HierarchicalData
        containing the generated covariate measurements in addition to any
        covariates that it the parameter `covars` already contained if it was
        passed.
    """

    #--------------------------------------------------------------------------

    ### Load data

    # Ignore loading if stack is given
    if img_seg is None:

        # Report
        if verbose: print "Loading data..."

        # Try loading the segmentation stack
        try:
            img_seg = imread(fpath_seg)
        except:
            print "Attempting to load seg stack failed with this error:"
            raise

    # Check dimensionality
    if not img_seg.ndim == 3:
        raise IOError("Expected a 3D segmentation stack, got " +
                      str(img_seg.ndim) + "D instead.")


    #--------------------------------------------------------------------------

    ### Prepare covariate class

    # Ignore if class is given
    if covars is None:

        # Instantiate empty hierarchical class
        covars = HierarchicalData()


    #--------------------------------------------------------------------------

    ### Extract simple prim covariates

    if verbose: print "Extracting simple prim covariates..."

    # Number of cells [covars.img.sample.cellnum]
    cell_ids = np.unique(img_seg)
    cell_ids = cell_ids[1:]
    cell_num = cell_ids.size
    covars.img.sample.cellnum = cell_num

    # Total prim volume [covars.img.sample.volume]
    covars.img.sample.volume = np.sum(img_seg != 0)


    #--------------------------------------------------------------------------

    ### Extract rosette state

    #if verbose: print "Extracting rosette covariates..."

    # TODO!
    # - Number of rosettes
    # - Rosette positions (centroids/lumina)
    # - Distance between rosettes
    # - Size of leading region
    # - Timing in deposition cycle
    pass


    #--------------------------------------------------------------------------

    ### Report and return results

    if verbose: print "Processing complete!"
    return covars


#------------------------------------------------------------------------------

# FUNCTION: EXTRACT PER-CELL COVARIATES AT THE TISSUE LEVEL FROM SEGMENTATION

def get_img_covars_tissue(fpath_seg, img_seg=None, covars=None,
                          verbose=False):
    """Extract per-cell tissue-scale covariates from segmentation mask.

    Currently extracts the following covariates:
        - covars.img.tissue.centroids
        - covars.img.tissue.bboxes
        - covars.img.tissue.neighbor_ids
        - covars.img.tissue.neighbor_num
        - covars.img.tissue.neighbor_contact_areas
        - covars.img.tissue.outside_contact_area

    To be implemented:
        - covars.img.tissue.rosettes.* [see TODO in code]
        - ...?

    Parameters
    ----------
    fpath_seg : string
        The path (either local from cwd or global) to a tif file containing
        a 3D single-cell segmentation stack.
    img_seg : 3D numpy array, optional, default None
        3D single-cell segmentation stack to be used instead of loading the
        stack from the file at fpath_seg. If this is passed, fpath_seg is
        ignored.
    covars : HierarchicalData instance, optional, default None
        An instance of katachi.utilities.hierarchical_data.HierarchicalData to
        which the determined covariates will be added. If this is None, a new
        instance is created.
    verbose : bool, optional, default False
        If True, more information is printed.

    Returns
    -------
    covars : HierarchicalData instance
        An instance of katachi.utilities.hierarchical_data.HierarchicalData
        containing the generated covariate measurements in addition to any
        covariates that it the parameter `covars` already contained if it was
        passed.
    """

    #--------------------------------------------------------------------------

    ### Load data

    # Ignore loading if stack is given
    if img_seg is None:

        # Report
        if verbose: print "Loading data..."

        # Try loading the segmentation stack
        try:
            img_seg = imread(fpath_seg)
        except:
            print "Attempting to load seg stack failed with this error:"
            raise

    # Check dimensionality
    if not img_seg.ndim == 3:
        raise IOError("Expected a 3D segmentation stack, got " +
                      str(img_seg.ndim) + "D instead.")


    #--------------------------------------------------------------------------

    ### Prepare covariate class

    # Ignore if class is given
    if covars is None:

        # Instantiate empty hierarchical class
        covars = HierarchicalData()


    #--------------------------------------------------------------------------

    ### Extract position-based covariates

    if verbose: print "Extracting position-based covariates..."

    # Prep: Get cell labels
    cell_ids = np.unique(img_seg)
    cell_ids = cell_ids[1:]

    # Centroid positions in image space [covars.img.tissue.centroids]
    centroids = np.array(ndi.measurements.center_of_mass(img_seg!=0, img_seg,
                                                         index=cell_ids))
    covars.img.tissue.centroids = centroids

    # Get bounding boxes
    bboxes = ndi.find_objects(img_seg)
    covars.img.tissue.bboxes = bboxes


    #--------------------------------------------------------------------------

    ### Extract neighbor/boundary-based covariates

    if verbose: print "Extracting neighbor-based covariates..."

    # For each cell...
    neighbor_ids = []
    neighbor_num = []
    neighbor_contact = []
    outside_contact  = []
    for cell_index, cell_id in enumerate(cell_ids):

        # Crop to bounding box to make it more efficient
        # NOTE: There is a very small risk in this of loosing a neighbor!
        crop = img_seg[bboxes[cell_index][0],
                       bboxes[cell_index][1],
                       bboxes[cell_index][2]]

        # Get outer cell shell
        shell = np.logical_xor(crop==cell_id,
                               ndi.binary_dilation(crop==cell_id))

        # Get neighbors
        n_ids = list(np.unique(crop[shell]))
        if 0 in n_ids:
            n_ids = n_ids[1:]
        neighbor_ids.append(n_ids)
        neighbor_num.append(len(n_ids))

        # Get neighbor contact areas
        n_contacts = []
        for n_id in n_ids:
            n_contacts.append( np.sum(crop[shell] == n_id) )
        neighbor_contact.append(n_contacts)

        # Get outside contact area
        outside_contact.append( np.sum(crop[shell]==0) )

    # Add to covariates
    covars.img.tissue.neighbor_ids = neighbor_ids
    covars.img.tissue.neighbor_num = np.array(neighbor_num)
    covars.img.tissue.neighbor_contact_areas = neighbor_contact
    covars.img.tissue.outside_contact_area   = np.array(outside_contact)


    #--------------------------------------------------------------------------

    ### Extract rosette-based covariates

    #if verbose: print "Extracting rosette-based covariates..."

    # TODO!
    # - Distance of centroid from each rosette/lumen
    # - Distance of centroid from tip
    # - Distance of closest voxel from each rosette/lumen
    # - Distance of closest voxel from tip
    # - Rosette assignment
    pass


    #--------------------------------------------------------------------------

    ### Report and return results

    if verbose: print "Processing complete!"
    return covars


#------------------------------------------------------------------------------

# FUNCTION: EXTRACT PER-CELL COVARIATES AT CELL LEVEL FROM SEGMENTATION

def get_img_covars_cell_seg(fpath_seg, fpath_meta,
                            img_seg=None, metadata=None, covars=None,
                            verbose=False):
    """Extract per-cell cell-scale covariates from segmentation mask.

    Currently extracts the following covariates:
        - covars.img.cell.volume_vxl
        - covars.img.cell.volume_um3
        - covars.img.cell.surface_area_vxl
        - covars.img.cell.marching_cubes_lewiner
        - covars.img.cell.surface_area_um2
        - covars.img.cell.sphericity_surfratio
        - covars.img.cell.roundness_volsurfextentsratio

    To be implemented:
        - ...?

    Parameters
    ----------
    fpath_seg : string
        The path (either local from cwd or global) to a tif file containing
        a 3D single-cell segmentation stack.
    fpath_meta : string
        The path (either local from cwd or global) to a pkl file containing
        metadata, including the TFOR centroid positions as computed in
        katachi.tools.find_TFOR.
    img_seg : 3D numpy array, optional, default None
        3D single-cell segmentation stack to be used instead of loading the
        stack from the file at fpath_seg. If this is passed, fpath_seg is
        ignored.
    metadata : metadata dictionary, optional, default None
        A dictionary of metadata to be used instead of the file at fpath_meta.
        If this is passed, fpath_meta is ignored.
    covars : HierarchicalData instance, optional, default None
        An instance of katachi.utilities.hierarchical_data.HierarchicalData to
        which the determined covariates will be added. If this is None, a new
        instance is created.
    verbose : bool, optional, default False
        If True, more information is printed.

    Returns
    -------
    covars : HierarchicalData instance
        An instance of katachi.utilities.hierarchical_data.HierarchicalData
        containing the generated covariate measurements in addition to any
        covariates that it the parameter `covars` already contained if it was
        passed.
    """

    #--------------------------------------------------------------------------

    ### Load data

    # Load segmentation
    if img_seg is None:

        # Report
        if verbose: print "Loading data..."

        # Try loading the segmentation stack
        try:
            img_seg = imread(fpath_seg)
        except:
            print "Attempting to load seg stack failed with this error:"
            raise

    # Check dimensionality
    if not img_seg.ndim == 3:
        raise IOError("Expected a 3D segmentation stack, got " +
                      str(img_seg.ndim) + "D instead.")

    # Load metadata
    if metadata is None:

        # Report
        if verbose: print "Loading metadata..."

        # Try loading the landmark data
        try:
            with open(fpath_meta, 'rb') as infile:
                metadata = pickle.load(infile)
        except:
            print "Attempting to load metadata failed with this error:"
            raise


    #--------------------------------------------------------------------------

    ### Prepare covariate class

    # Ignore if class is given
    if covars is None:

        # Instantiate empty hierarchical class
        covars = HierarchicalData()


    #--------------------------------------------------------------------------

    ### Extract cell level covariates from segmentation

    if verbose: print "Extracting cell level segmentation covariates..."

    # Prep: Get cell labels and bounding boxes
    cell_ids = np.unique(img_seg)
    cell_ids = cell_ids[1:]
    bboxes = ndi.find_objects(img_seg)

    # Volume in voxels [covars.img.cell.volume_vxl]
    volumes_vxl = ndi.measurements.sum(img_seg!=0, img_seg, index=cell_ids)
    covars.img.cell.volume_vxl = volumes_vxl

    # Volume in microns cubed [covars.img.cell.volume_um3]
    volumes_um3 = volumes_vxl * np.product(metadata['resolution'])
    covars.img.cell.volume_um3 = volumes_um3


    ## Surface area in voxels [covars.img.cell.surface_area_vxl]

    # For each cell...
    surface_areas_vxl = []
    for cell_index, cell_id in enumerate(cell_ids):

        # Crop to bounding box to make it more efficient
        crop = img_seg[bboxes[cell_index][0],
                       bboxes[cell_index][1],
                       bboxes[cell_index][2]]

        # Get inner cell shell
        shell = np.logical_xor(crop==cell_id,
                               ndi.binary_erosion(crop==cell_id))

        # Get surface area
        surface_areas_vxl.append( np.sum(shell) )

    # Add to covariates
    covars.img.cell.surface_area_vxl = np.array(surface_areas_vxl)


    ## Meshed surface based on Lewiner's Marching Cubes Algorithm
    ## [covars.img.cell.marching_cubes_lewiner]
    ## [covars.img.cell.surface_area_um2]

    # For each cell...
    meshes = []
    surface_areas_um2 = []
    for cell_index, cell_id in enumerate(cell_ids):

        # Get cell mask
        mask = img_seg==cell_id

        # Marching cubes
        out = marching_cubes_lewiner(mask, spacing=metadata['resolution'],
                                     step_size=2)

        # Handle resulting mesh
        verts, faces, normals, values = out
        mesh = {'verts' : verts, 'faces' : faces,
                'normals' : normals, 'values' : values}
        meshes.append(mesh)

        # Get surface area
        surf = mesh_surface_area(verts, faces)
        surface_areas_um2.append(surf)

    # Add to covariates
    surface_areas_um2 = np.array(surface_areas_um2)
    covars.img.cell.marching_cubes_lewiner = meshes
    covars.img.cell.surface_area_um2 = surface_areas_um2


    # Sphericity as ratio of actual surface area to surface area of volume-matched sphere
    # [covars.img.cell.sphericity_surfratio]
    sphere_radii   = ( (3/4) * (volumes_um3/np.pi) )**(1/3)
    sphere_surfs   = 4 * np.pi * sphere_radii**2
    surface_ratios = sphere_surfs / surface_areas_um2
    covars.img.cell.sphericity_surfratio = surface_ratios

    # Roundness as ratio of volume to surface area normalized by major bounding box extents
    # [covars.img.cell.roundness_volsurfextentsratio]
    extents_pca = np.empty((len(cell_ids),3))
    for i in range(len(cell_ids)):
        cell_pca = PCA()
        lms_pca = cell_pca.fit_transform(meshes[i]['verts'])
        extents_pca[i,:] = np.max(lms_pca, axis=0) - np.min(lms_pca, axis=0)
    extents_geom = np.product(extents_pca, axis=1)**(1/3)
    vol_surf_extent_ratios = volumes_um3 / (surface_areas_um2 * extents_geom)
    covars.img.cell.roundness_volsurfextentsratio = vol_surf_extent_ratios


    #--------------------------------------------------------------------------

    ### Report and return results

    if verbose: print "Processing complete!"
    return covars


#------------------------------------------------------------------------------

# FUNCTION: EXTRACT PER-CELL COVARIATES AT CELL LEVEL FROM INTENSITY

def get_img_covars_cell_int(fpath_seg, fpath_int, channel_name, mem_d,
                            img_seg=None, img_int=None, covars=None,
                            verbose=False):
    """Extract per-cell cell-scale covariates from an intensity channel.

    Currently extracts the following covariates:
        - covars.img.cell.<channel_name>.sum_total
        - covars.img.cell.<channel_name>.mean_total
        - covars.img.cell.<channel_name>.sum_membrane
        - covars.img.cell.<channel_name>.mean_membrane
        - covars.img.cell.<channel_name>.sum_inside
        - covars.img.cell.<channel_name>.mean_inside
        - covars.img.cell.<channel_name>.sum_apical
        - covars.img.cell.<channel_name>.mean_apical
        - covars.img.cell.<channel_name>.sum_basal
        - covars.img.cell.<channel_name>.mean_basal

    To be implemented:
        - covars.img.cell.<channel_name>.object_counts [see TODO in code]
        - ...?

    Parameters
    ----------
    fpath_seg : string
        The path (either local from cwd or global) to a tif file containing
        a 3D single-cell segmentation stack.
    fpath_int : string
        The path (either local from cwd or global) to a tif file containing
        a 3D fluorescence intensity stack.
    channel_name : string
        The intensity channel name, i.e. the name to be used as attribute in
        the hiararchical covars class for accessing measurements derived from
        the given intensity stack.
    mem_d : int
        Thickness (in voxels) of cell shell to be used for measuring intensity
        in the membrane region.
    img_seg : 3D numpy array, optional, default None
        3D single-cell segmentation stack to be used instead of loading the
        stack from the file at fpath_seg. If this is passed, fpath_seg is
        ignored.
    img_int : 3D numpy array, optional, default None
        3D fluorescence intensity stack to be used instead of loading the stack
        from the file at fpath_int. If this is passed, fpath_int is ignored.
    covars : HierarchicalData instance, optional, default None
        An instance of katachi.utilities.hierarchical_data.HierarchicalData to
        which the determined covariates will be added. If this is None, a new
        instance is created.
    verbose : bool, optional, default False
        If True, more information is printed.

    Returns
    -------
    covars : HierarchicalData instance
        An instance of katachi.utilities.hierarchical_data.HierarchicalData
        containing the generated covariate measurements in addition to any
        covariates that it the parameter `covars` already contained if it was
        passed.
    """

    #--------------------------------------------------------------------------

    ### Load data

    # Ignore loading if segmentation stack is given
    if img_seg is None:

        # Report
        if verbose: print "Loading segmentation data..."

        # Try loading the segmentation stack
        try:
            img_seg = imread(fpath_seg)
        except:
            print "Attempting to load seg stack failed with this error:"
            raise

    # Ignore loading if intensity stack is given
    if img_int is None:

        # Report
        if verbose: print "Loading intensity data..."

        # Try loading the segmentation stack
        try:
            img_int = imread(fpath_int)
        except:
            print "Attempting to load seg stack failed with this error:"
            raise

    # Check shape
    if not img_int.shape == img_seg.shape:
        raise IOError("The shape of the intensity image must be identical to "+
                      "that of the segmentation image!")


    #--------------------------------------------------------------------------

    ### Prepare covariate class

    # Ignore if class is given
    if covars is None:

        # Instantiate empty hierarchical class
        covars = HierarchicalData()


    #--------------------------------------------------------------------------

    ### Extract cell level covariates from intensity

    if verbose: print "Extracting cell level intensity covariates..."

    # Prep: Get cell labels and bounding boxes
    cell_ids = np.unique(img_seg)
    cell_ids = cell_ids[1:]
    bboxes = ndi.find_objects(img_seg)

    # Mean and sum of total intensity
    sum_total  = ndi.measurements.sum(img_int, img_seg, index=cell_ids)
    mean_total = ndi.measurements.mean(img_int, img_seg, index=cell_ids)
    covars.img.cell._gad(channel_name).sum_total  = sum_total
    covars.img.cell._gad(channel_name).mean_total = mean_total

    # For each cell...
    sum_membrane  = []
    mean_membrane = []
    sum_inside    = []
    mean_inside   = []
    sum_apical    = []
    mean_apical   = []
    sum_basal     = []
    mean_basal    = []
    for cell_index, cell_id in enumerate(cell_ids):

        # Crop to bounding box to make it more efficient
        crop_seg = img_seg[bboxes[cell_index][0],
                           bboxes[cell_index][1],
                           bboxes[cell_index][2]]
        crop_int = img_int[bboxes[cell_index][0],
                           bboxes[cell_index][1],
                           bboxes[cell_index][2]]


        # Get inner cell shell of given thickness (mem_d)
        core  = ndi.binary_erosion(crop_seg==cell_id, iterations=mem_d)
        shell = np.logical_xor(crop_seg==cell_id, core)

        # Get membrane values
        sum_membrane.append( np.sum(crop_int[shell]) )
        mean_membrane.append( np.mean(crop_int[shell]) )

        # Get inner values
        sum_inside.append( np.sum(crop_int[core]) )
        mean_inside.append( np.mean(crop_int[core]) )

        # Get (sort of) apical values
        midslice = crop_seg.shape[0] // 2
        crop_seg_apical = crop_seg[midslice:, :, :] == cell_id
        crop_int_apical = crop_int[midslice:, :, :]
        sum_apical.append( np.sum(crop_int_apical[crop_seg_apical]) )
        mean_apical.append( np.mean(crop_int_apical[crop_seg_apical]) )

        # Get (sort of) basal values
        crop_seg_basal = crop_seg[:midslice, :, :] == cell_id
        crop_int_basal = crop_int[:midslice, :, :]
        sum_basal.append( np.sum(crop_int_basal[crop_seg_basal]) )
        mean_basal.append( np.mean(crop_int_basal[crop_seg_basal]) )

    # Add to covariates
    covars.img.cell._gad(channel_name).sum_membrane = np.array(sum_membrane)
    covars.img.cell._gad(channel_name).mean_membrane = np.array(mean_membrane)
    covars.img.cell._gad(channel_name).sum_inside = np.array(sum_inside)
    covars.img.cell._gad(channel_name).mean_inside = np.array(mean_inside)
    covars.img.cell._gad(channel_name).sum_apical = np.array(sum_apical)
    covars.img.cell._gad(channel_name).mean_apical = np.array(mean_apical)
    covars.img.cell._gad(channel_name).sum_basal = np.array(sum_basal)
    covars.img.cell._gad(channel_name).mean_basal = np.array(mean_basal)

    # TODO: Object counts over threshold series
    # NOTE: This is a simple idea for a latent feature of structure in the
    #       fluorescence distribution. However, it is neither common nor easy
    #       to interpret, so including it is not a priority.
    pass


    #--------------------------------------------------------------------------

    ### Report and return results

    if verbose: print "Processing complete!"
    return covars


#------------------------------------------------------------------------------




