# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 17:13:47 2017

@author:    Jonas Hartmann @ Gilmour group @ EMBL Heidelberg

@descript:  Prepares the data structure for a new tif image stack.
            This includes the following steps:
              - Generation of a unique ID
              - Generation of a subdirectory for the stack
              - Generation of a metadata file containing user-given metadata
              - For multi-channel stacks, channels are also saved separately
"""

#------------------------------------------------------------------------------

# IMPORTS

# External
from __future__ import division
import os, shutil, pickle, string
import numpy as np
from tifffile import imread, imsave


# Internal
from katachi.utilities import IDgenerator


#------------------------------------------------------------------------------

# FUNCTION: INITIALIZE NEW IMAGE STACK

def initialize_stack(fpath, idpath, meta_dict, verbose=False):
    """Intialize the data structure for a new tif image stack.
    
    This is the first step when adding any new data to the analysis.
    
    The input and target file are checked for consistency, a unique ID is
    generated and the stack is labeled with this ID and moved to a matching
    subdirectory. The subdirectory's name is simply the ID, whereas the new
    file name for the stack is `ID+"_"+previous_file_name`.
    
    In addition, a `ID+"_stack_metadata.pkl"` file is produced in the subdir,
    which contains the stack's metadata as a dictionary.
    
    Parameters
    ----------
    fpath : string
        The path (either local from cwd or global) to the new image stack for
        which the data structure should be initialized. The stack must be
        saved as an imageJ-style tif file.
    idpath : string or None
        Path of the text file containing previously generated IDs.
        Necessary to ensure that newly generated IDs are unique.
        If None, a UserWarning is produced and an ID returned without checking.
    meta_dict : dict 
        A dictionary containing the initial (user-defined) metadata for the
        stack. Section Notes below lists the keys that must be included.
    verbose : bool, optional, default False
        If True, more information is printed.
    
    Notes
    -----
    The meta_dict dictionary must contain the following keys:
    - 'channels'   : A list of strings naming the channels in order. Must not 
                     contain characters that cannot be used in file names.
    - 'resolution' : A list of floats denoting the voxel size of the input
                     stack in order ZYX.
    It may optionally contain other entries as well.
    """


    #--------------------------------------------------------------------------
    
    ### Check that the input (in particular meta_dict) make sense
    
    # Report
    if verbose: print "Checking input..."
    
    # Check for presence of essential keys in meta_dict
    essential_keys = ['channels', 'resolution']
    for e_key in essential_keys:
        if not e_key in meta_dict.keys():
            raise IOError("Essential key '" + e_key + 
                          "' is missing from meta_dict.")
        
    # Specific sanity checks
    
    if not isinstance(meta_dict['channels'], list):
        raise IOError("meta_dict['channels'] is expected to be a list.")
        
    if not len(meta_dict['channels']) >= 1:
        raise IOError("meta_dict['channels'] must be of length >= 1.")
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)

    for channel in meta_dict['channels']:
        if not isinstance(channel, str):
            raise IOError("meta_dict['channels'] contains non-string objects.")
    
    for channel in meta_dict['channels']:
        if not all([char in valid_chars for char in channel]):
            raise IOError("Channel "+channel+" in meta_dict contains invalid"+
                          " characters. Use only chars valid for file names!")
    
    if not len(meta_dict['resolution']) == 3:
        raise IOError("meta_dict['resolution'] should have length 3 (for each"+
                      " dimension of the 3D stack). Currently got "+
                      str(len(meta_dict['resolution'])) + ".")
                
    for r,res in enumerate(meta_dict['resolution']):
        try:
            meta_dict['resolution'][r] = float(res)
        except:
            print "Attempt to convert resolutions to float failed with error:"
            raise
    
    
    #--------------------------------------------------------------------------
    
    ### Load the file and double-check that it is a valid target
    
    # Add .tif to filename if necessary
    if not fpath.endswith('.tif'):
        fpath = fpath + '.tif'
    
    # Try loading the file
    try:
        img = imread(fpath)
        img = np.rollaxis(img, 1)
    except:
        print "Attempting to load stack failed with this error:"
        raise
    
    # Report
    if verbose: print "-- Loaded stack of shape", img.shape
    
    # Check the dimensions and number of channels
    if img.ndim < 3:
        raise IOError("A 3D stack is expected. " +
                      "2D images are currently not supported. " +
                      "Stack shape was " + str(img.shape) + ".")
        
    if len(meta_dict['channels']) == 1:
        if not img.ndim == 3:
            raise IOError("Expected 3D stack with only one channel. " +
                          "Time courses are not supported yet. " + 
                          "Stack shape was " + str(img.shape) + ". ")
    
    else:
        if img.ndim == 3:
            raise IOError("Expected 4D stack (3D plus multiple channels). " + 
                          "Stack shape was " + str(img.shape) + ". ")
        elif img.ndim > 4:
            raise IOError("Expected 4D stack (3D plus multiple channels). " +
                          "Time courses are not supported yet. " + 
                          "Stack shape was " + str(img.shape) + ". ")    
    
        if len(meta_dict['channels']) != img.shape[0]:
            raise IOError("Expected " + str(len(meta_dict['channels'])) + 
                          " channels. Got " + str(img.shape[0]) + ". " + 
                          "Stack shape was " + str(img.shape) + ". ")
                      
    
    #--------------------------------------------------------------------------
    
    ### Generate a unique ID
    
    # Report
    if verbose: print "Initializing stack data structure..."
    
    # Generate
    hex_id = IDgenerator.generate_id(idpath=idpath, length=10, 
                                     save=True, verbose=verbose)
    
    
    #--------------------------------------------------------------------------
    
    ### Generate subdirectory, move file and label it with the ID
    
    # Generate subdirectory
    fdir, fname = os.path.split(fpath)
    
    # Check if subdir already exists
    tdir = os.path.join(fdir, hex_id)
    if os.path.isdir(tdir):
        raise IOError("The dir "+tdir+" already exists!")
    
    # Create the subdir
    os.mkdir(tdir)   
    
    # Move and label the file
    if len(meta_dict['channels']) > 1:
        shutil.move(fpath, os.path.join(tdir, hex_id+"_"+fname))
    else:
        shutil.move(fpath, os.path.join(tdir, hex_id+"_"+fname[:-4]+"_"+
                                              meta_dict['channels'][0]+".tif"))
    
    
    #--------------------------------------------------------------------------
    
    ### For multi-channel stacks: also save the channels individually

    if len(meta_dict['channels']) > 1:

        # Report
        if verbose: print "Writing single-channel stacks..."
        
        # Write    
        try:
            for c,cname in enumerate(meta_dict['channels']):
                cpath = os.path.join(tdir, hex_id + "_" +
                                     fname[:-4]  + "_" + 
                                     cname + ".tif")
                imsave(cpath, img[c,...], bigtiff=True)
        except:
            print "Writing single-channel stack failed with this error:"
            raise
    
    
    #--------------------------------------------------------------------------
    
    ### Generate the prim metadata file
        
    outfpath = os.path.join(tdir, hex_id+"_stack_metadata.pkl")
    with open(outfpath, 'wb') as outfile:
        pickle.dump(meta_dict, outfile, pickle.HIGHEST_PROTOCOL)    
    
    
    #--------------------------------------------------------------------------
    
    ### Report and return
    
    if verbose: print "Data structure for stack "+hex_id+" initialized."
    
    return
    
    
#------------------------------------------------------------------------------



