# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 17:47:30 2017

@author:    Jonas Hartmann @ Gilmour group @ EMBL Heidelberg

@descript:  Generates a (unique) hexadecimal ID.
"""


#------------------------------------------------------------------------------

# IMPORTS

# External
from __future__ import division
from random import choice
from warnings import warn


#------------------------------------------------------------------------------

# FUNCTION: GENERATE A RANDOM HEX NUMBER

def _hexgen(length):
    """Randomly generates a string of hexadecimal digits."""
    
    hex_id = ''.join(choice('0123456789ABCDEF') for n in xrange(length))
    return hex_id


#------------------------------------------------------------------------------

# FUNCTION: GENERATE UNIQUE ID

def generate_id(idpath=None, length=10, save=True, verbose=False):
    """Generates a unique string (called ID) of hexadecimal digits.

    Parameters
    ----------
    idpath : string or None, optional, default None
        Path of the text file containing previously generated IDs.
        Necessary to ensure that newly generated IDs are unique.
        If None, a UserWarning is produced and an ID returned without checking.
    length : int, optional, default 10
        Length of the generated ID.
    save : bool, optional, default True
        If True, the newly generated ID is saved to the id_file.
    verbose : bool, optional, default False
        If True, more information is printed.
    
    Returns
    -------
    str
        An ID of hexadecimal digits.
    """
    
    #--------------------------------------------------------------------------
    
    ### If no ID file is given, first warn and then produce and return an ID
    
    if idpath is None:
        warn("In generate_id: " +
             "No id_file provided -- cannot guarantee that ID is unique!")
        return _hexgen(length)


    #--------------------------------------------------------------------------
    
    ### Otherwise, produce a unique ID and return it
    
    # Import existing IDs from id_file
    try:
        with open(idpath,"r") as infile:
            known_ids = [line.strip() for line in infile.readlines()]
    except:
        print("Attempting to load existing IDs from id_file failed " +
              "with this error:")
        raise

    # Generate IDs until a new unique one is found
    while True:

        # Generate ID
        hex_id = _hexgen(length)

        # Check if ID not yet present
        if hex_id not in known_ids:

            # Add the new ID to the id_file
            if save:
                with open(idpath,"a") as outfile:
                    outfile.write(hex_id+"\n")

            # Report and return
            if verbose: print "-- Generated unique ID:", hex_id
            return hex_id
        
    
#------------------------------------------------------------------------------



