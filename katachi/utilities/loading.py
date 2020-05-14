# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 21:47:32 2017

@author:    Jonas Hartmann @ Gilmour group @ EMBL Heidelberg

@descript:  Data loading utility for data analysis downstream of feature
            extraction by cluster-based embedding.
"""

#------------------------------------------------------------------------------

# IMPORTS

# External
from __future__ import division
import os, pickle
import numpy as np
from tifffile import imread


#------------------------------------------------------------------------------

# Core Data Loader Class

class DataLoader:
    """Allows retrieval of paths to relevant datasets and subsequent loading of
    selected data of various types."""

    def __init__(self, path=None, recurse=False, verbose=True):
        """Initializes a DataLoader object. If a path is given, the files in it
        are indexed using the `find_imports` method.
        
        Parameters
        ----------
        path : str, optional, default None
            The files within this path are indexed using `find_imports`.
            If None, the DataLoader object is instantiated empty.
        recurse : bool, optional, default False
            See `find_imports`.
        verbose : bool, optional, default True
            See `find_imports`.
        """

        # The information that is tracked
        self.IDs  = []
        self.data = {}

        # Find available data
        if path is not None:
            self.find_imports(path, recurse=recurse, verbose=verbose)


    #--------------------------------------------------------------------------

    def find_imports(self, path, recurse=False, verbose=True):
        """Index files in `path`. For a file to be indexed, it must share its
        ID (the first 10 letters of the file name) with the ID of a stack
        metadata file (`<ID>_stack_metadata.pkl`) that is among the files to be 
        indexed. Indexed files can later be loaded with `load_dataset`.
        
        Parameters
        ----------
        path : str
            The files within this path are indexed using `find_imports`.
        recurse : bool, optional, default False
            If True, files are indexed within all subdirectories of `path`.
        verbose : bool, optional, default True
            If True, the number of files indexed is reported.
        """

        # List file paths without recursion
        if not recurse:
            fnames = os.listdir(path)
            fpaths = [os.path.join(path, f) for f in fnames]

        # List file paths with recursion
        else:
            fpaths = [os.path.join(w[0],f)
                      for w in os.walk(path)
                      for f in w[2]]

        # Find stack metadata files
        fp_meta = [fp for fp in fpaths if fp.endswith('stack_metadata.pkl')]

        # Handle finding nothing
        if not fp_meta:
            raise IOError("No valid metadata files found in target directory.")

        # Get IDs
        new_IDs = [os.path.split(fp)[1][:10] for fp in fp_meta
                   if os.path.split(fp)[1][:10] not in self.IDs]
        self.IDs += new_IDs

        # For each ID, find relevant data
        for ID in new_IDs:
            self.data[ID] = [fp for fp in fpaths
                             if os.path.split(fp)[1].startswith(ID)]

        # Report
        if verbose:
            print "Added", len(new_IDs), "new IDs to the library.",
            print "Total IDs in library:", len(self.IDs)


    #--------------------------------------------------------------------------

    def load_dataset(self, suffix, IDs='all', filetype='guess',
                     custom_loader=None, force_list=False, force_dict=False):
        """Find all files currently indexed within this object and load any of
        them that have a given suffix as one dataset.
        
        Parameters
        ----------
        suffix : str or list of strings
            Files to be loaded are determined based on whether their filename
            (including file ending) ends in this string (or in any of the 
            strings within the list, if a list of strings is given).
        IDs : string 'all', string ID, or list of string IDs
            If 'all', all files indexed and matching the suffix are loaded.
            If a 10-character ID string, only files matching the suffix and 
            featuring that ID are loaded. If a list of ID strings, only files
            matching the suffix and featuring one of the IDs are loaded.
        filetype : string, optional, default 'guess'
            If 'image', the files are expected to be tiff images that can be 
            loaded with `tifffile.imread`.
            If 'array', the file are expected to be numpy array files that can
            be loaded with `np.load`.
            If 'pickle', the files are expected to be pickle filea that can be
            loaded with `pickle.load`.
            If 'guess', the filetype is guessed based on the file ending (i.e.
            '.tif' for 'image', '.npy' for 'numpy' or '.pkl' for 'pickle').
            If 'custom', a custom loading function must be provided.
        custom_loader : callable or None, optional, default None
            If not None and filetype is set to 'custom', this must be a
            callable that accepts a file path and returns the loaded content
            from the file as output.
        force_list : bool, optional, default False
            If True, forces the output to be returned as a list with each entry
            corresponding to one loaded file.
        force_dict : bool, optional, default False
            If True, forces the output to be returned as a dict with IDs as
            keys for each loaded file.
            
        Returns
        -------
        data : list, dict, or array
            Loaded data in the form of a list where each entry corresponds to
            a loaded file (if `force_list` is True) or a dict with IDs as keys
            (if `force_dict` is True and as default for all file types except
            'array'). By default, files with filetype 'array' are concatenated
            and returned as a single array.
        IDs : list of 10-character strings
            The IDs for the loaded data. The order corresponds to the order of
            `data` if that is a list.
        data_idx : array or None
            If `filetype` is 'array', data_idx is an array of indices that map
            each entry in the `data` array to its ID in the `IDs` list. Thus,
            `data_idx` will have shape `data.shape[0]`. If `filetype` is not
            'array' or if `force_list` or `force_dict` are True, this returns
            None instead.
        """

        # Handle number of suffices
        if type(suffix)==str:
            suffix = [suffix]
        if not type(suffix)==list:
            raise ValueError("suffix must be a string or list of strings.")

        # Handle output format
        if force_list and force_dict:
            raise ValueError("force_list and force_dict cannot both be True.")

        # Handle IDs
        if IDs == 'all':
            IDs = self.IDs
        elif type(IDs) == str:
            IDs = [IDs]
        elif type(IDs) == list:
            pass
        else:
            raise ValueError("Invalid value for kwarg `IDs` encountered.")

        # Handle IDs that have not been searched
        for ID in IDs:
            if ID not in self.IDs:
                raise ValueError("ID "+ID+" not in library.")

        # Get relevant paths based on suffices (and subselect corresp. IDs)
        data_paths  = []
        matched_IDs = []
        for ID in IDs:
            matched_paths = [dp for dp in self.data[ID]
                             if any([dp.endswith(s) for s in suffix])]
            if len(matched_paths) > 0:
                data_paths += matched_paths
                matched_IDs.append(ID)
        IDs = matched_IDs

        # Error when none found
        if not data_paths:
            raise IOError("No paths found that match the specifications.")

        # Handle filetype
        guesses = {".tif" : "image",
                   ".npy" : "array",
                   ".pkl" : "pickle"}
        if filetype=='guess':
            if data_paths[0][-4:] in guesses.keys():
                filetype = guesses[data_paths[0][-4:]]
            else:
                raise ValueError("Could not guess file type from file ending!")
        elif filetype not in ["image", "array", "pickle", "custom"]:
            raise ValueError("Invalid value for kwarg `filetype` encountered.")

        # Load image data
        if filetype == 'image':

            # Load
            data = {ID:imread(dpath) for ID,dpath in zip(IDs, data_paths)}

            # Handle output formatting
            if force_list:
                data = [data[ID] for ID in IDs]

            # Return
            return data, IDs, None

        # Load array data
        if filetype == 'array':

            # Load
            data = [np.load(dpath) for dpath in data_paths]

            # Handle output formatting
            if force_dict:
                data = {ID:data[i] for i,ID in enumerate(IDs)}
                data_idx = None
            elif force_list:
                data_idx = None
            else:
                data_idx = np.array([j for j,d in enumerate(data)
                                     for i in range(d.shape[0])])
                data = np.concatenate(data)

            # Return
            return data, IDs, data_idx

        # Load pickle data
        if filetype == 'pickle':

            # Load
            data = dict()
            for ID, dpath in zip(IDs, data_paths):
                with open(dpath, 'rb') as infile:
                    inobj = pickle.load(infile)
                data[ID] = inobj

            # Handle output formatting
            if force_list:
                data = [data[ID] for ID in IDs]

            # Return
            return data, IDs, None

        # Load custom
        if filetype == 'custom':

            # Handle missing or incorrect loader object
            if not callable(custom_loader):
                raise ValueError("custom_loader must be a callable if "+
                                 "filetype is set to custom.")

            # Load
            data = dict()
            for ID, dpath in zip(IDs, data_paths):
                data[ID] = custom_loader(dpath)

            # Handle output formatting
            if force_list:
                data = [data[ID] for ID in IDs]

            # Return
            return data, IDs, None


#------------------------------------------------------------------------------

# CLASS: ADJUSTED DATA LOADER FOR IDR DATA...

class DataLoaderIDR:
    """Allows retrieval of paths to relevant datasets and subsequent loading of
    selected data of various types, specifically for the format as used in the
    data submission to the Image Data Repository (IDR)."""

    def __init__(self, path=None, recurse=False, verbose=True):
        """Initializes a DataLoaderIDR object. If a path is given, the files in
        it are indexed using the `find_imports` method.
        
        Parameters
        ----------
        path : str, optional, default None
            The files within this path are indexed using `find_imports`.
            If None, the DataLoader object is instantiated empty.
        recurse : bool, optional, default False
            See `find_imports`.
        verbose : bool, optional, default True
            See `find_imports`.
        """

        # The information that is tracked
        self.IDs  = []
        self.data = {}

        # Find available data
        if path is not None:
            self.find_imports(path, recurse=recurse, verbose=verbose)

    #--------------------------------------------------------------------------

    def find_imports(self, path, recurse=False, verbose=True):
        """Index files in `path`. For a file to be indexed, it must share its
        ID (the first 10 letters of the file name) with the ID of a stack
        reference file (`<ID>_other_measurements.tsv`) that is among the files 
        to be indexed. Indexed files can later be loaded with `load_dataset`.
        
        Parameters
        ----------
        path : str
            The files within this path are indexed using `find_imports`.
        recurse : bool, optional, default False
            If True, files are indexed within all subdirectories of `path`.
        verbose : bool, optional, default True
            If True, the number of files indexed is reported.
        """

        # List file paths without recursion
        if not recurse:
            fnames = os.listdir(path)
            fpaths = [os.path.join(path, f) for f in fnames]

        # List file paths with recursion
        else:
            fpaths = [os.path.join(w[0],f)
                      for w in os.walk(path)
                      for f in w[2]]

        # Find stack metadata files
        fp_meta = [fp for fp in fpaths if fp.endswith('_other_measurements.tsv')]

        # Handle finding nothing
        if not fp_meta:
            raise IOError("No valid metadata files found in target directory.")

        # Get IDs
        new_IDs = [os.path.split(fp)[1][:10] for fp in fp_meta
                   if os.path.split(fp)[1][:10] not in self.IDs]
        self.IDs += new_IDs

        # For each ID, find relevant data
        for ID in new_IDs:
            self.data[ID] = [fp for fp in fpaths
                             if os.path.split(fp)[1].startswith(ID)]

        # Report
        if verbose:
            print "Added", len(new_IDs), "new IDs to the library.",
            print "Total IDs in library:", len(self.IDs)

    #--------------------------------------------------------------------------

    def load_dataset(self, suffix, IDs='all', filetype='tsv',
                     force_df=False, force_list=False, force_dict=False):
        """Find all files currently indexed within this object and load any of
        them that have a given suffix as one dataset.
        
        Parameters
        ----------
        suffix : str or list of strings
            Files to be loaded are determined based on whether their filename
            (including file ending) ends in this string (or in any of the 
            strings within the list, if a list of strings is given).
        IDs : string 'all', string ID, or list of string IDs
            If 'all', all files indexed and matching the suffix are loaded.
            If a 10-character ID string, only files matching the suffix and 
            featuring that ID are loaded. If a list of ID strings, only files
            matching the suffix and featuring one of the IDs are loaded.
        filetype : string, optional, default 'tsv'
            DataLoaderIDR only supports loading tab-separated tsv files where 
            the first row contains column headers and the first two columns
            contain sample IDs and cell IDs, respectively. This is the format
            in which numerical data was deposited in the IDR.
        force_df : bool, optional, default False
            If True, forces the output to be returned as a pandas dataframe.
        force_list : bool, optional, default False
            If True, forces the output to be returned as a list with each entry
            corresponding to one loaded file.
        force_dict : bool, optional, default False
            If True, forces the output to be returned as a dict with IDs as
            keys for each loaded file.
            
        Returns
        -------
        data : list, dict, or array
            Loaded data in the form of a list where each entry corresponds to
            a loaded file (if `force_list` is True) or a dict with IDs as keys
            (if `force_dict` is True) or a pandas dataframe that concatenates
            the data from all samples (if `force_df` is True). By default, 
            files with filetype 'tsv' are loaded as numpy arrays and concat-
            enated into a single array.
        IDs : list of 10-character strings
            The IDs for the loaded data. The order corresponds to the order of
            `data` if that is a list.
        data_idx : array or None
            An array of indices that map each entry in the `data` array to its 
            ID in the `IDs` list. Thus, `data_idx` will have shape 
            `data.shape[0]`. If `force_list` or `force_dict` are True, this 
            returns None instead.
        """

        # Handle number of suffices
        if type(suffix)==str:
            suffix = [suffix]
        if not type(suffix)==list:
            raise ValueError("suffix must be a string or list of strings.")

        # Handle output format
        if force_list and force_dict:
            raise ValueError("force_list and force_dict cannot both be True.")

        # Handle IDs
        if IDs == 'all':
            IDs = self.IDs
        elif type(IDs) == str:
            IDs = [IDs]
        elif type(IDs) == list:
            pass
        else:
            raise ValueError("Invalid value for kwarg `IDs` encountered.")

        # Handle IDs that have not been searched
        for ID in IDs:
            if ID not in self.IDs:
                raise ValueError("ID "+ID+" not in library.")

        # Get relevant paths based on suffices (and subselect corresp. IDs)
        data_paths  = []
        matched_IDs = []
        for ID in IDs:
            matched_paths = [dp for dp in self.data[ID]
                             if any([dp.endswith(s) for s in suffix])]
            if len(matched_paths) > 0:
                data_paths += matched_paths
                matched_IDs.append(ID)
        IDs = matched_IDs

        # Error when none found
        if not data_paths:
            raise IOError("No paths found that match the specifications.")

        # Handle filetype
        if not filetype=='tsv':
            raise NotImplementedError("DataLoaderIDR only supports 'tsv' files.")

        # Load tsv data
        if filetype == 'tsv':
            
            # For each path...
            data = []
            for dpath in data_paths:
                
                # Get number of columns
                with open(dpath, 'r') as infile:
                    cols = infile.readline().strip().split('\t')
                    ncols = len(cols)
                    
                # Load the data
                data.append( np.loadtxt(dpath, delimiter='\t', skiprows=1, 
                                        usecols=range(2,ncols)) )

            # Handle output formatting
            if force_dict:
                data = {ID:data[i] for i,ID in enumerate(IDs)}
                data_idx = None
            elif force_list:
                data_idx = None
            else:
                data_idx = np.array([j for j,d in enumerate(data)
                                     for i in range(d.shape[0])])
                data = np.concatenate(data)
            if force_df:
                import pandas as pd
                data = pd.DataFrame(data, columns=cols[2:])

            # Return
            return data, IDs, data_idx
        

#------------------------------------------------------------------------------



