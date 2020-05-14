# -*- coding: utf-8 -*-
"""
Created on Wed Jan 03 14:30:30 2018

@author:    Jonas Hartmann @ Gilmour group @ EMBL Heidelberg

@descript:  Class to organize hierarchical data as a tree.
            Modified from Uwe Schmitt, original here:
                code.activestate.com/recipes/286150-hierarchical-data-objects
            Used under PSF permissive free software license.
"""


#------------------------------------------------------------------------------

# CLASS: HIERARCHICAL DATA STRUCTURE

class HierarchicalData(object):
    """ Organizes hierarchical data as a tree. For convenience, inner nodes
        need not be constructed explicitly.
    """

    def __init__(self):
        # self._d stores subtrees
        self._d = {}

    def __getattr__(self, name):
        # only attributes not starting with "_" are organized in the tree
        if not name.startswith("_"):
            return self._d.setdefault(name, HierarchicalData())
        raise AttributeError("object %r has no attribute %s" % (self, name))

    def __getstate__(self):
        # for pickling
        return self._d, self._attributes()

    def __setstate__(self, tp):
        # for unpickling
        d,l = tp
        self._d = d
        for name,obj in l: setattr(self, name, obj)

    def _merge(self, other):
        # for merging with another tree
        d,l = other._d, other._attributes()
        for key in d.keys():
            if key in self._d.keys():
                self._d[key]._merge(d[key])
            else:
                self._d[key] = d[key]
        for name,obj in l: setattr(self, name, obj)

    def _attributes(self):
        # return 'leaves' of the data tree
        return [(s, getattr(self, s)) for s in dir(self)
                if not s.startswith("_") and not s in self._d.keys()]

    def _getLeaves(self, prefix=""):
        # getLeaves tree, starting with self
        # prefix stores name of tree node above
        prefix = prefix and prefix + "."
        rv = {}
        atl = self._d.keys()
        for at in atl:
            ob = getattr(self, at)
            trv = ob._getLeaves(prefix+at)
            rv.update(trv)
        for at, ob in self._attributes():
            rv[prefix+at] = ob
        return rv

    def _getPaths(self):
        return self._getLeaves().keys()

    def _getAttrDynamic(self, name):
        # dynamic attributing
        if not name.startswith("_"):
            # Recursion over multiple branches
            if "." in name:
                split_name = name.split(".")
                root   = split_name[0]
                branch = ".".join(split_name[1:])
                return self._gad(root)._gad(branch)
            # Distinguish branches and leaves
            elif isinstance(getattr(self, name), HierarchicalData):
                return self.__getattr__(name)
            else:
                return getattr(self, name)
        raise ValueError("name must not start with an underscore.")

    def _gad(self, name):
        # short alias for dynamic attributing
        return self._getAttrDynamic(name)

    def _setAttrDynamic(self, name, obj):
        # setting an attribute dynamically (hard -> allow branch overwrite)
        if not name.startswith("_"):
            # Recursion over multiple branches
            if "." in name:
                split_name = name.split(".")
                root   = split_name[0]
                branch = ".".join(split_name[1:])
                return self._gad(root)._sad(branch, obj)
            # Set the attribute
            else:
                setattr(self, name, obj)
                return
        raise ValueError("name must not start with an underscore.")

    def _sad(self, name, obj):
        # short alias for setting attributes dynamically
        self._setAttrDynamic(name, obj)

    def __str__(self):
        # easy to read string representation of data
        rl = []
        for k,v in self._getLeaves().items():
            rl.append("%s = %s" %  (k,v))
        return "\n".join(rl)

    def __dir__(self):
        # better autocompletion in IPython
        return self.__dict__.keys() + self._d.keys()


#------------------------------------------------------------------------------

# FUNCTION: GET DICT MAPPING OF LEAVE PATHS

def getLeaves(ob, pre=""):
    """ getLeaves tree, returns dictionary mapping
        paths from root to leafs to value of leafs
    """
    return ob._getLeaves(pre)


#------------------------------------------------------------------------------



