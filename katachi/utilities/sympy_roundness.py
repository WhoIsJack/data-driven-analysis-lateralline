# -*- coding: utf-8 -*-
"""
Created on Wed Jan 09 15:52:48 2019

@author:    Jonas Hartmann @ Gilmour group @ EMBL Heidelberg

@descript:  Sympy-based function (with numpy speed-up) to compute the roundness
            of an outline-derived point cloud based on the deviation of its
            points from a circumscribed ellipsoid.

@WARNING:   When running this code in a parallelized fashion (using dask), the
            sympy part sometimes sporadically raises an obscure RuntimeError.
            Apparently sympy violates thread-safety somewhere. Rerunning the
            code in question will usually work, although several attempts may
            be required (see function `compute_roundness_deviation` for an
            example of how this can be handled).
"""

#------------------------------------------------------------------------------

# IMPORTS

# External
from __future__ import division
#import os, pickle
#from warnings import warn
#from time import sleep
import numpy as np
import sympy as sym
from sympy.utilities.lambdify import lambdify
from sklearn.decomposition import PCA


#------------------------------------------------------------------------------

# FUNCTION: Define symbolic intersection of line and ellipsoid
#           and the relevant distances between points

def line_ellipsoid_symbolic():
    """Symbolic intersection of a line with an ellipsoid for the purpose of
    measuring distances between a point within the ellipsoid and its projection
    onto the ellipsoid's surface.

    The line is defined by the origin (0,0,0) and the given point of interest
    (x0,y0,z0). The ellipsoid is defined by its principal semi-axes (a,b,c) and
    is expected to be aligned according to those semi-axes (a aligns with X,
    b aligns with Y, c aligns with Z).

    First, the intersection points of the line with the ellipsoid (x,y,z) are
    found (there are two solutions), then the following measures are derived:
        1. `maxdist`: the distance from (0,0,0) to (x,y,z)
        2. `dist1`:   1st solution for distance from (x0,y0,z0) to (x,y,z)
        3. `dist2`:   2nd solution for distance from (x0,y0,z0) to (x,y,z)

    Returns
    -------
    symbols : tuple of generated sympy symbol objects
        The arrangement of the tuple is (a,b,c,x0,y0,z0,x,y,z).
    maxdist : sympy symbolic expression
        See description above.
    dist1 : sympy symbolic expression
        See description above.
    dist2 : sympy symbolic expression
        See description above.
        
    WARNING
    -------
    Because sympy is apparently not thread-safe, this should not be run in a
    parallelized fashion, as it may throw a RuntimeError or may even simply run
    forever without producing an error or a result!
    """

    # Define symbols
    x,y,z    = sym.symbols('x,y,z')
    x0,y0,z0 = sym.symbols('x0,y0,z0')
    a,b,c    = sym.symbols('a,b,c')

    # Define system of three equations for line-ellipsoid intersection
    eq1 = - 1 + (z**2/c**2) + (y**2/b**2) + (x**2/a**2)
    eq2 = (z/y) - (z0/y0)
    eq3 = (z/x) - (z0/x0)

    # Find the two solutions
    sol1, sol2 = sym.solve([eq1, eq2, eq3], [x,y,z])

    # Get the Euclidean distance from origin to intersection
    # Note: This is the maximum distance a given landmark could be away from
    #       its projection onto the ellipsoid!
    maxdist = sym.sqrt((sol1[0])**2+(sol1[1])**2+(sol1[2])**2)

    # Get the Euclidean dist from landmark to intersection
    # Note: This is the ACTUAL distance a given landmark is away from its
    #       projection onto the ellipsoid!
    dist1 = sym.sqrt((sol1[0]-x0)**2+(sol1[1]-y0)**2+(sol1[2]-z0)**2)
    dist2 = sym.sqrt((sol2[0]-x0)**2+(sol2[1]-y0)**2+(sol2[2]-z0)**2)

    # Return result
    return (a,b,c,x0,y0,z0,x,y,z), maxdist, dist1, dist2


#------------------------------------------------------------------------------

# FUNCTION: Convert symbolic to numpy function for massive speed gains

def line_ellipsoid_numpy(symbols, maxdist, dist1, dist2):
    """Uses sympy.utilities.lambdify.lambdify to convert the symbolic sympy
    expressions generated in `line_ellipsoid_symbolic` into numpy functions.
    This yields a massive speed-boost when the expressions are evaluated over
    many input values in a numpy array.

    Parameters
    ----------
    symbols : tuple of sympy symbol objects
        The arrangement of the tuple is (a,b,c,x0,y0,z0,x,y,z).
    maxdist : sympy symbolic expression
        See doc string of line_ellipsoid_symbolic.
    dist1 : sympy symbolic expression
        See doc string of line_ellipsoid_symbolic.
    dist2 : sympy symbolic expression
        See doc string of line_ellipsoid_symbolic.

    Returns
    -------
    np_maxdist : function
        A numpy version of maxdist, as returned by `line_ellipsoid_symbolic`.
    np_dist1 : function
        A numpy version of dist1, as returned by `line_ellipsoid_symbolic`.
    np_dist2 : function
        A numpy version of dist2, as returned by `line_ellipsoid_symbolic`.
        
    WARNING
    -------
    Because sympy is apparently not thread-safe, this usually causes a runtime
    error when run in a parallelized fashion.
    """

    # Convert to numpy functions
    np_maxdist = lambdify(symbols[:6], maxdist, modules="numpy")
    np_dist1   = lambdify(symbols[:6], dist1,   modules="numpy")
    np_dist2   = lambdify(symbols[:6], dist2,   modules="numpy")

    # Return result
    return np_maxdist, np_dist1, np_dist2


#------------------------------------------------------------------------------

# FUNCTION: Computation of deviation-based roundness measure
#           for an array of point clouds

def compute_roundness_deviation(clouds, aligned=False, semi_axes=None):
    """Compute a roundness measure for a 3D point cloud based on the deviation
    of points from the cloud's circumscribed ellipsoid.

    Specifically, each point of the cloud is projected along a ray originating
    from the ellipsoid's center onto the ellipsoid's surface. The distance of
    the point from its surface projection point is measured and divided by the
    maximum possible distance (i.e. the distance from the origin to the surface
    projection point), yielding a relative deviation. The mean of relative
    deviations across all points of the cloud is then subtracted from 1 to
    yield the final measure.

    The final measure has the following properties:
        - It is a float between 0.0 and 1.0
        - As the cloud approaches a perfect sphere, it goes to 1.0
        - As the cloud approaches a perfect 'anti-sphere', it goes to 0.0*

    Parameters
    ----------
    clouds : numpy array of shape (n_clouds, n_points, 3)
        An array containing n_clouds point clouds with n_points each in 3
        dimensions.
    pca_done : bool, default False
        For this approach to work properly, the point cloud must be aligned
        with the major axes of the circumscribed ellipsoid with the dimensions
        ordered by the major axes' extents. Here, this is accomplished by
        performing a PCA on each input cloud. If the input clouds have already
        been aligned (by PCA or in some other way), `aligned` can be set to
        `True` and the PCA step is skipped.
    semi_axes : None or numpy array of shape (n_clouds, 3), default None
        Length of the three principal semi-axes of the circumscribed ellipsoid,
        ordered from largest to smallest. If None, this is computed as half of
        the extents of each point cloud (in the aligned space).

    Returns
    -------
    roundness_deviation : numpy array of shape (n_clouds)
        Deviation-based roundness measure for each input point cloud.

    Footnotes
    ---------
    *In practice, this is 0.0-ish with the current implementation, but physical
    objects never approach perfect anti-spheres, anyway.
    """

    ## Solve the sympy equations and convert solution to fast numpy lambda
    ## Note: Because sympy is apparently not thread-safe, doing this within
    ##       dask caused a million and one issues. I therefore ultimately went
    ##       with the stupid man's solution below, which works here because the
    ##       solutions to the equations are not dependent on the input.
    #symbols, maxdist, dist1, dist2 = line_ellipsoid_symbolic()
    #np_maxdist, np_dist1, np_dist2 = line_ellipsoid_numpy(symbols, maxdist, 
    #                                                      dist1, dist2)

    # The stupid man's solution to sympy's lack of thread-safety
    def np_maxdist(a,b,c,x0,y0,z0):
        return np.sqrt(a**2*b**2*c**2*x0**2/(a**2*b**2*z0**2 + a**2*c**2*y0**2 + b**2*c**2*x0**2) + a**2*b**2*c**2*y0**2/(a**2*b**2*z0**2 + a**2*c**2*y0**2 + b**2*c**2*x0**2) + a**2*b**2*c**2*z0**2/(a**2*b**2*z0**2 + a**2*c**2*y0**2 + b**2*c**2*x0**2))

    def np_dist1(a,b,c,x0,y0,z0):
        return np.sqrt((-a*b*c*x0*np.sqrt(1/(a**2*b**2*z0**2 + a**2*c**2*y0**2 + b**2*c**2*x0**2)) - x0)**2 + (-a*b*c*y0*np.sqrt(1/(a**2*b**2*z0**2 + a**2*c**2*y0**2 + b**2*c**2*x0**2)) - y0)**2 + (-a*b*c*z0*np.sqrt(1/(a**2*b**2*z0**2 + a**2*c**2*y0**2 + b**2*c**2*x0**2)) - z0)**2)

    def np_dist2(a,b,c,x0,y0,z0):
        return np.sqrt((a*b*c*x0*np.sqrt(1/(a**2*b**2*z0**2 + a**2*c**2*y0**2 + b**2*c**2*x0**2)) - x0)**2 + (a*b*c*y0*np.sqrt(1/(a**2*b**2*z0**2 + a**2*c**2*y0**2 + b**2*c**2*x0**2)) - y0)**2 + (a*b*c*z0*np.sqrt(1/(a**2*b**2*z0**2 + a**2*c**2*y0**2 + b**2*c**2*x0**2)) - z0)**2)

    # Prep output array
    roundness_deviation = np.empty(clouds.shape[0])

    # For each cell...
    for i in range(clouds.shape[0]):

        # Get cloud
        cloud = clouds[i,...]

        # If required: transform to PCA space
        if not aligned:
            cloud = PCA().fit_transform(cloud)

        # If required: compute semi-axes
        # Note: Deriving this from the PCA extents is a bit rough but it works!
        if semi_axes is None:
            semi = (np.max(cloud, axis=0) - np.min(cloud, axis=0)) / 2.0
        else:
            semi = semi_axes[i]

        # Get the origin-intersection distances
        maxdists = np_maxdist(semi[0], semi[1], semi[2],
                              cloud[:,0], cloud[:,1], cloud[:,2])

        # Get the lm-intersection distances
        dists1 = np_dist1(semi[0], semi[1], semi[2],
                          cloud[:,0], cloud[:,1], cloud[:,2])
        dists2 = np_dist2(semi[0], semi[1], semi[2],
                          cloud[:,0], cloud[:,1], cloud[:,2])

        # Find the smaller (correct) distances
        dists = np.vstack([dists1, dists2])
        dists = dists.min(axis=0)

        # Compute the relative distances
        # Note: This goes to 0 for perfect spheres (dist goes to 0)
        #       This goes to 1 for perfect anti-spheres (dist goes to maxdist)
        relative_dists = dists / maxdists

        # Get the mean and invert it
        roundness_dev = 1 - np.mean(relative_dists)

        # Keep result
        roundness_deviation[i] = roundness_dev

    # Return result
    return roundness_deviation


#------------------------------------------------------------------------------



