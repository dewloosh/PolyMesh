import numpy as np
from numpy import ndarray
from numba import njit, prange

__cache = True


@njit(nogil=True, parallel=True, cache=__cache)
def tet_vol_bulk(ecoords: ndarray) -> ndarray:
    """
    Calculates volumes of several tetrahedra.
    
    Parameters
    ----------
    ecoords : numpy.ndarray
        A 3d float array of shape (nE, nNE, 3) of 
        nodal coordinates for several elements. Here nE
        is the number of nodes and nNE is the number of
        nodes per element.
        
    Returns
    -------
    numpy.ndarray
        1d float array of volumes.
        
    Note
    ----
    This only returns exact results for linear cells. For 
    nonlinear cells, use objects that calculate the volumes
    using numerical integration.  
    """
    nE = len(ecoords)
    res = np.zeros(nE, dtype=ecoords.dtype)
    for i in prange(nE):
        v1 = ecoords[i, 1] - ecoords[i, 0]
        v2 = ecoords[i, 2] - ecoords[i, 0]
        v3 = ecoords[i, 3] - ecoords[i, 0]
        res[i] = np.dot(np.cross(v1, v2), v3)
    return np.abs(res) / 6


@njit(nogil=True, cache=__cache)
def lcoords_tet() -> ndarray:
    """
    Returns coordinates of the master element
    of a simplex in 3d.
    """
    return np.array([
        [0., 0., 0.],
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.]
    ])


@njit(nogil=True, cache=__cache)
def nat_to_loc_tet(acoord: np.ndarray) -> ndarray:
    """
    Transformation from natural to local coordinates 
    within a tetrahedra.

    Notes
    -----
    This function is numba-jittable in 'nopython' mode.
    """
    return acoord.T @ lcoords_tet()
