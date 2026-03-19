"""
Computational Physics
Directory: src/calculus

Code: Vector calculus operators — gradient, divergence, curl
      All use central differences internally.

Author: Barlev Raymond
"""
import numpy as np


def gradient(f, r, h=1e-5):
    """
    Gradient of a scalar field f at position r = [x, y, z].

    Parameters
    ----------
    f : callable, f(r) -> float
    r : array-like, shape (3,)
    h : float, step size

    Returns
    -------
    np.ndarray, shape (3,)
    """
    x, y, z = r
    dx = (f(np.array([x+h, y, z])) - f(np.array([x-h, y, z]))) / (2*h)
    dy = (f(np.array([x, y+h, z])) - f(np.array([x, y-h, z]))) / (2*h)
    dz = (f(np.array([x, y, z+h])) - f(np.array([x, y, z-h]))) / (2*h)
    return np.array([dx, dy, dz])


def divergence(g, r, h=1e-5):
    """
    Divergence of a vector field g at position r = [x, y, z].

    Parameters
    ----------
    g : callable, g(r) -> np.ndarray shape (3,)
    r : array-like, shape (3,)
    h : float, step size

    Returns
    -------
    float
    """
    x, y, z = r
    dgx_dx = (g(np.array([x+h, y, z]))[0] - g(np.array([x-h, y, z]))[0]) / (2*h)
    dgy_dy = (g(np.array([x, y+h, z]))[1] - g(np.array([x, y-h, z]))[1]) / (2*h)
    dgz_dz = (g(np.array([x, y, z+h]))[2] - g(np.array([x, y, z-h]))[2]) / (2*h)
    return dgx_dx + dgy_dy + dgz_dz


def curl(g, r, h=1e-5):
    """
    Curl of a vector field g at position r = [x, y, z].

    Parameters
    ----------
    g : callable, g(r) -> np.ndarray shape (3,)
    r : array-like, shape (3,)
    h : float, step size

    Returns
    -------
    np.ndarray, shape (3,)
    """
    x, y, z = r
    dgz_dy = (g(np.array([x, y+h, z]))[2] - g(np.array([x, y-h, z]))[2]) / (2*h)
    dgy_dz = (g(np.array([x, y, z+h]))[1] - g(np.array([x, y, z-h]))[1]) / (2*h)
    dgx_dz = (g(np.array([x, y, z+h]))[0] - g(np.array([x, y, z-h]))[0]) / (2*h)
    dgz_dx = (g(np.array([x+h, y, z]))[2] - g(np.array([x-h, y, z]))[2]) / (2*h)
    dgy_dx = (g(np.array([x+h, y, z]))[1] - g(np.array([x-h, y, z]))[1]) / (2*h)
    dgx_dy = (g(np.array([x, y+h, z]))[0] - g(np.array([x, y-h, z]))[0]) / (2*h)
    return np.array([dgz_dy - dgy_dz, dgx_dz - dgz_dx, dgy_dx - dgx_dy])
