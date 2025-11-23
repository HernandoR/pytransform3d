"""Euler angles."""

import numpy as np

from ..array_api import get_array_namespace, check_array_type
from ._angle import active_matrices_from_angles


def active_matrices_from_intrinsic_euler_angles(
    basis1, basis2, basis3, e, out=None
):
    """Compute active rotation matrices from intrinsic Euler angles.

    Parameters
    ----------
    basis1 : int
        Basis vector of first rotation. 0 corresponds to x axis, 1 to y axis,
        and 2 to z axis.

    basis2 : int
        Basis vector of second rotation. 0 corresponds to x axis, 1 to y axis,
        and 2 to z axis.

    basis3 : int
        Basis vector of third rotation. 0 corresponds to x axis, 1 to y axis,
        and 2 to z axis.

    e : array-like, shape (..., 3)
        Euler angles

    out : array, shape (..., 3, 3), optional (default: new array)
        Output array to which we write the result

    Returns
    -------
    Rs : array, shape (..., 3, 3)
        Rotation matrices
    """
    e = check_array_type(e, "e")
    xp = get_array_namespace(e)
    
    R_shape = e.shape + (3,)
    # Flatten for batch processing
    flat_shape = (-1,) + e.shape[-1:]
    e_flat = xp.reshape(e, flat_shape)
    
    R_alpha = active_matrices_from_angles(basis1, e_flat[..., 0])
    R_beta = active_matrices_from_angles(basis2, e_flat[..., 1])
    R_gamma = active_matrices_from_angles(basis3, e_flat[..., 2])

    if out is None:
        out = xp.zeros(R_shape)

    # Use einsum for batch matrix multiplication
    temp = xp.reshape(xp.matmul(R_alpha, R_beta), (-1, 3, 3))
    result = xp.reshape(xp.matmul(temp, R_gamma), R_shape)
    out[:] = result

    return out


def active_matrices_from_extrinsic_euler_angles(
    basis1, basis2, basis3, e, out=None
):
    """Compute active rotation matrices from extrinsic Euler angles.

    Parameters
    ----------
    basis1 : int
        Basis vector of first rotation. 0 corresponds to x axis, 1 to y axis,
        and 2 to z axis.

    basis2 : int
        Basis vector of second rotation. 0 corresponds to x axis, 1 to y axis,
        and 2 to z axis.

    basis3 : int
        Basis vector of third rotation. 0 corresponds to x axis, 1 to y axis,
        and 2 to z axis.

    e : array-like, shape (..., 3)
        Euler angles

    out : array, shape (..., 3, 3), optional (default: new array)
        Output array to which we write the result

    Returns
    -------
    Rs : array, shape (..., 3, 3)
        Rotation matrices
    """
    e = check_array_type(e, "e")
    xp = get_array_namespace(e)
    
    R_shape = e.shape + (3,)
    # Flatten for batch processing
    flat_shape = (-1,) + e.shape[-1:]
    e_flat = xp.reshape(e, flat_shape)
    
    R_alpha = active_matrices_from_angles(basis1, e_flat[..., 0])
    R_beta = active_matrices_from_angles(basis2, e_flat[..., 1])
    R_gamma = active_matrices_from_angles(basis3, e_flat[..., 2])

    if out is None:
        out = xp.zeros(R_shape)

    # Use einsum for batch matrix multiplication (gamma * beta * alpha)
    temp = xp.reshape(xp.matmul(R_gamma, R_beta), (-1, 3, 3))
    result = xp.reshape(xp.matmul(temp, R_alpha), R_shape)
    out[:] = result

    return out
