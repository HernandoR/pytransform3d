"""Utility functions for rotations."""

import math

import numpy as np

from ..array_api import get_array_namespace, check_array_type
from ._constants import unitz, eps


def check_axis_index(name, i):
    """Checks axis index.

    Parameters
    ----------
    name : str
        Name of the axis. Required for the error message.

    i : int from [0, 1, 2]
        Index of the axis (0: x, 1: y, 2: z)

    Raises
    ------
    ValueError
        If basis is invalid
    """
    if i not in [0, 1, 2]:
        raise ValueError("Axis index %s (%d) must be in [0, 1, 2]" % (name, i))


def norm_vector(v):
    """Normalize vector.

    Parameters
    ----------
    v : array-like, shape (n,)
        nd vector

    Returns
    -------
    u : array, shape (n,)
        nd unit vector with norm 1 or the zero vector
    """
    v = check_array_type(v, "v")
    xp = get_array_namespace(v)
    norm = xp.linalg.vector_norm(v)
    if norm == 0.0:
        return v

    return xp.asarray(v) / norm


def perpendicular_to_vectors(a, b):
    """Compute perpendicular vector to two other vectors.

    Parameters
    ----------
    a : array-like, shape (3,)
        3d vector

    b : array-like, shape (3,)
        3d vector

    Returns
    -------
    c : array, shape (3,)
        3d vector that is orthogonal to a and b
    """
    a = check_array_type(a, "a")
    b = check_array_type(b, "b")
    xp = get_array_namespace(a, b)
    # Use linalg.cross if available (for better torch compatibility)
    if hasattr(xp.linalg, 'cross'):
        return xp.linalg.cross(a, b)
    return xp.cross(a, b)


def perpendicular_to_vector(a):
    """Compute perpendicular vector to one other vector.

    There is an infinite number of solutions to this problem. Thus, we
    restrict the solutions to [1, 0, z] and return [0, 0, 1] if the
    z component of a is 0.

    Parameters
    ----------
    a : array-like, shape (3,)
        3d vector

    Returns
    -------
    b : array, shape (3,)
        A 3d vector that is orthogonal to a. It does not necessarily have
        unit length.
    """
    a = check_array_type(a, "a")
    xp = get_array_namespace(a)
    if abs(a[2]) < eps:
        return xp.asarray(unitz, copy=True)
    # Now that we solved the problem for [x, y, 0], we can solve it for all
    # other vectors by restricting solutions to [1, 0, z] and find z.
    # The dot product of orthogonal vectors is 0, thus
    # a[0] * 1 + a[1] * 0 + a[2] * z == 0 or -a[0] / a[2] = z
    return xp.asarray([1.0, 0.0, -a[0] / a[2]])


def angle_between_vectors(a, b, fast=False):
    """Compute angle between two vectors.

    Parameters
    ----------
    a : array-like, shape (n,)
        nd vector

    b : array-like, shape (n,)
        nd vector

    fast : bool, optional (default: False)
        Use fast implementation instead of numerically stable solution

    Returns
    -------
    angle : float
        Angle between a and b
    """
    a = check_array_type(a, "a")
    b = check_array_type(b, "b")
    xp = get_array_namespace(a, b)
    if len(a) != 3 or fast:
        return xp.acos(
            xp.clip(
                xp.sum(a * b) / (xp.linalg.vector_norm(a) * xp.linalg.vector_norm(b)),
                -1.0,
                1.0,
            )
        )
    # Use linalg.cross if available (for better torch compatibility)
    if hasattr(xp.linalg, 'cross'):
        cross_prod = xp.linalg.cross(a, b)
    else:
        cross_prod = xp.cross(a, b)
    return xp.atan2(xp.linalg.vector_norm(cross_prod), xp.sum(a * b))


def vector_projection(a, b):
    """Orthogonal projection of vector a on vector b.

    Parameters
    ----------
    a : array-like, shape (3,)
        Vector a that will be projected on vector b

    b : array-like, shape (3,)
        Vector b on which vector a will be projected

    Returns
    -------
    a_on_b : array, shape (3,)
        Vector a
    """
    a = check_array_type(a, "a")
    b = check_array_type(b, "b")
    xp = get_array_namespace(a, b)
    b_norm_squared = xp.sum(b * b)
    if b_norm_squared == 0.0:
        return xp.zeros(3)
    return xp.sum(a * b) * b / b_norm_squared


def plane_basis_from_normal(plane_normal):
    """Compute two basis vectors of a plane from the plane's normal vector.

    Note that there are infinitely many solutions because any rotation of the
    basis vectors about the normal is also a solution. This function
    deterministically picks one of the solutions.

    The two basis vectors of the plane together with the normal form an
    orthonormal basis in 3D space and could be used as columns to form a
    rotation matrix.

    Parameters
    ----------
    plane_normal : array-like, shape (3,)
        Plane normal of unit length.

    Returns
    -------
    x_axis : array, shape (3,)
        x-axis of the plane.

    y_axis : array, shape (3,)
        y-axis of the plane.
    """
    plane_normal = check_array_type(plane_normal, "plane_normal")
    xp = get_array_namespace(plane_normal)
    if abs(plane_normal[0]) >= abs(plane_normal[1]):
        # x or z is the largest magnitude component, swap them
        length = math.sqrt(
            plane_normal[0] * plane_normal[0]
            + plane_normal[2] * plane_normal[2]
        )
        x_axis = xp.asarray(
            [-plane_normal[2] / length, 0.0, plane_normal[0] / length]
        )
        y_axis = xp.asarray(
            [
                plane_normal[1] * x_axis[2],
                plane_normal[2] * x_axis[0] - plane_normal[0] * x_axis[2],
                -plane_normal[1] * x_axis[0],
            ]
        )
    else:
        # y or z is the largest magnitude component, swap them
        length = math.sqrt(
            plane_normal[1] * plane_normal[1]
            + plane_normal[2] * plane_normal[2]
        )
        x_axis = xp.asarray(
            [0.0, plane_normal[2] / length, -plane_normal[1] / length]
        )
        y_axis = xp.asarray(
            [
                plane_normal[1] * x_axis[2] - plane_normal[2] * x_axis[1],
                -plane_normal[0] * x_axis[2],
                plane_normal[0] * x_axis[1],
            ]
        )
    return x_axis, y_axis
