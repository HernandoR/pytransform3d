"""Batch operations for axis-angle representation."""

import numpy as np

from ..array_api import get_array_namespace
from ..rotations import norm_angle
from ._utils import norm_vectors


def norm_axis_angles(a):
    """Normalize axis-angle representation.

    Parameters
    ----------
    a : array-like, shape (..., 4)
        Axis of rotation and rotation angle: (x, y, z, angle)

    Returns
    -------
    a : array, shape (..., 4)
        Axis of rotation and rotation angle: (x, y, z, angle). The length
        of the axis vector is 1 and the angle is in [0, pi). No rotation
        is represented by [1, 0, 0, 0].
    """
    a = np.asarray(a)
    xp = get_array_namespace(a)

    # Handle the case of only one axis-angle instance
    only_one = a.ndim == 1
    a = xp.astype(
        xp.reshape(a, (-1, a.shape[-1])) if a.ndim == 1 else a,
        xp.float64,
        copy=False,
    )

    angles = a[..., 3]
    norm = xp.sqrt(xp.sum(a[..., :3] ** 2, axis=-1))

    no_rot_mask = (angles == 0.0) | (norm == 0.0)
    rot_mask = ~no_rot_mask

    res = xp.empty_like(a)
    no_rot_value = xp.asarray(
        [1.0, 0.0, 0.0, 0.0], dtype=a.dtype, device=getattr(a, "device", None)
    )
    res[no_rot_mask, :] = no_rot_value
    res[rot_mask, :3] = a[rot_mask, :3] / xp.reshape(norm[rot_mask], (-1, 1))

    angle_normalized = norm_angle(angles)

    negative_angle_mask = angle_normalized < 0.0
    res[negative_angle_mask, :3] *= -1.0
    angle_normalized = xp.where(
        negative_angle_mask, -angle_normalized, angle_normalized
    )

    res[rot_mask, 3] = angle_normalized[rot_mask]

    if only_one:
        res = res[0]
    return res


def matrices_from_compact_axis_angles(A=None, axes=None, angles=None, out=None):
    """Compute rotation matrices from compact axis-angle representations.

    This is called exponential map or Rodrigues' formula.

    This typically results in an active rotation matrix.

    Parameters
    ----------
    A : array-like, shape (..., 3)
        Axes of rotation and rotation angles in compact representation:
        angle * (x, y, z)

    axes : array, shape (..., 3)
        If the unit axes of rotation have been precomputed, you can pass them
        here.

    angles : array, shape (...)
        If the angles have been precomputed, you can pass them here.

    out : array, shape (..., 3, 3), optional (default: new array)
        Output array to which we write the result

    Returns
    -------
    Rs : array, shape (..., 3, 3)
        Rotation matrices
    """
    if angles is None:
        A_arr = np.asarray(A)
        xp = get_array_namespace(A_arr)
        thetas = xp.sqrt(xp.sum(A_arr**2, axis=-1))
    else:
        thetas = np.asarray(angles)
        xp = get_array_namespace(thetas)

    if axes is None:
        omega_unit = norm_vectors(A)
    else:
        omega_unit = axes
        if angles is None:
            xp = get_array_namespace(omega_unit)

    c = xp.cos(thetas)
    s = xp.sin(thetas)
    ci = 1.0 - c
    ux = omega_unit[..., 0]
    uy = omega_unit[..., 1]
    uz = omega_unit[..., 2]

    uxs = ux * s
    uys = uy * s
    uzs = uz * s
    ciux = ci * ux
    ciuy = ci * uy
    ciuxuy = ciux * uy
    ciuxuz = ciux * uz
    ciuyuz = ciuy * uz

    if out is None:
        out = xp.empty(
            A.shape[:-1] + (3, 3),
            dtype=omega_unit.dtype,
            device=getattr(omega_unit, "device", None),
        )

    out[..., 0, 0] = ciux * ux + c
    out[..., 0, 1] = ciuxuy - uzs
    out[..., 0, 2] = ciuxuz + uys
    out[..., 1, 0] = ciuxuy + uzs
    out[..., 1, 1] = ciuy * uy + c
    out[..., 1, 2] = ciuyuz - uxs
    out[..., 2, 0] = ciuxuz - uys
    out[..., 2, 1] = ciuyuz + uxs
    out[..., 2, 2] = ci * uz * uz + c

    return out
