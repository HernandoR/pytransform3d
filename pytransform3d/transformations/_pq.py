"""Position+quaternion operations."""

import numpy as np

from ..array_api import get_array_namespace, check_array_type
from ._transform import transform_from
from .. import rotations


def check_pq(pq):
    """Input validation for position and orientation quaternion.

    Parameters
    ----------
    pq : array-like, shape (7,)
        Position and orientation quaternion: (x, y, z, qw, qx, qy, qz)

    Returns
    -------
    pq : array, shape (7,)
        Validated position and orientation quaternion:
         (x, y, z, qw, qx, qy, qz)

    Raises
    ------
    ValueError
        If input is invalid
    """
    pq = check_array_type(pq, "pq")
    xp = get_array_namespace(pq)
    pq = xp.asarray(pq, dtype=xp.float64)
    if pq.ndim != 1 or pq.shape[0] != 7:
        raise ValueError(
            "Expected position and orientation quaternion in a "
            "1D array, got array-like object with shape %s" % (pq.shape,)
        )
    return pq


def pq_slerp(start, end, t):
    """Spherical linear interpolation of position and quaternion.

    We will use spherical linear interpolation (SLERP) for the quaternion and
    linear interpolation for the position.

    Parameters
    ----------
    start : array-like, shape (7,)
        Position and orientation quaternion: (x, y, z, qw, qx, qy, qz)

    end : array-like, shape (7,)
        Position and orientation quaternion: (x, y, z, qw, qx, qy, qz)

    t : float in [0, 1]
        Position between start and end

    Returns
    -------
    pq_t : array-like, shape (7,)
        Position and orientation quaternion: (x, y, z, qw, qx, qy, qz)

    See Also
    --------
    dual_quaternion_sclerp :
        An alternative approach is screw linear interpolation (ScLERP) with
        dual quaternions.

    pytransform3d.rotations.axis_angle_slerp :
        SLERP for axis-angle representation.

    pytransform3d.rotations.quaternion_slerp :
        SLERP for quaternions.

    pytransform3d.rotations.rotor_slerp :
        SLERP for rotors.
    """
    start = check_pq(start)
    end = check_pq(end)
    xp = get_array_namespace(start, end)
    # Split position and quaternion
    start_p, start_q = start[:3], start[3:]
    end_p, end_q = end[:3], end[3:]
    q_t = rotations.quaternion_slerp(start_q, end_q, t, shortest_path=True)
    p_t = start_p + t * (end_p - start_p)
    return xp.concat([p_t, q_t])


def transform_from_pq(pq):
    """Compute transformation matrix from position and quaternion.

    Parameters
    ----------
    pq : array-like, shape (7,)
        Position and orientation quaternion: (x, y, z, qw, qx, qy, qz)

    Returns
    -------
    A2B : array, shape (4, 4)
        Transformation matrix from frame A to frame B
    """
    pq = check_pq(pq)
    return transform_from(rotations.matrix_from_quaternion(pq[3:]), pq[:3])


def dual_quaternion_from_pq(pq):
    """Compute dual quaternion from position and quaternion.

    Parameters
    ----------
    pq : array-like, shape (7,)
        Position and orientation quaternion: (x, y, z, qw, qx, qy, qz)

    Returns
    -------
    dq : array, shape (8,)
        Unit dual quaternion to represent transform:
        (pw, px, py, pz, qw, qx, qy, qz)
    """
    pq = check_pq(pq)
    xp = get_array_namespace(pq)
    real = pq[3:]
    # Create position quaternion: (0, x, y, z)
    pos_quat = xp.concat([xp.asarray([0.0]), pq[:3]])
    dual = 0.5 * rotations.concatenate_quaternions(pos_quat, real)
    return xp.concat([real, dual])
