"""Axis-angle representation."""

import math

import numpy as np
from numpy.testing import assert_array_almost_equal

from ..array_api import get_array_namespace, check_array_type
from ._angle import norm_angle
from ._constants import eps
from ._utils import norm_vector, perpendicular_to_vector


def check_axis_angle(a):
    """Input validation of axis-angle representation.

    Parameters
    ----------
    a : array-like, shape (4,)
        Axis of rotation and rotation angle: (x, y, z, angle)

    Returns
    -------
    a : array, shape (4,)
        Validated axis of rotation and rotation angle: (x, y, z, angle)

    Raises
    ------
    ValueError
        If input is invalid
    """
    a = check_array_type(a, "a")
    xp = get_array_namespace(a)
    a = xp.asarray(a, dtype=xp.float64)
    if a.ndim != 1 or a.shape[0] != 4:
        raise ValueError(
            "Expected axis and angle in array with shape (4,), "
            "got array-like object with shape %s" % (a.shape,)
        )
    return norm_axis_angle(a)


def check_compact_axis_angle(a):
    """Input validation of compact axis-angle representation.

    Parameters
    ----------
    a : array-like, shape (3,)
        Axis of rotation and rotation angle: angle * (x, y, z)

    Returns
    -------
    a : array, shape (3,)
        Validated axis of rotation and rotation angle: angle * (x, y, z)

    Raises
    ------
    ValueError
        If input is invalid
    """
    a = check_array_type(a, "a")
    xp = get_array_namespace(a)
    a = xp.asarray(a, dtype=xp.float64)
    if a.ndim != 1 or a.shape[0] != 3:
        raise ValueError(
            "Expected axis and angle in array with shape (3,), "
            "got array-like object with shape %s" % (a.shape,)
        )
    return norm_compact_axis_angle(a)


def norm_axis_angle(a):
    """Normalize axis-angle representation.

    Parameters
    ----------
    a : array-like, shape (4,)
        Axis of rotation and rotation angle: (x, y, z, angle)

    Returns
    -------
    a : array, shape (4,)
        Axis of rotation and rotation angle: (x, y, z, angle). The length
        of the axis vector is 1 and the angle is in [0, pi). No rotation
        is represented by [1, 0, 0, 0].
    """
    a = check_array_type(a, "a")
    xp = get_array_namespace(a)
    angle = a[3]
    norm = xp.linalg.vector_norm(a[:3])
    if angle == 0.0 or norm == 0.0:
        return xp.asarray([1.0, 0.0, 0.0, 0.0])

    res = xp.empty(4, dtype=a.dtype if hasattr(a, 'dtype') else xp.float64)
    res[:3] = a[:3] / norm

    angle = norm_angle(angle)
    if angle < 0.0:
        angle *= -1.0
        res[:3] *= -1.0

    res[3] = angle

    return res


def norm_compact_axis_angle(a):
    """Normalize compact axis-angle representation.

    Parameters
    ----------
    a : array-like, shape (3,)
        Axis of rotation and rotation angle: angle * (x, y, z)

    Returns
    -------
    a : array, shape (3,)
        Axis of rotation and rotation angle: angle * (x, y, z).
        The angle is in [0, pi). No rotation is represented by [0, 0, 0].
    """
    a = check_array_type(a, "a")
    xp = get_array_namespace(a)
    angle = xp.linalg.vector_norm(a)
    if angle == 0.0:
        return xp.zeros(3, dtype=a.dtype if hasattr(a, 'dtype') else xp.float64)
    axis = a / angle
    return axis * norm_angle(angle)


def compact_axis_angle_near_pi(a, tolerance=1e-6):
    r"""Check if angle of compact axis-angle representation is near pi.

    When the angle :math:`\theta = \pi`, both :math:`\hat{\boldsymbol{\omega}}`
    and :math:`-\hat{\boldsymbol{\omega}}` result in the same rotation. This
    ambiguity could lead to problems when averaging or interpolating.

    Parameters
    ----------
    a : array-like, shape (3,)
        Axis of rotation and rotation angle: angle * (x, y, z).

    tolerance : float
        Tolerance of this check.

    Returns
    -------
    near_pi : bool
        Angle is near pi.
    """
    a = check_array_type(a, "a")
    xp = get_array_namespace(a)
    theta = xp.linalg.vector_norm(a)
    return bool(abs(float(theta) - xp.pi) < tolerance)


def assert_axis_angle_equal(a1, a2, *args, **kwargs):
    """Raise an assertion if two axis-angle are not approximately equal.

    Usually we assume that the rotation axis is normalized to length 1 and
    the angle is within [0, pi). However, this function ignores these
    constraints and will normalize the representations before comparison.
    See numpy.testing.assert_array_almost_equal for a more detailed
    documentation of the other parameters.

    Parameters
    ----------
    a1 : array-like, shape (4,)
        Axis of rotation and rotation angle: (x, y, z, angle)

    a2 : array-like, shape (4,)
        Axis of rotation and rotation angle: (x, y, z, angle)

    args : tuple
        Positional arguments that will be passed to
        `assert_array_almost_equal`

    kwargs : dict
        Positional arguments that will be passed to
        `assert_array_almost_equal`
    """
    a1 = norm_axis_angle(a1)
    a2 = norm_axis_angle(a2)
    xp = get_array_namespace(a1, a2)
    # required despite normalization in case of 180 degree rotation
    if xp.any(xp.sign(a1) != xp.sign(a2)):
        a1 = -a1
        a1 = norm_axis_angle(a1)
    # Convert to numpy for assertion
    a1_np = np.asarray(a1) if not isinstance(a1, np.ndarray) else a1
    a2_np = np.asarray(a2) if not isinstance(a2, np.ndarray) else a2
    assert_array_almost_equal(a1_np, a2_np, *args, **kwargs)


def assert_compact_axis_angle_equal(a1, a2, *args, **kwargs):
    """Raise an assertion if two axis-angle are not approximately equal.

    Usually we assume that the angle is within [0, pi). However, this function
    ignores this constraint and will normalize the representations before
    comparison. See numpy.testing.assert_array_almost_equal for a more detailed
    documentation of the other parameters.

    Parameters
    ----------
    a1 : array-like, shape (3,)
        Axis of rotation and rotation angle: angle * (x, y, z)

    a2 : array-like, shape (3,)
        Axis of rotation and rotation angle: angle * (x, y, z)

    args : tuple
        Positional arguments that will be passed to
        `assert_array_almost_equal`

    kwargs : dict
        Positional arguments that will be passed to
        `assert_array_almost_equal`
    """
    a1 = check_array_type(a1, "a1")
    a2 = check_array_type(a2, "a2")
    xp = get_array_namespace(a1, a2)
    
    angle1 = xp.linalg.vector_norm(a1)
    angle2 = xp.linalg.vector_norm(a2)
    # required despite normalization in case of 180 degree rotation
    if (
        abs(float(angle1)) == xp.pi
        and abs(float(angle2)) == xp.pi
        and xp.any(xp.sign(a1) != xp.sign(a2))
    ):
        a1 = -a1
    a1 = norm_compact_axis_angle(a1)
    a2 = norm_compact_axis_angle(a2)
    # Convert to numpy for assertion
    a1_np = np.asarray(a1) if not isinstance(a1, np.ndarray) else a1
    a2_np = np.asarray(a2) if not isinstance(a2, np.ndarray) else a2
    assert_array_almost_equal(a1_np, a2_np, *args, **kwargs)


def axis_angle_from_two_directions(a, b):
    """Compute axis-angle representation from two direction vectors.

    The rotation will transform direction vector a to direction vector b.
    The direction vectors don't have to be normalized as this will be
    done internally. Note that there is more than one possible solution.

    Parameters
    ----------
    a : array-like, shape (3,)
        First direction vector

    b : array-like, shape (3,)
        Second direction vector

    Returns
    -------
    a : array, shape (4,)
        Axis of rotation and rotation angle: (x, y, z, angle). The angle is
        constrained to [0, pi].
    """
    a = norm_vector(a)
    b = norm_vector(b)
    xp = get_array_namespace(a, b)
    
    cos_angle = xp.sum(a * b)
    if abs(-1.0 - float(cos_angle)) < eps:
        # For 180 degree rotations we have an infinite number of solutions,
        # but we have to pick one axis.
        axis = perpendicular_to_vector(a)
    else:
        # Use linalg.cross if available (PyTorch), otherwise use cross
        if hasattr(xp.linalg, 'cross'):
            axis = xp.linalg.cross(a, b)
        else:
            axis = xp.cross(a, b)
    
    axis_norm = norm_vector(axis)
    # Clamp cos_angle to [-1, 1] to avoid numerical issues with arccos
    cos_angle_clamped = xp.clip(cos_angle, -1.0, 1.0)
    angle = xp.acos(cos_angle_clamped)
    aa = xp.concat([axis_norm, xp.asarray([angle])])
    return norm_axis_angle(aa)


def matrix_from_axis_angle(a):
    r"""Compute rotation matrix from axis-angle.

    This is called exponential map or Rodrigues' formula.

    .. math::

        \boldsymbol{R}(\hat{\boldsymbol{\omega}}, \theta)
        =
        Exp(\hat{\boldsymbol{\omega}} \theta)
        =
        \cos{\theta} \boldsymbol{I}
        + \sin{\theta} \left[\hat{\boldsymbol{\omega}}\right]
        + (1 - \cos{\theta})
        \hat{\boldsymbol{\omega}}\hat{\boldsymbol{\omega}}^T
        =
        \boldsymbol{I}
        + \sin{\theta} \left[\hat{\boldsymbol{\omega}}\right]
        + (1 - \cos{\theta}) \left[\hat{\boldsymbol{\omega}}\right]^2

    This typically results in an active rotation matrix.

    Parameters
    ----------
    a : array-like, shape (4,)
        Axis of rotation and rotation angle: (x, y, z, angle)

    Returns
    -------
    R : array, shape (3, 3)
        Rotation matrix
    """
    a = check_axis_angle(a)
    xp = get_array_namespace(a)
    
    ux, uy, uz = a[0], a[1], a[2]
    theta = a[3]
    c = math.cos(float(theta))
    s = math.sin(float(theta))
    ci = 1.0 - c
    
    R = xp.asarray(
        [
            [ci * ux * ux + c, ci * ux * uy - uz * s, ci * ux * uz + uy * s],
            [ci * uy * ux + uz * s, ci * uy * uy + c, ci * uy * uz - ux * s],
            [ci * uz * ux - uy * s, ci * uz * uy + ux * s, ci * uz * uz + c],
        ]
    )

    # This is equivalent to
    # R = (xp.eye(3) * xp.cos(a[3]) +
    #      (1.0 - xp.cos(a[3])) * a[:3, xp.newaxis].dot(a[xp.newaxis, :3]) +
    #      cross_product_matrix(a[:3]) * xp.sin(a[3]))
    # or
    # w = cross_product_matrix(a[:3])
    # R = np.eye(3) + np.sin(a[3]) * w + (1.0 - np.cos(a[3])) * w.dot(w)

    return R


def matrix_from_compact_axis_angle(a):
    r"""Compute rotation matrix from compact axis-angle.

    This is called exponential map or Rodrigues' formula.

    .. math::

        Exp(\hat{\boldsymbol{\omega}} \theta)
        =
        \cos{\theta} \boldsymbol{I}
        + \sin{\theta} \left[\hat{\boldsymbol{\omega}}\right]
        + (1 - \cos{\theta})
        \hat{\boldsymbol{\omega}}\hat{\boldsymbol{\omega}}^T
        =
        \boldsymbol{I}
        + \sin{\theta} \left[\hat{\boldsymbol{\omega}}\right]
        + (1 - \cos{\theta}) \left[\hat{\boldsymbol{\omega}}\right]^2

    This typically results in an active rotation matrix.

    Parameters
    ----------
    a : array-like, shape (3,)
        Axis of rotation and rotation angle: angle * (x, y, z)

    Returns
    -------
    R : array-like, shape (3, 3)
        Rotation matrix
    """
    a = axis_angle_from_compact_axis_angle(a)
    return matrix_from_axis_angle(a)


def axis_angle_from_compact_axis_angle(a):
    """Compute axis-angle from compact axis-angle representation.

    We usually assume active rotations.

    Parameters
    ----------
    a : array-like, shape (3,)
        Axis of rotation and rotation angle: angle * (x, y, z).

    Returns
    -------
    a : array, shape (4,)
        Axis of rotation and rotation angle: (x, y, z, angle). The angle is
        constrained to [0, pi].
    """
    a = check_compact_axis_angle(a)
    xp = get_array_namespace(a)
    
    angle = xp.linalg.vector_norm(a)

    if float(angle) == 0.0:
        return xp.asarray([1.0, 0.0, 0.0, 0.0])

    axis = a / angle
    return xp.concat([axis, xp.asarray([angle])])


def compact_axis_angle(a):
    r"""Compute 3-dimensional axis-angle from a 4-dimensional one.

    In the 3-dimensional axis-angle representation, the 4th dimension (the
    rotation) is represented by the norm of the rotation axis vector, which
    means we map :math:`\left( \hat{\boldsymbol{\omega}}, \theta \right)` to
    :math:`\boldsymbol{\omega} = \theta \hat{\boldsymbol{\omega}}`.

    This representation is also called rotation vector or exponential
    coordinates of rotation.

    Parameters
    ----------
    a : array-like, shape (4,)
        Axis of rotation and rotation angle: (x, y, z, angle).

    Returns
    -------
    a : array, shape (3,)
        Axis of rotation and rotation angle: angle * (x, y, z) (compact
        representation).
    """
    a = check_axis_angle(a)
    return a[:3] * a[3]


def quaternion_from_axis_angle(a):
    """Compute quaternion from axis-angle.

    This operation is called exponential map.

    We usually assume active rotations.

    Parameters
    ----------
    a : array-like, shape (4,)
        Axis of rotation and rotation angle: (x, y, z, angle)

    Returns
    -------
    q : array, shape (4,)
        Unit quaternion to represent rotation: (w, x, y, z)
    """
    a = check_axis_angle(a)
    xp = get_array_namespace(a)
    
    half_angle = 0.5 * a[3]

    w = xp.cos(half_angle)
    xyz = xp.sin(half_angle) * a[:3]
    return xp.concat([xp.asarray([w]), xyz])


def quaternion_from_compact_axis_angle(a):
    """Compute quaternion from compact axis-angle (exponential map).

    We usually assume active rotations.

    Parameters
    ----------
    a : array-like, shape (4,)
        Axis of rotation and rotation angle: angle * (x, y, z)

    Returns
    -------
    q : array, shape (4,)
        Unit quaternion to represent rotation: (w, x, y, z)
    """
    a = axis_angle_from_compact_axis_angle(a)
    return quaternion_from_axis_angle(a)


def mrp_from_axis_angle(a):
    r"""Compute modified Rodrigues parameters from axis-angle representation.

    .. math::

        \boldsymbol{\psi} = \tan \left(\frac{\theta}{4}\right)
        \hat{\boldsymbol{\omega}}

    Parameters
    ----------
    a : array-like, shape (4,)
        Axis of rotation and rotation angle: (x, y, z, angle)

    Returns
    -------
    mrp : array, shape (3,)
        Modified Rodrigues parameters.
    """
    a = check_axis_angle(a)
    xp = get_array_namespace(a)
    return xp.tan(0.25 * a[3]) * a[:3]
