"""Rotation matrices."""

import warnings

import numpy as np
from numpy.testing import assert_array_almost_equal

from ..array_api import get_array_namespace, check_array_type
from ._axis_angle import compact_axis_angle
from ._utils import norm_vector, perpendicular_to_vectors, vector_projection


def check_matrix(R, tolerance=1e-6, strict_check=True):
    r"""Input validation of a rotation matrix.

    We check whether R multiplied by its inverse is approximately the identity
    matrix

    .. math::

        \boldsymbol{R}\boldsymbol{R}^T = \boldsymbol{I}

    and whether the determinant is positive

    .. math::

        det(\boldsymbol{R}) > 0

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix

    tolerance : float, optional (default: 1e-6)
        Tolerance threshold for checks. Default tolerance is the same as in
        assert_rotation_matrix(R).

    strict_check : bool, optional (default: True)
        Raise a ValueError if the rotation matrix is not numerically close
        enough to a real rotation matrix. Otherwise we print a warning.

    Returns
    -------
    R : array, shape (3, 3)
        Validated rotation matrix

    Raises
    ------
    ValueError
        If input is invalid

    See Also
    --------
    norm_matrix : Enforces orthonormality of a rotation matrix.
    robust_polar_decomposition
        A more expensive orthonormalization method that spreads the error more
        evenly between the basis vectors.
    """
    R = check_array_type(R, "R")
    xp = get_array_namespace(R)
    R = xp.asarray(R, dtype=xp.float64)
    if R.ndim != 2 or R.shape[0] != 3 or R.shape[1] != 3:
        raise ValueError(
            "Expected rotation matrix with shape (3, 3), got "
            "array-like object with shape %s" % (R.shape,)
        )
    RRT = xp.matmul(R, xp.matrix_transpose(R))
    eye = xp.asarray(xp.eye(3), dtype=xp.float64)
    if not xp.allclose(RRT, eye, atol=tolerance):
        error_msg = (
            "Expected rotation matrix, but it failed the test "
            "for inversion by transposition. matmul(R, R.T) "
            "gives %r" % RRT
        )
        if strict_check:
            raise ValueError(error_msg)
        warnings.warn(error_msg, UserWarning, stacklevel=2)
    R_det = xp.linalg.det(R)
    if R_det < 0.0:
        error_msg = (
            "Expected rotation matrix, but it failed the test "
            "for the determinant, which should be 1 but is %g; "
            "that is, it probably represents a rotoreflection" % R_det
        )
        if strict_check:
            raise ValueError(error_msg)
        warnings.warn(error_msg, UserWarning, stacklevel=2)
    return R


def matrix_requires_renormalization(R, tolerance=1e-6):
    r"""Check if a rotation matrix needs renormalization.

    This function will check if :math:`R R^T \approx I`.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix that should be orthonormal.

    tolerance : float, optional (default: 1e-6)
        Tolerance for check.

    Returns
    -------
    required : bool
        Indicates if renormalization is required.

    See Also
    --------
    norm_matrix : Orthonormalize rotation matrix.
    robust_polar_decomposition
        A more expensive orthonormalization method that spreads the error more
        evenly between the basis vectors.
    """
    R = check_array_type(R, "R")
    xp = get_array_namespace(R)
    R = xp.asarray(R, dtype=float)
    RRT = xp.matmul(R, xp.matrix_transpose(R))
    eye = xp.asarray(xp.eye(3), dtype=R.dtype)
    return not xp.allclose(RRT, eye, atol=tolerance)


def norm_matrix(R):
    r"""Orthonormalize rotation matrix.

    A rotation matrix is defined as

    .. math::

        \boldsymbol R =
        \left( \begin{array}{ccc}
            r_{11} & r_{12} & r_{13}\\
            r_{21} & r_{22} & r_{23}\\
            r_{31} & r_{32} & r_{33}\\
        \end{array} \right)
        \in SO(3)

    and must be orthonormal, which results in 6 constraints:

    * column vectors must have unit norm (3 constraints)
    * and must be orthogonal to each other (3 constraints)

    A more compact representation of these constraints is
    :math:`\boldsymbol R^T \boldsymbol R = \boldsymbol I`.

    Because of numerical problems, a rotation matrix might not satisfy the
    constraints anymore. This function will enforce them with Gram-Schmidt
    orthonormalization optimized for 3 dimensions.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix with small numerical errors.

    Returns
    -------
    R : array, shape (3, 3)
        Orthonormalized rotation matrix.

    See Also
    --------
    check_matrix : Checks orthonormality of a rotation matrix.
    matrix_requires_renormalization
        Checks if a rotation matrix needs renormalization.
    robust_polar_decomposition
        A more expensive orthonormalization method that spreads the error more
        evenly between the basis vectors.
    """
    R = check_array_type(R, "R")
    xp = get_array_namespace(R)
    R = xp.asarray(R)
    c2 = R[:, 1]
    c3 = norm_vector(R[:, 2])
    # Use linalg.cross if available
    if hasattr(xp.linalg, 'cross'):
        c1_unnorm = xp.linalg.cross(c2, c3)
        c1 = norm_vector(c1_unnorm)
        c2_unnorm = xp.linalg.cross(c3, c1)
    else:
        c1_unnorm = xp.cross(c2, c3)
        c1 = norm_vector(c1_unnorm)
        c2_unnorm = xp.cross(c3, c1)
    c2 = norm_vector(c2_unnorm)
    # Stack columns
    return xp.stack([c1, c2, c3], axis=1)


def assert_rotation_matrix(R, *args, **kwargs):
    """Raise an assertion if a matrix is not a rotation matrix.

    The two properties :math:`\\boldsymbol{I} = \\boldsymbol{R R}^T` and
    :math:`det(R) = 1` will be checked. See
    numpy.testing.assert_array_almost_equal for a more detailed documentation
    of the other parameters.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix

    args : tuple
        Positional arguments that will be passed to
        `assert_array_almost_equal`

    kwargs : dict
        Positional arguments that will be passed to
        `assert_array_almost_equal`
    """
    assert_array_almost_equal(np.dot(R, R.T), np.eye(3), *args, **kwargs)
    assert_array_almost_equal(np.linalg.det(R), 1.0, *args, **kwargs)


def matrix_from_two_vectors(a, b):
    """Compute rotation matrix from two vectors.

    We assume that the two given vectors form a plane so that we can compute
    a third, orthogonal vector with the cross product.

    The x-axis will point in the same direction as a, the y-axis corresponds
    to the normalized vector rejection of b on a, and the z-axis is the
    cross product of the other basis vectors.

    Parameters
    ----------
    a : array-like, shape (3,)
        First vector, must not be 0

    b : array-like, shape (3,)
        Second vector, must not be 0 or parallel to a

    Returns
    -------
    R : array, shape (3, 3)
        Rotation matrix

    Raises
    ------
    ValueError
        If vectors are parallel or one of them is the zero vector
    """
    a = check_array_type(a, "a")
    b = check_array_type(b, "b")
    xp = get_array_namespace(a, b)
    
    if xp.linalg.vector_norm(a) == 0:
        raise ValueError("a must not be the zero vector.")
    if xp.linalg.vector_norm(b) == 0:
        raise ValueError("b must not be the zero vector.")

    c = perpendicular_to_vectors(a, b)
    if xp.linalg.vector_norm(c) == 0:
        raise ValueError("a and b must not be parallel.")

    a = norm_vector(a)

    b_on_a_projection = vector_projection(b, a)
    b_on_a_rejection = b - b_on_a_projection
    b = norm_vector(b_on_a_rejection)

    c = norm_vector(c)

    return xp.stack([a, b, c], axis=1)


def quaternion_from_matrix(R, strict_check=True):
    """Compute quaternion from rotation matrix.

    We usually assume active rotations.

    .. warning::

        When computing a quaternion from the rotation matrix there is a sign
        ambiguity: q and -q represent the same rotation.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix

    strict_check : bool, optional (default: True)
        Raise a ValueError if the rotation matrix is not numerically close
        enough to a real rotation matrix. Otherwise we print a warning.

    Returns
    -------
    q : array, shape (4,)
        Unit quaternion to represent rotation: (w, x, y, z)
    """
    R = check_matrix(R, strict_check=strict_check)
    xp = get_array_namespace(R)
    q = xp.empty(4, dtype=R.dtype)

    # Source:
    # http://www.euclideanspace.com/maths/geometry/rotations/conversions
    trace = xp.trace(R)
    if trace > 0.0:
        sqrt_trace = xp.sqrt(1.0 + trace)
        q[0] = 0.5 * sqrt_trace
        q[1] = 0.5 / sqrt_trace * (R[2, 1] - R[1, 2])
        q[2] = 0.5 / sqrt_trace * (R[0, 2] - R[2, 0])
        q[3] = 0.5 / sqrt_trace * (R[1, 0] - R[0, 1])
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            sqrt_trace = xp.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            q[0] = 0.5 / sqrt_trace * (R[2, 1] - R[1, 2])
            q[1] = 0.5 * sqrt_trace
            q[2] = 0.5 / sqrt_trace * (R[1, 0] + R[0, 1])
            q[3] = 0.5 / sqrt_trace * (R[0, 2] + R[2, 0])
        elif R[1, 1] > R[2, 2]:
            sqrt_trace = xp.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            q[0] = 0.5 / sqrt_trace * (R[0, 2] - R[2, 0])
            q[1] = 0.5 / sqrt_trace * (R[1, 0] + R[0, 1])
            q[2] = 0.5 * sqrt_trace
            q[3] = 0.5 / sqrt_trace * (R[2, 1] + R[1, 2])
        else:
            sqrt_trace = xp.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            q[0] = 0.5 / sqrt_trace * (R[1, 0] - R[0, 1])
            q[1] = 0.5 / sqrt_trace * (R[0, 2] + R[2, 0])
            q[2] = 0.5 / sqrt_trace * (R[2, 1] + R[1, 2])
            q[3] = 0.5 * sqrt_trace
    return q


def axis_angle_from_matrix(R, strict_check=True, check=True):
    """Compute axis-angle from rotation matrix.

    This operation is called logarithmic map. Note that there are two possible
    solutions for the rotation axis when the angle is 180 degrees (pi).

    We usually assume active rotations.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix

    strict_check : bool, optional (default: True)
        Raise a ValueError if the rotation matrix is not numerically close
        enough to a real rotation matrix. Otherwise we print a warning.

    check : bool, optional (default: True)
        Check if rotation matrix is valid

    Returns
    -------
    a : array, shape (4,)
        Axis of rotation and rotation angle: (x, y, z, angle). The angle is
        constrained to [0, pi].
    """
    if check:
        R = check_matrix(R, strict_check=strict_check)
    else:
        R = check_array_type(R, "R")
    xp = get_array_namespace(R)
    
    cos_angle = (xp.trace(R) - 1.0) / 2.0
    # Convert to scalar for min/max, then back to array type
    cos_angle_float = float(cos_angle)
    angle_float = float(xp.acos(xp.asarray(min(max(-1.0, cos_angle_float), 1.0))))
    angle = xp.asarray(angle_float)

    if angle_float == 0.0:  # R == eye(3)
        return xp.asarray([1.0, 0.0, 0.0, 0.0], dtype=R.dtype)

    a = xp.empty(4, dtype=R.dtype)

    # We can usually determine the rotation axis by inverting Rodrigues'
    # formula. Subtracting opposing off-diagonal elements gives us
    # 2 * sin(angle) * e,
    # where e is the normalized rotation axis.
    axis_unnormalized = xp.asarray(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]
    )

    if abs(angle_float - float(xp.pi)) < 1e-4:  # trace(R) close to -1
        # The threshold 1e-4 is a result from this discussion:
        # https://github.com/dfki-ric/pytransform3d/issues/43
        # The standard formula becomes numerically unstable, however,
        # Rodrigues' formula reduces to R = I + 2 (ee^T - I), with the
        # rotation axis e, that is, ee^T = 0.5 * (R + I) and we can find the
        # squared values of the rotation axis on the diagonal of this matrix.
        # We can still use the original formula to reconstruct the signs of
        # the rotation axis correctly.

        # In case of floating point inaccuracies:
        R_diag = xp.clip(xp.asarray([R[0, 0], R[1, 1], R[2, 2]]), -1.0, 1.0)

        eeT_diag = 0.5 * (R_diag + 1.0)
        signs = xp.sign(axis_unnormalized)
        signs = xp.where(signs == 0.0, 1.0, signs)
        a[:3] = xp.sqrt(eeT_diag) * signs
    else:
        a[:3] = axis_unnormalized
        # The norm of axis_unnormalized is 2.0 * sin(angle), that is, we
        # could normalize with a[:3] = a[:3] / (2.0 * sin(angle)),
        # but the following is much more precise for angles close to 0 or pi:
    a[:3] /= xp.linalg.vector_norm(a[:3])

    a[3] = angle
    return a


def compact_axis_angle_from_matrix(R, check=True):
    """Compute compact axis-angle from rotation matrix.

    This operation is called logarithmic map. Note that there are two possible
    solutions for the rotation axis when the angle is 180 degrees (pi).

    We usually assume active rotations.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix

    check : bool, optional (default: True)
        Check if rotation matrix is valid

    Returns
    -------
    a : array, shape (3,)
        Axis of rotation and rotation angle: angle * (x, y, z). The angle is
        constrained to [0, pi].
    """
    a = axis_angle_from_matrix(R, check=check)
    return compact_axis_angle(a)
