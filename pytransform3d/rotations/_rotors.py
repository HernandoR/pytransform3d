"""Rotor operations."""

import numpy as np

from ..array_api import get_array_namespace, check_array_type
from ._constants import unitx, unity, unitz, eps
from ._quaternion import concatenate_quaternions, q_prod_vector
from ._utils import norm_vector, perpendicular_to_vector


def check_rotor(rotor):
    """Input validation of rotor.

    Parameters
    ----------
    rotor : array-like, shape (4,)
        Rotor: (a, b_yz, b_zx, b_xy)

    Returns
    -------
    rotor : array, shape (4,)
        Validated rotor (with unit norm): (a, b_yz, b_zx, b_xy)

    Raises
    ------
    ValueError
        If input is invalid
    """
    rotor = check_array_type(rotor, "rotor")
    xp = get_array_namespace(rotor)
    rotor = xp.asarray(rotor, dtype=xp.float64)
    if rotor.ndim != 1 or rotor.shape[0] != 4:
        raise ValueError(
            "Expected rotor with shape (4,), got "
            "array-like object with shape %s" % (rotor.shape,)
        )
    return norm_vector(rotor)


def wedge(a, b):
    r"""Outer product of two vectors (also exterior or wedge product).

    .. math::

        B = a \wedge b

    Parameters
    ----------
    a : array-like, shape (3,)
        Vector: (x, y, z)

    b : array-like, shape (3,)
        Vector: (x, y, z)

    Returns
    -------
    B : array, shape (3,)
        Bivector that defines the plane that a and b form together:
        (b_yz, b_zx, b_xy)
    """
    a = check_array_type(a, "a")
    b = check_array_type(b, "b")
    xp = get_array_namespace(a, b)
    # Use linalg.cross if available (PyTorch), otherwise use cross
    if hasattr(xp.linalg, 'cross'):
        return xp.linalg.cross(a, b)
    else:
        return xp.cross(a, b)


def plane_normal_from_bivector(B):
    """Convert bivector to normal vector of a plane.

    Parameters
    ----------
    B : array-like, shape (3,)
        Bivector that defines a plane: (b_yz, b_zx, b_xy)

    Returns
    -------
    n : array, shape (3,)
        Unit normal of the corresponding plane: (x, y, z)
    """
    return norm_vector(B)


def geometric_product(a, b):
    r"""Geometric product of two vectors.

    The geometric product consists of the symmetric inner / dot product and the
    antisymmetric outer product (see :func:`~pytransform3d.rotations.wedge`) of
    two vectors.

    .. math::

        ab = a \cdot b + a \wedge b

    The inner product contains the cosine and the outer product contains the
    sine of the angle of rotation from a to b.

    Parameters
    ----------
    a : array-like, shape (3,)
        Vector: (x, y, z)

    b : array-like, shape (3,)
        Vector: (x, y, z)

    Returns
    -------
    ab : array, shape (4,)
        A multivector (a, b_yz, b_zx, b_xy) composed of scalar and bivector
        (b_yz, b_zx, b_xy) that form the geometric product of vectors a and b.
    """
    a = check_array_type(a, "a")
    b = check_array_type(b, "b")
    xp = get_array_namespace(a, b)
    dot_product = xp.sum(a * b)
    return xp.concat([xp.asarray([dot_product]), wedge(a, b)])


def rotor_reverse(rotor):
    """Invert rotor.

    Parameters
    ----------
    rotor : array-like, shape (4,)
        Rotor: (a, b_yz, b_zx, b_xy)

    Returns
    -------
    reverse_rotor : array, shape (4,)
        Reverse of the rotor: (a, b_yz, b_zx, b_xy)

    See Also
    --------
    q_conj : Quaternion conjugate, which is the same operation.
    """
    rotor = check_rotor(rotor)
    xp = get_array_namespace(rotor)
    return xp.concat([xp.asarray([rotor[0]]), -rotor[1:]])


def concatenate_rotors(rotor1, rotor2):
    """Concatenate rotors.

    Suppose we want to apply two extrinsic rotations given by rotors
    R1 and R2 to a vector v. We can either apply R2 to v and then R1 to
    the result or we can concatenate R1 and R2 and apply the result to v.

    Parameters
    ----------
    rotor1 : array-like, shape (4,)
        Rotor: (a, b_yz, b_zx, b_xy)

    rotor2 : array-like, shape (4,)
        Rotor: (a, b_yz, b_zx, b_xy)

    Returns
    -------
    rotor : array, shape (4,)
        rotor1 applied to rotor2: (a, b_yz, b_zx, b_xy)

    See Also
    --------
    concatenate_quaternions : Concatenate quaternions, which is the same
                              operation.
    """
    rotor1 = check_rotor(rotor1)
    rotor2 = check_rotor(rotor2)
    return concatenate_quaternions(rotor1, rotor2)


def rotor_apply(rotor, v):
    r"""Apply rotor to vector.

    .. math::

        v' = R v R^*

    Parameters
    ----------
    rotor : array-like, shape (4,)
        Rotor: (a, b_yz, b_zx, b_xy)

    v : array-like, shape (3,)
        Vector: (x, y, z)

    Returns
    -------
    v : array, shape (3,)
        Rotated vector

    See Also
    --------
    q_prod_vector : The same operation with a different name.
    """
    rotor = check_rotor(rotor)
    return q_prod_vector(rotor, v)


def matrix_from_rotor(rotor):
    """Compute rotation matrix from rotor.

    Parameters
    ----------
    rotor : array-like, shape (4,)
        Rotor: (a, b_yz, b_zx, b_xy)

    Returns
    -------
    R : array, shape (3, 3)
        Rotation matrix
    """
    rotor = check_rotor(rotor)
    xp = get_array_namespace(rotor)
    col1 = rotor_apply(rotor, unitx)
    col2 = rotor_apply(rotor, unity)
    col3 = rotor_apply(rotor, unitz)
    return xp.stack([col1, col2, col3], axis=1)


def rotor_from_two_directions(v_from, v_to):
    """Construct the rotor that rotates one vector to another.

    Parameters
    ----------
    v_from : array-like, shape (3,)
        Unit vector (will be normalized internally)

    v_to : array-like, shape (3,)
        Unit vector (will be normalized internally)

    Returns
    -------
    rotor : array, shape (4,)
        Rotor: (a, b_yz, b_zx, b_xy)
    """
    v_from = norm_vector(v_from)
    v_to = norm_vector(v_to)
    xp = get_array_namespace(v_from, v_to)
    cos_angle_p1 = 1.0 + xp.sum(v_from * v_to)
    if float(cos_angle_p1) < eps:
        # There is an infinite number of solutions for the plane of rotation.
        # This solution works with our convention, since the rotation axis is
        # the same as the plane bivector.
        plane = perpendicular_to_vector(v_from)
    else:
        plane = wedge(v_from, v_to)
    multivector = xp.concat([xp.asarray([cos_angle_p1]), plane])
    return norm_vector(multivector)


def rotor_from_plane_angle(B, angle):
    r"""Compute rotor from plane bivector and angle.

    Parameters
    ----------
    B : array-like, shape (3,)
        Unit bivector (b_yz, b_zx, b_xy) that represents the plane of rotation
        (will be normalized internally)

    angle : float
        Rotation angle

    Returns
    -------
    rotor : array, shape (4,)
        Rotor: (a, b_yz, b_zx, b_xy)
    """
    B = norm_vector(B)
    xp = get_array_namespace(B)
    a = xp.cos(angle / 2.0)
    sina = xp.sin(angle / 2.0)
    return xp.concat([xp.asarray([a]), sina * B])
