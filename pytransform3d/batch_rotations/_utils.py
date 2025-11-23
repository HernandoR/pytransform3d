"""Utility functions."""

import numpy as np

from ..array_api import get_array_namespace, check_array_type


def norm_vectors(V, out=None):
    """Normalize vectors.

    Parameters
    ----------
    V : array-like, shape (..., n)
        nd vectors

    out : array, shape (..., n), optional (default: new array)
        Output array to which we write the result

    Returns
    -------
    V_unit : array, shape (..., n)
        nd unit vectors with norm 1 or zero vectors
    """
    V = check_array_type(V, "V")
    xp = get_array_namespace(V)
    V = xp.asarray(V)
    norms = xp.linalg.vector_norm(V, axis=-1)
    if out is None:
        out = xp.empty_like(V)
    # Avoid division by zero with maximum(..., smallest positive float).
    # The norm is zero only when the vector is zero so this case does not
    # require further processing.
    tiny = xp.asarray(np.finfo(float).tiny)
    out[...] = V / xp.maximum(norms[..., xp.newaxis], tiny)
    return out


def angles_between_vectors(A, B):
    """Compute angle between two vectors.

    Parameters
    ----------
    A : array-like, shape (..., n)
        nd vectors

    B : array-like, shape (..., n)
        nd vectors

    Returns
    -------
    angles : array, shape (...)
        Angles between pairs of vectors from A and B
    """
    A = check_array_type(A, "A")
    B = check_array_type(B, "B")
    xp = get_array_namespace(A, B)
    A = xp.asarray(A)
    B = xp.asarray(B)
    n_dims = A.shape[-1]
    A_norms = xp.linalg.vector_norm(A, axis=-1)
    B_norms = xp.linalg.vector_norm(B, axis=-1)
    # Use element-wise multiplication and sum for dot product
    AdotB = xp.sum(A * B, axis=-1)
    return xp.acos(xp.clip(AdotB / (A_norms * B_norms), -1.0, 1.0))


def cross_product_matrices(V):
    """Generate the cross-product matrices of vectors.

    The cross-product matrix :math:`\\boldsymbol{V}` satisfies the equation

    .. math::

        \\boldsymbol{V} \\boldsymbol{w} = \\boldsymbol{v} \\times
        \\boldsymbol{w}

    It is a skew-symmetric (antisymmetric) matrix, i.e.
    :math:`-\\boldsymbol{V} = \\boldsymbol{V}^T`.

    Parameters
    ----------
    V : array-like, shape (..., 3)
        3d vectors

    Returns
    -------
    V_cross_product_matrices : array, shape (..., 3, 3)
        Cross-product matrices of V
    """
    V = check_array_type(V, "V")
    xp = get_array_namespace(V)
    V = xp.asarray(V)

    instances_shape = V.shape[:-1]
    V_matrices = xp.empty(instances_shape + (3, 3))

    V_matrices[..., 0, 0] = 0.0
    V_matrices[..., 0, 1] = -V[..., 2]
    V_matrices[..., 0, 2] = V[..., 1]
    V_matrices[..., 1, 0] = V[..., 2]
    V_matrices[..., 1, 1] = 0.0
    V_matrices[..., 1, 2] = -V[..., 0]
    V_matrices[..., 2, 0] = -V[..., 1]
    V_matrices[..., 2, 1] = V[..., 0]
    V_matrices[..., 2, 2] = 0.0

    return V_matrices
