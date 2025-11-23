"""Polar decomposition."""

import numpy as np

from ..array_api import get_array_namespace, check_array_type
from ._axis_angle import matrix_from_compact_axis_angle


def robust_polar_decomposition(A, n_iter=20, eps=np.finfo(float).eps):
    r"""Orthonormalize rotation matrix with robust polar decomposition.

    Robust polar decomposition [1]_ [2]_ is a computationally more costly
    method, but it spreads the error more evenly between the basis vectors
    in comparison to Gram-Schmidt orthonormalization (as in
    :func:`norm_matrix`).

    Robust polar decomposition finds an orthonormal matrix that minimizes the
    Frobenius norm

    .. math::

        ||\boldsymbol{A} - \boldsymbol{R}||^2

    between the input :math:`\boldsymbol{A}` that is not orthonormal and the
    output :math:`\boldsymbol{R}` that is orthonormal.

    Parameters
    ----------
    A : array-like, shape (3, 3)
        Matrix that contains a basis vector in each column. The basis does not
        have to be orthonormal.

    n_iter : int, optional (default: 20)
        Maximum number of iterations for which we refine the estimation of the
        rotation matrix.

    eps : float, optional (default: np.finfo(float).eps)
        Precision for termination criterion of iterative refinement.

    Returns
    -------
    R : array, shape (3, 3)
        Orthonormalized rotation matrix.

    See Also
    --------
    norm_matrix
        The cheaper default orthonormalization method that uses Gram-Schmidt
        orthonormalization optimized for 3 dimensions.

    References
    ----------
    .. [1] Selstad, J. (2019). Orthonormalization.
       https://zalo.github.io/blog/polar-decomposition/

    .. [2] Müller, M., Bender, J., Chentanez, N., Macklin, M. (2016).
       A Robust Method to Extract the Rotational Part of Deformations.
       In MIG '16: Proceedings of the 9th International Conference on Motion in
       Games, pp. 55-60, doi: 10.1145/2994258.2994269.
    """
    A = check_array_type(A, "A")
    xp = get_array_namespace(A)
    current_R = xp.eye(3)
    
    for _ in range(n_iter):
        # Use linalg.cross if available (PyTorch), otherwise use cross
        # Note: axisa, axisb, axisc parameters are not in array API standard
        # We need to manually compute cross products along specific axes
        # For column-wise cross products: cross(current_R[:, i], A[:, i])
        cross_prods = []
        for i in range(3):
            if hasattr(xp.linalg, 'cross'):
                cp = xp.linalg.cross(current_R[:, i], A[:, i])
            else:
                cp = xp.cross(current_R[:, i], A[:, i])
            cross_prods.append(cp)
        column_vector_cross_products = xp.stack(cross_prods, axis=1)
        
        column_vector_dot_products_sum = xp.sum(current_R * A)
        omega = xp.sum(column_vector_cross_products, axis=1) / (
            abs(float(column_vector_dot_products_sum)) + eps
        )
        
        if float(xp.linalg.vector_norm(omega)) < eps:
            break
        current_R = xp.matmul(matrix_from_compact_axis_angle(omega), current_R)
    return current_R
