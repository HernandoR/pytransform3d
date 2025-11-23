"""Jacobians of SO(3)."""

import math

import numpy as np

from ..array_api import get_array_namespace, check_array_type
from ._rot_log import cross_product_matrix


def left_jacobian_SO3(omega):
    r"""Left Jacobian of SO(3) at theta (angle of rotation).

    .. math::

        \boldsymbol{J}(\theta)
        =
        \frac{\sin{\theta}}{\theta} \boldsymbol{I}
        + \left(\frac{1 - \cos{\theta}}{\theta}\right)
        \left[\hat{\boldsymbol{\omega}}\right]
        + \left(1 - \frac{\sin{\theta}}{\theta} \right)
        \hat{\boldsymbol{\omega}} \hat{\boldsymbol{\omega}}^T

    Parameters
    ----------
    omega : array-like, shape (3,)
        Compact axis-angle representation.

    Returns
    -------
    J : array, shape (3, 3)
        Left Jacobian of SO(3).

    See also
    --------
    left_jacobian_SO3_series :
        Left Jacobian of SO(3) at theta from Taylor series.

    left_jacobian_SO3_inv :
        Inverse left Jacobian of SO(3) at theta (angle of rotation).
    """
    omega = check_array_type(omega, "omega")
    xp = get_array_namespace(omega)
    theta = xp.linalg.vector_norm(omega)
    
    # Get epsilon for the array type
    if hasattr(xp, 'finfo'):
        eps = xp.finfo(xp.float64).eps
    else:
        eps = float(np.finfo(float).eps)
    
    if float(theta) < eps:
        return left_jacobian_SO3_series(omega, 10)
    
    omega_unit = omega / theta
    omega_matrix = cross_product_matrix(omega_unit)
    
    theta_val = float(theta)
    return (
        xp.eye(3)
        + (1.0 - math.cos(theta_val)) / theta_val * omega_matrix
        + (1.0 - math.sin(theta_val) / theta_val) * xp.matmul(omega_matrix, omega_matrix)
    )


def left_jacobian_SO3_series(omega, n_terms):
    """Left Jacobian of SO(3) at theta from Taylor series.

    Parameters
    ----------
    omega : array-like, shape (3,)
        Compact axis-angle representation.

    n_terms : int
        Number of terms to include in the series.

    Returns
    -------
    J : array, shape (3, 3)
        Left Jacobian of SO(3).

    See Also
    --------
    left_jacobian_SO3 : Left Jacobian of SO(3) at theta (angle of rotation).
    """
    omega = check_array_type(omega, "omega")
    xp = get_array_namespace(omega)
    J = xp.eye(3)
    pxn = xp.eye(3)
    px = cross_product_matrix(omega)
    for n in range(n_terms):
        pxn = xp.matmul(pxn, px) / (n + 2)
        J = J + pxn
    return J


def left_jacobian_SO3_inv(omega):
    r"""Inverse left Jacobian of SO(3) at theta (angle of rotation).

    .. math::

        \boldsymbol{J}^{-1}(\theta)
        =
        \frac{\theta}{2 \tan{\frac{\theta}{2}}} \boldsymbol{I}
        - \frac{\theta}{2} \left[\hat{\boldsymbol{\omega}}\right]
        + \left(1 - \frac{\theta}{2 \tan{\frac{\theta}{2}}}\right)
        \hat{\boldsymbol{\omega}} \hat{\boldsymbol{\omega}}^T

    Parameters
    ----------
    omega : array-like, shape (3,)
        Compact axis-angle representation.

    Returns
    -------
    J_inv : array, shape (3, 3)
        Inverse left Jacobian of SO(3).

    See Also
    --------
    left_jacobian_SO3 : Left Jacobian of SO(3) at theta (angle of rotation).

    left_jacobian_SO3_inv_series :
        Inverse left Jacobian of SO(3) at theta from Taylor series.
    """
    omega = check_array_type(omega, "omega")
    xp = get_array_namespace(omega)
    theta = xp.linalg.vector_norm(omega)
    
    # Get epsilon for the array type
    if hasattr(xp, 'finfo'):
        eps = xp.finfo(xp.float64).eps
    else:
        eps = float(np.finfo(float).eps)
    
    if float(theta) < eps:
        return left_jacobian_SO3_inv_series(omega, 10)
    
    omega_unit = omega / theta
    omega_matrix = cross_product_matrix(omega_unit)
    
    theta_val = float(theta)
    return (
        xp.eye(3)
        - 0.5 * omega_matrix * theta_val
        + (1.0 - 0.5 * theta_val / math.tan(theta_val / 2.0))
        * xp.matmul(omega_matrix, omega_matrix)
    )


def left_jacobian_SO3_inv_series(omega, n_terms):
    """Inverse left Jacobian of SO(3) at theta from Taylor series.

    Parameters
    ----------
    omega : array-like, shape (3,)
        Compact axis-angle representation.

    n_terms : int
        Number of terms to include in the series.

    Returns
    -------
    J_inv : array, shape (3, 3)
        Inverse left Jacobian of SO(3).

    See Also
    --------
    left_jacobian_SO3_inv :
        Inverse left Jacobian of SO(3) at theta (angle of rotation).
    """
    from scipy.special import bernoulli

    omega = check_array_type(omega, "omega")
    xp = get_array_namespace(omega)
    J_inv = xp.eye(3)
    pxn = xp.eye(3)
    px = cross_product_matrix(omega)
    b = bernoulli(n_terms + 1)
    for n in range(n_terms):
        pxn = xp.matmul(pxn, px / (n + 1))
        J_inv = J_inv + b[n + 1] * pxn
    return J_inv
