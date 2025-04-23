"""Random rotations."""

import torch

# from ._axis_angle import matrix_from_compact_axis_angle
# from ._matrix import check_matrix, norm_matrix
from ._utils import norm_vector


def random_vector(rng=torch.Generator().manual_seed(0), n=3):
    r"""Generate an nd vector with normally distributed components.

    Each component will be sampled from :math:`\mathcal{N}(\mu=0, \sigma=1)`.

    Parameters
    ----------
    rng : torch.Generator, optional (default: random seed 0)
        Random number generator

    n : int, optional (default: 3)
        Number of vector components

    Returns
    -------
    v : Tensor shape (n,)
        Random vector
    """
    return torch.randn(n, generator=rng)  # type: ignore[no-untyped-call]


def random_axis_angle(rng=torch.Generator().manual_seed(0)):
    r"""Generate random axis-angle.

    The angle will be sampled uniformly from the interval :math:`[0, \pi)`
    and each component of the rotation axis will be sampled from
    :math:`\mathcal{N}(\mu=0, \sigma=1)` and then the axis will be normalized
    to length 1.

    Parameters
    ----------
    rng : torch.Generator, optional (default: random seed 0)
        Random number generator

    Returns
    -------
    a : Tensor shape (4,)
        Axis of rotation and rotation angle: (x, y, z, angle)
    """
    angle = torch.pi * rng.random()
    a = torch.tensor([0, 0, 0, angle])
    a[:3] = norm_vector(torch.randn(3, generator=rng))
    return a


def random_compact_axis_angle(rng=torch.Generator().manual_seed(0)):
    r"""Generate random compact axis-angle.

    The angle will be sampled uniformly from the interval :math:`[0, \pi)`
    and each component of the rotation axis will be sampled from
    :math:`\mathcal{N}(\mu=0, \sigma=1)` and then the axis will be normalized
    to length 1.

    Parameters
    ----------
    rng : torch.Generator, optional (default: random seed 0)
        Random number generator

    Returns
    -------
    a : Tensor shape (3,)
        Axis of rotation and rotation angle: angle * (x, y, z)
    """
    a = random_axis_angle(rng)
    return a[:3] * a[3]


def random_quaternion(rng=torch.Generator().manual_seed(0)):
    """Generate random quaternion.

    Parameters
    ----------
    rng : torch.Generator, optional (default: random seed 0)
        Random number generator

    Returns
    -------
    q : Tensor shape (4,)
        Unit quaternion to represent rotation: (w, x, y, z)
    """
    return norm_vector(torch.randn(4, generator=rng))


def random_matrix(rng=torch.Generator().manual_seed(0), mean=torch.eye(3), cov=torch.eye(3)):
    r"""Generate random rotation matrix.

    Generate :math:`\Delta \boldsymbol{R}_{B_{i+1}{B_i}}
    \boldsymbol{R}_{{B_i}A}`, with :math:`\Delta \boldsymbol{R}_{B_{i+1}{B_i}}
    = Exp(\hat{\omega} \theta)` and :math:`\hat{\omega}\theta \sim
    \mathcal{N}(\boldsymbol{0}_3, \boldsymbol{\Sigma}_{3 \times 3})`.
    The mean :math:`\boldsymbol{R}_{{B_i}A}` and the covariance
    :math:`\boldsymbol{\Sigma}_{3 \times 3}` are parameters of the function.

    Note that uncertainty is defined in the global frame B, not in the
    body frame A.

    Parameters
    ----------
    rng : torch.Generator, optional (default: random seed 0)
        Random number generator.

    mean : array-like, shape (3, 3), optional (default: I)
        Mean rotation matrix.

    cov : array-like, shape (3, 3), optional (default: I)
        Covariance of noise in exponential coordinate space.

    Returns
    -------
    R : Tensor shape (3, 3)
        Rotation matrix
    """
    raise NotImplementedError
    # mean = check_matrix(mean)
    # a = rng.multivariate_normal(mean=torch.zeros(3), cov=cov)
    # delta = matrix_from_compact_axis_angle(a)
    # return norm_matrix(torch.dot(delta, mean))
