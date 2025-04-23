import warnings

# import numpy as np
import pytest
import torch

# from numpy.testing import assert_close
from torch.testing import assert_close

# import pytransform3d.rotations as pr
import pytransform3d.t.rotations as ptr


def test_norm_vector():
    """Test normalization of vectors."""
    # rng = torch.Generator().manual_seed(0)
    rng = torch.Generator().manual_seed(0)
    for n in range(1, 6):
        v = ptr.random_vector(rng, n)
        v = torch.from_numpy(v)
        u = ptr.norm_vector(v)
        assert pytest.approx(torch.linalg.norm(u)) == 1


def test_norm_zero_vector():
    """Test normalization of zero vector."""
    normalized = ptr.norm_vector(torch.zeros(3))
    assert torch.isfinite(torch.linalg.norm(normalized))


def test_perpendicular_to_vectors():
    """Test function to compute perpendicular to vectors."""
    rng = torch.Generator().manual_seed(0)
    a = ptr.norm_vector(ptr.random_vector(rng))
    a1 = ptr.norm_vector(ptr.random_vector(rng))
    b = ptr.norm_vector(ptr.perpendicular_to_vectors(a, a1))
    c = ptr.norm_vector(ptr.perpendicular_to_vectors(a, b))
    assert pytest.approx(ptr.angle_between_vectors(a, b)) == torch.pi / 2.0
    assert pytest.approx(ptr.angle_between_vectors(a, c)) == torch.pi / 2.0
    assert pytest.approx(ptr.angle_between_vectors(b, c)) == torch.pi / 2.0
    assert_close(ptr.perpendicular_to_vectors(b, c), a)
    assert_close(ptr.perpendicular_to_vectors(c, a), b)


def test_perpendicular_to_vector():
    """Test function to compute perpendicular to vector."""
    assert (
        pytest.approx(
            ptr.angle_between_vectors(
                ptr.unitx, ptr.perpendicular_to_vector(ptr.unitx)
            )
        )
        == torch.pi / 2.0
    )
    assert (
        pytest.approx(
            ptr.angle_between_vectors(
                ptr.unity, ptr.perpendicular_to_vector(ptr.unity)
            )
        )
        == torch.pi / 2.0
    )
    assert (
        pytest.approx(
            ptr.angle_between_vectors(
                ptr.unitz, ptr.perpendicular_to_vector(ptr.unitz)
            )
        )
        == torch.pi / 2.0
    )
    rng = torch.Generator().manual_seed(0)
    for _ in range(5):
        a = ptr.norm_vector(ptr.random_vector(rng))
        assert (
            pytest.approx(
                ptr.angle_between_vectors(a, ptr.perpendicular_to_vector(a))
            )
            == torch.pi / 2.0
        )
        b = a - torch.tensor([a[0], 0.0, 0.0])
        assert (
            pytest.approx(
                ptr.angle_between_vectors(b, ptr.perpendicular_to_vector(b))
            )
            == torch.pi / 2.0
        )
        c = a - torch.tensor([0.0, a[1], 0.0])
        assert (
            pytest.approx(
                ptr.angle_between_vectors(c, ptr.perpendicular_to_vector(c))
            )
            == torch.pi / 2.0
        )
        d = a - torch.tensor([0.0, 0.0, a[2]])
        assert (
            pytest.approx(
                ptr.angle_between_vectors(d, ptr.perpendicular_to_vector(d))
            )
            == torch.pi / 2.0
        )


def test_angle_between_vectors():
    """Test function to compute angle between two vectors."""
    v = torch.tensor([1, 0, 0])
    a = torch.tensor([0, 1, 0, torch.pi / 2])
    R = ptr.matrix_from_axis_angle(a)
    vR = torch.dot(R, v)
    assert pytest.approx(ptr.angle_between_vectors(vR, v)) == a[-1]
    v = torch.tensor([0, 1, 0])
    a = torch.tensor([1, 0, 0, torch.pi / 2])
    R = ptr.matrix_from_axis_angle(a)
    vR = torch.dot(R, v)
    assert pytest.approx(ptr.angle_between_vectors(vR, v)) == a[-1]
    v = torch.tensor([0, 0, 1])
    a = torch.tensor([1, 0, 0, torch.pi / 2])
    R = ptr.matrix_from_axis_angle(a)
    vR = torch.dot(R, v)
    assert pytest.approx(ptr.angle_between_vectors(vR, v)) == a[-1]


def test_angle_between_close_vectors():
    """Test angle between close vectors.

    See issue #47.
    """
    a = torch.tensor([0.9689124217106448, 0.24740395925452294, 0.0, 0.0])
    b = torch.tensor([0.9689124217106448, 0.247403959254523, 0.0, 0.0])
    angle = ptr.angle_between_vectors(a, b)
    assert pytest.approx(angle) == 0.0


def test_angle_to_zero_vector_is_nan():
    """Test angle to zero vector."""
    a = torch.tensor([1.0, 0.0])
    b = torch.tensor([0.0, 0.0])
    with warnings.catch_warnings(record=True) as w:
        angle = ptr.angle_between_vectors(a, b)
        # todo: no waning is shown
        assert len(w) == 1
    assert torch.isnan(angle)


def test_vector_projection_on_zero_vector():
    """Test projection on zero vector."""
    rng = torch.Generator().manual_seed(23)
    for _ in range(5):
        a = ptr.random_vector(rng, 3)
        a_on_b = ptr.vector_projection(a, torch.zeros(3))
        assert_close(a_on_b, torch.zeros(3))


def test_vector_projection():
    """Test orthogonal projection of one vector to another vector."""
    a = torch.ones(3)
    a_on_unitx = ptr.vector_projection(a, ptr.unitx)
    assert_close(a_on_unitx, ptr.unitx)
    assert (
        pytest.approx(ptr.angle_between_vectors(a_on_unitx, ptr.unitx)) == 0.0
    )

    a2_on_unitx = ptr.vector_projection(2 * a, ptr.unitx)
    assert_close(a2_on_unitx, 2 * ptr.unitx)
    assert (
        pytest.approx(ptr.angle_between_vectors(a2_on_unitx, ptr.unitx)) == 0.0
    )

    a_on_unity = ptr.vector_projection(a, ptr.unity)
    assert_close(a_on_unity, ptr.unity)
    assert (
        pytest.approx(ptr.angle_between_vectors(a_on_unity, ptr.unity)) == 0.0
    )

    minus_a_on_unity = ptr.vector_projection(-a, ptr.unity)
    assert_close(minus_a_on_unity, -ptr.unity)
    assert (
        pytest.approx(ptr.angle_between_vectors(minus_a_on_unity, ptr.unity))
        == torch.pi
    )

    a_on_unitz = ptr.vector_projection(a, ptr.unitz)
    assert_close(a_on_unitz, ptr.unitz)
    assert (
        pytest.approx(ptr.angle_between_vectors(a_on_unitz, ptr.unitz)) == 0.0
    )

    unitz_on_a = ptr.vector_projection(ptr.unitz, a)
    assert_close(unitz_on_a, torch.ones(3) / 3.0)
    assert pytest.approx(ptr.angle_between_vectors(unitz_on_a, a)) == 0.0

    unitx_on_unitx = ptr.vector_projection(ptr.unitx, ptr.unitx)
    assert_close(unitx_on_unitx, ptr.unitx)
    assert (
        pytest.approx(ptr.angle_between_vectors(unitx_on_unitx, ptr.unitx))
        == 0.0
    )


def test_plane_basis_from_normal():
    x, y = ptr.plane_basis_from_normal(ptr.unitx)
    R = torch.column_stack((x, y, ptr.unitx))
    ptr.assert_rotation_matrix(R)

    x, y = ptr.plane_basis_from_normal(ptr.unity)
    R = torch.column_stack((x, y, ptr.unity))
    ptr.assert_rotation_matrix(R)

    x, y = ptr.plane_basis_from_normal(ptr.unitz)
    R = torch.column_stack((x, y, ptr.unitz))
    ptr.assert_rotation_matrix(R)

    rng = torch.Generator().manual_seed(25)
    for _ in range(5):
        normal = ptr.norm_vector(rng.standard_normal(3))
        x, y = ptr.plane_basis_from_normal(normal)
        R = torch.column_stack((x, y, normal))
        ptr.assert_rotation_matrix(R)


if __name__ == "__main__":
    pytest.main([__file__])
