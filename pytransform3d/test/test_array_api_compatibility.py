"""Tests for array API compatibility across different backends."""

import numpy as np
import pytest

# Import array API utilities
from pytransform3d.array_api import (
    get_array_namespace,
    check_array_type,
    ensure_numpy_array,
)

# Import functions that have been migrated to array API
from pytransform3d.rotations import (
    norm_vector,
    perpendicular_to_vectors,
    perpendicular_to_vector,
    angle_between_vectors,
    vector_projection,
    plane_basis_from_normal,
    norm_angle,
    active_matrix_from_angle,
    passive_matrix_from_angle,
    quaternion_from_angle,
)

# Try to import torch for testing
try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TestArrayAPIUtilities:
    """Test array API utility functions."""

    def test_get_array_namespace_numpy(self):
        """Test getting numpy namespace."""
        arr = np.array([1.0, 2.0, 3.0])
        xp = get_array_namespace(arr)
        assert hasattr(xp, "asarray")
        assert hasattr(xp, "zeros")

    def test_get_array_namespace_list(self):
        """Test that lists are converted to numpy."""
        lst = [1.0, 2.0, 3.0]
        xp = get_array_namespace(lst)
        assert hasattr(xp, "asarray")

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_get_array_namespace_torch(self):
        """Test getting torch namespace."""
        arr = torch.tensor([1.0, 2.0, 3.0])
        xp = get_array_namespace(arr)
        assert hasattr(xp, "asarray")
        assert hasattr(xp, "zeros")

    def test_check_array_type_scalar_warning(self, caplog):
        """Test that scalars are converted with warning."""
        result = check_array_type(5.0, "test_param")
        assert isinstance(result, (np.ndarray, float, np.floating))
        assert any(
            "scalar" in record.message.lower() for record in caplog.records
        )

    def test_check_array_type_list_warning(self, caplog):
        """Test that lists are converted with warning."""
        result = check_array_type([1.0, 2.0], "test_param")
        assert isinstance(result, np.ndarray)
        assert any(
            "list" in record.message.lower() for record in caplog.records
        )

    def test_ensure_numpy_array_numpy(self):
        """Test ensure_numpy_array with numpy array."""
        arr = np.array([1.0, 2.0])
        result = ensure_numpy_array(arr, "test")
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, arr)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_ensure_numpy_array_torch_raises(self):
        """Test ensure_numpy_array raises for torch tensor."""
        arr = torch.tensor([1.0, 2.0])
        with pytest.raises(ValueError, match="must be a numpy array"):
            ensure_numpy_array(arr, "test")


class TestRotationsUtilsArrayAPI:
    """Test rotations._utils functions with different array backends."""

    def test_norm_vector_numpy(self):
        """Test norm_vector with numpy."""
        v = np.array([3.0, 4.0, 0.0])
        result = norm_vector(v)
        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(result, [0.6, 0.8, 0.0])
        np.testing.assert_allclose(np.linalg.norm(result), 1.0)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_norm_vector_torch(self):
        """Test norm_vector with torch."""
        v = torch.tensor([3.0, 4.0, 0.0], dtype=torch.float64)
        result = norm_vector(v)
        assert isinstance(result, torch.Tensor)
        torch.testing.assert_close(
            result, torch.tensor([0.6, 0.8, 0.0], dtype=torch.float64)
        )
        torch.testing.assert_close(
            torch.linalg.vector_norm(result),
            torch.tensor(1.0, dtype=torch.float64),
        )

    def test_perpendicular_to_vectors_numpy(self):
        """Test perpendicular_to_vectors with numpy."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        result = perpendicular_to_vectors(a, b)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [0.0, 0.0, 1.0])

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_perpendicular_to_vectors_torch(self):
        """Test perpendicular_to_vectors with torch."""
        a = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        b = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
        result = perpendicular_to_vectors(a, b)
        assert isinstance(result, torch.Tensor)
        torch.testing.assert_close(
            result, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
        )

    def test_angle_between_vectors_numpy(self):
        """Test angle_between_vectors with numpy."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        result = angle_between_vectors(a, b)
        assert isinstance(result, (np.ndarray, float, np.floating))
        np.testing.assert_allclose(result, np.pi / 2, rtol=1e-6)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_angle_between_vectors_torch(self):
        """Test angle_between_vectors with torch."""
        a = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        b = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
        result = angle_between_vectors(a, b)
        assert isinstance(result, torch.Tensor)
        torch.testing.assert_close(
            result,
            torch.tensor(torch.pi / 2, dtype=torch.float64),
            rtol=1e-6,
            atol=1e-8,
        )

    def test_vector_projection_numpy(self):
        """Test vector_projection with numpy."""
        a = np.array([1.0, 1.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        result = vector_projection(a, b)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1.0, 0.0, 0.0])

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_vector_projection_torch(self):
        """Test vector_projection with torch."""
        a = torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64)
        b = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        result = vector_projection(a, b)
        assert isinstance(result, torch.Tensor)
        torch.testing.assert_close(
            result, torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        )


class TestRotationsAngleArrayAPI:
    """Test rotations._angle functions with different array backends."""

    def test_norm_angle_numpy(self):
        """Test norm_angle with numpy."""
        a = np.array([3.5 * np.pi])
        result = norm_angle(a)
        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(result, [-0.5 * np.pi], rtol=1e-6)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_norm_angle_torch(self):
        """Test norm_angle with torch."""
        a = torch.tensor(3.5 * np.pi, dtype=torch.float64)
        result = norm_angle(a)
        assert isinstance(result, torch.Tensor)
        torch.testing.assert_close(
            result,
            torch.tensor(-0.5 * np.pi, dtype=torch.float64),
            rtol=1e-6,
            atol=1e-8,
        )

    def test_active_matrix_from_angle_numpy(self):
        """Test active_matrix_from_angle with numpy."""
        angle = 0.5
        R = active_matrix_from_angle(2, angle)
        assert isinstance(R, np.ndarray)
        assert R.shape == (3, 3)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_active_matrix_from_angle_torch(self):
        """Test active_matrix_from_angle with torch."""
        angle = torch.tensor(0.5, dtype=torch.float64)
        R = active_matrix_from_angle(2, angle)
        assert isinstance(R, torch.Tensor)
        assert R.shape == (3, 3)

    def test_quaternion_from_angle_numpy(self):
        """Test quaternion_from_angle with numpy."""
        angle = 0.5
        q = quaternion_from_angle(0, angle)
        assert isinstance(q, np.ndarray)
        assert q.shape == (4,)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_quaternion_from_angle_torch(self):
        """Test quaternion_from_angle with torch."""
        angle = torch.tensor(0.5, dtype=torch.float64)
        q = quaternion_from_angle(0, angle)
        assert isinstance(q, torch.Tensor)
        assert q.shape == (4,)


class TestVisualizationValidation:
    """Test that visualization functions validate numpy arrays."""

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_plot_basis_rejects_torch(self):
        """Test that plot_basis rejects torch tensors."""
        from pytransform3d.rotations import plot_basis

        R = torch.eye(3, dtype=torch.float64)
        with pytest.raises(ValueError, match="must be a numpy array"):
            plot_basis(R=R)

    def test_plot_basis_accepts_numpy(self):
        """Test that plot_basis accepts numpy arrays."""
        from pytransform3d.rotations import plot_basis
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        R = np.eye(3)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        result = plot_basis(ax=ax, R=R)
        plt.close(fig)
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestRotationsMatrixArrayAPI:
    """Test rotations._matrix functions with different array backends."""

    def test_check_matrix_numpy(self):
        """Test check_matrix with numpy."""
        from pytransform3d.rotations import check_matrix
        
        R = np.eye(3)
        result = check_matrix(R)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 3)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_check_matrix_torch(self):
        """Test check_matrix with torch."""
        from pytransform3d.rotations import check_matrix
        
        R = torch.eye(3, dtype=torch.float64)
        result = check_matrix(R)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 3)

    def test_norm_matrix_numpy(self):
        """Test norm_matrix with numpy."""
        from pytransform3d.rotations import norm_matrix
        
        R = np.eye(3) + np.random.randn(3, 3) * 0.01
        result = norm_matrix(R)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 3)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_norm_matrix_torch(self):
        """Test norm_matrix with torch."""
        from pytransform3d.rotations import norm_matrix
        
        R = torch.eye(3, dtype=torch.float64) + torch.randn(3, 3, dtype=torch.float64) * 0.01
        result = norm_matrix(R)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 3)

    def test_matrix_from_two_vectors_numpy(self):
        """Test matrix_from_two_vectors with numpy."""
        from pytransform3d.rotations import matrix_from_two_vectors
        
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        result = matrix_from_two_vectors(a, b)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 3)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_matrix_from_two_vectors_torch(self):
        """Test matrix_from_two_vectors with torch."""
        from pytransform3d.rotations import matrix_from_two_vectors
        
        a = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        b = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
        result = matrix_from_two_vectors(a, b)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 3)

    def test_quaternion_from_matrix_numpy(self):
        """Test quaternion_from_matrix with numpy."""
        from pytransform3d.rotations import quaternion_from_matrix
        
        R = np.eye(3)
        result = quaternion_from_matrix(R)
        assert isinstance(result, np.ndarray)
        assert result.shape == (4,)
        np.testing.assert_allclose(result, [1.0, 0.0, 0.0, 0.0], atol=1e-6)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_quaternion_from_matrix_torch(self):
        """Test quaternion_from_matrix with torch."""
        from pytransform3d.rotations import quaternion_from_matrix
        
        R = torch.eye(3, dtype=torch.float64)
        result = quaternion_from_matrix(R)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (4,)
        torch.testing.assert_close(
            result, 
            torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64),
            atol=1e-6, rtol=1e-6
        )

    def test_axis_angle_from_matrix_numpy(self):
        """Test axis_angle_from_matrix with numpy."""
        from pytransform3d.rotations import axis_angle_from_matrix
        
        R = np.eye(3)
        result = axis_angle_from_matrix(R)
        assert isinstance(result, np.ndarray)
        assert result.shape == (4,)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_axis_angle_from_matrix_torch(self):
        """Test axis_angle_from_matrix with torch."""
        from pytransform3d.rotations import axis_angle_from_matrix
        
        R = torch.eye(3, dtype=torch.float64)
        result = axis_angle_from_matrix(R)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (4,)


class TestRotationsEulerArrayAPI:
    """Test rotations._euler functions with different array backends."""

    def test_norm_euler_numpy(self):
        """Test norm_euler with numpy."""
        from pytransform3d.rotations import norm_euler
        
        e = np.array([0.1, 0.2, 0.3])
        result = norm_euler(e, 0, 1, 2)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_norm_euler_torch(self):
        """Test norm_euler with torch."""
        from pytransform3d.rotations import norm_euler
        
        e = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
        result = norm_euler(e, 0, 1, 2)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3,)

    def test_matrix_from_euler_numpy(self):
        """Test matrix_from_euler with numpy."""
        from pytransform3d.rotations import matrix_from_euler
        
        e = np.array([0.1, 0.2, 0.3])
        result = matrix_from_euler(e, 0, 1, 2, True)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 3)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_matrix_from_euler_torch(self):
        """Test matrix_from_euler with torch."""
        from pytransform3d.rotations import matrix_from_euler
        
        e = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
        result = matrix_from_euler(e, 0, 1, 2, True)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 3)

    def test_euler_near_gimbal_lock_numpy(self):
        """Test euler_near_gimbal_lock with numpy."""
        from pytransform3d.rotations import euler_near_gimbal_lock
        
        e = np.array([0.1, 0.0, 0.3])  # Near gimbal lock
        result = euler_near_gimbal_lock(e, 0, 1, 0, tolerance=0.1)
        assert isinstance(result, (bool, np.bool_))

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_euler_near_gimbal_lock_torch(self):
        """Test euler_near_gimbal_lock with torch."""
        from pytransform3d.rotations import euler_near_gimbal_lock
        
        e = torch.tensor([0.1, 0.0, 0.3], dtype=torch.float64)
        result = euler_near_gimbal_lock(e, 0, 1, 0, tolerance=0.1)
        # Result can be bool, np.bool_, or torch.Tensor
        assert result is not None
