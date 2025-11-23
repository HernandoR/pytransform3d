"""
Array API Compatibility Example
=================================

This example demonstrates how pytransform3d now supports multiple array backends
through the Array API standard, including NumPy, PyTorch, JAX, and others.
"""

print(__doc__)

import numpy as np

# Try to import optional backends
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch not available - skipping torch examples")

from pytransform3d import rotations as pr

# %%
# Using NumPy (default behavior)
# -------------------------------
#
# The traditional way - all operations use NumPy arrays.

print("\n" + "="*60)
print("NumPy Examples")
print("="*60)

# Create a vector and normalize it
v_numpy = np.array([3.0, 4.0, 0.0])
v_normalized = pr.norm_vector(v_numpy)
print(f"NumPy input type: {type(v_numpy)}")
print(f"NumPy output type: {type(v_normalized)}")
print(f"Normalized vector: {v_normalized}")
print(f"Norm: {np.linalg.norm(v_normalized)}")

# Compute angle between vectors
a_numpy = np.array([1.0, 0.0, 0.0])
b_numpy = np.array([0.0, 1.0, 0.0])
angle = pr.angle_between_vectors(a_numpy, b_numpy)
print(f"\nAngle between x and y axes: {angle:.4f} radians ({np.degrees(angle):.1f}°)")

# Generate rotation matrix
R_numpy = pr.active_matrix_from_angle(2, np.pi/4)  # Rotation around z-axis
print(f"\nRotation matrix type: {type(R_numpy)}")
print(f"Shape: {R_numpy.shape}")

# %%
# Using PyTorch
# -------------
#
# The same operations work seamlessly with PyTorch tensors!

if HAS_TORCH:
    print("\n" + "="*60)
    print("PyTorch Examples")
    print("="*60)
    
    # Create a PyTorch tensor and normalize it
    v_torch = torch.tensor([3.0, 4.0, 0.0], dtype=torch.float64)
    v_normalized_torch = pr.norm_vector(v_torch)
    print(f"PyTorch input type: {type(v_torch)}")
    print(f"PyTorch output type: {type(v_normalized_torch)}")
    print(f"Normalized vector: {v_normalized_torch}")
    print(f"Norm: {torch.linalg.vector_norm(v_normalized_torch)}")
    
    # Compute angle between vectors
    a_torch = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    b_torch = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    angle_torch = pr.angle_between_vectors(a_torch, b_torch)
    print(f"\nAngle between x and y axes: {angle_torch:.4f} radians")
    print(f"Type: {type(angle_torch)}")
    
    # Generate rotation matrix with PyTorch
    angle_torch_val = torch.tensor(torch.pi/4, dtype=torch.float64)
    R_torch = pr.active_matrix_from_angle(2, angle_torch_val)
    print(f"\nRotation matrix type: {type(R_torch)}")
    print(f"Shape: {R_torch.shape}")
    print(f"Device: {R_torch.device}")
    
    # You can also move tensors to GPU if available
    if torch.cuda.is_available():
        v_gpu = v_torch.cuda()
        v_norm_gpu = pr.norm_vector(v_gpu)
        print(f"\nGPU tensor type: {type(v_norm_gpu)}")
        print(f"GPU device: {v_norm_gpu.device}")
    else:
        print("\nCUDA not available - skipping GPU example")

# %%
# Automatic Conversion with Warnings
# -----------------------------------
#
# When you pass scalars or lists, they are automatically converted to NumPy
# arrays with a warning message.

print("\n" + "="*60)
print("Automatic Conversion Examples")
print("="*60)

import logging
# Enable warnings to see conversion messages
logging.basicConfig(level=logging.WARNING, format='%(name)s: %(message)s')

# Using a list (will be converted to NumPy with warning)
v_list = [3.0, 4.0, 0.0]
v_from_list = pr.norm_vector(v_list)
print(f"List input converted to: {type(v_from_list)}")

# Using a scalar (will be converted to NumPy with warning)
angle_scalar = pr.norm_angle(3.5 * np.pi)
print(f"Scalar input converted to: {type(angle_scalar)}")

# %%
# Visualization Functions Require NumPy
# --------------------------------------
#
# Visualization functions explicitly require NumPy arrays for compatibility
# with matplotlib.

print("\n" + "="*60)
print("Visualization Validation")
print("="*60)

if HAS_TORCH:
    # Try to plot with PyTorch tensor (will raise ValueError)
    try:
        R_torch = torch.eye(3, dtype=torch.float64)
        pr.plot_basis(R=R_torch)
    except ValueError as e:
        print(f"✓ Correctly rejected PyTorch tensor: {str(e)[:60]}...")

# Plot with NumPy (works fine)
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for this example
import matplotlib.pyplot as plt

R_numpy = np.eye(3)
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
pr.plot_basis(ax=ax, R=R_numpy, s=1.0)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Rotation Basis Vectors')
print("✓ NumPy array accepted for visualization")

# Save the figure
plt.savefig('array_api_example.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved visualization to 'array_api_example.png'")

# %%
# Benefits of Array API Support
# ------------------------------
#
# 1. **Seamless Integration**: Use pytransform3d with your preferred array library
# 2. **GPU Acceleration**: Leverage PyTorch or JAX for GPU-accelerated computations
# 3. **Automatic Differentiation**: Use PyTorch's autograd for gradient-based optimization
# 4. **Type Safety**: Functions preserve and return the same array type as input
# 5. **Backward Compatibility**: Existing NumPy code continues to work unchanged

print("\n" + "="*60)
print("Summary")
print("="*60)
print("✓ Array API support enables using pytransform3d with multiple backends")
print("✓ NumPy, PyTorch, JAX, and other Array API compatible libraries supported")
print("✓ Automatic type preservation - input type = output type")
print("✓ Visualization functions require NumPy for matplotlib compatibility")
print("✓ Scalar and list inputs auto-converted to NumPy with warnings")
