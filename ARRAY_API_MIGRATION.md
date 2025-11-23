# Array API Migration Guide

## Overview

This document describes the ongoing migration of pytransform3d to support the [Python Array API standard](https://data-apis.org/array-api/latest/), enabling compatibility with multiple array backends including NumPy, PyTorch, JAX, CuPy, and others.

## Goals

1. **Backend Flexibility**: Allow users to choose their preferred array library (NumPy, PyTorch, JAX, etc.)
2. **Type Preservation**: Functions return the same array type as their inputs
3. **GPU Acceleration**: Enable GPU computations through PyTorch or JAX without code changes
4. **Backward Compatibility**: Existing NumPy-based code continues to work unchanged
5. **Clear Boundaries**: Visualization functions explicitly require NumPy arrays (matplotlib dependency)

## Implementation Strategy

### Core Principles

1. **Minimal Changes**: Only modify code necessary for array API compatibility
2. **Type Awareness**: Use `get_array_namespace()` to detect the array backend
3. **Validation**: Add type checking with clear error messages
4. **Logging**: Warn when converting scalars/lists to NumPy arrays

### Module-by-Module Approach

Migration is being done systematically, one module at a time:

1. **Core Utilities First**: Start with widely-used utility functions
2. **Visualization Validation**: Add NumPy-only checks to plot functions
3. **Comprehensive Testing**: Add tests for each migrated module
4. **Documentation**: Update docstrings and examples

## Migration Status

### ✅ Completed Modules

#### `array_api.py` (Core Utilities)
- `get_array_namespace()` - Detect array backend
- `check_array_type()` - Validate and convert inputs
- `ensure_numpy_array()` - Enforce NumPy for visualization
- Comprehensive logging for type conversions

#### `rotations/_utils.py` (6 functions) ✅
- `norm_vector()` - Normalize vectors
- `perpendicular_to_vectors()` - Cross product
- `perpendicular_to_vector()` - Find perpendicular vector
- `angle_between_vectors()` - Compute angles
- `vector_projection()` - Project vectors
- `plane_basis_from_normal()` - Compute basis vectors

#### `rotations/_angle.py` (4 functions) ✅
- `norm_angle()` - Normalize angles
- `active_matrix_from_angle()` - Create rotation matrices
- `passive_matrix_from_angle()` - Create rotation matrices
- `quaternion_from_angle()` - Create quaternions

#### `rotations/_matrix.py` (7 functions) ✅
- `check_matrix()` - Rotation matrix validation
- `matrix_requires_renormalization()` - Check if matrix needs renormalization
- `norm_matrix()` - Orthonormalize rotation matrix
- `matrix_from_two_vectors()` - Create rotation matrix from two vectors
- `quaternion_from_matrix()` - Convert rotation matrix to quaternion
- `axis_angle_from_matrix()` - Convert rotation matrix to axis-angle
- `compact_axis_angle_from_matrix()` - Convert to compact axis-angle

#### `rotations/_euler.py` (4 functions) ✅
- `norm_euler()` - Normalize Euler angle range
- `euler_near_gimbal_lock()` - Check if Euler angles are near gimbal lock
- `matrix_from_euler()` - Compute rotation matrix from Euler angles
- `assert_euler_equal()` - Assert two Euler angles are equal

#### `rotations/_mrp.py` (4 functions) ✅
- `check_mrp()` - Input validation of modified Rodrigues parameters
- `norm_mrp()` - Normalize MRP angle range
- `mrp_near_singularity()` - Check if MRPs are near singularity
- `mrp_double()` - Get alternative MRP representation

#### `batch_rotations/_utils.py` (3 functions) ✅
- `norm_vectors()` - Batch vector normalization
- `angles_between_vectors()` - Batch angle computation
- `cross_product_matrices()` - Cross product matrix generation

#### `rotations/_plot.py` (3 functions - validation added) ✅
- `plot_basis()` - Requires NumPy arrays
- `plot_axis_angle()` - Requires NumPy arrays
- `plot_bivector()` - Requires NumPy arrays

### 🔄 In Progress

#### `rotations/_quaternion.py` (20/20 functions completed, 100%) ✅
All functions migrated:
- `check_quaternion()` - Quaternion validation
- `check_quaternions()` - Batch quaternion validation
- `quaternion_requires_renormalization()` - Check if quaternion needs renormalization
- `quaternion_double()` - Create equivalent quaternion
- `pick_closest_quaternion_impl()` - Resolve quaternion ambiguity
- `quaternion_integrate()` - Integrate angular velocities to quaternions
- `quaternion_gradient()` - Compute time derivatives of quaternions
- `concatenate_quaternions()` - Quaternion multiplication
- `q_prod_vector()` - Apply quaternion rotation to vector
- `q_conj()` - Quaternion conjugate
- `quaternion_dist()` - Distance between quaternions
- `matrix_from_quaternion()` - Convert to rotation matrix
- `axis_angle_from_quaternion()` - Convert to axis-angle
- `quaternion_xyzw_from_wxyz()` - Convention conversion
- `quaternion_wxyz_from_xyzw()` - Convention conversion
- Plus additional conversion and operation functions

#### `rotations/_axis_angle.py` (15/15 functions completed, 100%) ✅
All functions migrated:
- `check_axis_angle()` - Axis-angle validation
- `check_compact_axis_angle()` - Compact axis-angle validation
- `norm_axis_angle()` - Normalize axis-angle
- `norm_compact_axis_angle()` - Normalize compact axis-angle
- `compact_axis_angle_near_pi()` - Check if angle near pi
- `assert_axis_angle_equal()` - Assert equality with sign handling
- `assert_compact_axis_angle_equal()` - Assert equality for compact form
- `axis_angle_from_two_directions()` - Compute from direction vectors
- `matrix_from_axis_angle()` - Convert to rotation matrix (Rodrigues)
- `axis_angle_from_compact_axis_angle()` - Convert from compact form
- `quaternion_from_axis_angle()` - Convert to quaternion (exponential map)
- `mrp_from_axis_angle()` - Convert to modified Rodrigues parameters
- Plus additional conversion functions

#### `batch_rotations` (Partially complete)
Some functions already support array API:
- `batch_concatenate_quaternions()`
- Functions in `_quaternion.py`

### ⏳ Pending Modules

#### `rotations` (Remaining ~60 functions)
- `_slerp.py` - Spherical interpolation (~6 functions)
- `_rotors.py` - Rotor operations (~8 functions)
- `_rot_log.py` - Rotation logarithm (~4 functions)
- `_random.py` - Random rotations (~5 functions)
- `_jacobians.py` - Jacobian matrices (~4 functions)
- `_polar_decomp.py` - Polar decomposition (~1 function)
- Remaining functions in `_quaternion.py` (~15 functions)
- Remaining functions in `_axis_angle.py` (~11 functions)

#### Other Core Modules
- `transformations/` - SE(3) transformations
- `trajectories/` - Trajectory operations
- `coordinates.py` - Coordinate system conversions
- `camera.py` - Camera models
- `_geometry.py` - Geometric primitives
- `uncertainty/` - Uncertainty propagation

## Usage Examples

### NumPy (Default Behavior)

```python
import numpy as np
from pytransform3d import rotations as pr

# All operations use NumPy
v = np.array([3.0, 4.0, 0.0])
v_norm = pr.norm_vector(v)
print(type(v_norm))  # <class 'numpy.ndarray'>
```

### PyTorch

```python
import torch
from pytransform3d import rotations as pr

# Same code, different array type
v = torch.tensor([3.0, 4.0, 0.0], dtype=torch.float64)
v_norm = pr.norm_vector(v)
print(type(v_norm))  # <class 'torch.Tensor'>

# Works on GPU too!
v_gpu = v.cuda()
v_norm_gpu = pr.norm_vector(v_gpu)
print(v_norm_gpu.device)  # cuda:0
```

### JAX (Future)

```python
import jax.numpy as jnp
from pytransform3d import rotations as pr

# Will work once JAX support is tested
v = jnp.array([3.0, 4.0, 0.0])
v_norm = pr.norm_vector(v)
print(type(v_norm))  # DeviceArray
```

### Automatic Conversion

Scalars and lists are automatically converted to NumPy with a warning:

```python
# Scalar input (warning logged)
angle = pr.norm_angle(3.5 * np.pi)  # Warning: converting scalar to numpy

# List input (warning logged)
v = [3.0, 4.0, 0.0]
v_norm = pr.norm_vector(v)  # Warning: converting list to numpy
```

### Visualization Functions

Visualization functions explicitly require NumPy arrays:

```python
import torch
from pytransform3d import rotations as pr

# This will raise ValueError
R_torch = torch.eye(3)
pr.plot_basis(R=R_torch)  # ValueError: must be a numpy array

# This works
R_numpy = np.eye(3)
pr.plot_basis(R=R_numpy)  # OK
```

## Testing

### Test Coverage

Comprehensive tests have been added in `pytransform3d/test/test_array_api_compatibility.py`:

- Array API utilities tests
- NumPy backend tests
- PyTorch backend tests
- Visualization validation tests
- Type preservation tests

Run tests with:

```bash
# All array API tests
pytest pytransform3d/test/test_array_api_compatibility.py -v

# With PyTorch
pytest pytransform3d/test/test_array_api_compatibility.py -v

# Skip PyTorch tests if not installed
pytest pytransform3d/test/test_array_api_compatibility.py -v -m "not torch"
```

## Migration Pattern

When migrating a new function, follow this pattern:

### 1. Import Array API Utilities

```python
from ..array_api import get_array_namespace, check_array_type
```

### 2. Check and Convert Inputs

```python
def my_function(array_input):
    # Validate and convert if needed (with warning for scalars/lists)
    array_input = check_array_type(array_input, "array_input")
    
    # Get the appropriate array namespace
    xp = get_array_namespace(array_input)
```

### 3. Use Namespace Functions

```python
    # Replace np.* with xp.*
    result = xp.asarray(...)
    norm = xp.linalg.vector_norm(...)
    cos_val = xp.cos(...)
```

### 4. Array API Considerations

Some operations need special handling:

```python
# Use linalg.cross when available (for torch)
if hasattr(xp.linalg, 'cross'):
    result = xp.linalg.cross(a, b)
else:
    result = xp.cross(a, b)

# Use sum instead of dot for compatibility
dot_product = xp.sum(a * b)

# Use vector_norm instead of norm
norm = xp.linalg.vector_norm(v)
```

### 5. For Visualization Functions

```python
from ..array_api import ensure_numpy_array

def plot_something(array_input):
    # Enforce NumPy only
    array_input = ensure_numpy_array(array_input, "array_input")
    # ... rest of plotting code ...
```

## Benefits

1. **Performance**: Use GPU acceleration with PyTorch or JAX
2. **Flexibility**: Choose the best array library for your use case
3. **Interoperability**: Seamlessly integrate with ML frameworks
4. **Gradients**: Use automatic differentiation with PyTorch or JAX
5. **Future-Proof**: Compatible with emerging array libraries

## Known Limitations

1. **Visualization**: Only NumPy arrays supported (matplotlib requirement)
2. **Legacy Code**: Some old functions may not yet support array API
3. **Testing**: JAX and CuPy support not yet fully tested
4. **Dependencies**: array-api-compat required

## Contributing

To contribute to the array API migration:

1. Choose a module from the "Pending" list
2. Follow the migration pattern above
3. Add tests in `test_array_api_compatibility.py`
4. Update this documentation
5. Submit a pull request

## References

- [Python Array API Standard](https://data-apis.org/array-api/latest/)
- [array-api-compat Documentation](https://github.com/data-apis/array-api-compat)
- [NumPy Array API Support](https://numpy.org/neps/nep-0047-array-api-standard.html)
- [PyTorch Array API](https://pytorch.org/docs/stable/torch.html)

## Version History

- **v1.0.0** (2025-01): Initial array API migration started
  - Core utilities implemented
  - rotations._utils migrated
  - rotations._angle migrated
  - Visualization validation added
  - Comprehensive tests added
  - PyTorch support verified
