# Comprehensive Array API Migration Plan

## Overview
Migrating 26 core computational modules (403 numpy operations) across 6 packages.

## Status Summary
- **Rotations:** ✅ 100% Complete (14 modules, 88 functions)
- **Other Packages:** 🔄 2/26 files (8%)

## Phase 1: TRANSFORMATIONS (8 files, 163 ops) - CRITICAL

### Completed
1. ✅ _pq.py (6 ops) - Position+quaternion
2. ✅ _random.py (9 ops) - Random generation

### In Progress  
3. ⏳ _plot.py (10 ops) - Plotting utilities
4. ⏳ _transform_operations.py (14 ops) - Transform operations
5. ⏳ _transform.py (20 ops) - Transform utilities
6. ⏳ _dual_quaternion.py (21 ops) - Dual quaternions
7. ⏳ _jacobians.py (21 ops) - Jacobians
8. ⏳ _screws.py (62 ops) - Screw theory

## Phase 2: BATCH_ROTATIONS (4 files, 48 ops) - HIGH PRIORITY

1. ⏳ _angle.py (4 ops)
2. ⏳ _euler.py (8 ops)
3. ⏳ _axis_angle.py (11 ops)
4. ⏳ _matrix.py (25 ops)

## Phase 3: TRAJECTORIES (3 files, 61 ops) - HIGH PRIORITY

1. ⏳ _transforms.py (19 ops)
2. ⏳ _random.py (20 ops)
3. ⏳ _screws.py (22 ops)

## Phase 4: STANDALONE MODULES (3 files, 67 ops) - MEDIUM

1. ⏳ camera.py (19 ops)
2. ⏳ urdf.py (21 ops)
3. ⏳ coordinates.py (27 ops)

## Phase 5: TRANSFORM_MANAGER (3 files, 22 ops) - MEDIUM

1. ⏳ _transform_manager.py (4 ops)
2. ⏳ _transform_graph_base.py (6 ops)
3. ⏳ _temporal_transform_manager.py (12 ops)

## Phase 6: UNCERTAINTY (5 files, 42 ops) - LOW PRIORITY

1. ⏳ _invert.py (1 op)
2. ⏳ _frechet_mean.py (7 ops)
3. ⏳ _plot.py (7 ops)
4. ⏳ _fusion.py (13 ops)
5. ⏳ _composition.py (14 ops)

## Migration Progress
- Files completed: 2/26 (8%)
- Operations migrated: 15/403 (4%)
- Estimated remaining effort: ~388 operations across 24 files

## Notes
- Each file requires: import addition, function-by-function migration, testing
- Large files (_screws.py with 62 ops) require extra attention
- All migrations maintain backward compatibility
- Progressive testing after each batch
