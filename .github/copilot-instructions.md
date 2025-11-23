# Copilot Instructions for pytransform3d

## Repository Overview

**pytransform3d** is a Python library for transformations in three dimensions. It provides operations for rotation (orientation) and translation (position) representations, conversions between representations, visualization with matplotlib, and tools like TransformManager for managing complex transformation chains.

### Key Stats
- **Language**: Python 3.8+
- **Size**: Medium (~65 Python files across 11 modules)
- **Type**: Scientific computing library
- **Primary Dependencies**: NumPy, SciPy, matplotlib, lxml, Open3D (optional)
- **Testing Framework**: pytest with pytest-cov
- **Linting**: flake8, ruff, black (configured but not enforced in CI)
- **Documentation**: Sphinx with sphinx-gallery

## Environment Management with uv

This project uses **uv** for dependency management. uv is an extremely fast Python package and project manager written in Rust. It replaces pip, pip-tools, poetry, pyenv, virtualenv, and more. It includes both a pip-compatible CLI (prepend `uv` to pip commands) and a first-class project interface with lockfiles and workspace support.

### Installing uv
If uv is not installed:
```bash
pip install uv
```

For more installation options, see [uv installation docs](https://docs.astral.sh/uv/getting-started/installation/).

### Key uv Commands

**ALWAYS use `uv run` to execute Python commands** in this project to ensure the correct virtual environment and dependencies:

```bash
# Sync dependencies (creates/updates .venv)
uv sync

# Install the package in editable mode (REQUIRED before testing)
uv pip install -e .

# Run tests
MPLBACKEND=Agg uv run pytest

# Run Python scripts
uv run python script.py

# Install additional packages
uv pip install package-name
```

**Critical**: The package MUST be installed in editable mode (`uv pip install -e .`) before running tests, otherwise imports will fail with `ModuleNotFoundError: No module named 'pytransform3d'`.

### uv Capabilities

uv provides several advantages for this project:
- **Fast dependency resolution**: Written in Rust for performance
- **Lockfile support**: `uv.lock` ensures reproducible environments
- **Pip compatibility**: Can use `uv pip` as a drop-in replacement for pip
- **Project management**: Handles virtual environments automatically with `uv run`
- **Workspace support**: Manages complex project structures

For comprehensive documentation, see the [uv documentation](https://docs.astral.sh/uv/) or [uv for LLMs](https://docs.astral.sh/uv/llms.txt).

## Build, Test, and Validation

### Bootstrap/Setup (First Time)

1. Install uv: `pip install uv`
2. Sync dependencies: `uv sync` (takes ~60-90 seconds, downloads large packages like Open3D ~427MB)
3. Install system dependency: `sudo apt-get install -y graphviz` (required for pydot graph exports)
4. Install package in editable mode: `uv pip install -e .` (REQUIRED - do not skip)

### Running Tests

**ALWAYS set `MPLBACKEND=Agg` when running tests** to avoid display issues:

```bash
# Run all tests (takes ~9 seconds, 393 tests)
MPLBACKEND=Agg uv run pytest

# Run with coverage report (generates htmlcov/index.html)
MPLBACKEND=Agg uv run pytest

# Run specific test file
MPLBACKEND=Agg uv run pytest pytransform3d/rotations/test/test_angle.py

# Run with verbose output
MPLBACKEND=Agg uv run pytest -v

# Stop on first failure
MPLBACKEND=Agg uv run pytest -x
```

**Test Configuration**: Tests are configured in `setup.cfg` under `[tool:pytest]` section. Coverage omits plotting utilities and editor modules (see `.coveragerc`).

### Linting

The project uses flake8 for critical checks in CI, with ruff and black configured in `pyproject.toml` but not enforced:

```bash
# Critical flake8 checks (as used in CI)
uv run flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

# Warning-level flake8 checks
uv run flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

# Ruff linting (optional, see pyproject.toml for config)
uv run ruff check . --select=E,F,UP,B,SIM,I

# Black formatting (optional, line-length=80)
uv run black --check pytransform3d examples
```

**Known linting notes**:
- Some ambiguous variable names (E741) exist for mathematical clarity (e.g., `O` for origin, `I` for identity matrix)
- Ruff configuration targets Python 3.8+ with specific rule selections in `pyproject.toml`

### Building Documentation

Documentation requires additional dependencies:

```bash
# Install doc dependencies
uv pip install -e '.[doc]'

# Build HTML documentation
cd doc
make html

# Output location: doc/build/html/index.html
```

Documentation uses Sphinx with sphinx-gallery for examples. Build may take several minutes as it runs all example scripts.

## Project Structure

### Root Files
- `pyproject.toml` - Main project configuration (dependencies, tool config for black/ruff)
- `setup.py` - Legacy setup file (still used for editable installs)
- `setup.cfg` - pytest configuration
- `.coveragerc` - Coverage configuration (omits plot_utils, editor, visualizer)
- `uv.lock` - Locked dependency versions
- `requirements.txt` - Basic runtime dependencies

### Source Layout (`pytransform3d/`)

**Core Modules** (each has own `test/` subdirectory):
- `rotations/` - Rotation representations (matrix, quaternion, axis-angle, Euler, MRP, rotor)
- `transformations/` - Transformation operations (homogeneous matrices, dual quaternions, screws)
- `batch_rotations/` - Vectorized batch operations for rotations
- `transform_manager/` - TransformManager class for transformation graphs
- `trajectories/` - Trajectory representations and operations
- `uncertainty/` - Uncertainty handling for transformations
- `plot_utils/` - Matplotlib visualization utilities (omitted from coverage)
- `visualizer/` - Open3D visualization interface (omitted from coverage)

**Other Key Files**:
- `camera.py` - Camera models and projections
- `coordinates.py` - Coordinate system utilities
- `urdf.py` - URDF file parsing (requires lxml)
- `editor.py` - Qt-based transformation editor (omitted from coverage)

### Test Structure
- Tests are located in `test/` subdirectories within each module
- Integration tests in `pytransform3d/test/`
- Test data in `test/test_data/`
- All test files follow pattern `test_*.py`

### Examples
- `examples/plots/` - Matplotlib plotting examples
- `examples/visualizations/` - Open3D visualization examples  
- `examples/animations/` - Animation examples
- `examples/apps/` - Interactive applications

## CI/CD and Validation

### GitHub Actions Workflows

**`.github/workflows/python-package.yml`** (runs on all pushes except gh-pages):
- Tests Python 3.8, 3.9, 3.10, 3.11, 3.12, 3.13
- Installs: graphviz (apt), flake8, package with `[test]` extras, pydot, trimesh
- Runs flake8 with critical checks
- Runs: `MPLBACKEND=Agg pytest`
- Uploads coverage to codecov

**`.github/workflows/docs.yml`** (runs on main branch):
- Uses pixi (not uv) for doc builds
- Builds documentation with sphinx
- Deploys to GitHub Pages

**CircleCI** (`.circleci/config.yml`):
- Python 3.10 on Docker
- Installs: graphviz, gfortran, libopenblas-dev, liblapack-dev
- Creates venv and installs with `pip install -e .[test] pydot trimesh`
- Runs: `MPLBACKEND=Agg pytest`

### Replicating CI Locally

To replicate GitHub Actions checks locally:

```bash
# 1. Install system dependencies
sudo apt-get install -y graphviz

# 2. Install Python dependencies
uv pip install -e '.[test]' pydot trimesh

# 3. Run flake8 critical checks
uv run flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

# 4. Run tests
MPLBACKEND=Agg uv run pytest
```

**Expected Results**:
- flake8: Should pass with no critical errors
- pytest: All 393 tests should pass in ~9 seconds
- One test (`test_png_export`) requires graphviz system package

## Common Issues and Workarounds

### ModuleNotFoundError for pytransform3d
**Cause**: Package not installed in editable mode  
**Fix**: Run `uv pip install -e .` before testing

### graphviz/dot not found
**Cause**: System package graphviz not installed  
**Fix**: `sudo apt-get install -y graphviz`  
**Affected Test**: `test_transform_manager.py::test_png_export`

### Display/GUI errors during tests
**Cause**: matplotlib trying to use GUI backend  
**Fix**: Always set `MPLBACKEND=Agg` environment variable before running tests

### Large download during uv sync
**Expected**: Open3D is ~427MB, first sync takes 60-90 seconds. This is normal.

### Permission errors on Windows (in tests)
**Known Issue**: Some transform_manager tests have workarounds for Windows permission issues when cleaning up temporary PNG files. See `test_transform_manager.py` for examples.

## Development Workflow

### Making Changes

1. **Setup**: Follow bootstrap steps if not done already
2. **Branch**: Create feature/fix branch (never push directly to main)
3. **Code**: Make changes to source files
4. **Test early**: Run relevant tests frequently with `MPLBACKEND=Agg uv run pytest <test_file>`
5. **Lint**: Check code with flake8 for critical issues
6. **Test all**: Run full test suite before committing
7. **Document**: Add/update docstrings following NumPy style
8. **Pull Request**: Target branch is typically `develop` (not `main`)

### Adding New Features

**Required for new features** (see CONTRIBUTING.md):
- NumPy-style docstrings
- Unit tests covering all branches
- API documentation entry
- Consider adding an example script

### Code Style

- **Line length**: 80 characters (black/ruff config)
- **Docstrings**: NumPy style (enforced for public API)
- **Type hints**: PEP 484 stubs in `.pyi` files for key modules
- **Python version**: Support 3.8+ (check pyproject.toml target-version)

## Key Dependencies and Versions

**Runtime** (from pyproject.toml):
- numpy >= 1.24.4
- scipy >= 1.10.1
- matplotlib >= 3.7.5
- lxml >= 5.4.0
- array-api-compat >= 1.9

**Optional**:
- pydot >= 3.0.4 (graph visualization, requires graphviz)
- trimesh >= 4.6.8 (mesh processing)
- pycollada >= 0.9 (COLLADA file support)
- open3d >= 0.19.0 (3D visualization)

**Testing**:
- pytest >= 8.3.5
- pytest-cov >= 5.0.0

## Quick Reference Commands

```bash
# Setup (first time)
pip install uv && uv sync && sudo apt-get install -y graphviz && uv pip install -e .

# Run tests
MPLBACKEND=Agg uv run pytest

# Run tests with coverage
MPLBACKEND=Agg uv run pytest --cov=pytransform3d --cov-report html

# Critical lint check (CI requirement)
uv run flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

# Build docs
uv pip install -e '.[doc]' && cd doc && make html

# Install with all optional dependencies
uv pip install -e '.[all]'
```

## Trust These Instructions

These instructions have been validated by running the actual commands and observing their behavior. If you encounter issues not documented here, it may indicate:
1. A change in the codebase since these instructions were written
2. A platform-specific issue (these were validated on Linux)
3. An environmental difference (Python version, system packages)

In such cases, investigate the discrepancy and update these instructions if needed. Otherwise, trust and follow these instructions precisely for the most efficient workflow.
