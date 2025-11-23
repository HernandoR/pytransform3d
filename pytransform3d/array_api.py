"""Array API compatibility utilities for pytransform3d.

This module provides utilities for working with different array backends
(NumPy, PyTorch, JAX, etc.) through the Array API standard.
"""
import array_api_compat as xarray
import numpy as np
import logging
from functools import wraps


logger = logging.getLogger(__name__)


def _np_arraylize_if_list(*args):
    """Convert list arguments to numpy arrays."""
    converted = []
    for arg in args:
        if isinstance(arg, list):
            converted.append(np.asarray(arg))
        else:
            converted.append(arg)
    return converted


def _np_arraylize_scalar_or_list(*args, warn=True):
    """Convert scalar or list arguments to numpy arrays with optional warning.
    
    Parameters
    ----------
    *args : various
        Arguments to convert
    warn : bool, optional (default: True)
        Whether to log a warning when converting scalars or lists
        
    Returns
    -------
    converted_args : list
        Arguments with scalars and lists converted to numpy arrays
    """
    converted = []
    for arg in args:
        if isinstance(arg, (int, float)):
            if warn:
                logger.warning(
                    "Converting scalar %s to numpy array. "
                    "Consider passing array directly.", type(arg).__name__
                )
            converted.append(np.asarray(arg))
        elif isinstance(arg, list):
            if warn:
                logger.warning(
                    "Converting list to numpy array. "
                    "Consider passing array directly."
                )
            converted.append(np.asarray(arg))
        else:
            converted.append(arg)
    return converted


def get_array_namespace(*args, **kwargs):
    """Get the array namespace of the position arguments.

    For list instances, converts to numpy and returns numpy namespace.
    For array API compatible arrays, returns the appropriate namespace.

    Parameters
    ----------
    *args : array-like
        Arrays to get the namespace from
    **kwargs : dict
        Additional keyword arguments passed to array_namespace

    Returns
    -------
    namespace : module
        Array API compatible namespace (numpy, torch, jax.numpy, etc.)
    """
    args = _np_arraylize_if_list(*args)
    return xarray.array_namespace(*args, **kwargs)


def check_array_type(arr, param_name="array"):
    """Check if input is a valid array type.
    
    Parameters
    ----------
    arr : array-like
        Input to check
    param_name : str, optional (default: "array")
        Name of the parameter for error messages
        
    Returns
    -------
    arr : array
        The input array (possibly converted from list with warning)
        
    Raises
    ------
    TypeError
        If input is not array-like
    """
    if isinstance(arr, (int, float)):
        logger.warning(
            "Parameter '%s' received scalar %s, converting to numpy array. "
            "Consider passing array directly.",
            param_name, type(arr).__name__
        )
        return np.asarray(arr)
    elif isinstance(arr, list):
        logger.warning(
            "Parameter '%s' received list, converting to numpy array. "
            "Consider passing array directly.",
            param_name
        )
        return np.asarray(arr)
    return arr


# decorator to transfer list args to numpy
def np_arraylize_if_list(func):
    """Decorator to convert list arguments to numpy arrays.
    
    This decorator converts any list arguments to numpy arrays before
    calling the decorated function.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        args = _np_arraylize_if_list(*args)
        return func(*args, **kwargs)

    return wrapper


def ensure_numpy_array(arr, param_name="array"):
    """Ensure input is a numpy array, raising ValueError if not.
    
    This is for visualization functions that require numpy arrays.
    
    Parameters
    ----------
    arr : array-like
        Input to check
    param_name : str, optional (default: "array")
        Name of the parameter for error messages
        
    Returns
    -------
    arr : numpy.ndarray
        The input array
        
    Raises
    ------
    ValueError
        If input is not a numpy array
    TypeError
        If input is not array-like
    """
    if not isinstance(arr, np.ndarray):
        raise ValueError(
            f"Parameter '{param_name}' must be a numpy array for visualization "
            f"functions. Got {type(arr).__name__} instead. "
            "Please convert your array to numpy before calling this function."
        )
    return arr
