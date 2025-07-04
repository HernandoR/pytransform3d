import array_api_compat as xarray
import numpy as np
from functools import wraps


def _np_arraylize_if_list(*args):
    return [np.asarray(arg) if isinstance(arg, list) else arg for arg in args]


def get_array_namespace(*args, **kwargs):
    """
    Get the array namespace of the position arguments.

    for list instance, return numpy.
    """
    args = _np_arraylize_if_list(*args)
    return xarray.array_namespace(*args, **kwargs)


# decorator to transfer list args to numpy
def np_arraylize_if_list(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        args = _np_arraylize_if_list(*args)
        return func(*args, **kwargs)

    return wrapper
