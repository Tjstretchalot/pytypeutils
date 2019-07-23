"""Checks related to numpy arrays"""
import numpy as np
from pytypeutils.utils import check_tensorlikes

def check_ndarrays(**kwargs):
    """Checks to verify the given arguments are numpy arrays with the given
    specifications. The keys are used for error messages and the values are
    tuples of the form (arr, expected shape, expected dtype). The expected
    shape may be None not to check shape information, or a tuple of dimension
    descriptions which can be None (for an any-size dimension), a str (for
    an any-size dimension with a name for error messages), an int (for a
    dimension of a particular size), or a tuple (str, int) where the str is
    the name of the dimension and the int is the size of the dimension. The
    dtype may be a tuple of numpy datatypes as strings or types, or None for
    any dtype.

    Example:

    ```
    import pytypeutils as tus
    import typing
    import numpy as np

    def feature_means(pts: np.ndarray):
        '''Returns the mean of the features for the given points,
        where pts is in the shape (num_samples, num_features)'''
        tus.check_ndarrays(
            pts=(pts, ('num_samples', 'num_features'), ('float32', 'float64')))
        res = pts.mean(0)
        tus.check_ndarrays(
            res=(res, [('num_features', pts.shape[1])], pts.dtype)
        )
        return res

    ```
    """
    check_tensorlikes(np.ndarray, **kwargs)
