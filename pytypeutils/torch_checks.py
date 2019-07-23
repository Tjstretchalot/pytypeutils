"""Contains type checking functions particular for torch tensors"""

import torch
from pytypeutils.utils import check_tensorlikes

def check_tensors(**kwargs):
    """Checks to verify the given arguments are torch tensors with the given
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
    import torch

    def feature_means(pts: torch.tensor):
        '''Returns the mean of the features for the given points,
        where pts is in the shape (num_samples, num_features)'''
        tus.check_tensors(
            pts=(pts, ('num_samples', 'num_features'), ('float32', 'float64')))
        res = pts.mean(0)
        tus.check_tensors(
            res=(res, [('num_features', pts.shape[1])], pts.dtype)
        )
        return res

    ```
    """
    for key in list(kwargs):
        val = kwargs[key]
        if len(val) > 2:
            if isinstance(val[2], str):
                val = list(val)
                val[2] = getattr(torch, val[2])
                kwargs[key] = tuple(val)
            elif isinstance(val[2], (list, tuple)):
                val = list(val)
                newtypes = []
                for typ in val[2]:
                    if isinstance(typ, str):
                        newtypes.append(getattr(torch, typ))
                    else:
                        newtypes.append(typ)
                val[2] = newtypes
                kwargs[key] = tuple(val)

    check_tensorlikes(torch.Tensor, **kwargs)
