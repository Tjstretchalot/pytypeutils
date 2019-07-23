"""Various generic functions and imports for pytypeutils."""
import random
import sys

def check(**kwargs):
    """Each keyword argument corresponds to an argument whose type will be
    checked. The key is used for printing out error messages and the value
    is a tuple describing the object to be checked and how it is to be checked.

    Example:

    ```
    import pytypeutils as tus
    import typing

    def foo(a: int, b: typing.Union[int, float]):
        tus.check(a=(a, int), b=(b, (int, float)))
    ```
    """
    for key, val in kwargs.items():
        if not isinstance(val, (list, tuple)) or len(val) != 2:
            raise ValueError(f'check(**{kwargs}) has {key}={val} but should be'
                             + ' key=(val, types)')
        if not isinstance(val[0], val[1]):
            _in = ' in ' if isinstance(val[1], (tuple, list)) else ' '
            raise ValueError(f'expected {key} is{_in}{val[1]} but got {val[0]}'
                             + f' (type({key})={type(val[0])})')

def check_listlike(**kwargs):
    """Verifies that the given list-like objects have contents of a particular
    type. The keys are used for error messages, the values should be a tuple
    of the form (list, content types, optional length or (minl, maxl)).

    If there are more than 100 elements in any of the lists, 100 elements are
    sampled at random.

    Example:
    ```
    import pytypeutils as tus
    import typing

    def foo(a: typing.List[str], b: typing.List[typing.Union[int, float]]):
        tus.check(a=(a, (tuple, list)), b=(b, (tuple, list)))
        tus.check_listlike(a=(a, str), b=(b, (int, float), (0, 5)))
        # all elements of a are str
        # b has 0-5 elements inclusive, each of which is an int or float
    ```
    """
    for key, val in kwargs.items():
        if not isinstance(val, (list, tuple)) or len(val) not in {2, 3}:
            raise ValueError(f'check_listlike(**{kwargs}) has {key}={val} but '
                             + 'should be key=(list, content types[, length])')

        arr = val[0]
        types = val[1]
        if len(val) > 2:
            len_info = val[2]
            if isinstance(len_info, int):
                len_info = (len_info, len_info)
        else:
            len_info = None

        if len(arr) < 100:
            inds = range(len(arr))
        else:
            inds = random.sample(range(len(arr)), 100)

        for i in inds:
            check(**{f'{key}[{i}]': (arr[i], types)})

        if len_info is not None:
            if (len_info[0] is not None and len(arr) < len_info[0] or
                    len_info[1] is not None and len(arr) > len_info[1]):
                _inf = 'inf'
                raise ValueError(f'expected len({key}) in interval '
                                 + f'[{len_info[0] or 0}, {len_info[1] or _inf}] but got '
                                 + f'{len(arr)}')

def check_callable(**kwargs):
    """Verifies that every value is callable. If not, raises an error"""
    for key, val in kwargs.items():
        if not callable(val):
            raise ValueError(f'expected {key} is callable, got {val} '
                             + f'(type={type(val)}')

have_numpy = False
try:
    import numpy
    have_numpy = True
except ImportError:
    pass

if have_numpy:
    from pytypeutils.np_checks import check_ndarrays

have_torch = False
try:
    import torch
    have_torch = True
except ImportError:
    pass

if have_torch:
    from pytypeutils.torch_checks import check_tensors
