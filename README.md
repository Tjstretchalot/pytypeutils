# PyTypeUtils

A runtime type-checking library which is not based on annotations and has
support for numpy and scipy.

## Features

This has documentation, unit tests for all API functions, and readable error
messages. This detects numpy and torch and, if they exist, provides
additional functions for each.

## Installation

`pip install pytypeutils`

## Usage

As a precondition test:

```py
import pytypeutils as tus
import typing

def foo(a: int, b: typing.Union[int, float]):
    tus.check(a=(a, int), b=(b, (int, float)))
    return 7

foo(3, 3) # no error!
foo(3, 3.0) # no error!
foo(3.0, 3) # ValueError: expected a is <class 'int'> but got 3.0 (type(a)=<class 'float'>)
```

Inside unit tests:

```py
import unittest
import pytypeutils as tus

def foo(a: int):
    return 2 * a

class MyTester(unittest.TestCase):
    def test_foo_giveint_returnsint(self):
        res = foo(2)
        tus.check(res=(res, int))
```

This detects if numpy is installed and adds `pytypeutils.check_ndarrays` if
it is:

```py
import pytypeutils as tus
import typing
import numpy as np

def feature_means(pts: np.ndarray):
    """Returns the mean of the features for the given points,
    where pts is in the shape (num_samples, num_features)"""
    tus.check_ndarrays(
        pts=(pts, ('num_samples', 'num_features'), ('float32', 'float64')))
    res = pts.mean(0)
    tus.check_ndarrays(
        res=(res, [('num_features', pts.shape[1])], pts.dtype)
    )
    return res

feature_means(np.random.uniform(0, 1, (10, 4))) # works


feature_means(np.random.uniform(0, 1, (10, 4, 2)))
# ValueError: expected pts.shape is (num_samples=any, num_features=any)
# but has shape (10, 4, 2)
```

This detects if torch is installed and adds `pytypeutils.check_tensors` if
it is:

```py
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

feature_means(torch.randn(10, 4)) # works

feature_means(torch.randn(10, 4, 2))
# ValueError: expected pts.shape is (num_samples=any, num_features=any)
# but has shape (10, 4, 2)
```

## Why this type library?

Annotation type libraries, such as [pytypes](https://pypi.org/project/pytypes/) and
[typeguard](https://github.com/agronholm/typeguard) can be great if it's reasonable
to express things in terms of typing hints. However, this can often be tedious and
it doesn't translate well to unit tests, quick snippets, or for a quick debug
command in the middle of a function call. This library lets you get reasonable type
errors which you can write and optionally remove quickly, without having to do
type checking on an entire functions parameters. Furthermore, the built-in extensions
for numpy and pytorch reduce a tremendous amount of boilerplate if you are using
those libraries and want some sanity checks.

## API

```Text
>>> help(pytypeutils)
Help on package pytypeutils:

NAME
    pytypeutils - Various generic functions and imports for pytypeutils.

PACKAGE CONTENTS
    np_checks
    torch_checks
    utils

FUNCTIONS
    check(**kwargs)
        Each keyword argument corresponds to an argument whose type will be
        checked. The key is used for printing out error messages and the value
        is a tuple describing the object to be checked and how it is to be checked.

        Example:

        ```
        import pytypeutils as tus
        import typing

        def foo(a: int, b: typing.Union[int, float]):
            tus.check(a=(a, int), b=(b, (int, float)))
        ```

    check_callable(**kwargs)
        Verifies that every value is callable. If not, raises an error

    check_listlike(**kwargs)
        Verifies that the given list-like objects have contents of a particular
        type. The keys are used for error messages, the values should be a tuple
        of the form (list, content types, optional length or (minl, maxl)).

        If there are more than 100 elements in any of the lists, 100 elements are
        sampled at random.

        Example:

        import pytypeutils as tus
        import typing

        def foo(a: typing.List[str], b: typing.List[typing.Union[int, float]]):
            tus.check(a=(a, (tuple, list)), b=(b, (tuple, list)))
            tus.check_listlike(a=(a, str), b=(b, (int, float), (0, 5)))
            # all elements of a are str
            # b has 0-5 elements inclusive, each of which is an int or float
```

```
>>> help(pytypeutils.check_ndarrays)
Help on function check_ndarrays in module pytypeutils.np_checks:

check_ndarrays(**kwargs)
    Checks to verify the given arguments are numpy arrays with the given
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

```
>>> help(pytypeutils.check_tensors)
Help on function check_tensors in module pytypeutils.torch_checks:

check_tensors(**kwargs)
    Checks to verify the given arguments are torch tensors with the given
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