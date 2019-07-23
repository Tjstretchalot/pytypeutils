"""Utility functions which are meant for ease of the implementation of
pytypeutils moreso than external usage."""

from pytypeutils import check

def make_shape_str(expected_shape) -> str:
    """Converts the given description of a shape (where each element
    corresponds to a description of a single dimension) into its
    human-readable counterpart.

    Args:
        expected_shape (tuple[any]): the expected shape of the thing

    Returns:
        str: a pretty description of the expected shape
    """
    result = []
    for item in expected_shape:
        if item is None:
            result.append('any')
        elif isinstance(item, int):
            result.append(str(item))
        elif isinstance(item, str):
            result.append(f'{item}=any')
        else:
            name, amt = item
            if amt is None:
                result.append(f'{name}=any')
            else:
                result.append(f'{name}={amt}')
    return '(' + ', '.join(result) + ')'

def check_shape(actual, expected) -> bool:
    """Verifies that the actual shape tuple matches the expected shape
    tuple. Every element in actual should be an int, and every element in
    expected should be a description of a dimension (str, int, or (str, int)).

    Args:
        actual (tuple[int]): actual shape
        expected (tuple[union[int, str, tuple[int, str]]]): expected shape

    Returns:
        bool: True if the two shapes match, false otherwise
    """
    if len(actual) != len(expected):
        return False
    for ind, act in enumerate(actual):
        exp = expected[ind]
        if isinstance(exp, int):
            if act != exp:
                return False
        elif isinstance(exp, (tuple, list)):
            if isinstance(exp[1], int) and act != exp[1]:
                return False
    return True

def check_tensorlikes(typ, **kwargs):
    """Verifies that each of the given tensor-like objects matches the
    corresponding type, shape, and dtype.
    """
    for key, val in kwargs.items():
        check(**{key: (val[0], typ)})
        if (len(val) > 1 and val[1] is not None and
                not check_shape(val[0].shape, val[1])):
            raise ValueError(
                f'expected {key}.shape is {make_shape_str(val[1])} but has '
                + f'shape {str(tuple(val[0].shape))}')
        if (len(val) > 2 and val[2] is not None):
            if isinstance(val[2], (list, tuple)):
                found = val[0].dtype in val[2]
                _in = ' in '
            else:
                found = val[0].dtype == val[2]
                _in = ' '
            if not found:
                raise ValueError(f'expected {key}.dtype is{_in}{val[2]} but '
                                 + f'got {val[0].dtype}')