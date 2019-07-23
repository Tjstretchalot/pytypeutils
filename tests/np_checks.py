"""Tests for things within the np_checks folder"""
import unittest
import pytypeutils as tus

import numpy as np

def _like(shape, dtype):
    return np.random.uniform(0, 1, shape).astype(dtype)

class TestCheckNDArraysMethod(unittest.TestCase):
    def test_checkndarrays_noargs_pass(self):
        tus.check_ndarrays()

    def test_checkndarrays_emptytensor_pass(self):
        tus.check_ndarrays(a=(np.ndarray(0), None, None))

    def test_checkndarrays_withstrtype_pass(self):
        tus.check_ndarrays(
            a=(np.ndarray(0, dtype='float64'), None, 'float64'))

    def test_checkndarrays_withstrtype_fail(self):
        with self.assertRaises(ValueError):
            tus.check_ndarrays(
                a=(np.ndarray(0, dtype='float64'), None, 'float32'))

    def test_checkndarrays_withnptype_pass(self):
        tus.check_ndarrays(
            a=(np.ndarray(0, dtype='float64'), None, np.float64))

    def test_checkndarrays_withnptype_fail(self):
        with self.assertRaises(ValueError):
            tus.check_ndarrays(
                a=(np.ndarray(0, dtype='float64'), None, np.float32))

    def test_checkndarrays_numdims_pass(self):
        tus.check_ndarrays(
            a=(_like((10, 4), 'float32'), (None, None), None)
        )

    def test_checkndarrays_numdims_fail(self):
        with self.assertRaises(ValueError):
            tus.check_ndarrays(
                a=(_like((10, 4, 2), 'float32'), (None, None), None)
            )


    def test_checkndarrays_namednumdims_pass(self):
        tus.check_ndarrays(
            a=(_like((10, 4), 'float32'), ('dim1', 'dim2'), None)
        )

    def test_checkndarrays_namednumdims_fail(self):
        with self.assertRaises(ValueError):
            tus.check_ndarrays(
                a=(_like((10, 4, 2), 'float32'), ('dim1', 'dim2'), None)
            )

    def test_checkndarrays_exactdims_pass(self):
        tus.check_ndarrays(
            a=(_like((10, 4), 'float32'), (10, 4), None)
        )

    def test_checkndarrays_exactdims_fail(self):
        with self.assertRaises(ValueError):
            tus.check_ndarrays(
                a=(_like((10, 5), 'float32'), (10, 4), None)
            )

    def test_checkndarrays_exactnameddims_pass(self):
        tus.check_ndarrays(
            a=(_like((10, 4), 'float32'), (('dim1', 10), 4), None)
        )

    def test_checkndarrays_exactnameddims_fail(self):
        with self.assertRaises(ValueError):
            tus.check_ndarrays(
                a=(_like((10, 4), 'float32'), (('dim1', 11), 4), None)
            )
