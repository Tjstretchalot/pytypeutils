"""Verifies that check_tensors works correctly. This is small because
np_checks tests virtually the same implementation"""
import unittest
import torch
import pytypeutils as tus

class TestCheckTensorsMethod(unittest.TestCase):
    def test_checkndarrays_noargs_pass(self):
        tus.check_tensors()

    def test_checkndarrays_emptytensor_pass(self):
        tus.check_tensors(a=(torch.rand(2, 4), None, None))

    def test_checkndarrays_complete_pass(self):
        tus.check_tensors(
            a=(torch.rand(2, 4), (('dim1', 2), ('dim2', 4)), torch.float)
        )

    def test_checkndarrays_strdtype_pass(self):
        tus.check_tensors(
            a=(torch.rand(2, 4), (('dim1', 2), ('dim2', 4)), 'float32')
        )

    def test_checkndarrays_baddim_fail(self):
        with self.assertRaises(ValueError):
            tus.check_tensors(
                a=(torch.rand(3, 4), (('dim1', 2), ('dim2', 4)), torch.float)
            )

    def test_checkndarrays_baddimstr_fail(self):
        with self.assertRaises(ValueError):
            tus.check_tensors(
                a=(torch.rand(3, 4), (('dim1', 2), ('dim2', 4)), 'float32')
            )

    def test_checkndarrays_baddtype_fail(self):
        with self.assertRaises(ValueError):
            tus.check_tensors(
                a=(torch.rand(2, 4), (('dim1', 2), ('dim2', 4)), torch.int32)
            )

    def test_checkndarrays_multdtype_pass(self):
        tus.check_tensors(
            a=(torch.rand(2, 4), (('dim1', 2), ('dim2', 4)), ('int32', 'float32'))
        )
