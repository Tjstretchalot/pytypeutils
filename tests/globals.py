"""Test for the global callables"""
import unittest

import pytypeutils as tus

class TestCheckMethod(unittest.TestCase):
    """Tests for the check() method"""

    def test_check_noargs_pass(self):
        tus.check()

    def test_check_intisint_pass(self):
        tus.check(a=(5, int))

    def test_check_intisfloat_fail(self):
        with self.assertRaises(ValueError):
            tus.check(a=(5, float))

    def test_check_intisintorfloat_pass(self):
        tus.check(a=(5, (int, float)))

    def test_check_npintisint_fail(self):
        try:
            import numpy as np
        except ImportError:
            return

        with self.assertRaises(ValueError):
            tus.check(a=(np.int32(5), int))

    def test_check_npfloatisfloat_fail(self):
        try:
            import numpy as np
        except ImportError:
            return

        with self.assertRaises(ValueError):
            tus.check(a=(np.float32(5), float))

    def test_check_torchintisint_fail(self):
        try:
            import torch
        except ImportError:
            return

        with self.assertRaises(ValueError):
            tus.check(a=(torch.tensor([5], dtype=torch.int32), float))

    def test_check_twoargsbothgood_pass(self):
        tus.check(a=('a', str), b=(2, int))

    def test_check_twoargsonegood_fail(self):
        with self.assertRaises(ValueError):
            tus.check(a=('a', str), b=(2, str))

    def test_check_twoargsbothbad_fail(self):
        with self.assertRaises(ValueError):
            tus.check(a=('a', int), b=(2, str))

class TestCheckListlikeMethod(unittest.TestCase):
    """Tests for check_listlike"""

    def test_checklistlike_noargs_pass(self):
        tus.check_listlike()

    def test_checklistlike_emptylist_pass(self):
        tus.check_listlike(a=([], str))

    def test_checklistlike_singleint_pass(self):
        tus.check_listlike(a=([1], int))

    def test_checklistlike_singleint_fail(self):
        with self.assertRaises(ValueError):
            tus.check_listlike(a=([1], str))

    def test_checklistlike_manystr_pass(self):
        tus.check_listlike(a=(['a', 'b'], str))

    def test_checklistlike_mixed_fail(self):
        with self.assertRaises(ValueError):
            tus.check_listlike(a=(['a', 5], str))

    def test_checklistlike_multiplelist_pass(self):
        tus.check_listlike(
            a=([3.5], float),
            b=(['john', 'doe'], str)
        )

    def test_checklistlike_multiplelist_fail(self):
        with self.assertRaises(ValueError):
            tus.check_listlike(
                a=([3.5], float),
                b=(['john'], float)
            )

    def test_checklistlike_exactlen_pass(self):
        tus.check_listlike(a=([3.5], float, 1))

    def test_checklistlike_exactlen_fail(self):
        with self.assertRaises(ValueError):
            tus.check_listlike(a=([3.5], float, 2))

    def test_checklistlike_minlen_pass(self):
        tus.check_listlike(a=([3.5, 2.5, 1.0], float, (2, None)))

    def test_checklistliek_minlen_fail(self):
        with self.assertRaises(ValueError):
            tus.check_listlike(a=([3.5, 2.5, 1.0], float, (4, None)))

    def test_checklistlike_maxlen_pass(self):
        tus.check_listlike(a=([3.5, 2.5, 1.0], float, (None, 5)))

    def test_checklistliek_maxlen_fail(self):
        with self.assertRaises(ValueError):
            tus.check_listlike(a=([3.5, 2.5, 1.0], float, (0, 2)))

class TestCheckCallableMethod(unittest.TestCase):
    def test_checkcallable_noarg_pass(self):
        tus.check_callable()

    def test_checkcallable_class_pass(self):
        tus.check_callable(a=TestCheckCallableMethod)

    def tests_checkcallable_int_fail(self):
        with self.assertRaises(ValueError):
            tus.check_callable(a=5)

    def test_checkcallable_function_pass(self):
        def foo():
            pass
        tus.check_callable(a=foo)

    def test_checkcallable_lambda_pass(self):
        tus.check_callable(a=lambda x: True)
