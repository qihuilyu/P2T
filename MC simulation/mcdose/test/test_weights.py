import unittest
import os
from os.path import join as pjoin
from collections import namedtuple

import numpy as np

from setup_tests import test_data
from mcdose import weights

DataPair = namedtuple('DataPair', ('inputs', 'labels'))

class TestWeights(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        inputs= np.arange(1, 11)[:, None, None, None, None] * np.ones((3, 5, 5, 1), dtype=np.float32)
        cls.data = DataPair(
            inputs=inputs,
            labels=inputs
        )

    def test_sample_lin_norm_sum(self):
        w = weights.sample_lin_norm_sum()(*self.data)
        gt = np.arange(1,11, dtype=float)/10.0
        self.assertTrue(np.allclose(w, gt), 'Arrays do not agree')

    def test_sample_exp_norm_sum(self):
        decay_rate = 5.0
        w = weights.sample_exp_norm_sum(decay_rate)(*self.data)
        gt = np.exp(decay_rate * (np.arange(1,11, dtype=float)/10.0 - 1))
        self.assertTrue(np.allclose(w, gt), 'Arrays do not agree')


if __name__ == '__main__':
    unittest.main()
