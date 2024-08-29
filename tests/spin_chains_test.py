import numpy as np
import torch

import unittest
import sys
import os

tensoractpath = os.path.dirname(os.path.abspath(os.getcwd()))
sys.path.append(os.path.join(tensoractpath, "tensoract"))

from tensoract.models.spin_chains import *
from tensoract.core.lptn import *

class TestSpinChains(unittest.TestCase):

    def test_TransverseIsing(self):
        
        dtype = torch.double
        N = np.random.randint(5,10)
        g = 2 * np.random.random()
        model = TransverseIsing(N, g)
        self.assertTrue(np.allclose(model.mpo.to_matrix(), model.H_full.toarray()))

        all_up = LPTN.gen_polarized_spin_chain(N, polarization='+z', dtype=dtype)
        self.assertEqual(model.energy(all_up), -1*(N-1))

        all_right = LPTN.gen_polarized_spin_chain(N, polarization='+x', dtype=dtype)
        self.assertAlmostEqual(model.energy(all_right), -g*N)

    def test_Heisenberg(self):

        rng = np.random.default_rng()
        N = rng.integers(3,8)
        g = 2 * rng.uniform()
        J = 2 * rng.uniform(size=3)
        dtype = torch.double
        model = Heisenberg(N, J, g)
        self.assertTrue(np.allclose(model.mpo.to_matrix(), model.H_full.toarray()))

if __name__ == '__main__':
    unittest.main()