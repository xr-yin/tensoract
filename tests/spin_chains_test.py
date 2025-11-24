import numpy as np
import torch

import unittest

from tensoract.models.spin_chains import *
from tensoract.core.lptn import *

class TestSpinChains(unittest.TestCase):

    def test_TransverseIsing(self):
        
        dtype = torch.complex128
        N = np.random.randint(5,10)
        g = 2 * np.random.random()
        model = TransverseIsing(N, g)
        self.assertTrue(np.allclose(model.mpo.to_matrix(), model.H_full().toarray()))

        all_up = LPTN.gen_polarized_spin_chain(N, polarization='+z', dtype=dtype)
        self.assertEqual(model.energy(all_up).item(), -1*(N-1))

        all_right = LPTN.gen_polarized_spin_chain(N, polarization='+x', dtype=dtype)
        # TODO: check the error in the energy calculation
        self.assertAlmostEqual(model.energy(all_right).item(), -g*N, places=6)

        # inhomogeneous case
        rng = np.random.default_rng()
        g = 2 * rng.uniform(size=N)
        J = 2 * rng.uniform(size=N-1)
        model = TransverseIsing(N, g, J)
        self.assertTrue(np.allclose(model.mpo.to_matrix(), model.H_full().toarray()))

    def test_Heisenberg(self):

        rng = np.random.default_rng()
        N = rng.integers(3,8)
        g = 2 * rng.uniform()
        Jx, Jy, Jz = 2 * rng.uniform(size=3)
        model = Heisenberg(N, Jx, Jy, Jz, g)
        self.assertTrue(np.allclose(model.mpo.to_matrix(), model.H_full().toarray()))
        
        # inhomogeneous case
        g = 2 * rng.uniform(size=N)
        Jx = 2 * rng.uniform(size=N-1)
        Jy = 2 * rng.uniform(size=N-1)
        Jz = 2 * rng.uniform(size=N-1)
        model = Heisenberg(N, Jx, Jy, Jz, g)
        self.assertTrue(np.allclose(model.mpo.to_matrix(), model.H_full().toarray()))

if __name__ == '__main__':
    unittest.main()