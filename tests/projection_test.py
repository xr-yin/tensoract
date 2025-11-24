import torch

import unittest

from tensoract.core.projection import *
from tensoract.core import MPO, LPTN

class TestRightBondTensors(unittest.TestCase):
    
    def test_load(self):
        N = torch.randint(3, 7, size=(1,))
        phy_dims = torch.randint(2, 5, size=(N,))
        O = MPO.gen_random_mpo(N, 12, phy_dims)
        O.orthonormalize('mixed', N//2)

        psi = LPTN.gen_random_state(N, 12, 12, phy_dims)
        psi.orthonormalize('mixed', N//2+1)

        phi = psi.copy()

        for i in range(N):
            phi[i] += torch.rand(1, device=phi.device, dtype=phi.dtype)
        phi.orthonormalize('mixed', N//2-1)

        print(O[0].dtype, psi[0].dtype, phi[0].dtype)

        Rs = RightBondTensors(N)
        Rs.load(phi, psi, O)
        self.assertEqual([_.shape for _ in Rs],
                         [(i,j,k) for i,j,k in zip(phi.bond_dims[1:], O.bond_dims[1:], psi.bond_dims[1:])])



if __name__ == '__main__':
    unittest.main()