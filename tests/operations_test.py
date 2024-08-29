import torch
from torch import randint, normal, cdouble

import unittest
import sys
import os
import logging
logging.basicConfig(level=logging.INFO)

tensoractpath = os.path.dirname(os.path.abspath(os.getcwd()))
sys.path.append(os.path.join(tensoractpath, "tensoract"))

from tensoract.core.operations import *
from tensoract.core.mpo import MPO
from tensoract.core.lptn import LPTN

class TestMergeSplit(unittest.TestCase):
    
    def test_merge_mpo(self):

        a_size = randint(1,8,size=(4,))
        b_size = randint(1,8,size=(4,))
        b_size[0] = a_size[1]
        dtype = cdouble
        a = normal(0, 1, size=a_size.tolist(), dtype=dtype)
        b = normal(0, 1, size=b_size.tolist(), dtype=dtype)
        c = merge(a,b)
        self.assertTrue(c.dtype==dtype)
        self.assertEqual(c.size(), 
                    torch.Size([a_size[0],b_size[1],a_size[2],b_size[2],a_size[3],b_size[3]]))
    
    def test_split_mpo(self):
        
        a_size = randint(1,8,size=(6,))
        for dtype in [torch.float, torch.double, torch.cfloat, torch.cdouble]:
            a = normal(0, 1, size=a_size.tolist(), dtype=dtype)
            for mode in ['left', 'right', 'sqrt']:
                b, c = split(a, mode, 0.)
                s = (b.shape[0], c.shape[1], 
                    b.shape[2], c.shape[2], 
                    b.shape[3], c.shape[3])
                self.assertTrue(b.dtype == c.dtype == dtype)
                self.assertEqual(s, tuple(a_size.tolist()))
                self.assertEqual(b.size()[1], c.size()[0])
    
    def test_splitandmerge_mpo(self):

        a_size = randint(1,8,size=(6,))
        a = normal(0, 1, size=a_size.tolist(), dtype=cdouble)
        for mode in ['left', 'right', 'sqrt']:
            b, c = split(a, mode, 0., renormalize=False)
            new_a = merge(b, c)
            self.assertTrue(torch.allclose(new_a, a))

class TestMultiplication(unittest.TestCase):

    def test_mul(self):

        A = self.O
        N = len(A)
        m_max = max(A.bond_dims)
        phy_dims = A.physical_dims

        # MPO x MPO
        ampo = MPO.gen_random_mpo(N, m_max, phy_dims, dtype=A.dtype, device=A.device)
        C = mul(A, ampo)
        self.assertTrue(torch.allclose(C.to_matrix(), A.to_matrix() @ ampo.to_matrix()))

    def test_inner(self):
        W = self.phi
        P = W.copy()
        for i in range(len(P)):
            P[i] += torch.rand(1, device=P.device, dtype=P.dtype)
        self.assertTrue(torch.allclose(inner(W, P), torch.trace(W.to_matrix().adjoint() @ P.to_matrix())))

    def test_apply_mpo_overwrite(self):

        O, phi = self.O, self.phi

        multibonds = phi.bond_dims * O.bond_dims
        logging.info(f'multiplied bond dimensions: {multibonds}')

        # regular matrix products for comparison
        ref = O.to_matrix() @ phi.to_matrix() # |v> = O|psi>

        norm = apply_mpo(O, phi, tol=1e-7, m_max=None, max_sweeps=3)
        logging.info(f'optimized bond dimensions: {phi.bond_dims}')

        self.assertAlmostEqual(norm, torch.linalg.norm(ref).item())    # <v|v> =? <v|v>**0.5
        
        r = ref / torch.linalg.norm(ref)
        self.assertTrue(torch.allclose(r, phi.to_matrix()))
        self.assertTrue(torch.allclose(r@r.adjoint(), phi.to_density_matrix()))
    
    def test_apply_mpo(self):

        O, phi = self.O, self.phi

        multibonds = phi.bond_dims * O.bond_dims
        logging.info(f'multiplied bond dimensions: {multibonds}')

        # regular matrix products for comparison
        ref = O.to_matrix() @ phi.to_matrix() # |v> = O|psi>

        phi, norm = apply_mpo(O, phi, tol=1e-7, m_max=None, max_sweeps=3, overwrite=False)
        logging.info(f'optimized bond dimensions: {phi.bond_dims}')

        phi[0] = norm * phi[0]
        self.assertAlmostEqual(norm, torch.linalg.norm(ref).item())    # <v|v> =? <v|v>**0.5
        
        self.assertTrue(torch.allclose(ref, phi.to_matrix()))
        self.assertTrue(torch.allclose(ref@ref.adjoint(), phi.to_density_matrix()))

    def setUp(self) -> None:

        device = 'cpu'
        dtype = torch.cdouble
        
        N = randint(3,7,size=(1,))
        phy_dims = randint(2, 5, size=(N,))

        self.O = MPO.gen_random_mpo(N, 9, phy_dims, dtype=dtype, device=device)
        self.O.orthonormalize('right')

        self.phi = LPTN.gen_random_state(N, 9, 6, phy_dims, dtype=dtype, device=device)
        self.phi.orthonormalize('right')

if __name__ == '__main__':
    unittest.main()