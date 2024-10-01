import torch

import unittest
import logging
import sys
import os
from copy import deepcopy

tensoractpath = os.path.dirname(os.path.abspath(os.getcwd()))
sys.path.append(os.path.join(tensoractpath, "tensoract"))

from tensoract.core.lptn import *
from tensoract.core.lptn import _load_right_bond_tensors

class TestLPTN(unittest.TestCase):

    def test_to_density_matrix(self):
        
        dm = self.psi.to_density_matrix()
        self.assertEqual(dm.shape, (torch.prod(self.phy_dims),)*2)
        self.assertTrue(torch.allclose(dm, dm.T.conj()))

    def test_orthonormalize(self):

        # $\rho = X X^\dagger$
        # when $X$ is orthonormalized, the density matrix has trace 1.
        self.psi.orthonormalize('right')
        self.assertAlmostEqual(torch.trace(self.psi.to_density_matrix()).item(), 1, 12)

    def test_purity(self):
        psi = self.psi
        psi.orthonormalize('right')
        self.assertTrue(torch.isclose(torch.linalg.norm(psi.to_density_matrix())**2, 
                                      psi.purity()))

    def setUp(self) -> None:

        device = 'cpu'
        dtype = torch.complex128
        N = torch.randint(3,8,size=(1,))
        self.phy_dims = torch.randint(2, 5, size=(N,))
        self.psi = LPTN.gen_random_state(N, 9, 9, self.phy_dims, dtype=dtype, device=device)

class TestMeasurements(unittest.TestCase):

    def test_site(self):

        device, dtype = self.device, self.dtype

        N = torch.randint(3,8,size=(1,))
        sz = torch.tensor([[1., 0.], [0., -1.]], device=device, dtype=dtype)
        
        for polar, val in [('+z', 1.), ('-z', -1.), ('+x', 0.)]:
            psi = LPTN.gen_polarized_spin_chain(N, 
                                                polarization=polar, 
                                                device=device, 
                                                dtype=dtype)
            
            self.assertTrue(torch.allclose(psi.site_expectation_value([sz]*N), 
                                           torch.full((N,), val, dtype=self.dtype, device=device)))
            self.assertTrue(torch.allclose(psi.site_expectation_value(sz, idx=N//2), 
                                           torch.tensor(val, dtype=self.dtype, device=device)))

        self.phy_dims = torch.randint(2, 5, size=(N,))
        psi = LPTN.gen_random_state(N, 9, 9, self.phy_dims, dtype=dtype, device=device)

        # local case
        idx = torch.randint(N,size=(1,))
        d = psi.physical_dims[idx]
        a = torch.rand(size=(d,d), device=device, dtype=dtype)
        sz = a.adjoint() + a  # sz is now a random hermitian operator
        complex_expect = psi.site_expectation_value(sz, idx=idx, drop_imag=False)
        self.assertAlmostEqual(complex_expect.imag.item(), 0.)
        self.assertTrue(torch.allclose(psi.site_expectation_value(sz, idx=idx, drop_imag=True), 
                                      complex_expect.real))
        
        # global case
        op_list = []
        for d in psi.physical_dims:
            a = torch.rand(size=(d,d), device=device, dtype=dtype)
            op_list.append(a.adjoint() + a)
        complex_expect = psi.site_expectation_value(op_list, drop_imag=False)
        self.assertTrue(torch.allclose(complex_expect.imag,     # check all imaginary parts are zero
                                       torch.zeros(size=(N,), device=device, dtype=torch.double)))
        self.assertTrue(torch.allclose(psi.site_expectation_value(op_list, drop_imag=True), 
                                       complex_expect.real))    # check the real parts agree

    def test_bond(self):
        pass

    def test_correlation(self):
        pass

    def test_measure(self):
        
        N = torch.randint(3,8,size=(1,))
        d = torch.randint(2,5,size=(1,))
        dtype = self.dtype
        psi = LPTN.gen_random_state(N, 9, 9, d.tile((N,)), dtype=dtype)

        num = torch.randint(1,4,size=(1,)) # number of operators
        if torch.rand(1) > 0.55:
            op_list = LPTN.gen_random_mpo(num, 2, d.tile((num,)), dtype=dtype)
            drop_imag = False
        else:
            op_list = LPTN.gen_random_mpo(num, 2, d.tile((num,)), dtype=dtype, hermitian=True)
            drop_imag = True
            logging.info('Hermitian operator')
        op_list = [op.squeeze() for op in op_list]

        # test local case
        idx = torch.randint(N,size=(1,))   # the site to be measured
        exp = psi.measure(op_list, idx=idx, drop_imag=drop_imag)
        self.assertTrue(len(exp) == len(op_list) == num)
        for op, v in zip(op_list, exp):
            self.assertTrue(torch.allclose(psi.site_expectation_value(op, idx=idx, drop_imag=drop_imag), v))

        # test global case
        exp = psi.measure(op_list, drop_imag=drop_imag)
        for op, v in zip(op_list, exp):
            self.assertTrue(torch.allclose(psi.site_expectation_value([op]*N, drop_imag=drop_imag), v))
    
    def setUp(self) -> None:
        self.device = 'cpu'
        self.dtype = torch.complex128

class TestCompress():

    def test_load_right_bond_tensors(self):
        psi = self.psi
        phi = deepcopy(psi)
        phi.orthonormalize('right')
        RBT = _load_right_bond_tensors(psi, phi)
        self.assertEqual(len(RBT), len(psi))
        self.assertEqual([_.shape for _ in RBT], 
                [(i,j) for i,j in zip(psi.bond_dims[1:],phi.bond_dims[1:])])

    def test_compress(self):
        self.psi.orthonormalize('right')
        phi, overlap = compress(self.psi, 1e-7, max(self.psi.bond_dims)-1, max_sweeps=2)
        self.assertAlmostEqual(overlap, 1)
        self.assertAlmostEqual(torch.linalg.norm(self.psi.to_matrix()-phi.to_matrix()), 0.)

    def setUp(self) -> None:

        N = torch.randint(3,8,size=(1,))
        self.phy_dims = torch.randint(2, 5, size=(N,))
        self.psi = LPTN.gen_random_state(N, 9, 9, self.phy_dims)

if __name__ == '__main__':

    unittest.main()