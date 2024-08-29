import torch

import unittest
import sys
import os

tensoractpath = os.path.dirname(os.path.abspath(os.getcwd()))
sys.path.append(os.path.join(tensoractpath, "tensoract"))

from tensoract.core.mpo import *

class TestMPO(unittest.TestCase):

    def test_orthonormalize(self):
        W = self.randomMPO
        dtype = self.dtype
        device = self.device
        W.orthonormalize('left')
        for A in W:
            A = torch.swapaxes(A, 0, 1)
            s = A.size()
            A = torch.reshape(A, (s[0], s[1]*s[2]*s[3]))
            self.assertTrue(torch.allclose(A @ A.adjoint(), 
                                           torch.eye(s[0], device=device, dtype=dtype)))
        W.orthonormalize('right')
        for A in W:
            s = A.size()
            A = torch.reshape(A, (s[0], s[1]*s[2]*s[3]))
            self.assertTrue(torch.allclose(A @ A.adjoint(), 
                                           torch.eye(s[0], device=device, dtype=dtype)))
        idx = torch.randint(len(W), size=(1,))
        W.orthonormalize('mixed', idx)
        for i in range(idx):
            A = torch.swapaxes(W[i], 0, 1)
            s = A.size()
            A = torch.reshape(A, (s[0], s[1]*s[2]*s[3]))
            self.assertTrue(torch.allclose(A @ A.adjoint(), 
                                           torch.eye(s[0], device=device, dtype=dtype)))
        for i in range(idx+1,len(W)):
            A = W[i]
            s = A.size()
            A = torch.reshape(A, (s[0], s[1]*s[2]*s[3]))
            self.assertTrue(torch.allclose(A @ A.adjoint(), 
                                           torch.eye(s[0], device=device, dtype=dtype)))

    def test_to_matrix(self):
        W = self.randomMPO
        self.assertEqual(W.to_matrix().size(), (torch.prod(W.physical_dims),)*2)

    def test_hc(self):
        W = self.randomMPO
        self.assertTrue(torch.allclose(W.to_matrix().adjoint(), W.hc().to_matrix()))

    def setUp(self) -> None:

        N = torch.randint(low=3, high=7, size=(1,))
        phy_dims = torch.randint(2, 5, size=(N,))
        self.dtype = torch.cdouble
        self.device = 'cpu'
        self.randomMPO = MPO.gen_random_mpo(N, 9, phy_dims, dtype=self.dtype, device=self.device)
        self.randomMPO2 = MPO.gen_random_mpo(N, 9, phy_dims, dtype=self.dtype, device=self.device)

if __name__ == '__main__':
    unittest.main()