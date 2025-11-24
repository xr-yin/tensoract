import torch

import unittest

from tensoract.core.mps import MPS

class TestMPS(unittest.TestCase):

    def test_to_array(self):
        mps = self.randomMPS
        arr = mps.to_array()
        N = len(mps)
        phy_dims = mps.physical_dims
        self.assertEqual(arr.dim(), 1)
        self.assertEqual(arr.size().numel(), torch.prod(phy_dims))

    def setUp(self) -> None:
        N = torch.randint(3,7,size=(1,))
        phy_dims = torch.randint(2, 5, size=(N,))
        self.randomMPS = MPS.gen_random_mps(N, 9, phy_dims, dtype=torch.complex128, device='cpu')
        
if __name__ == '__main__':
    unittest.main()