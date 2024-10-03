import torch

import unittest
import sys
import os

tensoractpath = os.path.dirname(os.path.abspath(os.getcwd()))
sys.path.append(os.path.join(tensoractpath, "tensoract"))

from tensoract.models.boson_chains import *

class TestBosonChains(unittest.TestCase):

    def test_BoseHubburd(self):

        with torch.no_grad():
            N = torch.randint(3, 7, (1,))
            d = torch.randint(2, 5, (1,))
            t, U, mu = torch.rand(size=(3,))
            model = BoseHubburd(N.item(), d.item(), t, U, mu, dtype=torch.double)
            # numpy uses double precesion
            self.assertTrue(torch.allclose(model.mpo.to_matrix(), 
                                        torch.from_numpy(model.H_full().toarray())))

    def test_DDBH(self):

        with torch.no_grad():            
            N = torch.randint(3, 7, (1,))
            d = torch.randint(2, 5, (1,))
            t, U, mu, F, gamma = torch.rand(size=(5,))
            model = DDBH(N.item(), d.item(), t, U, mu, F.item(), gamma)
            # numpy uses double precesion
            self.assertTrue(torch.allclose(model.mpo.to_matrix(), 
                                           torch.from_numpy(model.H_full().toarray())))
            F = torch.rand(size=(N,)).numpy()
            model = DDBH(N.item(), d.item(), t, U, mu, F, gamma)
            # numpy uses double precesion
            self.assertTrue(torch.allclose(model.mpo.to_matrix(), 
                                           torch.from_numpy(model.H_full().toarray())))


if __name__ == '__main__':
    unittest.main()