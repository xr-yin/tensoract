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
            t, U, mu, F = torch.rand(size=(4,))
            model = BoseHubburd(N.item(), d.item(), t.item(), U.item(), mu.item(), F=F.item())
            # numpy uses double precesion
            self.assertTrue(torch.allclose(model.mpo.to_matrix(), 
                                        torch.from_numpy(model.H_full().toarray())))
            
            # inhomogeneous case
            t = torch.rand(size=(N-1,))
            U = torch.rand(size=(N,))
            mu = torch.rand(size=(N,))
            mu = torch.rand(size=(N,))
            F = torch.rand(size=(N,))
            model = BoseHubburd(N.item(), d.item(), t, U, mu, F=F)
            # numpy uses double precesion
            self.assertTrue(torch.allclose(model.mpo.to_matrix(), 
                                        torch.from_numpy(model.H_full().toarray())))

    def test_DDBH(self):
        from tensoract.models.boson_chains import DDBH

        with torch.no_grad():            
            N = torch.randint(3, 7, (1,))
            d = torch.randint(2, 5, (1,))
            t, U, mu, F, gamma = torch.rand(size=(5,))
            model = DDBH(N.item(), d.item(), t.item(), U.item(), mu.item(), F.item(), gamma.item())
            # numpy uses double precesion
            self.assertTrue(torch.allclose(model.mpo.to_matrix(), 
                                           torch.from_numpy(model.H_full().toarray())))

            # compare with BoseHubburd
            F = 1.
            model1 = DDBH(N.item(), d.item(), t.item(), U.item(), mu.item(), F, gamma.item())
            model2 = BoseHubburd(N.item(), d.item(), t.item(), U.item(), mu.item(), F=F)
            self.assertTrue(torch.allclose(model1.mpo.to_matrix(), model2.mpo.to_matrix()))

            # inhomogeneous case
            t = torch.rand(size=(N-1,))
            U = torch.rand(size=(N,))
            mu = torch.rand(size=(N,))
            F = torch.rand(size=(N,))
            gamma = torch.rand(size=(N,))
            model = DDBH(N.item(), d.item(), t, U, mu, F, gamma)
            self.assertTrue(torch.allclose(model.mpo.to_matrix(), 
                                           torch.from_numpy(model.H_full().toarray())))


if __name__ == '__main__':
    unittest.main()