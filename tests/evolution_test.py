import torch
from torch.linalg import matrix_exp
from scipy.sparse.linalg import expm_multiply

import unittest
import sys
import os
import logging
logging.basicConfig(level=logging.INFO)

tensoractpath = os.path.dirname(os.path.abspath(os.getcwd()))
sys.path.append(os.path.join(tensoractpath, "tensoract"))

from tensoract import MPO, LPTN, mul, Heisenberg, DDBH
from tensoract.models.spin_chains import dissipative_testmodel
from tensoract.solvers.evolution import LindbladOneSite, contract_dissipative_layer

class TestDetach(unittest.TestCase):

    def test_make_coherent_layers(self):

        device, dtype = self.device, self.dtype

        N = torch.randint(3,8,size=(1,)).item()   # number of sites

        logging.info('the difference, namely the Trotter error, should decrease as we decrease the time step')
        
        for dt in (0.5, 0.3, 0.1):

            logging.info('unitary MPO for Heisenberg model')

            _ = LPTN.gen_polarized_spin_chain(N, polarization='+z', dtype=dtype, device=device)
            model = Heisenberg(N, [1., 1., 1.5], g=1.)
            lab = LindbladOneSite(_, model)
            lab.make_coherent_layer(dt)

            u = mul(lab.uMPO[1], lab.uMPO[0])
            v = mul(lab.uMPO[0], lab.uMPO[1])
            
            logging.info(f"dt={dt}: [H_o, H_e]: {torch.dist(u.to_matrix(), v.to_matrix())}")
            logging.info(f"dt={dt}: H_trotter-H_exact: {torch.dist(u.to_matrix().cpu(), matrix_exp(torch.from_numpy(-1j*dt*model.H_full.toarray())))}")
            # check unitarity
            self.assertTrue(torch.allclose(u.to_matrix() @ u.to_matrix().adjoint(), torch.eye(2**N, dtype=dtype, device=device)))
        
        for dt in (0.2, 0.1, 0.05):

            logging.info('unitary MPO for Bose Hubbard model')

            N, d = 4, 4

            _ = LPTN.gen_random_state(N, 5, 5, phy_dims=[d]*N, dtype=dtype, device=device)
            model = DDBH(N, d, t=0.2, U=1., mu=0.8, F=0.3, gamma=0.1)
            lab = LindbladOneSite(_, model)
            lab.make_coherent_layer(dt, eps=0.)
            print(lab.uMPO[1].bond_dims)
            print(lab.uMPO[0].bond_dims)
            u = mul(lab.uMPO[1], lab.uMPO[0])
            v = mul(lab.uMPO[0], lab.uMPO[1])
            
            logging.info(f"dt={dt}: [H_o, H_e]: {torch.dist(u.to_matrix(), v.to_matrix())}")
            logging.info(f"dt={dt}: H_trotter-H_exact: {torch.dist(u.to_matrix().cpu(), matrix_exp(torch.from_numpy(-1j*dt*model.H_full().toarray())))}")
            # check unitarity
            self.assertTrue(torch.allclose(u.to_matrix() @ u.to_matrix().adjoint(), torch.eye(d**N, dtype=dtype, device=device)))

    def test_make_dissipative_layer(self):

        device, dtype = self.device, self.dtype

        N = torch.randint(3,7,size=(1,)).item()
        dt = torch.rand(size=(1,)).item()
        psi = LPTN.gen_random_state(N, m_max=6, k_max=6, phy_dims=[2]*N, device=device, dtype=dtype)
        model = dissipative_testmodel(N)
        lab = LindbladOneSite(psi, model)
        lab.make_dissipative_layer(dt)
        id2 = torch.eye(2, dtype=dtype, device=device)
        for i, l in enumerate(lab.B_list):
            if i in model.indices:
                self.assertEqual(l.shape[2], 1)
                self.assertTrue(torch.allclose(l[:,:,0], id2))
            else:
                trace_preserving = torch.tensordot(l, l.conj(), dims=([0,2], [0,2]))
                self.assertTrue(torch.allclose(trace_preserving, id2))
        
    def test_contract_dissipative_layer(self):
        
        device, dtype = self.device, self.dtype

        N = torch.randint(3,7,size=(1,)).item()
        phy_dims = torch.randint(2,5, size=(N,))
        krauss_dims = torch.randint(1, 6, size=(N,))
        
        # generate a random dissipative layer
        As = []
        for i in range(N):
            size = (phy_dims[i], phy_dims[i], krauss_dims[i])
            As.append(torch.normal(0,1,size=size,dtype=dtype,device=device))
        
        # generate a random LPTN
        psi = LPTN.gen_random_state(N, 6, 4, phy_dims, dtype=dtype, device=device)
        psi.orthonormalize('right')
        bds, kds = psi.bond_dims, psi.krauss_dims

        # reshape the MPKO into a MPO 
        Ws = [a.swapaxes(1,2) for a in As]
        Ws = [a.reshape(1, 1, -1, a.shape[-1]) for a in Ws]
        O = MPO(Ws)

        # perform a MPO-MPS multiplication
        ref = mul(O, psi)

        # reshape ref back to a LPTN
        for i, a in enumerate(ref):
            self.assertEqual((a.shape[0], a.shape[1], a.shape[3]), 
                             (bds[i], bds[i+1], kds[i]))
            ref[i] = torch.reshape(a, a.shape[:2] + (phy_dims[i],-1))
        ref = LPTN(ref.As)

        contract_dissipative_layer(As, psi, keys=[1]*N)

        for i, a in enumerate(psi):
            self.assertEqual(a.shape, (bds[i], bds[i+1], phy_dims[i], kds[i]*krauss_dims[i]))

        self.assertTrue(torch.allclose(ref.to_density_matrix(), psi.to_density_matrix()))
        #self.assertTrue(torch.allclose(ref.to_matrix(), psi.to_matrix()))

    def setUp(self) -> None:
        
        self.device = 'cpu'
        self.dtype = torch.cdouble

if __name__ == "__main__":
    unittest.main()