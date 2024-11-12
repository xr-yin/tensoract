""" Algorithms for evolving a mixed state in locally purified tensor network (LPTN) 
representation according to the Lindblad master equation."""

import torch
from tqdm import trange
from torch.linalg import matrix_exp, eigh, svd, qr
from numpy.typing import ArrayLike

from typing import Any
import logging
import gc

from ..core import MPO, LPTN, split, mul, apply_mpo
from ..core.projection import *

__all__ = ['LindbladOneSite', 'mesolve']


def mesolve(
        model,
        psi0: LPTN,
        t0: float,
        tf: float,
        dt: float,
        m_max: int,
        k_max: int,
        *,
        e_ops: ArrayLike | list[ArrayLike] | None = None,
        options: dict[str, Any] | None = None
) -> None:  

    psi = psi0.copy()
    lab = LindbladOneSite(psi, model)
    
    if options is None:
        options = {}

    tol = options.get('tol', 1e-14)
    max_sweeps = options.get('max_sweeps', 1)
    store_final_state = options.get('store_final_state', True)

    # for now, only time-independent Hamiltonian is accepted
    # therefore only tf-t0 matters here
    N_tot = round((tf-t0)/(dt))

    expects = torch.empty(size=(len(e_ops), N_tot, len(psi)), dtype=torch.complex128)
    # track growth of the *entanglement and the purity
    purity = torch.empty(size=(N_tot,))
    S_ent = torch.empty(size=(N_tot,))  # TODO: another entropy measure is needed

    center = len(psi) // 2

    for i in trange(N_tot):

        lab.run(1, dt, tol, m_max, k_max, max_sweeps=max_sweeps)   
        # in the end, I want results ordered as the input e_ops
        # results[0] = [[], [], ... , []]
        # (len(e_ops), temporal, spatial)
        exp = psi.measure(e_ops)
        for j in range(len(e_ops)):
            expects[j,i,:] = exp[j]
        #purity[i] = psi.rho2trace()
        S_ent[i] = psi.entropy(idx=center)
        purity[i] = psi.rho2trace()

    results = {'expect': expects, 'purity': purity, 'S_ent': S_ent}

    if store_final_state:
        results.update({'final_state': psi})

    return results

class LindbladOneSite(object):
    r"""Handles the case when the Lindblad jump operators only act on single sites.

    Parameters
    ----------
    psi : LPTN
        the LPTN to be evolved, modification is done in place
    model : None
        1D model object, must have the following two properties: `hduo` and `Lloc`.

    Attributes
    ----------
    psi : LPTN
        see above
    hduo : list
        a list containing two-local Hamiltonian terms involving only nearest neighbors
    dims : list
        a list containing local Hilbert space dimensions of each site
    Lloc : list
        a list conataing local Lindblad jump operators
    dt : float
        time step
    overlaps : list
        a list containing the optimized overlaps between the updated LPTN and the exact LPTN
        at each time step

    Methods
    ----------
    run()
        primal method for simulating the Lindblad master equation with one-stie dissipators,
        using a variational approach to update the purification network at every time step.
    run_attach()
        a simpler simulating method, contracting the matrix product krauss operator and the 
        purification network at every time step. Warning: long runtime, still in test phase.
    make_coherent_layer()
        calculate the unitary time evolution operator at a MPO (MPU) from the local Hamiltonian
        terms
    make_dissipative_layer()
        calculate the Krauss representation of the local quantum channels
    TODO:
    time-dependent Hamiltonian
    """
    def __init__(self, psi: LPTN, model) -> None:
        self.psi = psi
        self.dims = psi.physical_dims
        self.hduo = model.hduo   # list of two-local terms
        self.Lloc = model.Lloc # list containing all (local) jump operators
        self.model_params = model.parameters()
        self.device = psi[0].device # device follow from the state psi
        self.dtype = psi[0].dtype # dtype follow from the state psi
        self.simulation_params = {}
        self.dt = None
        self._bot()

    def run(self, 
            Nsteps:int, 
            dt: float, 
            tol: float, 
            m_max: int, 
            k_max: int, 
            *,
            max_sweeps: int=1) -> None:
        
        Nbonds = len(self.psi)-1
        assert Nbonds == len(self.hduo)
        if dt != self.dt:
            self.make_coherent_layer(dt)
            self.make_dissipative_layer(dt)
            self.dt = dt
            self.simulation_params.update({'dt':dt, 'tol':tol, 'm_max':m_max, 'k_max':k_max})
        for i in range(Nsteps):
            # apply uMPO[0]
            lp = apply_mpo(self.uMPO[0], self.psi, tol, m_max, max_sweeps)  # --> right canonical form
            logging.debug(f'overlap={lp}')
            # apply bMPO
            contract_dissipative_layer(self.B_list, self.psi, self.B_keys)
            # STILL in right canonical form
            # CPTP maps preserve canonical forms!
            truncate_krauss_sweep(self.psi, tol, k_max) # --> left canonical form
            # apply uMPO[1]
            lp = apply_mpo(self.uMPO[1], self.psi, tol, m_max, max_sweeps)  # --> right canonical form
            # now in right canonical form
            logging.debug(f'overlap={lp}')
                
    def make_dissipative_layer(self, dt: float) -> None:
        """
        calculate the Kraus operators from the Lindblad jump operators, the resulting
        Kraus operators have the following shape
                   0|
                :.......:
                :   B   :--:
                :.......:  |
                   1|     2|

        0 : output (dim=d)
        1 : input  (dim=d)
        2 : Kraus  (dim<=d^2)
        """
        B_list = []
        B_keys = []
        for i, L in enumerate(self.Lloc):
            d = self.dims[i]
            if L is not None:
                B_keys.append(1)
                # calculate the dissipative part in superoperator form
                D = torch.kron(L,L.conj()) \
                - 0.5*(torch.kron(L.conj().T@L, torch.eye(d)) + torch.kron(torch.eye(d), L.T@L.conj()))
                # TODO: try the other decomposition and compare result operator sizes
                eDt = matrix_exp(D*dt)
                eDt = torch.reshape(eDt, (d,d,d,d))
                eDt = eDt.permute(0,2,1,3)
                eDt = torch.reshape(eDt, (d*d,d*d))
                assert torch.allclose(eDt, eDt.adjoint())
                # TODO: try cholesky from torch, quantify the error
                B = _cholesky(eDt)
                #B = cholesky(eDt)
                B = torch.reshape(B, (d,d,-1))
                B_list.append(B.to(self.device, dtype=self.dtype))
            else:
                B_keys.append(0)
                B_list.append(torch.eye(d, device=self.device, dtype=self.dtype)[:,:,None])
        self.B_list = B_list
        self.B_keys = B_keys

    def make_coherent_layer(self, dt:float, eps=0.) -> None:
        """
        Here we adopt the strong splitting exp(-i*H*t)~exp(-i*H_e*t/2)exp(-i*H_o*t)exp(-i*H_e*t/2)
        """
        N = len(self.psi)
        dtype, device = self.dtype, self.device
        half_e = [torch.eye(self.dims[i], dtype=dtype, device=device)[None,None,:,:] for i in range(N)]
        half_o = [torch.eye(self.dims[i], dtype=dtype, device=device)[None,None,:,:] for i in range(N)]
        # torch.tensordot(a, b) does not support type casting!
        for k, ls in enumerate([half_e, half_o]):  # even k = 0, odd k = 1
            for i in range(k,N-1,2):
                j = i+1
                di, dj = self.dims[i], self.dims[j]
                u2site = matrix_exp(-1j*self.hduo[i]*dt/2) # imaginary unit included!
                u2site = torch.reshape(u2site, (1,1,di,dj,di,dj))  # mL,mR,i,j,i*,j*
                ls[i], ls[j] = split(u2site, mode='left', tol=eps, renormalize=False)
                ls[i] = ls[i].to(device, dtype=dtype)
                ls[j] = ls[j].to(device, dtype=dtype)
        half_e = MPO(half_e)
        half_o = MPO(half_o)
        # TODO: do this variationally using apply_mpo
        # combine the dissipative part 
        self.uMPO = [mul(half_e, half_o), mul(half_o, half_e)]
        """
        uMPO1, norm1 = apply_mpo(half_e, half_o, tol=1e-7, m_max=None, max_sweeps=3, overwrite=False)
        uMPO1[0] = norm1*uMPO1[0]
        uMPO2, norm2 = apply_mpo(half_o, half_e, tol=1e-7, m_max=None, max_sweeps=3, overwrite=False)
        uMPO2[0] = norm2*uMPO2[0]
        self.uMPO = [uMPO1, uMPO2]
        """
        del half_e
        del half_o
        gc.collect()
        torch.cuda.empty_cache()

    def state_dict(self) -> dict:
        return {'psi': self.psi, 
                'model_params': self.model_params, 
                'simulation_params': self.simulation_params}

    def _bot(self) -> None:
        assert len(self.Lloc) == len(self.psi)


def _cholesky(a):
    """stablized cholesky decomposition of matrix a"""
    eigvals, eigvecs = eigh(a)
    logging.debug(f'smallest 5 eigenvalues  are {eigvals[:5]}')
    mask = eigvals > min(1e-15, abs(eigvals[0]))
    eigvals = eigvals[mask]
    eigvecs = eigvecs[:,mask]
    B_mat = eigvecs*torch.sqrt(eigvals)
    logging.debug(f'error during cholesky decomposition: {torch.dist(a, B_mat@B_mat.adjoint())}')
    return B_mat

def contract_dissipative_layer(O: list, psi: LPTN, keys: list) -> None:
    """Contract the dissipative layer of Kraus operators with the LPTN

    Parameters
    ----------
    O : list
        list containing the local one-site Kraus operator
    psi : LPTN
        the operand
    keys : list
        a binary list
    tol : float
        largest discarded singular value in each truncation step
    k_max : int 
        largest Kraus dimension allowed

    Return
    ----------
    phi : LPTN
        result of the product O|psi>
        
                    |k   0 output
                    O -----2 Kraus
    O |psi> =       |k*, 1
                    |k , 2
                ---psi----
                    |
    """
    assert len(psi) == len(O) 
    assert len(psi) == len(keys)
    for i in range(len(psi)):
        if keys[i]:
            psi[i] = torch.tensordot(psi[i], O[i], dims=([2],[1]))
            psi[i] = torch.swapaxes(psi[i], 2, 3)
            psi[i] = torch.reshape(psi[i], psi[i].shape[:-2]+(-1,))

def truncate_krauss_sweep(As:list, tol:float, k_max:int) -> None:
    """psi must be in right canonical form when passed in"""
    nbonds = len(As) - 1
    for i in range(nbonds):
        di, dj, dd, dk = As[i].shape
        u, svals, _ = svd(As[i].reshape(-1,dk), full_matrices=False)
        # svals should have unit norm
        pivot = min(torch.sum(svals**2 > tol), k_max)
        svals = svals[:pivot] / torch.linalg.norm(svals[:pivot])
        As[i] = torch.reshape(u[:,:pivot]*svals[:pivot], (di, dj, dd, -1)) # s, d, k, s'

        dk = As[i].shape[-1]    # reduced Kraus dim
        As[i], r = qr(torch.reshape(As[i].permute(0,2,3,1), (-1, dj)))  # TODO: check copy or view behavior
        # qr decomposition might change dj
        As[i] = As[i].reshape(di,dd,dk,-1).permute(0,3,1,2) # TODO: check copy or view behavior
        
        As[i+1] = torch.tensordot(r, As[i+1], dims=1)

    i = -1
    di, dj, dd, dk = As[i].shape
    u, svals, _ = svd(As[i].reshape(-1,dk), full_matrices=False)
    pivot = min(torch.sum(svals**2 > tol), k_max)
    svals = svals[:pivot] / torch.linalg.norm(svals[:pivot])
    As[i] = torch.reshape(u[:,:pivot]*svals[:pivot], (di, dj, dd, -1)) # s, d, k, s'

    dk = As[i].shape[-1]    # reduced Kraus dim
    As[i], r = qr(torch.reshape(As[i].permute(0,2,3,1), (-1, dj)))
    # qr decomposition might change dj
    As[i] = As[i].reshape(di,dd,dk,-1).permute(0,3,1,2)