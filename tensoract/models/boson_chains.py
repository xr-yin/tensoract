#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__='Xianrui Yin'

import torch
from scipy import sparse

from ..core import MPO

__all__ = ['BosonChain', 'BoseHubburd', 'DDBH']

class BosonChain(object):
    """1D homogeneous Boson chain
    """
    def __init__(self, 
                 N: int, 
                 d: int, 
                 *,
                 dtype: torch.dtype=torch.double) -> None:
        """we include the local operators as instance attributes instead of 
        class attributes due to the indeterminacy of local dimensions"""
        self._N = N
        self._dtype = dtype
        self.d = d
        self.nu = torch.zeros((d,d), dtype=dtype)
        self.bt = torch.diag(torch.arange(1,d, dtype=dtype)**0.5, -1)
        self.bn = torch.diag(torch.arange(1,d, dtype=dtype)**0.5, +1)
        self.num = torch.diag(torch.arange(d, dtype=dtype))
        self.bid = torch.eye(d, dtype=dtype)

    def H_full(self):
        N, d = self._N, self.d
        h_full = sparse.csr_matrix((d**N, d**N))
        for i, hh in enumerate(self.hduo):
            h_full += sparse.kron(sparse.eye(d**i), sparse.kron(hh, torch.eye(d**(N-2-i))))
        return h_full
    
    def L_full(self):
        """extend local one-site Lindblad operators into full space"""
        N, d = self._N, self.d
        Ls = []
        for i, L in enumerate(self.Lloc):
            if L is not None:
                Ls.append(sparse.kron(sparse.eye(d**i), sparse.kron(L, torch.eye(d**(N-1-i)))))
        return Ls
    
    def Liouvillian(self, H, *Ls):
        """
        calculate the Liouvillian (super)operator

        Paras:
            H: the Hamiltonian in the full Hilbert space
            *L: the Lindblad jump operator(s) in the full Hilbert space

        Return: the Liouvillian operator as a sparse matrix
        """
        Lv = self._Hsup(H)
        for L in Ls:
            Lv += self._Dsup(L)
        return Lv
    
    def _Dsup(self, L):
        """
        calculate the $L\otimes L^\bar - (L^\dagger L\otimes I + I\otimes L^T L^\bar)/2$
        """
        D = self.d**self._N
        return sparse.kron(L,L.conj()) \
            - 0.5*(sparse.kron(L.conj().T@L, sparse.eye(D)) + sparse.kron(sparse.eye(D), L.T@L.conj()))

    def _Hsup(self, H):
        """
        calculate the Hamiltonian superoperator $-iH \otimes I + iI \otimes H^T$
        """
        D = self.d**self._N
        return - 1j*(sparse.kron(H,sparse.eye(D)) - sparse.kron(sparse.eye(D), H.T))

    @property
    def dtype(self):
        return self._dtype
    
    def __len__(self):
        return self._N
    
class BoseHubburd(BosonChain):
    """
    1D Bose-Hubburd model with Hamiltonian
    H = -t \sum (b)

    Parameters
    ----------
    N : int
        size of the 1D-lattice
    d : int
        local Hilbert space dimension
    t : float or torch.complex
        hopping amplitude
    U : float
        onstie interaction strength (U>0 means replusive)
    mu : float
        chemical potential
    F : float or torch.complex
        coherent driving strength
    gamma : float
        coupling strength between the system and the environment
    """
    def __init__(self, 
                 N: int, 
                 d: int, 
                 t: float, 
                 U: float, 
                 mu: float, 
                 *, 
                 dtype: torch.dtype=torch.double) -> None:
        super().__init__(N, d, dtype=dtype)
        self.t = t
        self.U = U
        self.mu = mu

    @property
    def hduo(self):
        bt, bn = self.bt, self.bn
        n, id = self.num, self.bid
        t = self.t
        h_list = []
        for i in range(self._N - 1):
            UL = UR = 0.5 * self.U
            muL = muR = 0.5 * self.mu
            if i == 0: # first bond
                UL, muL = self.U, self.mu
            if i + 1 == self._N - 1: # last bond
                UR, muR = self.U, self.mu
            h = - t * (torch.kron(bt, bn) + torch.kron(bn, bt)) \
                - muL * torch.kron(n, id) \
                - muR * torch.kron(id, n) \
                + UL * torch.kron(n@(n-id), id)/2 \
                + UR * torch.kron(id, n@(n-id))/2
            # h is a matrix with legs ``(i, j), (i*, j*)``
            # reshape to a tensor with legs ``i, j, i*, j*``
            # reshape is carried out in evolution algorithms after exponetiation
            h_list.append(h)
        return h_list

    @property
    def mpo(self):
        t, U, mu = self.t, self.U, self.mu
        bt, bn= self.bt, self.bn
        n, nu, id = self.num, self.nu, self.bid
        with torch.no_grad():
            row1 = torch.stack([id, nu, nu, nu], dim=0)
            row2 = torch.stack([bn, nu, nu, nu], dim=0)
            row3 = torch.stack([bt, nu, nu, nu], dim=0)
            row4 = torch.stack([0.5*U*n@(n-id) - mu*n, -t*bt, -t*bn, id], dim=0)
        O = torch.stack([row1, row2, row3, row4], dim=0)
        Os = [O] * self._N
        Os[0] = O[None,-1,:,:,:]
        Os[-1] = O[:,0,None,:,:]
        return MPO(Os)
    
class DDBH(BoseHubburd):
    """class for driven-dissipative Bose-Hubburd model"""
    def __init__(self, 
                 N: int, 
                 d: int, 
                 t: float, 
                 U: float, 
                 mu: float, 
                 F: float, 
                 gamma: float) -> None:
        # dtype must be set to double here to ensure accuracy when prepare the 
        # unitary time evolution operator and the Kraus operators
        super().__init__(N, d, t, U, mu, dtype=torch.double)
        self.F = F
        self.gamma = gamma

    @property
    def hduo(self):
        bt, bn = self.bt, self.bn
        n, id = self.num, self.bid
        t = self.t
        h_list = []
        for i in range(self._N - 1):
            UL = UR = 0.5 * self.U
            muL = muR = 0.5 * self.mu
            FL = FR = 0.5 * self.F
            if i == 0: # first bond
                UL, muL, FL = self.U, self.mu, self.F
            if i + 1 == self._N - 1: # last bond
                UR, muR, FR = self.U, self.mu, self.F
            h = - t * (torch.kron(bt, bn) + torch.kron(bn, bt)) \
                - muL * torch.kron(n, id) \
                - muR * torch.kron(id, n) \
                + UL * torch.kron(n@(n-id), id)/2 \
                + UR * torch.kron(id, n@(n-id))/2 \
                + FL * torch.kron(bt, id) \
                + FR * torch.kron(id, bt) \
                + FL.conjugate() * torch.kron(bn, id) \
                + FR.conjugate() * torch.kron(id, bn)
            # h is a matrix with legs ``(i, j), (i*, j*)``
            # reshape to a tensor with legs ``i, j, i*, j*``
            # reshape is carried out in evolution algorithms after exponetiation
            h_list.append(h)
        return h_list
    
    @property
    def mpo(self):
        t, U, mu, F = self.t, self.U, self.mu, self.F
        bt, bn= self.bt, self.bn
        n, nu, id = self.num, self.nu, self.bid
        diag = 0.5*U*n@(n-id) - mu*n + F*bt + F.conjugate()*bn
        with torch.no_grad():
            row1 = torch.stack([id, nu, nu, nu], dim=0)
            row2 = torch.stack([bn, nu, nu, nu], dim=0)
            row3 = torch.stack([bt, nu, nu, nu], dim=0)
            row4 = torch.stack([diag, -t*bt, -t*bn, id], dim=0)
        O = torch.stack([row1, row2, row3, row4], dim=0)
        Os = [O] * self._N
        Os[0] = O[None,-1,:,:,:]
        Os[-1] = O[:,0,None,:,:]
        return MPO(Os)
    
    @property
    def Lloc(self):
        return [self.gamma**0.5 * self.bn] * self._N
    
    @property
    def Liouvillian(self):
        return super().Liouvillian(self.H_full(), *self.L_full())
    
    def parameters(self):
        return {'N': self._N, 
                'd': self.d, 
                't': self.t, 
                'U': self.U, 
                'mu': self.mu, 
                'F': self.F, 
                'gamma': self.gamma, 
                'dtype': self.dtype}