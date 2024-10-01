#!/usr/bin/env python3
# -*- coding: utf-8 -*-


__author__='Xianrui Yin'

import torch

from typing import Self, overload
from collections.abc import Sequence

from .mpo import MPO
from .operations import merge, split, mul, qr_step, inner

__all__ = ['LPTN', 'compress']

class LPTN(MPO):
    """Locally purified tensor networks
    
    Parameters
    ----------
        As : list
            list of local rank-4 tensors, each tensor has the following shape
                k
                |
            i---A---j       (i, j, k, l)
                |
                l
    
    Methods
    ----------
    site_expectation_value(idx=None)
        compute the expectation value of local (one-site) observables
    bond_expectation_value(idx=None)
        compute the expectation value of local two-site observables
    to_density_matrix()
    probabilities()
    """
    def __init__(self, As) -> None:
        super().__init__(As)

    @classmethod
    def gen_polarized_spin_chain(cls, 
                                 N: int, 
                                 polarization: str, 
                                 *,
                                 dtype: torch.dtype=torch.complex128, 
                                 device: torch.device=None) -> Self:
        if polarization not in ['+z','-z','+x']:
            raise ValueError('Only support polarization +z, -z or +x')
        A = torch.zeros([1,1,2,1], dtype=dtype, device=device)
        if polarization == '+z':
            A[0,0,0,0] = 1.
        elif polarization == '-z':
            A[0,0,1,0] = 1.
        else:
            A[0,0,0,0] = A[0,0,1,0] = 0.5**0.5
        return cls([A]*N)
    
    @classmethod
    def gen_random_state(cls,
                         N: int, 
                         m_max: int, 
                         k_max: int, 
                         phy_dims: torch.Tensor, 
                         *,
                         dtype: torch.dtype=torch.complex128, 
                         device: torch.device=None) -> Self:
        assert len(phy_dims) == N
        bond_dims = torch.randint(m_max//2, m_max, size=(N+1,))
        krauss_dims = torch.randint(k_max//2, k_max, size=(N,))
        bond_dims[0] = bond_dims[-1] = 1

        sizes = [(bond_dims[i],bond_dims[i+1],phy_dims[i],krauss_dims[i]) for i in range(N)]
        As = [torch.normal(0, 1, size=size, dtype=dtype, device=device) for size in sizes]
        return cls(As)
    
    @property
    def krauss_dims(self) -> torch.Tensor:
        return torch.tensor([A.shape[3] for A in self])
    
    @overload
    def site_expectation_value(self, op: torch.Tensor, *, idx: int, drop_imag=False) -> torch.Tensor: ...
    """if `idx` is an integer, `op` must be a single operator for this specific physical site."""

    @overload
    def site_expectation_value(self, op: Sequence[torch.Tensor], *, idx: None, drop_imag=False) -> torch.Tensor: ...
    """if `idx` is None, `op` must be a list of ordered local operators for every physical site."""

    def site_expectation_value(self, 
                               op : torch.Tensor | Sequence[torch.Tensor], 
                               *, 
                               idx: int | None=None, 
                               drop_imag=False) -> torch.Tensor:
        """
        Parameters
        ----------
        op : list or NDArray
            list of local operators or a single local operator
        idx : int or None
            index of the site for calculating the expectation value <O_i>
        drop_imag : bool
            if True, only returns the real part of the expectation value. This is 
            desired when the operators are Hermitian.

        Note
        ----
        The state is NOT cached and will be converted into right canonical form if idx 
        is None otherwise mixed canonical form
        """
        if idx is None:
            """measure every site"""
            assert len(op) == self._N
            exp = torch.zeros(size=(self._N,), dtype=self.dtype, device=self.device)
            self.orthonormalize(mode='right')
            for i in range(self._N-1):
                amp = self[i]   # amplitude in the Schmidt basis
                opc = torch.tensordot(amp, op[i], dims=([2],[1])) # apply local operator
                exp[i] = torch.tensordot(amp.conj(), opc.swapaxes(2,3), dims=4)
                self[i], self[i+1] = qr_step(self[i], self[i+1]) # move the orthogonality center
            amp = self[-1]
            opc = torch.tensordot(amp, op[-1], dims=([2],[1])) # apply local operator
            exp[-1] = torch.tensordot(amp.conj(), opc.swapaxes(2,3), dims=4)
            return exp.real if drop_imag else exp
        else:
            """measure only site #idx"""
            self.orthonormalize(mode='mixed', center_idx=idx)
            amp = self[idx]   # amplitude in the Schmidt basis
            opc = torch.tensordot(amp, op, dims=([2],[1])) # apply local operator
            res = torch.tensordot(amp.conj(), opc.swapaxes(2,3), dims=4)
            return res.real if drop_imag else res

    @overload
    def bond_expectation_value(self, op: Sequence[torch.Tensor], *, idx: None, drop_imag: bool=False) -> torch.Tensor: ...
    """if `idx` is None, `op` must be a list of ordered two-local operators for every pair of neighbouring site."""

    @overload
    def bond_expectation_value(self, op: torch.Tensor, *, idx: int, drop_imag: bool=False) -> torch.Tensor: ...
    """if `idx` is an integer, `op` must be a single two-local operator for this specific pair of neighbouring site."""

    def bond_expectation_value(self, 
                               op: torch.Tensor | Sequence[torch.Tensor], 
                               *, 
                               idx: int | None=None, 
                               drop_imag: bool=False) -> torch.Tensor:
        """
        Parameters
        ----------
        op : list or torch.Tensor
            list of local two-site operators or a single local two-site operator
        idx : int or None
            the index of the left site for calculating the expectation value <O_i O_i+1>
        drop_imag : bool
            if True, only returns the real part of the expectation value. This is 
            desired when the operators are Hermitian.

        Note
        ----
        The state is NOT cached and will be converted into right canonical form if idx 
        is None otherwise mixed canonical form
        """
        if idx is None:
            """measure every pair of sites"""
            assert len(op) == self._N-1
            exp = torch.zeros(size=(self._N-1,), dtype=self.dtype, device=self.device)
            self.orthonormalize(mode='right')
            for i in range(self._N-1):
                j = i+1
                amp = merge(self[i], self[j])   # amplitude in the Schmidt basis
                opc = torch.tensordot(amp, op[i], dims=([2,3],[2,3])) # apply local operator
                exp[i] = torch.tensordot(amp.conj().permute(0,1,4,5,2,3), opc, dims=6)
                self[i], self[i+1] = qr_step(self[i], self[i+1]) # move the orthogonality center
            return exp.real if drop_imag else exp
        else:
            """measure only site pair idx & idx+1"""
            self.orthonormalize(mode='mixed', center_idx = idx)
            amp = merge(self[idx],self[idx+1])   # amplitude in the Schmidt basis
            opc = torch.tensordot(amp, op, dims=([2,3],[2,3])) # apply local operator
            res = torch.tensordot(amp.conj().permute(0,1,4,5,2,3), opc, dims=6)
            return res.real if drop_imag else res
        
    def correlations(self, A, B, i, connected: bool=False) -> torch.Tensor:
        """calculate the correlation function C(i,j) = <A_i*B_j> for j >=i
        
        Parameters
        ----------
        A : torch.Tensor
            operator acting on site i
        B : torch.Tensor
            operator acting on site j
        i : int
            site index
        connected : bool
            if True, return the connected correlation function
            C_conn(i,j) = <A_i*B_j> - <A_i>*<B_j>

        Return
        ------
        corrs : torch.Tensor
            1D tensor contains C(j)
        """
        N = self._N
        assert i < N
        corrs = torch.zeros(size=(self._N-i,), dtype=self.dtype, device=self.device)
        # j = i
        self.orthonormalize(mode='mixed', center_idx=i)
        amp = self[i]   # amplitude in the Schmidt basis
        opc = torch.tensordot(amp, A @ B, dims=([2],[1]))
        corrs[0] = torch.tensordot(amp.conj(), opc.swapaxes(2,3), dims=4)
        # j > i
        # Alice stores the contraction from the left to site j (including A but excluding B)
        Alice = torch.tensordot(amp, A, dims=([2],[1]))
        Alice = torch.tensordot(amp.conj(), Alice, dims=([0,2,3],[0,3,2]))  # mR*, mR
        for j in range(i+1, N):
            amp = self[j]   # this is not an amplitude in the Schmidt basis since j != i
            Bob = torch.tensordot(amp, B, dims=([2],[1]))
            Bob = torch.tensordot(Alice, Bob, dims=([1], [0]))
            Bob = torch.tensordot(amp.conj(), Bob.swapaxes(2,3), dims=4)
            corrs[j-i] = Bob
            # update Alice
            Alice = torch.tensordot(Alice, amp, dims=([1], [0]))
            Alice = torch.tensordot(amp.conj(), Alice, dims=([0,2,3], [0,2,3]))
        if connected:
            corrs -= self.site_expectation_value(A, idx=i) * self.site_expectation_value([B]*N)[i:]
        return corrs

    def measure(self, op_list: list, *, idx: int | None=None, drop_imag: bool=False) -> list[torch.Tensor]:
        """Perform measurements successively on each site

        When only one operator is measured, one can call site_expectation_value().
        When there are more to be measured, this function will be faster bacause 
        here we only go through the entire system once and the measurements are 
        handled together.

        Parameters
        ----------
        op_list : list
            list of local operators
        idx : int or None
            if `idx` is None, the measurements are done on every site in order
            This is only to be used when every site has the same physical dimension
            if `idx` is an integer, the measurements are done on the specified 
            site only
        drop_imag : bool
            if True, only returns the real part of the expectation value. This is 
            desired when the operators are Hermitian.

        Return
        ----------
        exp : list, exp[i] stores the measurement results for operator op_list[i]

            if `idx` is None, the elements of the list are 1D Tensor of size (N,)
            if `idx` is an integer, the elements of the list are 0D Tensor (single number)

        Note
        ----
        The state is NOT cached and will be converted into right canonical form if idx 
        is None otherwise mixed canonical form
        """
        if idx is None:
            # TODO: dynamically determine the result data type
            exp = [torch.zeros(size=(self._N,), dtype=self.dtype, device=self.device) for op in op_list]
            self.orthonormalize(mode='right')
            for i in range(self._N-1):  # column index
                amp = self[i]   # amplitude in the Schmidt basis
                for j, op in enumerate(op_list):    # row index
                    opc = torch.tensordot(amp, op, dims=([2],[1])) # apply local operator
                    exp[j][i] = torch.tensordot(amp.conj(), opc.swapaxes(2,3), dims=4)
                self[i], self[i+1] = qr_step(self[i], self[i+1]) # move the orthogonality center
            amp = self[-1]
            for j, op in enumerate(op_list):
                opc = torch.tensordot(amp, op, dims=([2],[1])) # apply local operator
                exp[j][-1] = torch.tensordot(amp.conj(), opc.swapaxes(2,3), dims=4)

                if torch.allclose(op.adjoint(), op) and drop_imag:
                    exp[j] = exp[j].real

            return exp
        else:
            self.orthonormalize(mode='mixed', center_idx = idx)
            amp = self[idx]   # amplitude in the Schmidt basis
            exp = []
            for j, op in enumerate(op_list):
                opc = torch.tensordot(amp, op, dims=([2],[1])) # apply local operator
                exp.append(torch.tensordot(amp.conj(), opc.swapaxes(2,3), dims=4))
                if torch.allclose(op.adjoint(), op) and drop_imag:
                    exp[j] = exp[j].real
            return exp
        
    def entropy(self,idx: int | None=None) -> torch.Tensor:
        r"""the (pesudo) von Neumann entanglement entropy

        The (bipartite) entanglement entropy for a pure state divided into two subsytems A and B, is 
        defined as :math:`Tr_A(\rho_A log(\rho_A))`, where :math:`\rho_A` is the reduced density matrix on 
        subsystem A.
        If we have a mixed state that is described by purification :math:`X_{k_1 k_2 \dots}^{i_1 i_2 \dots}` 
        and the correspponding density operator :math:`\rho_{i_1 i_2 \dots}{i*_1 i*_2 \dots}`, the 
        generalization to the entanglement entropy above would be :math:`Tr_A(\rho_A log(\rho_A))`, where 
        :math:`\rho_A = Tr_B(\rho) = Tr_B (Tr_{k_1 k_2 \dots}(X X*))`. This function instead calculates 
        :math: Tr_B (X X*). 
        I emphasize that the zero entropy, which corresponds to trivial bond at the bipartition, naturally 
        leads to a factorizable density matrix :math: \rho = \rho_A \otimes \rho_B. However, the reverse is 
        not true, since a general unitary transformation acting on the the Kraus indices {k_1 k_2 \dots} can 
        give rise to higher bond dimensions while leave the density matrix untouched. For this reason, the word 
        'peusdo' was introduced in the name."""
        if idx is None:
            S = torch.empty(size=(self._N - 1,))
            self.orthonormalize(mode='right')
            for i in range(self._N - 1):
                self[i], self[i+1] = qr_step(self[i], self[i+1])    # orthogonality cneter at i+1
                mL = self[i+1].shape[0] # bond between i and i+1
                s = torch.linalg.svdvals(self[i+1].reshape(mL, -1))
                s = s[s>1.e-15]
                ss = s*s
                S[i] = -torch.sum(ss*torch.log(ss))
            return S
        else:
            assert isinstance(idx, int)
            self.orthonormalize(mode='mixed',center_idx=idx)
            mL = self[idx].shape[0] # bond between idx-1 and idx
            s = torch.linalg.svdvals(self[idx].reshape(mL, -1))
            s = s[s>1.e-15]
            ss = s*s
            return -torch.sum(ss*torch.log(ss))

    def to_density_matrix(self, full=True) -> torch.Tensor:
        r"""density matrix for the locally purified tensor network
        :math:`\rho = X  X^\dagger`
        """
        if full:
            return mul(self, self.hc()).to_matrix()
        else:
            return mul(self, self.hc())
    
    def purity(self):
        r"""The purity of the density operator
        :math:`Tr(\rho^2) = Tr(\rho \cdot \rho) = ||rho||_F^2`"""
        rho = mul(self, self.hc())
        return inner(rho, rho).real
    
    def probabilities(self):
        """probability amplitudes of each state in the ensemble
        """
        Nstates = torch.prod(self.krauss_dims)
        norms = torch.zeros(Nstates)
        kspace = torch.arange(Nstates).reshape(self.krauss_dims)
        for i in range(Nstates):
            loc = torch.where(kspace==i)
            As = []
            assert len(self) == len(loc)
            for A, idx in zip(self, loc):
                As.append(A[:,:,:,idx])
            # TODO: check here
            psi = LPTN(As)
            norms[i] = torch.linalg.norm(psi.to_density_matrix())**2        
        return norms

def compress(psi:LPTN, tol:float, m_max:int, max_sweeps:int=2) -> tuple[LPTN,float]:
    """variationally compress a LPTN by optimizing the trace norm |X'-X|, where X' is 
    the guess state

    Parameters
    ----------
    psi : MPS
        the MPS to be compressed
    tol : float
        the largest truncated singular value
    m_max : int
        maximum bond dimension
    k_max : int
        maximun kraus dimension
    max_sweeps : int
        maximum optimization sweeps

    Return
    ----------
    phi : MPS
        the compressed MPS, in the mixed canonial form centered on the 0-th site
        almost right canonical as one would say
    """
    N = len(psi)
    phi = psi.copy()  # overwrite set to False, first copy then orthonormalize
    phi.orthonormalize('left')
    # peform a SVD sweep from the right to left
    # to find the initial guess of the target state with the required dimension
    for i in range(N-1,0,-1):
        di, dj, dd, dk = phi[i].shape
        phi[i] = torch.reshape(phi[i], (di, dj*dd*dk))
        u, s, vt = torch.linalg.svd(phi[i], full_matrices=False)
        mask = s>10*tol
        phi[i] = vt[mask,:].reshape(-1,dj,dd,dk)
        phi[i-1] = torch.tensordot(u[:,mask]*s[mask], phi[i-1], dims=(0,1)).swapaxes(1,0)
    # now we arrive at a right canonical LPTN
    RBT = _load_right_bond_tensors(psi,phi)
    LBT = [torch.ones((1,1))] * N
    for n in range(max_sweeps):
        for i in range(N-1): # sweep from left to right
            j = i+1
            temp = merge(psi[i],psi[j])
            temp = torch.tensordot(LBT[i], temp, dims=(0,0))
            temp = torch.tensordot(RBT[j], temp, dims=(0,1)).swapaxes(0,1)
            phi[i], phi[j] = split(temp, 'right', tol, m_max)
            # compute left bond tensor L[j]
            LBT[j] = torch.tensordot(psi[i], LBT[i], dims=(0,0))
            LBT[j] = torch.tensordot(LBT[j], phi[i].conj(), dims=([1,2,3],[2,3,0]))
        for j in range(N-1,0,-1):  # sweep from right to left
            i = j-1
            temp = merge(psi[i],psi[j])
            temp = torch.tensordot(LBT[i], temp, dims=(0,0))
            temp = torch.tensordot(RBT[j], temp, dims=(0,1)).swapaxes(0,1)
            phi[i], phi[j] = split(temp, 'left', tol, m_max)
            # compute right bond tensor R[i]
            RBT[i] = torch.tensordot(psi[j], RBT[j], dims=(1,0))
            RBT[i] = torch.tensordot(RBT[i], phi[j].conj(), dims=([1,2,3],[2,3,1]))
        overlap = torch.tensordot(psi[0], RBT[0], dims=(1,0))
        overlap = torch.tensordot(overlap, phi[0].conj(), dims=([1,2,3],[2,3,1]))
        print(f'overlap after the {n+1} sweep(s): {overlap.item()}')
    return phi, overlap.item()

def _load_right_bond_tensors(psi:LPTN, phi:LPTN) -> list[torch.Tensor]:
    """Calculate the right bond tensors while contracting two LPTNs.
    RBT[i] is to the right of the LPTN[i].

    Parameters
    ----------
    psi : LPTN
        used as ket
    phi : LPTN 
        used as bra

    Return
    ----------
    RBT : list
        list of length N containing the right bond tensors, RBT[N-1] is trivial
    """
    assert len(psi) == len(phi)
    N = len(psi)
    RBT = [torch.ones((1,1))] * N
    for i in range(N-1,0,-1):
        RBT[i-1] = torch.tensordot(psi[i], RBT[i], dims=(1,0))
        RBT[i-1] = torch.tensordot(RBT[i-1], phi[i].conj(), dims=([1,2,3],[2,3,1]))
    return RBT