#!/usr/bin/env python3
# -*- coding: utf-8 -*-


__author__='Xianrui Yin'

import torch

from typing import Self

from .operations import qr_step, rq_step

__all__ = ['MPO']

class MPO(object):
    """class for matrix product operators

    Parameters
    ----------
    As : list 
        a list of rank-4 tensors, each tensor has the following shape

        k |
    i---- A ----j
        k*|

    i (j) is the left (right) bond leg and k (k*) is the ket (bra) physical leg
    the legs are ordered as `i, j, k, k*`

    Attributes
    ----------
    As : sequence of Tensors
        as described above

    Methods
    -------
    orthonormalize()
    conj()
    hc()
    to_matrix()
    to()
    """
    def __init__(self, As) -> None:
        self.As = As
        self._N = len(As)
        self.device = As[0].device
        self.dtype = As[0].dtype
        self.__bot()

    @classmethod
    def gen_random_mpo(cls, N:int, 
                       m_max:int, 
                       phy_dims:list, 
                       *,
                       dtype:torch.dtype=torch.complex128, 
                       hermitian:bool=False, 
                       device:torch.device=None) -> Self:
        assert len(phy_dims) == N
        bond_dims = torch.randint(1, m_max, size=(N+1,))
        bond_dims[0] = bond_dims[-1] = 1 

        sizes = [(bond_dims[i],bond_dims[i+1],phy_dims[i], phy_dims[i]) for i in range(N)]

        As = [torch.normal(0, 1, size=size, dtype=dtype, device=device, requires_grad=False) for size in sizes]

        if hermitian:
            As = [A + A.swapaxes(2,3).conj() for A in As]

        return cls(As)
    
    def copy(self) -> Self:
        return self.__class__([A.clone() for A in self])

    @property
    def bond_dims(self) -> torch.Tensor:
        return torch.tensor([A.shape[0] for A in self] + [self[-1].shape[1]])
    
    @property
    def physical_dims(self) -> torch.Tensor:
        return torch.tensor([A.shape[2] for A in self])
    
    def orthonormalize(self, mode: str, center_idx=None) -> None:
        r"""
        Transforming the MPS (MPO) into canaonical forms by doing successive QR decompositions.

        Parameters
        ----------
        mode : str
            'right', 'left', 'mixed'. When choosing 'mixed,' the corresponding index of the
            orthogonality center must be given
        center_idx : int
            the index of the orthogonality center

        Return
        ----------
        None

        Notes
        ----------
        scipy.linalg.qr, which we use here, only accepts 2-d arrays (matrices) as inputs to 
        be decomposed. Therefore, one must first combine the physical and matrix leg by doing 
        a reshape, before calling qr().
        
        On the other hand, numpy.linalg.qr can take in (N>2)-d arrays, which are regarded 
        as stacks of matrices residing on the last 2 dimensions. Consequently, one can call 
        qr() with the original tensors. In this regard, [physical, left bond, right bond] 
        indexing is preferred.
        """

        if mode == 'right':
            for i in range(self._N-1, 0,-1):
                    self[i-1], self[i] = rq_step(self[i-1], self[i])
            self[0] /= torch.linalg.norm(self[0].squeeze())
        elif mode == 'left':
            for i in range(self._N - 1):
                self[i], self[i+1] = qr_step(self[i], self[i+1])
            self[-1] /= torch.linalg.norm(self[-1].squeeze())
        elif mode == 'mixed':
            assert center_idx >= 0
            assert center_idx < self._N
            for i in range(center_idx):
                self[i], self[i+1] = qr_step(self[i], self[i+1])
            for i in range(self._N-1,center_idx,-1):
                self[i-1], self[i] = rq_step(self[i-1], self[i])
            self[center_idx] /= torch.linalg.norm(self[center_idx].squeeze())
        else:
                raise ValueError(
                    'Mode argument should be one of left, right or mixed')

    def conj(self) -> Self:
        """
        Return
        ------
        complex conjugate of the MPO
        """
        return MPO([A.conj() for A in self])
    
    def hc(self) -> Self:
        """
        Return
        ------
        Hermitian conjugate of the MPO
        """
        return MPO([A.swapaxes(2,3).conj() for A in self])

    def to_matrix(self) -> torch.Tensor:
        """
        convert the MPO into a dense matrix for best compatability. Users
        are free to further convert it into a sparse matrix to explore more
        efficient linear algebra algorithms.
        """
        full = self[0]
        for i in range(1,self._N):
            full = torch.tensordot(full, self[i], dims=([1],[0]))
            full = full.permute(0,3,1,4,2,5)
            di, dj, dk1, dk2, dk3, dk4 = full.shape
            full = torch.reshape(full, (di, dj, dk1*dk2, dk3*dk4))
        return full.squeeze()
    
    def to(self, *args, **kwargs)  -> None:
        """call torch.tensor.to()"""
        for i in range(self._N):
            self[i] = self[i].to(*args, **kwargs)
        self.device = self[0].device
        self.dtype = self[0].dtype
        
    def __len__(self):
        return self._N
    
    def __getitem__(self, idx: int):
        return self.As[idx]
    
    def __setitem__(self, idx: int, value):
        self.As[idx] = value
    
    def __iter__(self):
        return iter(self.As)
    
    def __bot(self) -> None:
        assert self.As[0].shape[0]==self.As[-1].shape[1]==1
        # check bond dims of neighboring tensors
        for i in range(self._N-1):
             assert self.As[i].shape[1] == self.As[i+1].shape[0]