import torch
from scipy import sparse

from collections.abc import Sequence

from ..core import MPO
from .base_model import NearestNeighborModel

__all__ = ['SpinChain', 'TransverseIsing', 'Heisenberg']

class SpinChain(NearestNeighborModel):
    """
    Base class for spin one-half chains

    Parameters
    ----------
    N : int
        The length of the spin chain.
    gamma : float | array
        The dissipation rates.
    l_ops : torch.Tensor or list of torch.Tensor
        The local Lindblad operators.

    Methods
    -------
    H_full()
        Returns the full Hamiltonian.
    L_full()
        Returns the list of full Lindblad operators.
    energy(psi)
        Returns the energy of the system.
    current(psi)
        Returns the particle current.
    Liouvillian(H, *Ls)
        Returns the Liouvillian superoperator.
    """
    _dtype = torch.complex128
    cid = torch.eye(2, dtype=_dtype)
    nu = torch.zeros([2,2], dtype=_dtype)
    sx = torch.tensor([[0., 1.], [1., 0.]], dtype=_dtype)
    sy = torch.tensor([[0., -1j], [1j, 0.]], dtype=_dtype)
    sz = torch.tensor([[1., 0.], [0., -1.]], dtype=_dtype)
    splus = torch.tensor([[0., 1.], [0., 0.]], dtype=_dtype)
    sminus = torch.tensor([[0., 0.], [1., 0.]], dtype=_dtype)
    
    def __init__(self, N:int, gamma, l_ops) -> None:
        self._N = N
        self.gamma = self.init_onsites(gamma, N)
        self.l_ops = self.init_l_ops(self.gamma, l_ops, N)

    def H_full(self):
        """extend local two-site Hamiltonian operators into full space"""
        N = self._N
        h_full = sparse.csr_matrix((2**N, 2**N))
        for i, hh in enumerate(self.h_ops):
            h_full += sparse.kron(sparse.eye(2**i), sparse.kron(hh, torch.eye(2**(N-2-i))))
        return h_full
    
    def L_full(self):
        """extend local one-site Lindblad operators into full space"""
        N = self._N
        Ls = []
        # translational invariant, l_ops is a single operator
        if not isinstance(self.l_ops, Sequence):
            for i in range(N):
                Ls.append(sparse.kron(sparse.eye(2**i), sparse.kron(self.l_ops, torch.eye(2**(N-1-i)))))
        # not translational invariant, l_ops is a list of operators
        else:
            for i, L in enumerate(self.l_ops):
                if L is not None:
                    Ls.append(sparse.kron(sparse.eye(2**i), sparse.kron(L, torch.eye(2**(N-1-i)))))
        return Ls
    
    def energy(self, psi):
        """the energy (expectaton value of the Hamiltonian) of the system"""
        assert len(psi) == self._N
        return torch.sum(psi.bond_expectation_value([h.reshape(2,2,2,2) for h in self.h_ops]))
    
    def current(self, psi):
        """particle current"""
        assert len(psi) == self._N
        Nbonds = self._N-1
        current_op = -1j*(torch.kron(self.splus, self.sminus) - torch.kron(self.sminus, self.splus))
        current_op = torch.reshape(current_op, (2,2,2,2))
        return psi.bond_expectation_value([current_op]*Nbonds)

    def Liouvillian(self, H, *Ls):
        """
        Calculate the Liouvillian superoperator.

        Parameters
        ----------
        H : sparse matrix
            The Hamiltonian in the full Hilbert space.
        *Ls : sparse matrix
            The Lindblad jump operator(s) in the full Hilbert space.

        Returns
        -------
            The Liouvillian superoperator.
        """
        Lv = self._Hsup(H)
        for L in Ls:
            Lv += self._Dsup(L)
        return Lv
    
    def _Dsup(self, L):
        """
        calculate the $L\otimes L^\bar - (L^\dagger L\otimes I + I\otimes L^T L^\bar)/2$
        """
        D = 2**self._N
        return sparse.kron(L,L.conj()) \
            - 0.5*(sparse.kron(L.conj().T@L, sparse.eye(D)) + sparse.kron(sparse.eye(D), L.T@L.conj()))

    def _Hsup(self, H):
        """
        calculate the Hamiltonian superoperator $-iH \otimes I + iI \otimes H^T$
        """
        D = 2**self._N
        return - 1j*(sparse.kron(H,sparse.eye(D)) - sparse.kron(sparse.eye(D),H.T))

    def __len__(self):
        return self._N

class TransverseIsing(SpinChain):
    """
    Class for the transverse-field Ising model.
    
    The Hamiltonian for the transverse-field Ising model is given by:
    
    .. math::
        H = -J \sum_{i} \sigma_i^z \sigma_{i+1}^z - g \sum_{i} \sigma_i^x
    
    Parameters
    ----------
    N : int
        The length of the spin chain.
    g : float | array
        The strength of the transverse field.
    J : float | array, optional
        The interaction strength between spins (default is 1).
    gamma : float | array, optional
        The dissipation rates (default is 0).
    
    Attributes
    ----------
    J : float
        The interaction strength between spins.
    g : float
        The strength of the transverse field.
    _N : int
        The length of the spin chain.
    
    Methods
    -------
    mpo()
        Constructs the Matrix Product Operator (MPO) representation of the Hamiltonian.
    h_ops()
        Constructs the list of local Hamiltonian operators for the spin chain.
    """

    def __init__(self, N:int, g, J=1., gamma=0., l_ops=None):
        super().__init__(N, gamma, l_ops)
        self.J = self.init_couplings(J, N)
        self.g = self.init_onsites(g, N)

    @property
    def mpo(self):
        sx, sz, nu, id = self.sx, self.sz, self.nu, self.cid
        J, g = self.J, self.g
        N = self._N
        p = N - 1

        Os = []

        for i in range(N):
            # pseudo-PBC to avoid the out-of-bounds indexing for the couplings J
            # This does not affect the final result since the last MPO tensor 
            # consists of only the first column
            row1 = torch.stack([id, nu, nu], dim=0)
            row2 = torch.stack([sz, nu, nu], dim=0)
            row3 = torch.stack([-g[i]*sx, -J[i%p]*sz, id], dim=0)
            O = torch.stack([row1, row2, row3], dim=0)
            if i == 0:
                Os.append(O[None,-1,:,:,:])
            elif i == N-1:
                Os.append(O[:,0,None,:,:])
            else:
                Os.append(O)
        return MPO(Os)
    
    @property
    def h_ops(self):
        sx, sz, id = self.sx, self.sz, self.cid
        J, g = self.J, self.g
        h_list = []
        for i in range(self._N - 1):
            gL = 0.5 * g[i]
            gR = 0.5 * g[i+1]
            if i == 0: # first bond
                gL = g[i]
            if i + 1 == self._N - 1: # last bond
                gR = g[i+1]
            h = - J[i] * torch.kron(sz, sz) \
                - gL * torch.kron(sx, id) \
                - gR * torch.kron(id, sx)
            # h is a matrix with legs ``(i, j), (i*, j*)``
            # reshape to a tensor with legs ``i, j, i*, j*``
            # reshape is carried out in evolution algorithms after exponetiation
            h_list.append(h)
        return h_list

    def parameters(self):
        """
        Returns the parameters of the model.

        Returns
        -------
        list
            The parameters of the model.
        """
        return {'J': self.J, 'g': self.g, 'gamma':self.gamma}

class Heisenberg(SpinChain):
    """
    1D spin-1/2 Heisenberg model.

    The Hamiltonian for the Heisenberg model is given by:

    .. math::
        H = -\sum \left( J_x S_x S_x + J_y S_y S_y + J_z S_z S_z + g S_x \right)

    Parameters
    ----------
    N : int
        The length of the spin chain.
    Jx, Jy, Jz : float | array
        Coupling constants.
    g : float | array
        Transverse field strength.
    gamma : float | array
        Dissipation rate.

    Attributes
    ----------
    mpo : MPO
        Matrix Product Operator representation of the Hamiltonian.
    h_ops : list of torch.Tensor
        List of Hamiltonian operators for each bond.
    l_ops : list of torch.Tensor
        List of Lindblad jump operators.

    Methods
    -------
    Liouvillian()
        Constructs the Liouvillian superoperator.
    """
    def __init__(self, N:int, Jx, Jy, Jz, g, gamma=0., l_ops=None):
        super().__init__(N, gamma, l_ops)
        self.Jx = self.init_couplings(Jx, N)
        self.Jy = self.init_couplings(Jy, N)
        self.Jz = self.init_couplings(Jz, N)
        self.g = self.init_onsites(g, N)

    @property
    def mpo(self):
        sx, sy, sz, nu, id = self.sx, self.sy, self.sz, self.nu, self.cid
        Jx, Jy, Jz = self.Jx, self.Jy, self.Jz
        g = self.g
        N = self._N
        p = N - 1

        Os = []

        for i in range(N):
            # pseudo-PBC to avoid the out-of-bounds indexing for the couplings J
            # This does not affect the final result since the last MPO tensor 
            # consists of only the first column
            row1 = torch.stack([id, nu, nu, nu, nu], dim=0)
            row2 = torch.stack([sx, nu, nu, nu, nu], dim=0)
            row3 = torch.stack([sy, nu, nu, nu, nu], dim=0)
            row4 = torch.stack([sz, nu, nu, nu, nu], dim=0)
            row5 = torch.stack([-g[i]*sx, -Jx[i%p]*sx, -Jy[i%p]*sy, -Jz[i%p]*sz, id], dim=0)
            O = torch.stack([row1, row2, row3, row4, row5], dim=0)
            if i == 0:
                Os.append(O[None,-1,:,:,:])
            elif i == N-1:
                Os.append(O[:,0,None,:,:])
            else:
                Os.append(O)
        return MPO(Os)
    
    @property
    def h_ops(self):
        sx, sy, sz, id = self.sx, self.sy, self.sz, self.cid
        Jx, Jy, Jz = self.Jx, self.Jy, self.Jz
        g = self.g
        h_list = []
        for i in range(self._N - 1):
            gL = 0.5 * g[i]
            gR = 0.5 * g[i+1]
            if i == 0: # first bond
                gL = g[i]
            if i + 1 == self._N - 1: # last bond
                gR = g[i+1]
            h = - Jx[i] * torch.kron(sx, sx) \
                - Jy[i] * torch.kron(sy, sy) \
                - Jz[i] * torch.kron(sz, sz) \
                - gL * torch.kron(sx, id) \
                - gR * torch.kron(id, sx)
            # h is a matrix with legs ``(i, j), (i*, j*)``
            # reshape to a tensor with legs ``i, j, i*, j*``
            # reshape is carried out in evolution algorithms after exponetiation
            h_list.append(h)
        return h_list
    
    def Liouvillian(self):
        return super().Liouvillian(self.H_full(), *self.L_full())

    def parameters(self):
        """
        Returns the parameters of the model.

        Returns
        -------
        list
            The parameters of the model.
        """
        return {'Jx': self.Jx, 'Jy': self.Jy, 'Jz': self.Jz, 'g': self.g, 'gamma':self.gamma}
    
class dissipative_testmodel(SpinChain):
    """A spin chain model with random local dissipators"""
    def __init__(self, N:int) -> None:
        indices = torch.randint(low=0, high=N, size=(2,))
        self._Lloc = [torch.normal(0, 1, size=(2, 2), dtype=torch.cdouble) \
                      for i in range(N)]
        for idx in indices:
            self._Lloc[idx] = None
        self.indices = indices
        self._N = N

    @property
    def h_ops(self):
        return [torch.zeros((4,4))] * (self._N-1)  # null Hamiltonian --> identity unitaries
    
    @property
    def l_ops(self):
        return self._Lloc