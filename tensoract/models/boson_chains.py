import torch
from scipy import sparse

from collections.abc import Sequence
import warnings

from ..core import MPO, LPTN
from .base_model import NearestNeighborModel

__all__ = ['BosonChain', 'BoseHubburd']

torch.set_default_dtype(torch.float64)

bid = lambda d: torch.eye(d) + 0j
occup = lambda d: torch.diag(torch.arange(d)) + 0j
create = lambda d: torch.diag(torch.arange(1,d)**0.5, -1) + 0j
annihilate = lambda d: torch.diag(torch.arange(1,d)**0.5, +1) +0j

class BosonChain(NearestNeighborModel):
    """
    1D homogeneous Boson chain model.

    Attributes
    ----------
    N : int
        The number of sites in the Boson chain.
    d : int
        The local dimension of each site.
    nu : torch.Tensor
        A tensor of zeros with shape (d, d) and specified dtype.
    bt : torch.Tensor
        The annihilation operator tensor.
    bn : torch.Tensor
        The creation operator tensor.
    num : torch.Tensor
        The number operator tensor.
    bid : torch.Tensor
        The identity operator tensor.

    Methods
    -------
    H_full()
        Extend local two-site Hamiltonian operators into the full space.
    L_full()
        Extend local one-site Lindblad operators into the full space.
    Liouvillian(H, *Ls)
        Calculate the Liouvillian (super)operator.
    dtype
        Property to get the data type of the tensors.
    __len__()
        Return the number of sites in the Boson chain.
    """

    def __init__(self, 
                 N: int, 
                 d: int, 
                 gamma: float | Sequence[float]=0.,
                 l_ops: torch.Tensor | Sequence[torch.Tensor]=None) -> None:
        """we include the local operators as instance attributes instead of 
        class attributes due to the indeterminacy of local dimensions"""
        self._N = N
        self.d = d
        self.nu = torch.zeros((d,d))
        self.bt = create(d)
        self.bn = annihilate(d)
        self.num = occup(d)
        self.bid = bid(d)
        self.gamma = self.init_onsites(gamma, N)
        self.l_ops = self.init_l_ops(self.gamma, l_ops, N)

    def H_full(self):
        """extend local two-site Hamiltonian operators into full space"""
        N, d = self._N, self.d
        h_full = sparse.csr_matrix((d**N, d**N))
        for i, hh in enumerate(self.h_ops):
            h_full += sparse.kron(sparse.eye(d**i), sparse.kron(hh, torch.eye(d**(N-2-i))))
        return h_full
    
    def L_full(self):
        """extend local one-site Lindblad operators into full space"""
        N, d = self._N, self.d
        Ls = []
        # translational invariant, l_ops is a single operator
        if not isinstance(self.l_ops, Sequence):
            for i in range(N):
                Ls.append(sparse.kron(sparse.eye(d**i), sparse.kron(self.l_ops, torch.eye(d**(N-1-i)))))
        # not translational invariant, l_ops is a list of operators
        else:
            for i, L in enumerate(self.l_ops):
                if L is not None:
                    Ls.append(sparse.kron(sparse.eye(d**i), sparse.kron(L, torch.eye(d**(N-1-i)))))
        return Ls
    
    def Liouvillian(self, H, *Ls):
        """calculate the Liouvillian (super)operator

        Parameters
        ----------
            H : the Hamiltonian in the full Hilbert space
            *L : the Lindblad jump operator(s) in the full Hilbert space

        Return
        ------
        the Liouvillian operator as a sparse matrix
        """
        Lv = self._Hsup(H)
        for L in Ls:
            Lv += self._Dsup(L)
        return Lv
    
    def _Dsup(self, L):
        r"""calculate the :math:`L \otimes \bar{L} - (L^\dagger L\otimes I + I\otimes L^T \bar{L})/2`
        """
        D = self.d**self._N
        return sparse.kron(L,L.conj()) \
            - 0.5*(sparse.kron(L.conj().T@L, sparse.eye(D)) + sparse.kron(sparse.eye(D), L.T@L.conj()))

    def _Hsup(self, H):
        """calculate the Hamiltonian superoperator :math:`-iH \otimes I + iI \otimes H^T`
        """
        D = self.d**self._N
        return - 1j*(sparse.kron(H,sparse.eye(D)) - sparse.kron(sparse.eye(D), H.T))
    
    def __len__(self):
        return self._N
    
class BoseHubburd(BosonChain):
    r"""1D (driven-dissipative) Bose-Hubburd model with Hamiltonian
    .. math ::
        H = - t \sum_{\langle i, j \rangle} (b_i^{\dagger} b_j + b_j^{\dagger} b_i)
            + \sum_{j} \left[ -\mu b_j^\dagger b_j + \frac{U}{2} b_j^\dagger b_j^\dagger b_j b_j 
            + F (b_j^\dagger + b_j \right]

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
                 F: float | Sequence[float] = 0.0, 
                 gamma: float | Sequence[float] = 0.0,
                 l_ops: torch.Tensor | Sequence[torch.Tensor] = None) -> None:
        super().__init__(N, d, gamma, l_ops)
        self.t = self.init_couplings(t, N)
        self.U = self.init_onsites(U, N)
        self.mu = self.init_onsites(mu, N)
        self.F = self.init_onsites(F,N)

    @property
    def h_ops(self):
        bt, bn = self.bt, self.bn
        n, id = self.num, self.bid
        h_list = []
        for i in range(self._N - 1):
            UL = 0.5 * self.U[i]
            UR = 0.5 * self.U[i+1]
            muL = 0.5 * self.mu[i]
            muR = 0.5 * self.mu[i+1]
            FL = 0.5 * self.F[i]
            FR = 0.5 * self.F[i+1]
            if i == 0: # first bond
                UL, muL, FL = self.U[i], self.mu[i], self.F[i]
            if i + 1 == self._N - 1: # last bond
                UR, muR, FR = self.U[i+1], self.mu[i+1], self.F[i+1]
            h = - self.t[i] * (torch.kron(bt, bn) + torch.kron(bn, bt)) \
                - muL * torch.kron(n, id) \
                - muR * torch.kron(id, n) \
                + UL * torch.kron(n@(n-id), id)/2 \
                + UR * torch.kron(id, n@(n-id))/2 \
                + FL * torch.kron(bt, id) \
                + FR * torch.kron(id, bt) \
                + FL.conj() * torch.kron(bn, id) \
                + FR.conj() * torch.kron(id, bn)
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
        
        N = self._N
        p = N - 1
        
        Os = []

        for i in range(N):
            # pseudo-PBC to avoid the out-of-bounds indexing for the couplings J
            # This does not affect the final result since the last MPO tensor 
            # consists of only the first column
            row1 = torch.stack([id, nu, nu, nu], dim=0)
            row2 = torch.stack([bn, nu, nu, nu], dim=0)
            row3 = torch.stack([bt, nu, nu, nu], dim=0)
            row4 = torch.stack([0.5*U[i]*n@(n-id) - mu[i]*n + F[i]*bt + F[i].conj()*bn, 
                                -t[i%p]*bt, -t[i%p]*bn, id], dim=0)
            O = torch.stack([row1, row2, row3, row4], dim=0)
            if i == 0:
                Os.append(O[None,-1,:,:,:])
            elif i == N-1:
                Os.append(O[:,0,None,:,:])
            else:
                Os.append(O)
        return MPO(Os)
    
    def Liouvillian(self):
        return super().Liouvillian(self.H_full(), *self.L_full())

    def energy(self, psi: LPTN):
        """the energy (expectaton value of the Hamiltonian) of the system"""
        assert len(psi) == self._N
        d = psi.physical_dims[0]
        return torch.sum(psi.bond_expectation_value([h.reshape(d,d,d,d).to(psi.dtype) for h in self.h_ops])).real
    
    def fluctuation(self, psi: LPTN, idx: int | None=None, *, normalize=True):
        """the (normalized) fluctuations in the occupation number 
        
        Parameters
        ----------
            psi : LPTN
                the state whose occupation number fluctuation is to be calculated
            idx : int or None
                if idx is None, compute the fluctuations on every site,
                if idx is int, compute the fluctuation only on this site
        
        TODO: calculate the fluctuation of the total occupation number :math:`N = \sum_i n_i` 
        and :math:`\delta N^2 = \langle N^2 \rangle -\langle N \rangle`
        """
        assert len(psi) == self._N
        num = self.num.to(dtype=psi.dtype)
        nums, nums_sq = psi.measure([num, num@num], idx=idx, drop_imag=True)
        if normalize:
            return (nums_sq - nums**2) / nums
        else:
            return nums_sq - nums**2
    
    def parameters(self):
        return {'N': self._N, 
                'd': self.d, 
                't': self.t, 
                'U': self.U, 
                'mu': self.mu, 
                'F': self.F, 
                'gamma': self.gamma} 

class DDBH(BoseHubburd):
    r"""class for driven-dissipative Bose-Hubburd model
    .. math ::
    H = \sum_{j} \left[ -\mu b_j^\dagger b_j 
        + \frac{U}{2} b_j^\dagger b_j^\dagger b_j b_j 
        + F (b_j^\dagger + b_j \right]
        - \t \sum_{\langle i, j \rangle} (b_i^\dagger b_j + b_j^\dagger b_i)"""
    def __init__(self, 
                 N: int, 
                 d: int, 
                 t: float, 
                 U: float, 
                 mu: float, 
                 F: float | Sequence[float], 
                 gamma: float | Sequence[float],
                 l_ops: torch.Tensor | Sequence[torch.Tensor] = None) -> None:
        warnings.warn("DDBH is deprecated and will be removed in future versions. Please use BoseHubburd instead.", DeprecationWarning, stacklevel=2)
        # dtype must be set to double here to ensure accuracy when prepare the 
        # unitary time evolution operator and the Kraus operators
        super().__init__(N, d, t, U, mu, F=F, gamma=gamma, l_ops=l_ops)

class InfiniteDDBH(DDBH):

    def __init__(self, 
                 N: int, 
                 d: int, 
                 t: float, 
                 U: float, 
                 mu: float, 
                 F: float | Sequence[float], 
                 gamma: float | Sequence[float], 
                 dtype: torch.dtype = torch.double) -> None:
        super().__init__(N, d, t, U, mu, F, gamma, dtype)

    @property
    def h_ops(self):
        bt, bn = self.bt, self.bn
        n, id = self.num, self.bid
        t, U, mu, F = self.t, self.U, self.mu, self.F
        h_list = []
        for i in range(self._N):    # N terms instead of N-1 terms
            UL = UR = 0.5 * U
            muL = muR = 0.5 * mu
            FL = 0.5 * F[i]
            FR = 0.5 * F[i+1]
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