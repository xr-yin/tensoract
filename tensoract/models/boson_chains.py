import torch
from scipy import sparse

from collections.abc import Sequence

from ..core import MPO, LPTN

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
        if not isinstance(self.Lloc, Sequence):
            for i in range(N):
                Ls.append(sparse.kron(sparse.eye(d**i), sparse.kron(self.Lloc, torch.eye(d**(N-1-i)))))
        else:
            for i, L in enumerate(self.Lloc):
                if L is not None:
                    Ls.append(sparse.kron(sparse.eye(d**i), sparse.kron(L, torch.eye(d**(N-1-i)))))
        return Ls
    
    def Liouvillian(self, H, *Ls):
        """calculate the Liouvillian (super)operator

        Parameters
        ----------
            H: the Hamiltonian in the full Hilbert space
            *L: the Lindblad jump operator(s) in the full Hilbert space

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

    @property
    def dtype(self):
        return self._dtype
    
    def __len__(self):
        return self._N
    
class BoseHubburd(BosonChain):
    r"""1D Bose-Hubburd model with Hamiltonian
    .. math ::
        H = - t \sum_{\langle i, j \rangle} (b_i^{\dagger} b_j + b_j^{\dagger} b_i)
            + \frac{U}{2} \sum_i n_i (n_i - 1) - \mu \sum_i n_i

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
    
    def energy(self, psi: LPTN):
        """the energy (expectaton value of the Hamiltonian) of the system"""
        assert len(psi) == self._N
        d = psi.physical_dims[0]
        return torch.sum(psi.bond_expectation_value([h.reshape(d,d,d,d).to(psi.dtype) for h in self.hduo])).real
    
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
                 dtype: torch.dtype=torch.double) -> None:
        # dtype must be set to double here to ensure accuracy when prepare the 
        # unitary time evolution operator and the Kraus operators
        super().__init__(N, d, t, U, mu, dtype=dtype)
        self.F = F
        self.gamma = gamma

    @property
    def hduo(self):
        bt, bn = self.bt, self.bn
        n, id = self.num, self.bid
        t, U, mu, F = self.t, self.U, self.mu, self.F
        h_list = []
        if not isinstance(F, Sequence):
            F = [F] * self._N
        for i in range(self._N - 1):
            UL = UR = 0.5 * U
            muL = muR = 0.5 * mu
            FL = 0.5 * F[i]
            FR = 0.5 * F[i+1]
            if i == 0: # first bond
                UL, muL, FL = U, mu, F[i]
            if i + 1 == self._N - 1: # last bond
                UR, muR, FR = U, mu, F[i+1]
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
        if not isinstance(F, Sequence):
            F = [F] * self._N
        Os = []
        for i in range(self._N):
            diag = 0.5*U*n@(n-id) - mu*n + F[i]*bt + F[i].conjugate()*bn
            with torch.no_grad():
                row1 = torch.stack([id, nu, nu, nu], dim=0)
                row2 = torch.stack([bn, nu, nu, nu], dim=0)
                row3 = torch.stack([bt, nu, nu, nu], dim=0)
                row4 = torch.stack([diag, -t*bt, -t*bn, id], dim=0)
            O = torch.stack([row1, row2, row3, row4], dim=0)
            if i == 0:
                O= O[None,-1,:,:,:]
            if i == self._N-1:
                O = O[:,0,None,:,:]
            Os.append(O)
        return MPO(Os)
    
    @property
    def Lloc(self):
        """Local Linblad jump operators describing photon losses"""
        if not isinstance(self.gamma, Sequence):
            return self.gamma**0.5 * self.bn
        else:
            return [gamma**0.5 * self.bn for gamma in self.gamma]
    
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
    def hduo(self):
        bt, bn = self.bt, self.bn
        n, id = self.num, self.bid
        t, U, mu, F = self.t, self.U, self.mu, self.F
        h_list = []
        if not isinstance(F, Sequence):
            F = [F] * self._N
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