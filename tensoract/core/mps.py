import torch

class MPS:
    """class for matrix product operators

    Parameters
    ----------
    As : list 
        a list of rank-3 tensors, each tensor has the following shape

        k |
    i---- A ----j

    i (j) is the left (right) bond leg and k is the physical leg
    the legs are ordered as `i, j, k`

    Attributes
    ----------
    As : sequence of Tensors
        as described above

    Methods
    -------
    conj()
    to_array()
    to()
    """
    def __init__(self, As) -> None:
        self.As = As
        self._N = len(As)
        self.device = As[0].device
        self.dtype = As[0].dtype
        self.__bot()

    @classmethod
    def gen_random_mps(cls, N:int, 
                       m_max:int, 
                       phy_dims:list, 
                       *,
                       dtype:torch.dtype=torch.complex128, 
                       device:torch.device='cuda',
                       grad=True) -> 'MPS':
        """Generate a random MPS, with bond dimensions up to m_max (exclusive). When max=2, it is equivalent to a product state."""
        assert len(phy_dims) == N
        bond_dims = torch.randint(1, m_max, size=(N+1,))
        bond_dims[0] = bond_dims[-1] = 1 

        sizes = [(bond_dims[i],bond_dims[i+1],phy_dims[i]) for i in range(N)]

        As = [torch.normal(0, 1, size=size, dtype=dtype, device=device, requires_grad=grad) for size in sizes]

        return cls(As)

    @classmethod
    def allups(cls, N:int, 
               phy_dim:int=2, 
               *,
               dtype:torch.dtype=torch.complex128, 
               device:torch.device='cuda',
               grad=True) -> 'MPS':
        """Generate the all-ups (0=up) MPS state."""
        sizes = [(1,1,phy_dim) for _ in range(N)]
        As = [torch.zeros(size, dtype=dtype, device=device, requires_grad=grad) for size in sizes]
        for A in As:
            A[0,0,0] = 1.0
        return cls(As)

    def copy(self) -> 'MPS':
        return self.__class__([A.clone() for A in self])

    @property
    def bond_dims(self) -> torch.Tensor:
        return torch.tensor([A.shape[0] for A in self] + [self[-1].shape[1]])

    @property
    def physical_dims(self) -> torch.Tensor:
        return torch.tensor([A.shape[2] for A in self])

    def conj(self) -> 'MPS':
        """Return the complex conjugate of the MPS."""
        return MPS([A.conj() for A in self.As])

    def to_array(self) -> torch.Tensor:
        """Convert the MPS to a 1D array."""
        res = torch.tensor(1.0, dtype=self.dtype, device=self.device).reshape(1,1)
        for A in self.As:
            res = torch.tensordot(res, A, dims=([-2],[0]))
        return res.flatten()

    def to(self, *args, **kwargs)  -> None:
        """call torch.tensor.to()"""
        for i in range(self._N):
            self[i] = self[i].to(*args, **kwargs)
        self.device = self[0].device
        self.dtype = self[0].dtype
        
    def __len__(self) -> int:
        return self._N

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.As[index]

    def __setitem__(self, index: int, value: torch.Tensor) -> None:
        self.As[index] = value

    def __iter__(self):
        return iter(self.As)

    def __bot(self) -> None:
        """sanity check"""
        assert self.As[0].shape[0] == 1
        assert self.As[-1].shape[1] == 1
        for i in range(len(self)-1):
            assert self.As[i].shape[1] == self.As[i+1].shape[0]