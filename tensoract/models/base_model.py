import torch

__all__ = ['NearestNeighborModel']

class NearestNeighborModel(object):
    """Base class for nearest neighbor models."""
    
    @staticmethod
    def init_couplings(couplings, num_sites):
        """Initialize the couplings."""
        # Check if couplings is a scalar
        if isinstance(couplings, (int, float)):
            # Expand the scalar to a list or tensor of size num_sites
            couplings = couplings * torch.ones(num_sites-1)
        elif hasattr(couplings, '__iter__'):
            if len(couplings) != num_sites-1:
                raise ValueError("Length of couplings must match the number of sites - 1")
        else:
            raise TypeError("Couplings must be a scalar or an iterable")

        return couplings
    
    @staticmethod
    def init_onsites(onsites, num_sites):
        """Initialize the onsite terms."""
        # Check if onsite is a scalar
        if isinstance(onsites, (int, float)):
            # Expand the scalar to a list or tensor of size num_sites
            onsites = onsites * torch.ones(num_sites)
        elif hasattr(onsites, '__iter__'):
            if len(onsites) != num_sites:
                raise ValueError("Length of onsite field strengths must match the number of sites")
        else:
            raise TypeError("Onsite field strengths must be a scalar or an iterable")
        
        return onsites
    
    @staticmethod
    def init_l_ops(gamma, l_ops, num_sites):
        """Initialize the local operators."""
        if l_ops is None:
            return [None] * num_sites
        
        # Check if l_ops is a scalar
        if isinstance(l_ops, torch.Tensor):
            # Expand the single operator to a list or tensors of size num_sites
            l_ops = [l_ops] * num_sites
        elif isinstance(l_ops, list):
            if len(l_ops) != num_sites:
                raise ValueError("Length of local operators must match the number of sites")
        else:
            raise TypeError("Local operators must be a single-site operator or a list of tensors whose size matches the number of sites")
        
        res = []
        for g, op in zip(gamma, l_ops):
            if g==0 or op==None:
                res.append(None)
            else:
                res.append(g**0.5*op)
        
        return res