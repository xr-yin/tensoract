import torch
from torch.linalg import qr

import logging

from .projection import *

import gc

__all__ = ['qr_step', 'rq_step', 'split', 'merge', 'mul', 'inner', 'apply_mpo']


def qr_step(ls: torch.Tensor, rs: torch.Tensor):
    r"""Move the orthogonality center one site to the right.
    
    Given two neighboring MPO tensors as following,
    \        2,k           2,k       \
    \         |             |        \
    \     ----ls----    ----rs----   \
    \     0,i |  1,j    0,i | 1,j    \
    \        3,l           3,l       \
    compute the QR decompostion of ls, and multiply r with rs.

    Parameters
    ----------
    ls : torch.Tensor, ndim==4
        local MPS tensor on the left, to be QR decomposed
    rs : torch.Tensor, ndim==4
        local MPS tensor on the right

    Return
    --------
    ls_new : torch.Tensor, ndim==4
        left orthonormal MPS tensor
    rs_new : torch.Tensor, ndim==4
        new orthogonality cneter
    """
    di, dj, dk, dl = ls.size()
    ls = ls.swapaxes(1,3).reshape(-1,dj) # stick i,k,l together, first need to switch j,k
    # compute QR decomposition of the left matrix
    ls_new, _r = qr(ls)
    ls_new = ls_new.view(di,dl,dk,-1).swapaxes(3,1)
    # multiply matrix R into the right matrix
    rs_new = torch.tensordot(_r, rs, dims=1)
    return ls_new, rs_new

def rq_step(ls: torch.Tensor, rs: torch.Tensor):
    r"""Move the orthogonality center one site to the left.
    
    Given two neighboring MPO tensors as following,
    \        2,k           2,k       \
    \         |             |        \
    \     ----ls----    ----rs----   \
    \     0,i |  1,j    0,i | 1,j    \
    \        3,l           3,l       \
    compute the QR decompostion of ls, and multiply r with rs.

    Parameters
    ----------
    ls : ndarray, ndim==3 (or 4)
        local MPS tensor on the left
    rs : ndarray, ndim==3 (or 4)
        local MPS tensor on the right, to be RQ decomposed

    Return
    ----------
    ls_new : ndarray, ndim==3 (or 4)
        new orthogonality cneter
    rs_new : ndarray, ndim==3 (or 4)
        right orthonormal MPS tensor
    """
    di, dj, dk, dl = rs.size()
    rs = rs.reshape(di,-1).T
    # compute RQ decomposition of the right matrix
    rs_new, _r = qr(rs)
    rs_new = rs_new.T.view(-1,dj,dk,dl)
    # multiply matrix R into the left matrix
    ls_new = torch.tensordot(ls, _r, dims=([1],[1])).permute(0,3,1,2)   
    return ls_new, rs_new

def split(theta: torch.Tensor, mode:str, tol:float=0., m_max:int=None, renormalize:bool=True):
    '''
    split a local  6-tensor into two parts by doing a SVD 
    and discard the impertinent singular values
    '''
    if mode not in ["left", "right", "sqrt"]:
        raise ValueError('unknown mode')
    
    di, dj, dk1, dk2, dl1, dl2 = theta.size()
    theta = theta.permute(0,2,4,1,3,5).reshape(di*dk1*dl1, dj*dk2*dl2)

    theta1, s, theta2 = torch.linalg.svd(theta, full_matrices=False)
    
    # sum over the mask
    keep = torch.sum(s**2 > tol*torch.linalg.norm(s))
    pivot = min(keep, m_max) if m_max else keep
    theta1, s, theta2 = theta1[:,:pivot], s[:pivot], theta2[:pivot,:]
    if renormalize:
        s = s / torch.linalg.norm(s)
    if mode == 'left':
        theta1 *= s
    elif mode == 'right':
        theta2 = torch.diag(s.to(theta2.dtype)) @ theta2
    else:
        _s = torch.sqrt(s)
        theta1 *= _s
        theta2 = torch.diag(_s.to(theta2.dtype)) @ theta2

    theta1 = theta1.view(di,dk1,dl1,-1).permute(0,3,1,2)
    theta2 = theta2.view(-1,dj,dk2,dl2)
    return theta1, theta2

def merge(theta1, theta2):
    '''
    merge two local tensors into one, the result has the following shape

            k1          k2             
            |           |
      i---theta1--//--theta2---j  
            |           |      
           (l1)        (l2)
    
                k1    k2
                |     |
          i------theta------j
                |     |
               (l1)  (l2)
    '''
    theta = torch.tensordot(theta1, theta2, dims=([1],[0]))
    theta = theta.permute(0,3,1,4,2,5) # i,j,k1,k2,l1,l2
    return theta

def mul(A, B):
    """
    Calculate the product of two MPOs by direct contraction. The dimensions 
    of the bonds will simply multiply. 

    Parameters:
        A: a MPO
        B: a MPO

    Return:
        A x B: MPO
        
                    |k   output
                ----A----
    A x B   =       |k*, 3
                    |k , 2
                ----B----
                    |k*  input
    """
    
    from .mpo import MPO
    Os = []
    for a, b in zip(A, B): 
        a0, a1, a2 = a.shape[:3]
        b0, b1 = b.shape[:2]
        O = torch.tensordot(a, b, dims=([3],[2]))
        O = torch.swapaxes(O, 1, 3)
        O = O.reshape(a0*b0, a2, a1*b1, -1).swapaxes(1, 2)
        Os.append(O)
    return MPO(Os)

def inner(ampo, bmpo):
    """Evaluating the inner product of two MPOs by bubbling, complexity=O(D^3)

    Parameters
    ----------
    amps : MPO
        the bra MPO
    bmps : MPO
        the ket MPO

    Return
    ----------
    the inner product <ampo|bmpo>
    """
    assert len(ampo) == len(bmpo)
    N = len(ampo)
    res = torch.tensordot(ampo[0].conj(),bmpo[0],dims=([0,2,3],[0,2,3]))
    for i in range(1,N):
        res = torch.tensordot(ampo[i].conj(), res, dims=([0],[0]))
        res = torch.tensordot(res, bmpo[i], dims=([1,2,3],[2,3,0]))
    return res.squeeze()

def apply_mpo(O, psi, tol: float, m_max: int, max_sweeps: int = 2, overwrite: bool = True):
    """Varationally calculate the product of a MPO and another MPO/LPTN.

    psi is modified in place, the result is the product O|psi>  and has the 
    same dtype as psi and O       
                    |k   output
                ----O----
    O |psi> =       |k*, 3
                    |k , 2
                ---psi----
                    |

    Parameters
    ----------
    O : MPO
        the operator
    psi : MPO, LPTN 
        the operand
    tol : float
        largest discarded singular value in each truncation step
    m_max : int 
        largest bond dimension allowed, default is None
    max_sweeps : int
        maximum number of optimization sweeps

    Return
    ------
    norm : float
        the optimized inner product <phi|O|psi> where |phi> ~ O|psi>
        The alias 'overlap' is only appropriate when |psi> is a unit 
        vector and O is unitary. In this case, <phi|O|psi> evaluates 
        to 1 whenever phi is a good approximation. In general cases, 
        'norm square' would be a better term since <phi|O|psi> ~ 
        <phi|phi>.
    """    
    
    N = len(psi)
    phi = psi.copy()    
    # assume psi is only changed slightly, useful when O is a time 
    # evolution operator for a small time step 
    phi.orthonormalize('right') 
    Rs = RightBondTensors(N, dtype=O[0].dtype, device=O[0].device)  
    Ls = LeftBondTensors(N, dtype=O[0].dtype, device=O[0].device)
    Rs.load(phi, psi, O)
    if psi.bond_dims[1:-1].min() >= m_max:
        # perform one-site variational update
        logging.info('perform one-site variational update')
        for n in range(max_sweeps):
            for i in range(N-1): # sweep from left to right
                eff_O = ProjOneSite(Ls[i], Rs[i], O[i])
                phi[i] = eff_O._matvec(psi[i])
                phi[i], phi[i+1] = qr_step(phi[i], phi[i+1])
                # update the left bond tensor LBT[j]
                Ls.update(i+1, phi[i].conj(), psi[i], O[i])
            phi[-1] /= torch.linalg.norm(phi[-1])
            for j in range(N-1,0,-1): # sweep from right to left
                eff_O = ProjOneSite(Ls[j], Rs[j], O[j])
                phi[j] = eff_O._matvec(psi[j])
                phi[j-1], phi[j] = rq_step(phi[j-1], phi[j])
                # update the right bond tensor RBT[i]
                Rs.update(j-1, phi[j].conj(), psi[j], O[j])
            phi[0] /= torch.linalg.norm(phi[0])
    else:
        # perform two-site variational update
        logging.info('perform two-site variational update')
        for n in range(max_sweeps):
            for i in range(N-1): # sweep from left to right
                j = i+1
                x = merge(psi[i], psi[j])
                eff_O = ProjTwoSite(Ls[i], Rs[j], O[i], O[j])
                x = eff_O._matvec(x)
                # split the result tensor
                phi[i], phi[j] = split(x, 'right', tol, m_max)
                # update the left bond tensor LBT[j]
                Ls.update(j, phi[i].conj(), psi[i], O[i])
            for j in range(N-1,0,-1): # sweep from right to left
                i = j-1
                x = merge(psi[i], psi[j])
                # contracting left block LBT[i]
                eff_O = ProjTwoSite(Ls[i], Rs[j], O[i], O[j])
                x = eff_O._matvec(x)
                # split the result tensor
                phi[i], phi[j] = split(x, 'left', tol, m_max)
                # update the right bond tensor RBT[i]
                Rs.update(i, phi[j].conj(), psi[j], O[j])
            
    norm = torch.tensordot(phi[0].conj(), Rs[0], dims=([1],[0]))
    norm = torch.tensordot(norm, O[0], dims=([1,3],[2,1]))
    norm = torch.tensordot(norm, psi[0], dims=([2,4,1],[1,2,3]))
    norm = norm.item()
    logging.info(f'norm after {n+1} sweep(s): {norm}')

    if overwrite:
        psi.As = phi.As
        del Rs
        del Ls
        del phi
        gc.collect()
        torch.cuda.empty_cache()
        return norm
    else:
        del Rs
        del Ls
        gc.collect()
        torch.cuda.empty_cache()
        return phi, norm