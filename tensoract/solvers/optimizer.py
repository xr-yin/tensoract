"""This module implements the disentangle algorithm based on PRB 98, 235163 (2018). Specifically, the algorithm 
looks for a fixed point in the two-site unitaries such that minimize the von Neumann entropy on each bonds. Different 
than the original method, the locally purified tensor network is used here. The dimensions of the auxiliary Hilbert 
space can blow up exponentially with time. The disentanglement method can mitigate this problem."""

__author__='Xianrui Yin'

import torch

import logging

from ..core import LPTN, merge, split

__all__ = ['cost', 'optimize', 'single_shot_disentangle']

def cost(u, theta):
    """cost function, sum of the quadrad of singular values

        i     j
       _|_   _|_
    ---| theta |---
        |     |
       |`_`u`_`|
        |     |  
    """
    ml, mr, i, j, kl, kr = theta.shape
    u = torch.reshape(u, (kl, kr, kl, kr))
    Utheta = torch.tensordot(theta, u, dims=([4,5],[2,3]))
    Utheta = torch.transpose(Utheta, (0,2,4,1,3,5))
    # turn theta into a simple matrix
    Utheta = torch.reshape(Utheta, (ml*i*kl, mr*j*kr))
    # compute trace((Utheta Utheta*)^2) * means hermitian conjugate here = dagger
    Utheta2 = Utheta @ Utheta.T.conj()
    return torch.trace(Utheta2 @ Utheta2.T.conj())

def optimize(theta: torch.Tensor, U_start: torch.tensor=None, eps: float = 1e-9, max_iter: int = 20):
    """Disentangle the effective two-site wave function in the Schmidt basis
    
        i     j
       _|_   _|_
    ---| theta |---
      kl|     |kr

       kl*   kr*
        |     |
       |`_`u`_`|
        |     |  
        kl    kr

    the legs of U are aranged as kl, kr, kl*, kr*

    Parameters
    ----------
    theta : ndarray
        effective two-site wave function
    eps : float
        the difference in the second Renyi entropy in two consecutive 
        iterations. The iteration stops if the difference is smaller 
        than eps, default to 1e-9.
    max_iter : int
        maximum iterations, default to 20.

    Return
    ----------
    theta : Tensor
        optimized effective two-site wave function
    U : Tensor, ndim=2
        optimized unitary matrix disentangling theta, this U can be saved 
        for later disentangle step (e.g. after another time step)
    s2 : optimized entropy
    """

    ml, mr, di, dj, kl, kr = theta.size()
    theta = torch.reshape(theta.swapdims(1,2), (ml*di, mr*dj, kl, kr))
    # theta has legs ml, mr, kl, kr

    if U_start is None:
        Uh = torch.eye(kl*kr, dtype=theta.dtype, device=theta.device)
    else:
        Uh = U_start

    s_old = torch.inf

    for i in range(max_iter):
        # compute the  contraction
        rhoL = torch.tensordot(theta, theta.conj(), dims=[(1,3), (1,3)])
        dS = torch.tensordot(rhoL, theta.conj(), dims=[(0,1), (0,2) ])
        dS = torch.tensordot(dS, theta, dims=[(0,2),(0,1)])    # dS has legs kl, kr, kl*, kr*
        dS = torch.reshape(dS, (kl*kr, kl*kr))

        # compute the unitary from a SVD of dS
        w, _, vh = torch.linalg.svd(dS, full_matrices=False)
        uh = w @ vh

        # update Uh
        Uh = uh @ Uh

        # split out the legs
        uh = torch.reshape(uh, (kl,kr,kl,kr))

        # update theta
        theta = torch.tensordot(theta, uh.conj(), dims=([2,3],[2,3]))  # !

        # sum of quadrad of all singular values (singular values square to probability)
        s2 = torch.trace(dS).real
        # compute second Renyi entropy
        s_new = -torch.log(s2)

        if abs(s_old-s_new) < eps:
            logging.info(f'renyi entropy converged at tol={eps} after {i+1} iterations')
            break

        s_old = s_new

    return torch.swapdims(theta.reshape(ml,di,mr,dj,kl,kr), 1, 2), Uh, s_new

def single_shot_disentangle(psi:LPTN, tol:float, m_max:int, eps:float, max_iter:int):
    """a single (back and forth) sweep of the system
    
    Parameters
    ----------
    psi : LPTN
        the state to be disentangled
    tol : float
        largest discarded weights
    m_max : int
        largest matrix bond dimension
    eps : float
        the difference in the second Renyi entropy in two consecutive 
        iterations. The iteration stops if the difference is smaller 
        than eps
    max_iter: int
        maximun iterations
    
    """
    Nbond = len(psi)-1
    psi.orthonormalize('right')
    for i in range(Nbond):
        j = i+1
        theta = merge(psi[i],psi[j])
        #kl, kr = theta.shape[4:]    # debug
        theta1, Uh, S2 = optimize(theta, eps=eps, max_iter=max_iter)
        #logging.debug(torch.linalg.norm(theta1 - torch.tensordot(theta, Uh.reshape(kl,kr,kl,kr).conj(), dims=[(4,5),(2,3)])))
        logging.debug(f'entropy at bond {i}-{j}: {S2}')
        psi[i], psi[j] = split(theta1, 'right', tol, m_max)
    for j in range(Nbond,0,-1):
        i = j-1
        theta = merge(psi[i],psi[j])
        #kl, kr = theta.shape[4:]    # debug
        theta1, Uh, S2 = optimize(theta, eps=eps, max_iter=max_iter)
        #logging.debug(torch.linalg.norm(theta1 - torch.tensordot(theta, Uh.reshape(kl,kr,kl,kr).conj(), dims=[(4,5),(2,3)])))
        logging.debug(f'entropy at bond {i}-{j}: {S2}')
        psi[i], psi[j] = split(theta1, 'left', tol, m_max)