"""This module studies the dynamical hysteresis of both the Gutzwiller ansatz and the 
ful locally purified density operator (LPDO)."""

import torch
import matplotlib.pyplot as plt

import os
import sys
import pickle
import logging
logging.basicConfig(level=logging.WARNING)

tensoractpath = os.path.dirname(os.path.abspath(os.getcwd()))
sys.path.append(os.path.join(tensoractpath, "tensoract"))

from tensoract import LPTN, DDBH, LindbladOneSite

def prep_initial_state(N : int, 
                       d : int=5, 
                       mu : float=0.1, 
                       m_max : int=8, 
                       k_max : int=8,
                       start = 'random'):
    """prepare the initial state for the dynamical hysteresis. The initial state
    needs to be a steady state for the chosen parameters (detuning).
    
    Parameters
    ----------
    N : int
        system size
    d : int
        local Hilbert space dimension
    mu : float
        the infimum of detuning
    m_max : int
        max matrix bond dimension, for Gutzwiller ansatz, m_max=1
    k_max : int
        max Kraus dimension
    start : str
        starting state for the preparation
    """    


    if start == 'random':
        # start with a random state
        psi0 = LPTN([torch.rand(size=(1,1,d,1), dtype=torch.complex128) for _ in range(N)])
        psi0.orthonormalize('right')
    else:
        # start with a vacuum state
        A = torch.zeros([1,1,d,1], dtype=torch.complex128)
        A[0,0,0,0] = 1.
        psi0 = LPTN([A.clone() for _ in range(N)])

    model = DDBH(N, d, t=0.2, U=1., mu=mu, F=0.25, gamma=0.3)
    lab = LindbladOneSite(psi0, model)
    
    Nsteps = 25
    dt = 0.8
    options = {'disent_step': 2,
               'disent_sweep': 36}
    
    name = f'init_N={N}'

    for i in range(3):
        lab.run(Nsteps*2**i, 
                dt/2**i, 
                m_max, 
                k_max, 
                e_ops=[model.num+0j,], 
                options=options)
    
    n_avg = lab.expects[0][:,N//4:-(N//4)].real.mean(dim=1)

    plt.plot(lab.times, lab.expects[0][:,N//2].real, label=f'site {N//2}')
    plt.plot(lab.times, n_avg, '--', label='mean')

    plt.xlabel('Time')
    plt.ylabel(r'$\langle n(t) \rangle$')
    plt.title(fr'$\mu={mu}~N={N}~D={m_max}~K={k_max}$')
    plt.legend()
    plt.grid(True)   
    plt.tight_layout()

    plt.savefig('.'.join([name, 'pdf']))
    # save n(t)
    torch.save(torch.stack([torch.tensor(lab.times), n_avg]), '.'.join([name, 'pt']))
    # save final state
    torch.save(psi0, '.'.join([name, 'sv']))

def simple_sweep(mu_range, 
                 psi0 : LPTN,
                 m_max : int, 
                 k_max : int, 
                 Nsteps : int, 
                 dt : float, 
                 dir : str, 
                 device='cpu'):
    
    if device == 'cuda':
        dtype = torch.complex64
    else:
        dtype = torch.complex128

    mu_range = mu_range.numpy()
    N = len(psi0)
    d = psi0.physical_dims[0]
    psi0.to(dtype=dtype, device=device)

    if not os.path.exists(dir):
        os.makedirs(dir)

    options = {'disent_step': 2,
               'disent_sweep': 32}

    model = DDBH(N, d, t=0.2, U=1., mu=0.1, F=0.25, gamma=0.3)
    nt = psi0.site_expectation_value([model.num.to(device, dtype)]*N)[N//4:-(N//4)].real.mean()

    for mode, step in zip(['forward', 'backward'], [1,-1]):

        final_occupations = []
        final_states = []

        for i, mu in enumerate(mu_range[::step]):
            
            mu = round(mu.item(), 2)
            print('mu:', mu)

            if i == 0:
                final_occupations.append(nt)
                final_states.append(psi0.copy())
                continue

            fname = ''.join([dir, mode, str(i)])

            try:
                psi0 = torch.load('.'.join([fname, 'sv']))
            except FileNotFoundError as e:
                model = DDBH(N, d, t=0.2, U=1., mu=mu, F=0.25, gamma=0.3)

                lab = LindbladOneSite(psi0, model)

                for r in range(3):
                    lab.run(Nsteps*2**r, dt/2**r, m_max, k_max, options=options)
            
                torch.save(psi0, '.'.join([fname, 'sv']))

            nt = psi0.site_expectation_value([model.num.to(device, dtype)]*N)[N//4:-(N//4)].real.mean()

            final_occupations.append(nt)
            final_states.append(psi0.copy())

        with open(''.join([dir, mode, '.pkl']), 'wb') as f:
            pickle.dump({'mu_range': mu_range[::step], 
                         'final_states': final_states, 
                         'final_occupations': final_occupations}, f)

def dynamical_gutzwiller():

    N = 50
    dt = 0.8
    m_max, k_max = 1, 8
    for Nsteps in [96, 192]:
        dir = f'/scratch/ge47jac/gw_N={N}/Nstep={Nsteps}/'
        mu_range = torch.arange(0.1, 0.7, 0.02)
        psi0 = torch.load(f'init_N={N}_gw.sv')
        simple_sweep(mu_range, psi0, m_max, k_max, Nsteps, dt, dir)

def dynamical_lpdo(N:int, list_Nsteps:list):

    dt = 0.8
    m_max, k_max = 8, 8
    for Nsteps in list_Nsteps:
        dir = f'/scratch/ge47jac/area_scalingN={N}/Nstep={Nsteps}/'
        mu_range = torch.arange(0.1, 0.5, 0.02)
        psi0 = torch.load(f'init_N={N}.sv')
        simple_sweep(mu_range, psi0, m_max, k_max, Nsteps, dt, dir)

if __name__ == '__main__':
    
    dynamical_lpdo(N=36)