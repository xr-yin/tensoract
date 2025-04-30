import matplotlib.pyplot as plt
import numpy as np
import torch

import os
import sys

tensoractpath = os.path.dirname(os.path.abspath(os.getcwd()))
sys.path.append(os.path.join(tensoractpath, "tensoract"))

from qutip import (Qobj, basis, expect, entropy_vn, fidelity, hilbert_dist, mesolve, qeye, sigmax, sigmay, sigmaz, tensor)

def sim_QTP(N:int, Jx:list, Jy:list, Jz:list, g:list, gamma:list, l_ops:str, axes):
    # initial state
    state_list = [basis(2, 1)] + [basis(2, 0)] * (N - 1)
    psi0 = tensor(state_list)

    # Setup operators for individual qubits
    sx_list, sy_list, sz_list = [], [], []
    for i in range(N):
        op_list = [qeye(2)] * N
        op_list[i] = sigmax()
        sx_list.append(tensor(op_list))
        op_list[i] = sigmay()
        sy_list.append(tensor(op_list))
        op_list[i] = sigmaz()
        sz_list.append(tensor(op_list))

    # Hamiltonian - Energy splitting terms
    H = 0
    for i in range(N):
        H -= g[i] * sz_list[i]

    # Interaction terms
    for n in range(N - 1):
        H -= Jx[n] * sx_list[n] * sx_list[n + 1]
        H -= Jy[n] * sy_list[n] * sy_list[n + 1]
        H -= Jz[n] * sz_list[n] * sz_list[n + 1]
    
    times = np.linspace(0, 100, 200)

    # collapse operators
    if l_ops == "dephasing":
        c_ops = [np.sqrt(gamma[i]) * sz_list[i] for i in range(N)]
    elif l_ops == "spin loss": 
        c_ops = [np.sqrt(gamma[i]) * (sx_list[i] - 1j*sy_list[i]) * 0.5 for i in range(N)]
    else:
        raise ValueError("Invalid collapse operator. Choose 'dephasing' or 'spin loss'.")

    # evolution
    result = mesolve(H, psi0, times, c_ops, [])

    # Expectation value
    exp_sz = expect(sz_list, result.states)
    # purity
    purity = np.array([(rho*rho).tr() for rho in result.states])
    print('purity', purity[-1])
    # rduced density matrix
    # Be aware that the full density matrix is not a pure state!
    reduced_dm = [rho.ptrace(list(range(N//2))) for rho in result.states]
    # entropy
    entropy_vN = np.array([entropy_vn(rho) for rho in reduced_dm])
    entropy_renyi = np.array([-np.log(np.sum(np.linalg.svdvals(rho.full())**2)) for rho in reduced_dm])

    # Plot the expecation value
    axes[0].plot(times, exp_sz[0], color='tab:green', linestyle='--', label=r"QTP $\langle \sigma_z^{0} \rangle$")
    axes[0].plot(times, exp_sz[-1], color='tab:blue', linestyle='--', label=r"QTP $\langle \sigma_z^{-1} \rangle$")
    # Plot the purity
    axes[1].plot(times, purity, color='tab:orange', linestyle='--', label=r"QTP $\mathcal{P}$")
    # Plot the entropy
    axes[2].plot(times, entropy_vN, color='tab:purple', linestyle='--', label=r"QTP $S_{vN}$")
    axes[2].plot(times, entropy_renyi, color='tab:red', linestyle='--', label=r"QTP $S_2$")

    return H, result.states[1:], times[1:]

def sim_TN(N:int, Jx:float|list, Jy:float|list, Jz:float|list, g:float|list, gamma:float|list, l_ops:str, axes):
    from tensoract import LPTN, SpinChain, Heisenberg, LindbladOneSite
    # initial state
    psi0 = LPTN.gen_polarized_spin_chain(N, polarization='+z')
    psi0[0] = torch.flip(psi0[0], dims=[2])

    if l_ops == "dephasing":
        l_ops=SpinChain.sz
    elif l_ops == "spin loss":
        l_ops=SpinChain.sminus
    else: 
        raise ValueError("Invalid collapse operator. Choose 'dephasing' or 'spin loss'.")
    
    model = Heisenberg(N, Jx, Jy, Jz, g, gamma, l_ops)

    options = {'tol':1e-9, 'disent_step': 1, 'disent_sweep':4, 'store_states': True}

    lab = LindbladOneSite(psi0, model)
    lab.run(200, 0.5, 8, 8, e_ops=[model.sz,], options=options)
    print(psi0.bond_dims, psi0.krauss_dims)

    times = lab.times
    axes[0].plot(times, lab.expects[0,:,0].real, color='tab:green', label=r"TN $\langle \sigma_z^{0} \rangle$")
    axes[0].plot(times, lab.expects[0,:,-1].real, color='tab:blue', label=r"TN $\langle \sigma_z^{-1} \rangle$")
    # Plot the purity
    axes[1].plot(times, lab.purity, color='tab:orange', label=r"TN $\mathcal{P}$")
    # Plot the entropy
    axes[2].plot(times, lab.entropy[:, N//2-1], color='tab:red', label=r"TN $S_2$")

    return model, lab.states

def main(N:int, Jx:float, Jy:float, Jz:float, g:float, gamma:float=0.02, l_ops:str="spin loss"):
    """
    Simulate the dissipative Heisenberg model using both TN and QTP methods.
    Args:
        N (int): Number of spins.
            The inital state is made of first spin pointing up and the rest pointing down.
            The magnetization per spin is therefore (N-2)/ N.
        Jx (float): Coupling constant for x direction.
        Jy (float): Coupling constant for y direction.
        Jz (float): Coupling constant for z direction.
        g (float): Energy splitting.
        gamma (float): Dissipation rate.
        l_ops (str): Type of collapse operator ('dephasing' or 'spin loss').

    Note:
        In case of dephasing, the total spin is conserved.
        In case of spin loss, the system will be driven to a trivial state with all spins pointing down.
        A non-trivial state will exist when Jx != Jy.
    """
    f, axes = plt.subplots(1, 3, sharex=True, figsize=(16, 5))
    TN_model, TN_states = sim_TN(N, Jx, Jy, Jz, g, gamma, l_ops, axes)
    QTP_model, QTP_states, times = sim_QTP(N, TN_model.Jx, TN_model.Jy, TN_model.Jz, TN_model.g, TN_model.gamma, l_ops, axes)

    #assert np.allclose(TN_model.H_full().toarray(), QTP_model.full()), "TN and QTP models do not match"

    # infidelity
    infidelity = np.array([1 - fidelity(Qobj(a.to_density_matrix(), dims=b.dims), b) for a, b in zip(TN_states, QTP_states)])
    hs_dist = np.array([hilbert_dist(Qobj(a.to_density_matrix(), dims=b.dims), b) for a, b in zip(TN_states, QTP_states)])

    print('infidelity', np.max(infidelity), infidelity[-1])
    print('hs_dist', np.max(hs_dist), hs_dist[-1])

    # inset ax
    inset_ax = axes[1].inset_axes([0.5, 0.5, 0.4, 0.4])  # [x, y, width, height]
    inset_ax.semilogy(times, infidelity, color='tab:orange', linestyle='--', label=r"$IF$")
    inset_ax.semilogy(times, hs_dist, color='tab:blue', linestyle='--', label=r"$HS$")
    inset_ax.legend()

    axes[0].set_ylabel(r"$\langle \sigma_z \rangle$")
    axes[1].set_ylabel(r"$\mathcal{P}$")
    axes[2].set_ylabel(r"$S$")

    for ax in axes.flat:
        ax.set_xlabel("Time")
        ax.legend(loc="lower right")

    f.suptitle(fr"{l_ops} with $\gamma$={gamma}")
    f.tight_layout()

    plt.savefig("data/heisenberg_dephasing_4.pdf")

if __name__ == "__main__":
    # small damping, large errors at intermediate times
    Jx = 0.1 * torch.pi
    Jy = 0.1 * torch.pi
    Jz = 0.1 * torch.pi
    g = 1.0 * torch.pi
    main(N=6, Jx=Jx, Jy=Jy, Jz=Jz, g=g, l_ops="dephasing")