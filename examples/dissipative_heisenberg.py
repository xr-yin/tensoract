import matplotlib.pyplot as plt
import numpy as np
import torch

import os
import sys

tensoractpath = os.path.dirname(os.path.abspath(os.getcwd()))
sys.path.append(os.path.join(tensoractpath, "tensoract"))

def sim_QTP(N:int, Jx:float|list, Jy:float|list, Jz:float|list, g:float|list, gamma:float|list, l_ops:str, ax):
    from qutip import (basis, expect, mesolve, qeye, sigmax, sigmay, sigmaz,tensor)
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

    # Plot the expecation value
    ax.plot(times, exp_sz[0], color='tab:green', linestyle='--', label=r"QTP $\langle \sigma_z^{0} \rangle$")
    ax.plot(times, exp_sz[-1], color='tab:blue', linestyle='--', label=r"QTP $\langle \sigma_z^{-1} \rangle$")

    return H

def sim_TN(N:int, Jx:float|list, Jy:float|list, Jz:float|list, g:float|list, gamma:float|list, l_ops:str, ax):
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

    options = {'disent_step': 1, 'disent_sweep':5}

    lab = LindbladOneSite(psi0, model)
    lab.run(200, 0.5, 10, 4, e_ops=[model.sz,], options=options)
    print(psi0.bond_dims, psi0.krauss_dims)

    times = lab.times
    ax.plot(times, lab.expects[0,:,0], color='tab:green', label=r"TN $\langle \sigma_z^{0} \rangle$")
    ax.plot(times, lab.expects[0,:,-1], color='tab:blue', label=r"TN $\langle \sigma_z^{-1} \rangle$")

    return model

def main():
    N = 5
    Jx = Jy = Jz = 0.1 * torch.pi
    g = 1.0 * torch.pi
    gamma = 0.02
    # jump operators
    # The total spin is conserved in the case of dephasing
    l_ops = "spin loss"

    f = plt.figure(figsize=(8, 8))
    TN_model = sim_TN(N, Jx, Jy, Jz, g, gamma, l_ops, f.gca())

    QTP_model = sim_QTP(N, TN_model.Jx, TN_model.Jy, TN_model.Jz, TN_model.g, TN_model.gamma, l_ops, ax=f.gca())

    assert np.allclose(TN_model.H_full().toarray(), QTP_model.full()), "TN and QTP models do not match"

    plt.legend()
    plt.xlabel("Time")
    plt.ylabel(r"$\langle \sigma_z \rangle$")
    plt.title(fr"{l_ops} with $\gamma$={gamma}")
    plt.savefig("data/heisenberg_loss.pdf")

if __name__ == "__main__":
    main()