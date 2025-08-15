import matplotlib.pyplot as plt
import numpy as np
import torch
from qutip import (Qobj, basis, expect, entropy_vn, fidelity, hilbert_dist, mesolve, qeye, create, destroy, tensor)
from tensoract import LPTN, BosonChain, DDBH, LindbladOneSite

def sim_QTP(N:int, d:int, J:list, U:list, mu:list, F:list, gamma:list, l_ops:str, axes):
    # initial state
    state_list = [basis(d, 1)] + [basis(d, 0)] * (N - 1)
    psi0 = tensor(state_list)

    # Setup operators for individual qubits
    bn_list, bt_list, n_list = [], [], []
    for i in range(N):
        op_list = [qeye(d)] * N
        op_list[i] = destroy(d)
        bn_list.append(tensor(op_list))
        op_list[i] = create(d)
        bt_list.append(tensor(op_list))
        op_list[i] = create(d) * destroy(d)
        n_list.append(tensor(op_list))

    # Hamiltonian
    H = 0

    # energy splitting terms
    for i in range(N):
        H -= mu[i] * n_list[i]
        H += 0.5 * U[i] *  bt_list[i] * bt_list[i] * bn_list[i] * bn_list[i]
        H += F[i] * bt_list[i] + F[i].conj() * bn_list[i]

    # Interaction terms
    for n in range(N - 1):
        H -= J[n] * (bt_list[n] * bn_list[n + 1] + bn_list[n] * bt_list[n + 1])
    
    times = np.linspace(0, 100, 200)

    # collapse operators
    if l_ops == "dephasing":
        c_ops = [np.sqrt(gamma[i]) * n_list[i] for i in range(N)]
    elif l_ops == "photon loss": 
        c_ops = [np.sqrt(gamma[i]) * bn_list[i] for i in range(N)]
    else:
        raise ValueError("Invalid collapse operator. Choose 'dephasing' or 'photon loss'.")

    # evolution
    result = mesolve(H, psi0, times, c_ops, [])

    # Expectation value
    exp_n = expect(n_list, result.states)

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
    axes[0].plot(times, exp_n[0], color='tab:green', linestyle='--', label=r"QTP $\langle n_{0} \rangle$")
    axes[0].plot(times, exp_n[-1], color='tab:blue', linestyle='--', label=r"QTP $\langle n_{-1} \rangle$")

    # Plot the purity
    axes[1].plot(times, purity, color='tab:orange', linestyle='--', label=r"QTP $\mathcal{P}$")

    # Plot the entropy
    axes[2].plot(times, entropy_vN, color='tab:purple', linestyle='--', label=r"QTP $S_{vN}$")
    axes[2].plot(times, entropy_renyi, color='tab:red', linestyle='--', label=r"QTP $S_2$")

    return H, result.states[1:], times[1:]

def sim_TN(N:int, d:int, J:float|list, U:float|list, mu:float|list, F:float|list, gamma:float|list, l_ops:str, axes):
    # start with a vacuum state
    A = torch.zeros([1,1,d,1], dtype=torch.complex128)
    A[0,0,0,0] = 1.
    B = torch.zeros([1,1,d,1], dtype=torch.complex128)
    B[0,0,1,0] = 1.
    psi0 = LPTN([B,] + [A.clone() for _ in range(N-1)])

    if l_ops == "dephasing":
        l_ops = BosonChain(N,d).num
    elif l_ops == "photon loss":
        l_ops = BosonChain(N,d).bn
    else:
        raise ValueError("Invalid collapse operator. Choose 'dephasing' or 'photon loss'.")
    
    model = DDBH(N, d, J, U, mu, F, gamma, l_ops)

    options = {'tol': 1e-9, 'disent_step': 1, 'disent_sweep':8, 'store_states': True}

    lab = LindbladOneSite(psi0, model)
    lab.run(200, 0.5, 8, 8, e_ops=[model.num+0j,], options=options)
    print(psi0.bond_dims, psi0.krauss_dims)

    times = lab.times
    axes[0].plot(times, lab.expects[0,:,0].real, color='tab:green', label=r"TN $\langle n_{0} \rangle$")
    axes[0].plot(times, lab.expects[0,:,-1].real, color='tab:blue', label=r"TN $\langle n_{-1} \rangle$")
    # Plot the purity
    axes[1].plot(times, lab.purity, color='tab:orange', label=r"TN $\mathcal{P}$")
    # Plot the entropy
    axes[2].plot(times, lab.entropy[:, N//2-1], color='tab:red', label=r"TN $S_2$")

    return model, lab.states

def main(N:int, d:int, J:float, U:float, mu:float, F:float, gamma:float, l_ops:str="photon loss"):
    f, axes = plt.subplots(1, 3, sharex=True, figsize=(16, 5))
    TN_model, TN_states = sim_TN(N, d, J, U, mu, F, gamma, l_ops, axes)
    QTP_model, QTP_states, times = sim_QTP(N, d, TN_model.t, TN_model.U, TN_model.mu, TN_model.F, TN_model.gamma, l_ops, axes)

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

    axes[0].set_ylabel(r"$\langle n \rangle$")
    axes[1].set_ylabel(r"$\mathcal{P}$")
    axes[2].set_ylabel(r"$S$")

    for ax in axes.flat:
        ax.set_xlabel("Time")
        ax.legend(loc="lower right")

    f.suptitle(fr"{l_ops} with $\gamma$={gamma}")
    f.tight_layout()

    plt.savefig("data/BH_loss.pdf")

if __name__ == "__main__":
    # small damping, large errors at intermediate times
    main(N=4, d=5, J=0.2, U=1., mu=0.5, F=0.25, gamma=0.3)