import matplotlib.pyplot as plt
import numpy as np
import torch

import os
import sys

tensoractpath = os.path.dirname(os.path.abspath(os.getcwd()))
sys.path.append(os.path.join(tensoractpath, "tensoract"))
    
from tensoract import LPTN, LindbladOneSite
from qutip import (basis, Qobj, expect, fidelity, hilbert_dist, mesolve, qeye, create, destroy, sigmax, sigmay, sigmaz,tensor)

def sim_QTP(d:int, alpha_l:float, alpha_cc:float, omega_c:float, omega_s:float, gamma:float, axes):
    # initial state
    state_list = [basis(2, 1), basis(d,0), basis(d,3), basis(2, 1)]
    psi0 = tensor(state_list)

    # Setup operators for individual qubits
    sm_list, sp_list, sz_list, cm_list, cp_list= [], [], [], [], []

    for i in range(2):
        op_list = [qeye(2), qeye(d), qeye(d), qeye(2)]

        op_list[i*3] = (sigmax() - 1j*sigmay()) / 2
        sm_list.append(tensor(op_list))

        op_list[i*3] = (sigmax() + 1j*sigmay()) / 2
        sp_list.append(tensor(op_list))

        op_list[i*3] = sigmaz()
        sz_list.append(tensor(op_list))
    
    for i in range(2):
        op_list = [qeye(2), qeye(d), qeye(d), qeye(2)]

        op_list[i+1] = destroy(d)
        cm_list.append(tensor(op_list))

        op_list[i+1] = create(d)
        cp_list.append(tensor(op_list))

    H = 0   # Hamiltonian

    for i in range(2):
        # energy splitting terms
        H += (omega_c * cp_list[i] * cm_list[i] + omega_s * sz_list[i])
        # spin-cavity interaction
        H += alpha_l * (sm_list[i] * cp_list[i] + sp_list[i] * cm_list[i])

    # cavity-cavity interaction
    H += alpha_cc * (cm_list[0] * cp_list[1] + cp_list[0] * cm_list[1])

    times = np.linspace(0, 20, 100)

    # collapse operators
    c_ops = [sm_list[0], cm_list[0], cm_list[1], sm_list[1]]
    c_ops = [np.sqrt(gamma) * op for op in c_ops]

    # evolution
    result = mesolve(H, psi0, times, c_ops, [])

    # Expectation value
    exp = expect([sz_list[0], cp_list[0]*cm_list[0], cp_list[1]*cm_list[1], sz_list[1]], result.states)
    # transform sigma_z from [-1, 1] to [0, 1]
    exp[0] = (exp[0] + 1) / 2
    exp[3] = (exp[3] + 1) / 2
    # purity
    purity = np.array([(rho*rho).tr() for rho in result.states])
    #entropy = np.array([rho.ptrace(list(range(N//2))).entropy() for rho in result.states])
    print('qtp:', [a[0] for a in exp], purity[-1])
    axes[1].plot(times, purity, color='tab:orange', linestyle='--', label=r"QTP $\mathcal{P}$")
    """
    # Plot the expecation value
    axes[0].semilogy(times, exp[0], color='tab:green', linestyle='-', label=r"QTP S1")
    axes[0].semilogy(times, exp[1], color='tab:blue', linestyle='-', label=r"QTP C1")
    axes[0].semilogy(times, exp[2], color='tab:blue', linestyle='--', label=r"QTP C2")
    axes[0].semilogy(times, exp[3], color='tab:green', linestyle='--', label=r"QTP S2")
    axes[0].semilogy(times, np.sum(exp, axis=0), color='tab:purple', linestyle='-', label=r"QTP total")
    # Plot the purity
    axes[1].plot(times, purity, color='tab:orange', linestyle='--', label=r"QTP $\mathcal{P}$")
    # Plot the entropy
    #axes[2].plot(times, entropy, color='tab:red', linestyle='--', label=r"QTP $S$")
    """
    return result.states, times

def sim_TN(d:int, alpha_l:float, alpha_cc:float, omega_c:float, omega_s:float, gamma:float, axes):
    # initial state
    S1 = torch.zeros(size=(1,1,2,1), dtype=torch.complex128)
    S2 = torch.zeros(size=(1,1,2,1), dtype=torch.complex128)
    S1[0,0,1,0] = S2[0,0,1,0] = 1
    C1 = torch.zeros(size=(1,1,d,1), dtype=torch.complex128)
    C2 = torch.zeros(size=(1,1,d,1), dtype=torch.complex128)
    C1[0,0,0,0] = C2[0,0,3,0] = 1
    psi0 = LPTN([S1, C1, C2, S2])

    initial = [psi0.copy(),]

    model = JCM(d, alpha_l, alpha_cc, omega_c, omega_s, gamma)
 
    options = {'tol':1e-12, 'disent_step': 1, 'disent_sweep':4, 'store_states': True}
    e_ops = [(model.sz + model.cid) / 2, model.num, model.num, (model.sz + model.cid) / 2]

    lab = LindbladOneSite(psi0, model)
    lab.run(100, 0.2, 30, 30, e_ops=e_ops, options=options)

    times = lab.times
    axes[0].semilogy(times, lab.expects[:,0], color='tab:green', label="TN S1")
    axes[0].semilogy(times, lab.expects[:,1], color='tab:blue', label="TN C1")
    axes[0].semilogy(times, lab.expects[:,2], color='tab:blue', label="TN C2")
    axes[0].semilogy(times, lab.expects[:,3], color='tab:green', label="TN S2")
    axes[0].semilogy(times, lab.expects.sum(-1), color='tab:purple', label="total")
    axes[0].semilogy(times, 3*torch.exp(-gamma*torch.tensor(times)), label='interpolation')
    # Plot the purity
    axes[1].plot(times, lab.purity, color='tab:orange', label=r"TN $\mathcal{P}$")
    # Plot the entropy
    axes[2].plot(times, lab.entropy[:, 4//2-1], color='tab:red', label=r"TN $S$")

    return initial + lab.states

class JCM():

    dtype = torch.complex128

    def __init__(self, d:int, alpha_l:float, alpha_cc:float, omega_c:float, omega_s:float, gamma:float):
        # parameters
        self.alpha_l = alpha_l
        self.alpha_cc = alpha_cc
        self.omega_c = omega_c
        self.omega_s = omega_s
        self.gamma = gamma
        self.d = d
        # operators
        self.bt = torch.diag(torch.arange(1,d, dtype=torch.double)**0.5, -1) + 0j
        self.bn = torch.diag(torch.arange(1,d, dtype=torch.double)**0.5, +1) + 0j
        self.num = torch.diag(torch.arange(d, dtype=torch.double)) + 0j
        self.bid = torch.eye(d, dtype=self.dtype)
        self.cid = torch.eye(2, dtype=self.dtype)
        self.sz = torch.tensor([[1., 0.], [0., -1.]], dtype=self.dtype)
        self.splus = torch.tensor([[0., 1.], [0., 0.]], dtype=self.dtype)
        self.sminus = torch.tensor([[0., 0.], [1., 0.]], dtype=self.dtype)

    @property
    def h_ops(self):
        S1C1 = self.alpha_l * (torch.kron(self.splus, self.bn) + torch.kron(self.sminus, self.bt)) \
             + self.omega_s * torch.kron(self.sz, self.bid) \
             + self.omega_c * torch.kron(self.cid, self.num)
        C2S2 = self.alpha_l * (torch.kron(self.bn, self.splus) + torch.kron(self.bt, self.sminus)) \
             + self.omega_s * torch.kron(self.bid, self.sz) \
             + self.omega_c * torch.kron(self.num, self.cid)
        C1C2 = self.alpha_cc * (torch.kron(self.bt, self.bn) + torch.kron(self.bn, self.bt)) \

        return [S1C1, C1C2, C2S2]
    
    def H_full(self):
        S1C1, C1C2, C2S2 = self.h_ops
        H = torch.kron(S1C1, torch.kron(self.bid,self.cid)) \
          + torch.kron(self.cid, torch.kron(C1C2,self.cid)) \
          + torch.kron(torch.kron(self.cid, self.bid), C2S2)
        return H
    
    @property
    def l_ops(self):
        ls = [self.sminus, self.bn, self.bn, self.sminus]
        return [self.gamma**0.5 * op for op in ls]
    
    def parameters(self,):
        return None

def main():

    gamma = 0.05
    alpha_l = 0.48
    alpha_cc = -1.0
    omega_c = 1.0
    omega_s = 1.0

    d = 4

    f, axes = plt.subplots(1, 3, figsize=(16, 5))

    TN_states = sim_TN(d, alpha_l, alpha_cc, omega_c, omega_s, gamma, axes)

    QTP_states, times = sim_QTP(d, alpha_l, alpha_cc, omega_c, omega_s, gamma, axes)

    #assert np.allclose(TN_model.H_full().numpy(), QTP_model.full()), "TN and QTP models do not match"

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
    
    axes[0].set_ylabel(r"$\langle N \rangle$")
    axes[1].set_ylabel(r"$\mathcal{P}$")
    axes[2].set_ylabel(r"$S$")

    axes[0].set_ylim(1e-2, d)
    for ax in axes.flat:
        ax.set_xlabel("Time")
        ax.legend(loc="lower right")
    plt.savefig("data/jc_model_r4.pdf")

if __name__ == "__main__":
    main()