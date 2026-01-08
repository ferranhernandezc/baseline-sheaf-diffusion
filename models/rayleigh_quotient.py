import torch
import torch_sparse

def compute_rayleigh_quotient_and_baseline(x_old, Lx, id_L):
    dirichlet_energy = torch.sum(torch.diag(torch.transpose(x_old, 0, 1) @ Lx))
    x_baseline_diffusion = torch_sparse.spmm(id_L[0], id_L[1], x_old.size(0), x_old.size(0), x_old)
    baseline_dirichlet_energy = torch.sum(torch.diag(torch.transpose(x_old, 0, 1) @ x_baseline_diffusion))
    norm = torch.sum(torch.diag(torch.transpose(x_old, 0, 1) @ x_old))
    return dirichlet_energy/norm, baseline_dirichlet_energy/norm
