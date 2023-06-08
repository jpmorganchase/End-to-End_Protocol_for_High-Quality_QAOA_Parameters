import typing
import numpy as np
import torch 
from .fur_torch import furxy_ring_torch, furxy_complete_torch

def apply_qaoa_furxy_ring_torch(sv: torch.Tensor, gammas: torch.Tensor, betas: torch.Tensor, hc_diag: torch.Tensor, n_qubits: int, n_trotters: int = 1) -> None:
    for gamma, beta in zip(gammas, betas):
        sv = sv * torch.exp(-0.5j * gamma * hc_diag)
        for _ in range(n_trotters):
            sv = furxy_ring_torch(sv, 0.5 * beta / n_trotters, n_qubits)
    return sv

def apply_qaoa_furxy_complete_torch(sv: torch.Tensor, gammas: torch.Tensor, betas: torch.Tensor, hc_diag: torch.Tensor, n_qubits: int, n_trotters: int = 1) -> None:
    for gamma, beta in zip(gammas, betas):
        sv = sv * torch.exp(-0.5j * gamma * hc_diag)
        for _ in range(n_trotters):
            sv = furxy_complete_torch(sv, 0.5 * beta / n_trotters, n_qubits)
    return sv