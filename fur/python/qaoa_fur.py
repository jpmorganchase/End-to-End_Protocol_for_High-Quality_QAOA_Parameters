import typing
import numpy as np
from .fur import furxy_ring, furxy_complete

def apply_qaoa_furxy_ring(sv: np.ndarray, gammas: typing.Sequence[float], betas: typing.Sequence[float], hc_diag: np.ndarray, n_qubits: int, n_trotters: int = 1) -> None:
    for gamma, beta in zip(gammas, betas):
        sv *= np.exp(-0.5j * gamma * hc_diag)
        for _ in range(n_trotters):
            furxy_ring(sv, 0.5 * beta / n_trotters, n_qubits)


def apply_qaoa_furxy_complete(sv: np.ndarray, gammas: typing.Sequence[float], betas: typing.Sequence[float], hc_diag: np.ndarray, n_qubits: int, n_trotters: int = 1) -> None:
    for gamma, beta in zip(gammas, betas):
        sv *= np.exp(-0.5j * gamma * hc_diag)
        for _ in range(n_trotters):
            furxy_complete(sv, 0.5 * beta / n_trotters, n_qubits)
