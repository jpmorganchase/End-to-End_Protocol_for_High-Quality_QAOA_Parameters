import typing
from dataclasses import dataclass
import numpy as np
import numba

from ..simulator_base import QAOAFastSimulatorBase

from . import csim


@numba.njit(parallel=True)
def combine_complex(real: np.ndarray, imag: np.ndarray) -> np.ndarray:
    return real + 1j * imag


@numba.njit(parallel=True)
def norm_squared(real: np.ndarray, imag: np.ndarray) -> np.ndarray:
    return real**2 + imag**2


@dataclass
class ComplexArray:
    
    real: np.ndarray
    imag: np.ndarray

    def get_complex(self) -> np.ndarray:
        return combine_complex(self.real, self.imag)

    def get_norm_squared(self) -> np.ndarray:
        return norm_squared(self.real, self.imag)


class QAOAFastSimulatorCBase(QAOAFastSimulatorBase):
    def __init__(self, n_qubits: int, costs: typing.Sequence[float]) -> None:
        super().__init__(n_qubits, np.asarray(costs, dtype="float"))

    @property
    def default_sv0(self):
        return ComplexArray(
            np.full(self.n_states, 1.0 / np.sqrt(self.n_states), dtype="float"),
            np.zeros(self.n_states, dtype="float"),
        )

    def _apply_qaoa(self, sv: np.ndarray, gammas: typing.Sequence[float], betas: typing.Sequence[float], **kwargs):
        raise NotImplementedError

    def simulate_qaoa(
        self,
        gammas: typing.Sequence[float],
        betas: typing.Sequence[float],
        sv0: typing.Optional[np.ndarray] = None,
        **kwargs,
    ) -> ComplexArray:
        """
        simulator QAOA circuit using FUR
        """
        sv = ComplexArray(sv0.real.astype("float"), sv0.imag.astype("float")) if sv0 is not None else self.default_sv0
        self._apply_qaoa(sv, gammas, betas, **kwargs)
        return sv


class QAOAFURXYRingSimulatorC(QAOAFastSimulatorCBase):
    def _apply_qaoa(self, sv: np.ndarray, gammas: typing.Sequence[float], betas: typing.Sequence[float], n_trotters: int = 1):
        csim.apply_qaoa_furxy_ring(
            sv.real,
            sv.imag,
            gammas,
            betas,
            self.hc_diag,
            self.n_qubits,
            n_trotters,
        )


class QAOAFURXYCompleteSimulatorC(QAOAFastSimulatorCBase):
    def _apply_qaoa(self, sv: np.ndarray, gammas: typing.Sequence[float], betas: typing.Sequence[float], n_trotters: int = 1):
        csim.apply_qaoa_furxy_complete(
            sv.real,
            sv.imag,
            gammas,
            betas,
            self.hc_diag,
            self.n_qubits,
            n_trotters,
        )


def furxy(sv: typing.Union[ComplexArray, np.ndarray], theta: float, q1: int, q2: int) -> ComplexArray:
    if not isinstance(sv, ComplexArray):
        sv = ComplexArray(sv.real.astype("float"), sv.imag.astype("float"))
    csim.furxy(sv.real, sv.imag, 0.5 * theta, q1, q2)
    return sv
