import typing
import numpy as np
import torch 
from ..simulator_base_torch import QAOAFastSimulatorBase_torch
from .qaoa_fur_torch import apply_qaoa_furxy_complete_torch, apply_qaoa_furxy_ring_torch


class QAOAFastSimulatorPythonBase_torch(QAOAFastSimulatorBase_torch):

    @property
    def default_sv0(self):
        return torch.full(self.n_states, 1.0 / torch.sqrt(self.n_states), dtype="torch.complex128")

    def _apply_qaoa(self, sv: torch.Tensor, gammas: torch.Tensor, betas: torch.Tensor, **kwargs):
        raise NotImplementedError

    def simulate_qaoa(
        self,
        gammas: torch.Tensor,
        betas: torch.Tensor,
        sv0: typing.Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        simulator QAOA circuit using FUR
        """
        # sv = torch.Tensor(sv0,dtype="torch.complex128") if sv0 is not None else self.default_sv0
        sv = sv0.type(torch.complex128)
        sv = self._apply_qaoa(sv, gammas, betas, **kwargs)
        return sv


class QAOAFURXYRingSimulator_torch(QAOAFastSimulatorPythonBase_torch):
    
    def _apply_qaoa(self, sv: torch.Tensor, gammas: torch.Tensor, betas: torch.Tensor, n_trotters: int = 1):
        sv = apply_qaoa_furxy_ring_torch(sv, gammas, betas, self.hc_diag, self.n_qubits, n_trotters=n_trotters)
        return sv
        


class QAOAFURXYCompleteSimulator_torch(QAOAFastSimulatorPythonBase_torch):
    
    def _apply_qaoa(self, sv: torch.Tensor, gammas: torch.Tensor, betas: torch.Tensor, n_trotters: int = 1):
        sv = apply_qaoa_furxy_complete_torch(sv, gammas, betas, self.hc_diag, self.n_qubits, n_trotters=n_trotters)
        return sv