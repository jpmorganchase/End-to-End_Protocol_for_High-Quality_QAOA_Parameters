import typing
import torch

class QAOAFastSimulatorBase_torch(object):
    def __init__(self, n_qubits: int, costs: torch.Tensor) -> None:
        self.n_qubits = n_qubits
        self.n_states = 2**n_qubits
        self.hc_diag = costs
        assert len(costs) == self.n_states
