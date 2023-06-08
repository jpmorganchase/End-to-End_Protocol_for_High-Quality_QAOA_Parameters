import typing


class QAOAFastSimulatorBase(object):
    def __init__(self, n_qubits: int, costs: typing.Sequence[float]) -> None:
        self.n_qubits = n_qubits
        self.n_states = 2**n_qubits
        self.hc_diag = costs
        assert len(costs) == self.n_states
