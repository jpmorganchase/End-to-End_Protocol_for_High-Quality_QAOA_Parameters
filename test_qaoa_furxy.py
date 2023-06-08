import numpy as np
import qiskit
from qiskit import QuantumCircuit, execute
from qiskit.providers.aer import AerSimulator
from fur import QAOAFURXYRingSimulatorC, QAOAFURXYCompleteSimulatorC, QAOAFURXYRingSimulator, QAOAFURXYCompleteSimulator
from fur.c.simulator import furxy

def test_rxy_c():
    for N in [4, 5]:
        beta = np.random.uniform(0, np.pi)
        q1 = 0
        q2 = 2

        sv0 = np.zeros(2**N)
        sv0[1] = 1

        sv_fur = furxy(sv0, beta, q1, q2).get_complex()

        qc = QuantumCircuit(N)
        qc.rx(np.pi, 0)

        qc.append(
            qiskit.circuit.library.XXPlusYYGate(
                beta
            ),
            [q1, q2],
        )

        qc.save_state()
        backend = AerSimulator(method="statevector")
        sv_qiskit = execute(qc, backend).result().get_statevector()

        assert sv_qiskit.equiv(sv_fur)


def test_rxy_ring_c():
    for N in [4, 5]:
        beta = np.random.uniform(0, np.pi)
        sim = QAOAFURXYRingSimulatorC(N, np.zeros(2**N))

        sv0 = np.zeros(2**N)
        sv0[1] = 1

        sv_fur = sim.simulate_qaoa([0], [beta], sv0=sv0).get_complex()

        qc = QuantumCircuit(N)
        qc.rx(np.pi, 0)

        for i in range(2):
            for q1 in range(i, N-1, 2):
                qc.append(
                    qiskit.circuit.library.XXPlusYYGate(
                        beta
                    ),
                    [q1, q1+1],
                )
        qc.append(
            qiskit.circuit.library.XXPlusYYGate(
                beta
            ),
            [0, N-1],
        )

        qc.save_state()
        backend = AerSimulator(method="statevector")
        sv_qiskit = execute(qc, backend).result().get_statevector()

        assert sv_qiskit.equiv(sv_fur)


def test_rxy_complete_c():
    for N in [4, 5]:
        beta = np.random.uniform(0, np.pi)
        sim = QAOAFURXYCompleteSimulatorC(N, np.zeros(2**N))

        sv0 = np.zeros(2**N)
        sv0[1] = 1

        sv_fur = sim.simulate_qaoa([0], [beta], sv0=sv0).get_complex()

        qc = QuantumCircuit(N)
        qc.rx(np.pi, 0)

        for q1 in range(N-1):
            for q2 in range(q1+1, N):
                qc.append(
                    qiskit.circuit.library.XXPlusYYGate(
                        beta
                    ),
                    [q1, q2],
                )

        qc.save_state()
        backend = AerSimulator(method="statevector")
        sv_qiskit = execute(qc, backend).result().get_statevector()

        assert sv_qiskit.equiv(sv_fur)


def test_rxy_ring_trotter_c():
    for N in [4, 5]:
        beta = np.random.uniform(0, np.pi)
        n_trotters = 2

        sim = QAOAFURXYRingSimulatorC(N, np.zeros(2**N))

        sv0 = np.zeros(2**N)
        sv0[1] = 1

        sv_fur = sim.simulate_qaoa([0], [beta], sv0=sv0, n_trotters=n_trotters).get_complex()

        qc = QuantumCircuit(N)
        qc.rx(np.pi, 0)


        for _ in range(n_trotters):
            for i in range(2):
                for q1 in range(i, N-1, 2):
                    qc.append(
                        qiskit.circuit.library.XXPlusYYGate(
                            beta / n_trotters
                        ),
                        [q1, q1+1],
                    )
            qc.append(
                qiskit.circuit.library.XXPlusYYGate(
                    beta / n_trotters
                ),
                [0, N-1],
            )


        qc.save_state()
        backend = AerSimulator(method="statevector")
        sv_qiskit = execute(qc, backend).result().get_statevector()

        assert sv_qiskit.equiv(sv_fur)


def test_rxy_complete_trotter_c():
    for N in [4, 5]:
        beta = np.random.uniform(0, np.pi)
        n_trotters = 2

        sim = QAOAFURXYCompleteSimulatorC(N, np.zeros(2**N))

        sv0 = np.zeros(2**N)
        sv0[1] = 1

        sv_fur = sim.simulate_qaoa([0], [beta], sv0=sv0, n_trotters=n_trotters).get_complex()

        qc = QuantumCircuit(N)
        qc.rx(np.pi, 0)

        for _ in range(n_trotters):
            for q1 in range(N-1):
                for q2 in range(q1+1, N):
                    qc.append(
                        qiskit.circuit.library.XXPlusYYGate(
                            beta / n_trotters
                        ),
                        [q1, q2],
                    )

        qc.save_state()
        backend = AerSimulator(method="statevector")
        sv_qiskit = execute(qc, backend).result().get_statevector()

        assert sv_qiskit.equiv(sv_fur)


def test_rxy_ring():
    for N in [4, 5]:
        beta = np.random.uniform(0, np.pi)
        sim = QAOAFURXYRingSimulator(N, np.zeros(2**N))

        sv0 = np.zeros(2**N)
        sv0[1] = 1

        sv_fur = sim.simulate_qaoa([0], [beta], sv0=sv0)

        qc = QuantumCircuit(N)
        qc.rx(np.pi, 0)
        
        for i in range(2):
            for q1 in range(i, N-1, 2):
                qc.append(
                    qiskit.circuit.library.XXPlusYYGate(
                        beta
                    ),
                    [q1, q1+1],
                )
        qc.append(
            qiskit.circuit.library.XXPlusYYGate(
                beta
            ),
            [0, N-1],
        )

        qc.save_state()
        backend = AerSimulator(method="statevector")
        sv_qiskit = execute(qc, backend).result().get_statevector()

        assert sv_qiskit.equiv(sv_fur)


def test_rxy_complete():
    for N in [4, 5]:
        beta = np.random.uniform(0, np.pi)
        sim = QAOAFURXYCompleteSimulator(N, np.zeros(2**N))

        sv0 = np.zeros(2**N)
        sv0[1] = 1

        sv_fur = -1j*sim.simulate_qaoa([0], [beta], sv0=sv0)

        qc = QuantumCircuit(N)
        qc.rx(np.pi, 0)

        for q1 in range(N-1):
            for q2 in range(q1+1, N):
                qc.append(
                    qiskit.circuit.library.XXPlusYYGate(
                        beta
                    ),
                    [q1, q2],
                )

        qc.save_state()
        backend = AerSimulator(method="statevector")
        sv_qiskit = execute(qc, backend).result().get_statevector()

        assert sv_qiskit.equiv(sv_fur)


def test_rxy_ring_trotter():
    for N in [4, 5]:
        beta = np.random.uniform(0, np.pi)
        n_trotters = 2

        sim = QAOAFURXYRingSimulator(N, np.zeros(2**N))

        sv0 = np.zeros(2**N)
        sv0[1] = 1

        sv_fur = sim.simulate_qaoa([0], [beta], sv0=sv0, n_trotters=n_trotters)

        qc = QuantumCircuit(N)
        qc.rx(np.pi, 0)

        for _ in range(n_trotters):
            for i in range(2):
                for q1 in range(i, N-1, 2):
                    qc.append(
                        qiskit.circuit.library.XXPlusYYGate(
                            beta / n_trotters
                        ),
                        [q1, q1+1],
                    )
            qc.append(
                qiskit.circuit.library.XXPlusYYGate(
                    beta / n_trotters
                ),
                [0, N-1],
            )

        qc.save_state()
        backend = AerSimulator(method="statevector")
        sv_qiskit = execute(qc, backend).result().get_statevector()

        assert sv_qiskit.equiv(sv_fur)


def test_rxy_complete_trotter():
    for N in [4, 5]:
        beta = np.random.uniform(0, np.pi)
        n_trotters = 2

        sim = QAOAFURXYCompleteSimulator(N, np.zeros(2**N))

        sv0 = np.zeros(2**N)
        sv0[1] = 1

        sv_fur = sim.simulate_qaoa([0], [beta], sv0=sv0, n_trotters=n_trotters)

        qc = QuantumCircuit(N)
        qc.rx(np.pi, 0)


        for _ in range(n_trotters):
            for q1 in range(N-1):
                for q2 in range(q1+1, N):
                    qc.append(
                        qiskit.circuit.library.XXPlusYYGate(
                            beta / n_trotters
                        ),
                        [q1, q2],
                    )

        qc.save_state()
        backend = AerSimulator(method="statevector")
        sv_qiskit = execute(qc, backend).result().get_statevector()

        assert sv_qiskit.equiv(sv_fur)

from optimizer import circuit_measurement_function
from circuit_utils import get_configuration_cost_kw
from utils import get_problem, get_adjusted_state, precompute_energies_parallel, generate_dicke_state_fast
from functools import partial
def test_po_dicke_ring():
    for N in [4, 5]:
        K = 3
        q = 0.5
        seed = 2053
        scale = 10 
        po_problem = get_problem(N=N,K=K,q=q,seed=seed,pre=scale)
        po_obj = partial(get_configuration_cost_kw, po_problem=po_problem)
        precomputed_energies = get_adjusted_state(precompute_energies_parallel(po_obj, N, 1))
        sim = QAOAFURXYRingSimulatorC(N, po_problem['scale']*precomputed_energies)
        for p in [1,2]:
            for T in [1,2]:
                gammas = np.random.rand(p)
                betas = np.random.rand(p)
                # qaoa_circuit = get_qaoa_circuit(
                #                     po_problem = po_problem,
                #                     gammas = gammas,
                #                     betas = betas,
                #                     depth = p,
                #                     ini="dicke", 
                #                     mixer="t-xy", # ring mixer
                #                     T=T,
                #                     ini_state=None, 
                #                     save_state=True,
                #                     minus=False, #minus for ring mixer
                #                 )
                obj = circuit_measurement_function(
                        po_problem = po_problem,
                        p=p,
                        ini='dicke',
                        mixer='t-xy',
                        T=T,
                        ini_state=None, 
                        save_state=True,
                        n_trials=1024, # number of shots if save_state is False
                        minus=False,
                    )
                ini_x = np.concatenate((gammas,betas),axis=0)
                f1 = obj(ini_x)

                sv = sim.simulate_qaoa(2*gammas, 4*betas, sv0 = generate_dicke_state_fast(N, K), n_trotters=T)
                f2 = sv.get_norm_squared().dot(precomputed_energies)

                assert np.isclose(f1,f2)
            
            