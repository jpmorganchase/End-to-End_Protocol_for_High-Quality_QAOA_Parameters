import qiskit
from qiskit import QuantumCircuit, execute, Aer, QuantumRegister
import numpy as np
import scipy.linalg as LA
from scipy.linalg import expm
from utils import (
    invert_counts,
    get_adjusted_state,
    state_to_ampl_counts,
    convert_bitstring_to_int,
    exact_fidelity,
    get_ring_xy_mixer,
    get_constrained_eigenpair,
    generata_dicke_state,
)

def get_cost_circuit(po_problem, qc, gamma):
    """
    Construct the problem Hamiltonian layer for QAOA circuit
    H = 0.5*q\sum_{i=1}^{n-1} \sum_{j=i+1}^n \sigma_{ij}Z_i Z_j + 0.5 \sum_i (-q\sum_{j=1}^n{\sigma_ij} + \mu_i) Z_i + Constant
    """
    q = po_problem["q"]
    means = po_problem["means"]
    cov = po_problem["cov"]
    N = po_problem["N"]
    for i in range(N):
        qc.rz( (means[i] - q * np.sum(cov[i, :])) * gamma, i) #there is a 0.5 inside rz and rzz
    for i in range(N-1):
        for j in range(i + 1, N):
            qc.rzz(q * cov[i, j] * gamma, i, j) 
    return qc

def get_configuration_cost_slow(po_problem, config):
    """
    Compute energy for single sample configuration
    f(x) = q \sigma_ij x_i x_j - \mu_i x_i
    """
    scale = po_problem["scale"]
    means = po_problem["means"] / scale
    cov = po_problem["cov"] / scale
    q = po_problem["q"]
    N = po_problem["N"]
    
    config = convert_bitstring_to_int(config)
    c_sigma = 0.0
    for i in range(N):
        for j in range(N):
            c_sigma += q * cov[i, j] * config[i] * config[j]

    c_mu = 0.0
    for i in range(N):
        c_mu += means[i] * config[i]

    cost = c_sigma - c_mu
    return cost

def get_configuration_cost(po_problem, config):
    """
    Compute energy for single sample configuration
    f(x) = q \sigma_ij x_i x_j - \mu_i x_i
    """
    if not isinstance(config, np.ndarray):
        config = convert_bitstring_to_int(config)
    scale = po_problem["scale"]
    means = po_problem["means"] / scale
    cov = po_problem["cov"] / scale
    return po_problem["q"] * config.dot(cov).dot(config) - means.dot(config)

def get_configuration_cost_kw(config, po_problem=None):
    """
    Convenience function for functools.partial
    e.g. po_obj = partial(get_configuration_cost, po_problem=po_problem)
    """
    return get_configuration_cost(po_problem, config)

### Initial State Design
def get_dicke_init(N, K):
    from dicke_state_utils import dicke_simple
    # can be other dicke state implementaitons
    return dicke_simple(N, K)

def get_dicke_from_circuit(N, K):
    circ = get_dicke_init(N, K)
    samples = measure_circuit(circ, save_state=True)
    return samples

def get_uniform_init(N):
    qc = QuantumCircuit(N)
    for i in range(N):
        qc.h(i)
    return qc

def get_ring_gs_ini(N, K):
    """Obtain the subspace GS of ring mixer"""
    H_ring = get_ring_xy_mixer(N, ring=True)
    _, eigen_vector = get_constrained_eigenpair(H_ring, N, K)
    desired_vector = eigen_vector[0] #ring_gs take the v[:,0], then minus should be False
    q = QuantumRegister(N)
    qc = QuantumCircuit(q)
    qc.initialize(desired_vector, [q[i] for i in range(N)])
    return qc

### Mixer Design
def get_mixer_grover(qc, beta, N, K):
    F = generata_dicke_state(N=N, K=K)
    H = np.outer(F, F)
    U = expm(-1j * beta * H)
    qc.unitary(U, range(N))

    return qc


def get_mixer_Txy(qc, beta, minus=False, T=None):
    """
    H_{even} = \sum_{i is even} (X_i X_{i+1} + Y_i Y_{i+1})
    H_{odd} = \sum_{i is odd} (X_i X_{i+1} + Y_i Y_{i+1})
    H_{last} = X_{end} X_0 + Y_{end} Y_0
    U = {exp[-i*angle*H_{even}] exp[-i*angle*H_{odd}] exp[-i*angle*H_{last}]} ^ T  #repeat T times
    """
    if minus == True:
        beta = -beta
    N = len(qc._qubits)
    if T is None:
        T = 1
    beta = beta / T
    for _ in range(int(T)):
        for i in range(0, N - 1, 2):
            # even Exp[-j*angle*(XX+YY)]
            # qc.append(qiskit.circuit.library.XXPlusYYGate(4 * beta), [i, i + 1])
            qc.append(qiskit.circuit.library.XXPlusYYGate(4 * beta), [N-2-i, N-1-i])
        for i in range(1, N - 1, 2):
            # odd Exp[-j*angle*(XX+YY)]
            # qc.append(qiskit.circuit.library.XXPlusYYGate(4 * beta), [i, i + 1])
            qc.append(qiskit.circuit.library.XXPlusYYGate(4 * beta), [N-2-i, N-1-i])
        # last uniary
        qc.append(qiskit.circuit.library.XXPlusYYGate(4 * beta), [N - 1, 0])
    return qc

def apply_mixer_Txy(sim, beta, minus=False, T=None):
    """
    Helper function for tests
    
    H_{even} = \sum_{i is even} (X_i X_{i+1} + Y_i Y_{i+1})
    H_{odd} = \sum_{i is odd} (X_i X_{i+1} + Y_i Y_{i+1})
    H_{last} = X_{end} X_0 + Y_{end} Y_0
    U = {exp[-i*angle*H_{even}] exp[-i*angle*H_{odd}] exp[-i*angle*H_{last}]} ^ T  #repeat T times
    """
    if minus == True:
        beta = -beta
    N = sim.n_qubits
    if T is None:
        T = 1
    beta = beta / T
    for _ in range(int(T)):
        for i in range(0, N - 1, 2):
            # even Exp[-j*angle*(XX+YY)]
            sim.apply_rxy(i, i + 1, 4*beta)
        for i in range(1, N - 1, 2):
            # odd Exp[-j*angle*(XX+YY)]
            sim.apply_rxy(i, i + 1, 4*beta)
        # last uniary
        sim.apply_rxy(0, N - 1, 4*beta)

def apply_mixer_Txy_yue(sim, beta, minus=False, T=None):
    """
    Helper function for tests
    
    H_{even} = \sum_{i is even} (X_i X_{i+1} + Y_i Y_{i+1})
    H_{odd} = \sum_{i is odd} (X_i X_{i+1} + Y_i Y_{i+1})
    H_{last} = X_{end} X_0 + Y_{end} Y_0
    U = {exp[-i*angle*H_{even}] exp[-i*angle*H_{odd}] exp[-i*angle*H_{last}]} ^ T  #repeat T times
    """
    if minus == True:
        beta = -beta
    N = sim.n_qubits
    if T is None:
        T = 1
    beta = beta / T
    for _ in range(int(T)):
        for i in range(2):
            for q1 in range(i, N+i, 2):
                sim.apply_rxy(q1, (q1+1)%N, beta)        
        
def get_mixer_exactXY(qc, beta, minus=False):
    """
    H = \sum_i (X_i X_{i+1} + Y_i Y_{i+1})
    U = exp[-i*angle*H]
    """
    N = len(qc._qubits)
    I = np.array([[1, 0], [0, 1]],dtype=complex)
    X = np.array([[0, 1], [1, 0]],dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]],dtype=complex)
    Z = np.array([[1, 0], [0, -1]],dtype=complex)
    H = np.zeros((2**N, 2**N),dtype=complex)  
    for i in range(N - 1):
        Hx, Hy = 1, 1
        for _ in range(i):
            Hx = np.kron(Hx, I)
            Hy = np.kron(Hy, I)
        Hx = np.kron(np.kron(Hx, X), X)
        Hy = np.kron(np.kron(Hy, Y), Y)
        for _ in range(i + 2, N):
            Hx = np.kron(Hx, I)
            Hy = np.kron(Hy, I)
        H = H + Hx + Hy
    H_x_bond, H_y_bond = X, Y
    for i in range(N - 2):
        H_x_bond = np.kron(H_x_bond, I)
        H_y_bond = np.kron(H_y_bond, I)
    H_x_bond = np.kron(H_x_bond, X)
    H_y_bond = np.kron(H_y_bond, Y)
    H = H + H_x_bond + H_y_bond
    if minus == True:
        H = -H
    U = expm(-1j * beta * H)
    qc.unitary(U, range(N))
    return qc


def get_mixer_exact_complete(qc, beta, minus=False):
    """
    H = \sum_{i,j=i+1} (X_i X_{j} + Y_i Y_{j})
    U = exp[-i*angle*H]
    """
    N = len(qc._qubits)
    I = np.array([[1, 0], [0, 1]])
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    H = np.zeros((2**N, 2**N))
    for i in range(N - 1):
        for j in range(i + 1, N):
            Hx, Hy = 1, 1
            for _ in range(i):
                Hx = np.kron(Hx, I)
                Hy = np.kron(Hy, I)
            Hx = np.kron(Hx, X)
            Hy = np.kron(Hy, Y)
            for _ in range(i + 1, j):
                Hx = np.kron(Hx, I)
                Hy = np.kron(Hy, I)
            Hx = np.kron(Hx, X)
            Hy = np.kron(Hy, Y)
            for _ in range(j + 1, N):
                Hx = np.kron(Hx, I)
                Hy = np.kron(Hy, I)
            H = H + Hx + Hy
    if minus == True:
        H = -H
    U = expm(-1j * beta * H)
    qc.unitary(U, range(N))
    return qc
   
def get_mixer_exact_partial_complete(qc, beta, chains, minus=False):
    """
    chains: list of list
    H = \sum_{i,j=i+1} (X_i X_{j} + Y_i Y_{j}) with i,j in one list of chains
    U = exp[-i*angle*H]
    """
    N = len(qc._qubits)
    I = np.array([[1, 0], [0, 1]],dtype=complex)
    X = np.array([[0, 1], [1, 0]],dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]],dtype=complex)
    Z = np.array([[1, 0], [0, -1]],dtype=complex)
    H = np.zeros((2**N, 2**N),dtype=complex)    
    n_chain = len(chains)
    ###
    for i_chain in range(n_chain):
        chain = chains[i_chain]
        for i in range(N):
            if i != N - 1:
                edge = [chain[i], chain[i + 1]]
                Hx, Hy = 1, 1
                for j in range(N):
                    if j == edge[0] or j == edge[1]:
                        Hx = np.kron(Hx, X)
                        Hy = np.kron(Hy, Y)
                    else:
                        Hx = np.kron(Hx, I)
                        Hy = np.kron(Hy, I)
                H = H + Hx + Hy
    if minus == True:
        H = -H
    U = expm(-1j * beta * H)
    qc.unitary(U, range(N))
    return qc

def get_mixer_RX(qc, beta):
    """ A layer of RX gates """
    N = len(qc._qubits)
    for i in range(N):
        qc.rx(2 * beta, i)
    return qc


def get_mixer_complete_chaintrotter_exact(qc, beta, minus=False):
    """
    U = {exp[-i*angle*H_chain]} ^ N_chain
    H_chain = \sum_{i,j in chain} (X_i X_{j} + Y_i Y_{j}) is implemented exactly
    """
    N = len(qc._qubits)
    I = np.array([[1, 0], [0, 1]])
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    assert N % 2 == 0  # for even complete graph only
    chains = []
    chain = [0, 1]
    for i in range(N // 2 - 1):
        chain.append(N - 1 - i)
        chain.append(2 + i)
    chains.append(chain)
    for i in range(N // 2 - 1):
        chain = [i + 1 for i in chains[-1]]
        for (idx, item) in enumerate(chain):
            if item > N - 1:
                chain[idx] = chain[idx] - N
        chains.append(chain)
    for chain in chains:
        H = np.zeros((2**N, 2**N))
        for i in range(N):
            if i != N - 1:
                edge = [chain[i], chain[i + 1]]
            Hx, Hy = 1, 1
            for j in range(N):
                if j == edge[0] or j == edge[1]:
                    Hx = np.kron(Hx, X)
                    Hy = np.kron(Hy, Y)
                else:
                    Hx = np.kron(Hx, I)
                    Hy = np.kron(Hy, I)
            H = H + Hx + Hy
        if minus == True:
            H = -H
        U = expm(-1j * beta * H)
        qc.unitary(U, range(N))  
    return qc

def get_mixer_partial_Tchaintrotter_1xy(qc, beta, chains, T=None, minus=False):
    """
    H_chain = \sum_{i,j in chain} (X_i X_{j} + Y_i Y_{j})
    U = {exp[-i*angle*H_chain]} ^ N_chain  # for N-size complete graph, N_chain = N // 2
    exp[-i*angle*H_chain] is implemented by parity troterrization (no boundary condition)
    """
    if minus == True:
        beta = -beta
    N = len(qc._qubits)
    if T is None:
        T = 1
    beta = beta / T
    assert N % 2 == 0  # for even complete graph only
    for _ in range(int(T)):
        for chain in chains:
            for i in range(0, N - 1, 2):
                # even
                edge = [N-1-chain[i], N-1-chain[i + 1]]
                qc.append(
                    qiskit.circuit.library.XXPlusYYGate(
                        4 * beta
                    ),
                    [edge[0], edge[1]],
                )
            for i in range(1, N - 1, 2):
                # odd
                edge = [N-1-chain[i], N-1-chain[i + 1]]
                qc.append(
                    qiskit.circuit.library.XXPlusYYGate(4 * beta),
                    [edge[0], edge[1]],
                )
    return qc
 
def get_mixer_complete_Tchaintrotter_1xy(qc, beta, T=None, minus=False):
    """
    H_chain = \sum_{i,j in chain} (X_i X_{j} + Y_i Y_{j})
    U = {exp[-i*angle*H_chain]} ^ N_chain  # for N-size complete graph, N_chain = N // 2
    exp[-i*angle*H_chain] is implemented by parity troterrization (no boundary condition)
    """
    if minus == True:
        beta = -beta
    N = len(qc._qubits)
    if T is None:
        T = 1
    beta = beta / T
    assert N % 2 == 0  # for even complete graph only, so there is no gate index order issue as well
    chains = []
    chain = [0, 1]
    for i in range(N // 2 - 1):
        chain.append(N - 1 - i)
        chain.append(2 + i)
    chains.append(chain)
    for i in range(N // 2 - 1):
        chain = [i + 1 for i in chains[-1]]
        for (idx, item) in enumerate(chain):
            if item > N - 1:
                chain[idx] = chain[idx] - N
        chains.append(chain)
    for _ in range(int(T)):
        for chain in chains:
            for i in range(0, N - 1, 2):
                # even
                edge = [chain[i], chain[i + 1]]
                qc.append(
                    qiskit.circuit.library.XXPlusYYGate(4 * beta),
                    [edge[0], edge[1]],
                )
            for i in range(1, N - 1, 2):
                # odd
                edge = [chain[i], chain[i + 1]]
                qc.append(
                    qiskit.circuit.library.XXPlusYYGate(4 * beta),
                    [edge[0], edge[1]],
                )
    return qc

def get_qaoa_circuit(
    po_problem,
    gammas,
    betas,
    depth,
    ini="dicke",
    mixer="t-xy",
    T=None,
    ini_state=None,
    save_state=True,
    minus=False,
):
    """
    Put all ingredients together to build up a qaoa circuit
    Minus is for define mixer with a minus sign, for checking phase diagram
    """
    N = po_problem["N"]
    K = po_problem["K"]
    if ini_state is not None:
        q = QuantumRegister(N)
        circuit = QuantumCircuit(q)
        circuit.initialize(ini_state, [q[i] for i in range(N)])
    else:
        if ini.lower() == "dicke":
            circuit = get_dicke_init(N, K)
        elif ini.lower() == "uniform":
            circuit = get_uniform_init(N)
        elif ini.lower() == "ring_gs":
            circuit = get_ring_gs_ini(N, K)
        else:
            raise ValueError("Undefined initial circuit")
    for i in range(depth):
        circuit = get_cost_circuit(po_problem, circuit, gammas[i])
        if mixer.lower() == "t-xy":
            circuit = get_mixer_Txy(
                circuit, betas[i], minus=minus, T=T #minus should be false
            ) 
        elif mixer.lower() == "exact-xy":
            circuit = get_mixer_exactXY(
                circuit, betas[i], minus=minus, #minus should be false
            )  
        elif mixer.lower() == "t_chaintrotter_1xy":
            circuit = get_mixer_complete_Tchaintrotter_1xy(
                circuit, betas[i], T=T, minus=minus #minus should be true
            )
        # elif mixer.lower() == "complete_chaintrotter_exact":
        #     circuit = get_mixer_complete_chaintrotter_exact(
        #         circuit, betas[i], minus=minus
        #     )
        elif mixer.lower() == "exact-complete":
            circuit = get_mixer_exact_complete(
                circuit, betas[i], minus=minus, #minus should be true
            )
        elif mixer.lower() == "exact-complete1":
            circuit = get_mixer_exact_partial_complete(
                circuit, betas[i], [[0, 1, 5, 2, 4, 3]], minus=minus #minus should be true
            )
        elif mixer.lower() == "trotter-complete1":
            circuit = get_mixer_partial_Tchaintrotter_1xy(
                circuit, betas[i], [[0, 1, 5, 2, 4, 3]], T=T, minus=minus #minus should be true
            )
        elif mixer.lower() == "exact-complete2":
            circuit = get_mixer_exact_partial_complete(
                circuit, betas[i], [[1, 2, 0, 3, 5, 4]], minus=minus 
            )
        elif mixer.lower() == "trotter-complete2":
            circuit = get_mixer_partial_Tchaintrotter_1xy(
                circuit, betas[i], [[1, 2, 0, 3, 5, 4]], T=T, minus=minus 
            )
        elif mixer.lower() == "exact-complete3":
            circuit = get_mixer_exact_partial_complete(
                circuit, betas[i], [[2, 3, 1, 4, 0, 5]], minus=minus 
            )
        elif mixer.lower() == "trotter-complete3":
            circuit = get_mixer_partial_Tchaintrotter_1xy(
                circuit, betas[i], [[2, 3, 1, 4, 0, 5]], T=T, minus=minus 
            )
        elif mixer.lower() == "exact-complete12":
            circuit = get_mixer_exact_partial_complete(
                circuit, betas[i], [[0, 1, 5, 2, 4, 3], [1, 2, 0, 3, 5, 4]], minus=minus
            )
        elif mixer.lower() == "trotter-complete12":
            circuit = get_mixer_partial_Tchaintrotter_1xy(
                circuit, betas[i], [[0, 1, 5, 2, 4, 3], [1, 2, 0, 3, 5, 4]], T=T, minus=minus
            )
        elif mixer.lower() == "exact-complete13":
            circuit = get_mixer_exact_partial_complete(
                circuit, betas[i], [[0, 1, 5, 2, 4, 3],[2, 3, 1, 4, 0, 5]], minus=minus 
            )
        elif mixer.lower() == "trotter-complete13":
            circuit = get_mixer_partial_Tchaintrotter_1xy(
                circuit, betas[i], [[0, 1, 5, 2, 4, 3],[2, 3, 1, 4, 0, 5]], T=T, minus=minus
            )
        elif mixer.lower() == "exact-complete23":
            circuit = get_mixer_exact_partial_complete(
                circuit, betas[i], [[1, 2, 0, 3, 5, 4], [2, 3, 1, 4, 0, 5]], minus=minus
            )
        elif mixer.lower() == "trotter-complete23":
            circuit = get_mixer_partial_Tchaintrotter_1xy(
                circuit, betas[i], [[1, 2, 0, 3, 5, 4], [2, 3, 1, 4, 0, 5]], T=T, minus=minus 
            )
        elif mixer.lower() == "grover":
            circuit = get_mixer_grover(circuit, betas[i], N, K)
        elif mixer.lower() == "rx":
            circuit = get_mixer_RX(circuit, betas[i])
        else:
            raise ValueError("Undefined mixer circuit")
    if save_state is False:
        circuit.measure_all()
    return circuit


def measure_circuit(circuit, n_trials=1024, save_state=True):
    """Get the output from circuit, either measured samples or full state vector"""
    if save_state is False:
        backend = Aer.get_backend("qasm_simulator")
        job = execute(circuit, backend, shots=n_trials)
        result = job.result()
        bitstrings = invert_counts(result.get_counts())
        return bitstrings
    else:
        backend = Aer.get_backend("statevector_simulator")
        result = execute(circuit, backend).result()
        state = result.get_statevector()
        return get_adjusted_state(state)

        ##### equal #####
        # from qiskit.providers.aer import AerSimulator
        # circuit.save_state()
        # backend = AerSimulator(method="statevector")
        # sv_qiskit = execute(circuit, backend).result().get_statevector()
        ## assert np.allclose(sv_qiskit.data,state.data)
        # return get_adjusted_state(sv_qiskit)

def get_energy_expectation(po_problem, samples):
    """Compute energy expectation from measurement samples"""
    expectation_value = 0
    N_total = 0
    for (config, count) in samples.items():
        expectation_value += count * get_configuration_cost(po_problem, config)
        N_total += count
    expectation_value = expectation_value / N_total

    return expectation_value


def get_energy_expectation_sv(po_problem, samples):
    """Compute energy expectation from full state vector"""
    expectation_value = 0
    # convert state vector to dictionary
    samples = state_to_ampl_counts(samples)
    for (config, wf) in samples.items():
        expectation_value += (np.abs(wf) ** 2) * get_configuration_cost(
            po_problem, config
        )

    return expectation_value

def get_energy_std_sv(po_problem, samples):
    """Compute energy expectation from full state vector"""
    expectation_value = 0
    expectation_squared = 0
    # convert state vector to dictionary
    samples = state_to_ampl_counts(samples)
    for (config, wf) in samples.items():
        expectation_value += (np.abs(wf) ** 2) * get_configuration_cost(
            po_problem, config
        )
        expectation_squared += (np.abs(wf) ** 2) * (get_configuration_cost(po_problem, config)**2)
        
    variance = expectation_squared - expectation_value ** 2
    return np.sqrt(variance)


def get_energy_edge_contribution(po_problem, samples):
    """Compute energy contribution from each edge component"""
    N = po_problem["N"]
    scale = po_problem["scale"]
    means = po_problem["means"] / scale
    cov = po_problem["cov"] / scale
    q = po_problem["q"]
    edge_contributions = np.zeros((N, N), dtype=complex)
    Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    I = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
    for k1 in range(N):
        for k2 in range(k1 + 1, N):
            H = 1
            for i in range(N):
                if i == k1 or i == k2:
                    H = np.kron(H, Z)
                else:
                    H = np.kron(H, I)
            H = q / 2 * cov[k1, k2] * H
            edge_contributions[k1, k2] = samples @ H @ samples.conjugate()
    for k1 in range(N):
        H = 1
        for i in range(N):
            if i == k1:
                H = np.kron(H, Z)  # the order is important!
            else:
                H = np.kron(H, I)
        H = ((means[k1] - q * cov[k1, k1]) / 2 - q * sum(cov[k1, k1:])) * H
        edge_contributions[k1, k1] = samples @ H @ samples.conjugate()

    return edge_contributions

def run_circuit_phase(
    po_problem,
    p,
    delta,
    gs,
    ub=1,
    minus=False,
    T=None,
    ini="dicke",
    ini_state=None,
    mixer="t-xy",
    n_trials=1024,
    save_state=True,
):
    """Run the phase diagram simlation for a given p and delta"""
    gammas = np.zeros(p)
    betas = np.zeros(p)
    N = po_problem["N"]
    for i in range(p):
        f = (i + 1) / (p + 1) * ub
        gammas[i] = delta * f
        betas[i] = delta * (ub - f)

    circuit = get_qaoa_circuit(
        po_problem,
        ini=ini,
        mixer=mixer,
        gammas=gammas,
        betas=betas,
        depth=p,
        T=T,
        ini_state=ini_state,
        save_state=save_state,
        minus=minus,
    )
    samples = measure_circuit(circuit, n_trials=n_trials, save_state=save_state)
    if save_state is False:
        energy_expectation_value = get_energy_expectation(po_problem, samples)
    else:
        energy_expectation_value = get_energy_expectation_sv(po_problem, samples)

    fidelity = exact_fidelity(samples, gs)
    results = {}
    results["p"] = p
    results["delta"] = delta
    results["T"] = T
    results["ub"] = ub
    results["mixer"] = mixer
    results["ini"] = ini
    results["optimal_energy_measurement"] = energy_expectation_value
    results["fidelity"] = fidelity

    return results
