import argparse
import gc
import pickle
import warnings
from time import sleep, time

import networkx as nx
import numpy as np
from docplex.mp.model import Model
from evaluate_energy import get_evaluate_energy, load_problem
from qiskit.algorithms import QAOA
from qiskit.algorithms.optimizers import SPSA
from qiskit.opflow import AerPauliExpectation
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit_aer import AerSimulator
from qiskit_optimization.applications import Maxcut
from qiskit_optimization.problems.quadratic_program import QuadraticProgram
from qiskit_optimization.translators import from_docplex_mp
from qokit.parameter_utils import get_fixed_gamma_beta

warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument("--problem", type=str, default="maxcut")
parser.add_argument("-n", type=int, default=32)
parser.add_argument("-p", type=int, default=1)
parser.add_argument("-s", "--seed", type=int, default=1000)
parser.add_argument("-b", "--batch", type=int, default=0)
parser.add_argument("-d", "--delta", type=float, default=0.1)
parser.add_argument("--qiskit", default=False, action="store_true")
parser.add_argument("--cpu", default=False, action="store_true")
parser.add_argument("--no-aer", dest="aer", default=True, action="store_false")

args = parser.parse_args()
print(args)

problem = args.problem
n = args.n
p = args.p
seed_pool = list(range(args.batch * args.seed, (args.batch + 1) * args.seed))
delta = args.delta

data = []
for seed in seed_pool:
    start_time = time()
    data.append([])
    gamma, beta = get_fixed_gamma_beta(3, p)

    if args.qiskit:
        if args.problem == "maxcut":
            # class WeightedMaxcut(Maxcut):
            #     def to_quadratic_program(self) -> QuadraticProgram:
            #         mdl = Model(name="WeightedMaxcut")
            #         x = {
            #             i: mdl.binary_var(name=f"x_{i}")
            #             for i in range(self._graph.number_of_nodes())
            #         }
            #         objective = mdl.sum(
            #             self._graph.edges[i, j]["weight"] * x[i] * (1 - x[j])
            #             + self._graph.edges[i, j]["weight"] * x[j] * (1 - x[i])
            #             for i, j in self._graph.edges
            #         )
            #         mdl.maximize(objective)
            #         op = from_docplex_mp(mdl)
            #         return op

            instance, precomputed_energies = load_problem(problem, n, seed)
            problem = Maxcut(instance).to_quadratic_program()
        else:
            raise ValueError(f"Problem {problem} not implemented yet")

        backend = AerSimulator(
            method="statevector",
            device="CPU" if args.cpu else "GPU",
            # noise_model=noise_model,
            # fusion_enable=args.problem == "maxcut" or noise != "depolar" or n < 17,
            # blocking_enable=True,
            # blocking_qubits=30,
        )

        H, offset = problem.to_ising()

        algorithm_globals.random_seed = seed
        quantum_instance = QuantumInstance(
            backend=backend,
            seed_simulator=seed,
            seed_transpiler=seed,
            # optimization_level=0 if noise != "ideal" else None
        )

        algorithm = QAOA(
            SPSA(),
            reps=p,
            quantum_instance=quantum_instance,
            expectation=AerPauliExpectation() if args.aer else None,
        )
        algorithm._check_operator_ansatz(H)
        eval_func = algorithm.get_energy_evaluation(H)

        gamma = [g / 2 for g in gamma]
        # gamma, beta = [1.27627202/2], [-1.76714587/4]
        initial_value = eval_func(beta + gamma) + 7.5
        data[-1].append(initial_value)
        for i in range(p):
            gamma[i] += delta
            data[-1].append(eval_func(beta + gamma) - offset)
            gamma[i] -= delta
    else:
        instance, precomputed_energies = load_problem(problem, n, seed)
        # instance = nx.random_regular_graph(3, n, seed)
        eval_func = get_evaluate_energy(
            instance,
            precomputed_energies,
            p,
            objective="expectation",
            simulator="c" if args.cpu else "auto",
        )
        beta = [b * 4 for b in beta]

        initial_value = eval_func(gamma + beta)
        data[-1].append(initial_value)
        for i in range(p):
            gamma[i] += delta
            data[-1].append(eval_func(gamma + beta))
            gamma[i] -= delta

    print(f"{seed=}", data[-1], time() - start_time)
    del eval_func

pickle.dump(
    {"data": np.array(data), "args": args},
    open(
        f"data/{problem}/hardness/p{p}-q{n}-s{seed_pool[0]}-{seed_pool[-1]}-d{delta}.pckl",
        "wb",
    ),
)
