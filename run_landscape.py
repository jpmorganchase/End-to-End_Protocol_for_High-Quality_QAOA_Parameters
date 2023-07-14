# os.environ["OMP_NUM_THREADS"] = "32"
# os.environ["NUMBA_NUM_THREADS"] = "32"

import itertools
import os
import pickle
import time
from functools import partial
from math import pi
from pathlib import Path

import numba
import numpy as np
from oscar import BPDNReconstructor, CustomExecutor, Landscape
from qokit.fur import QAOAFURXSimulatorGPU, QAOAFURXYRingSimulatorC
from qokit.fur.c.utils import ComplexArray
from tqdm import tqdm

from circuit_utils import get_configuration_cost_kw
from optimizer import circuit_measurement_function
from utils import (
    generate_dicke_state_fast,
    get_adjusted_state,
    get_problem,
    get_real_problem,
    precompute_energies_parallel,
)


q = 0.5
sample_seed = 42
data_dir = "data/random"
os.makedirs(data_dir, exist_ok=True)
rng = np.random.default_rng(sample_seed)


def kbits(n, k):
    for bits in itertools.combinations(range(n), k):
        s = [0] * n
        for bit in bits:
            s[bit] = 1
        yield np.array(s)


def load_problem(n, seed):
    k = n // 2
    po_path = f"{data_dir}/po_problem_rule_{n}_{k}_{q}_seed{seed}.pckl"
    energy_path = f"{data_dir}/precomputed_energies_rule_{n}_{k}_{q}_seed{seed}.npy"
    if Path(po_path).exists():
        po_problem = pickle.load(open(po_path, "rb"))
    else:
        po_problem = get_real_problem(n, k, q, seed, pre=1)
        means_in_spins = np.array(
            [
                po_problem["means"][i] - po_problem["q"] * np.sum(po_problem["cov"][i, :])
                for i in range(len(po_problem["means"]))
            ]
        )
        scale = 1 / (
            np.sqrt(np.mean(((po_problem["q"] * po_problem["cov"]) ** 2).flatten()))
            + np.sqrt(np.mean((means_in_spins**2).flatten()))
        )
        po_problem["scale"] = scale
        po_problem["means"] = scale * po_problem["means"]
        po_problem["cov"] = scale * po_problem["cov"]

        min_constrained = float("inf")
        max_constrained = float("-inf")
        mean_constrained = 0
        total_constrained = 0
        po_obj = partial(get_configuration_cost_kw, po_problem=po_problem)
        for x in tqdm(kbits(n, k)):
            curr = po_obj(x)
            if curr < min_constrained:
                min_constrained = curr
                min_x = x
            if curr > max_constrained:
                max_constrained = curr
                max_x = x
            mean_constrained += curr
            total_constrained += 1.0
        mean_constrained /= total_constrained
        po_problem["feasible_min"] = min_constrained
        po_problem["feasible_max"] = max_constrained
        po_problem["feasible_min_x"] = min_x
        po_problem["feasible_max_x"] = max_x
        po_problem["feasible_mean"] = mean_constrained
        pickle.dump(po_problem, open(po_path, "wb"))

    if Path(energy_path).exists():
        precomputed_energies = np.load(energy_path)
    else:
        po_obj = partial(get_configuration_cost_kw, po_problem=po_problem)
        precomputed_energies = get_adjusted_state(precompute_energies_parallel(po_obj, n, 1)).real
        np.save(energy_path, precomputed_energies, allow_pickle=False)

    return po_problem, precomputed_energies


def evaluate_energy(theta, p, n, problem_seed, shots=None, sv_list=None, std_list=None):
    po_problem, precomputed_energies = load_problem(n, problem_seed)
    gamma, beta = theta[:p], theta[p:]
    if numba.cuda.is_available():
        simulator = QAOAFURXYRingSimulatorGPU
    else:
        simulator = QAOAFURXYRingSimulatorC
    sim = simulator(n, po_problem["scale"] * precomputed_energies)
    sv = sim.simulate_qaoa(gamma, beta, sv0=generate_dicke_state_fast(n, n // 2))
    if isinstance(sv, np.ndarray):
        sv = ComplexArray(sv.real, sv.imag)
    if sv_list is not None:
        sv_list.append(sv.get_complex())

    energy_mean = sv.get_norm_squared().dot(precomputed_energies).real
    if shots is None and std_list is None:
        return energy_mean

    energy_std = np.sqrt(
        (sv.get_norm_squared().dot(precomputed_energies**2) - energy_mean**2).real
    )
    if shots is not None:
        energy_std = energy_std / np.sqrt(shots)
    if std_list is not None:
        std_list.append(energy_std)
    if shots is None:
        return energy_mean
    return rng.normal(energy_mean, energy_std)


if __name__ == "__main__":
    depth_pool = [1]
    qubit_pool = [20]
    seed_pool = range(20, 30)
    # resolutions = [128, 32]
    # bounds = [(-2.2, -0.6), (0.9, 1.3)]
    resolutions = [128, 32]
    bounds = [(-4, 0), (0, 1)]

    for p, n, seed in itertools.product(depth_pool, qubit_pool, seed_pool):
        sv_list, std_list = None, None
        filename = f"data/landscapes/{p=}/{n=}/{p=}-{n=}-{seed=}-{bounds}-{resolutions}"
        landscape = Landscape(resolutions, bounds)
        try:
            executor = CustomExecutor(
                partial(
                    evaluate_energy,
                    p=p,
                    n=n,
                    problem_seed=seed,
                    shots=None,
                    sv_list=sv_list,
                    std_list=std_list,
                )
            )
            landscape.sample_and_run(executor, 1 / 16)
            landscape.reconstruct(BPDNReconstructor())
            # landscape.run_all(executor)
            landscape.interpolate(fill_value=np.max(landscape.reconstructed_landscape))
            landscape.save(filename + ".pckl")
            # std_landscape = Landscape(resolutions, bounds)
            # std_landscape.sampled_landscape = np.array(std_list)#.reshape(landscape.param_resolutions)
            # std_landscape._sampled_indices = landscape._sample_indices
            # std_landscape.reconstruct(BPDNReconstructor())
            # # std_landscape.true_landscape = np.array(std_list).reshape(landscape.param_resolutions)
            # std_landscape.interpolate(fill_value=np.max(std_landscape.reconstructed_landscape))
            # std_landscape.save(filename + "-std.pckl")
            # np.save(
            #     filename + "-sv.npy",
            #     np.array(sv_list).reshape(*landscape.param_resolutions, -1),
            # )
        except Exception as e:
            print(e)
