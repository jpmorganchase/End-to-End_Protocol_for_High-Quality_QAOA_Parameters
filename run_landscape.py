import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

import itertools
import os
import pickle
import time
from functools import partial
from math import pi
from pathlib import Path

import numpy as np
from oscar import BPDNReconstructor, CustomExecutor, Landscape
from tqdm import tqdm

from circuit_utils import get_configuration_cost_kw
from fur import QAOAFURXYRingSimulatorC
from optimizer import circuit_measurement_function
from utils import (
    generate_dicke_state_fast,
    get_adjusted_state,
    precompute_energies_parallel,
    get_problem,
)

q = 0.5
data_dir = "data/random"
os.makedirs(data_dir, exist_ok=True)


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
    if Path(po_path).exists() and Path(energy_path).exists():
        precomputed_energies = np.load(energy_path)
        po_problem = pickle.load(open(po_path, "rb"))
    else:
        po_problem_unscaled = get_problem(n, k, q, seed, pre=1)
        means_in_spins = np.array(
            [
                po_problem_unscaled["means"][i]
                - po_problem_unscaled["q"] * np.sum(po_problem_unscaled["cov"][i, :])
                for i in range(len(po_problem_unscaled["means"]))
            ]
        )
        scale = 1 / (
            np.sqrt(
                np.mean(((po_problem_unscaled["q"] * po_problem_unscaled["cov"]) ** 2).flatten())
            )
            + np.sqrt(np.mean((means_in_spins**2).flatten()))
        )
        print(f"scale = {scale}")
        po_problem = get_problem(N=n, K=k, q=q, seed=seed, pre=scale)

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
        precomputed_energies = get_adjusted_state(precompute_energies_parallel(po_obj, n, 1))

        np.save(energy_path, precomputed_energies, allow_pickle=False)
        pickle.dump(po_problem, open(po_path, "wb"))
    return po_problem, precomputed_energies


def evaluate_energy(theta, p, n, problem_seed, shots=None, return_std=False, sample_seed=42):
    po_problem, precomputed_energies = load_problem(n, problem_seed)
    gamma, beta = theta[:p], theta[p:]
    sim = QAOAFURXYRingSimulatorC(n, po_problem["scale"] * precomputed_energies)
    sv = sim.simulate_qaoa(gamma, beta, sv0=generate_dicke_state_fast(n, n // 2))
    energy_mean = sv.get_norm_squared().dot(precomputed_energies).real

    if shots is None and not return_std:
        return energy_mean
    energy_std = np.sqrt(
        (sv.get_norm_squared().dot(precomputed_energies**2) - energy_mean**2).real
    )
    if shots is None:
        return energy_std
    energy_std = energy_std / np.sqrt(shots)
    if return_std:
        return energy_std
    return np.random.default_rng(sample_seed).normal(energy_mean, energy_std)


if __name__ == "__main__":
    depth_pool = [1]
    qubit_pool = [18]
    seed_pool = range(3, 10)
    resolutions = [128, 32]
    bounds = [(-2.2, -0.6), (0.9, 1.3)]
    # resolutions = [64, 64]
    # bounds = [(-pi / 4, pi / 4), (-pi / 2, pi / 2)]

    for p, n, seed in itertools.product(depth_pool, qubit_pool, seed_pool):
        filename = f"data/landscapes/{p=}/{n=}/{p=}-{n=}-{seed=}-{bounds}-{resolutions}"
        for return_std in [False, True]:
            landscape = Landscape(resolutions, bounds)
            executor = CustomExecutor(
                partial(
                    evaluate_energy, p=p, n=n, problem_seed=seed, shots=None, return_std=return_std
                )
            )
            landscape.run_all(executor)
            landscape.interpolate(fill_value=np.max(landscape.true_landscape))
            if return_std:
                filename += "-std"
            landscape.save(filename + ".pckl")
