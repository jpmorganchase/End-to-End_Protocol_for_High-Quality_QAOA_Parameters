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
from qokit.fur import QAOAFURXYRingSimulatorGPU, QAOAFURXYRingSimulatorC
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
from evaluate_energy import get_evaluate_energy, load_problem


sample_seed = 42
data_dir = "data/random"
os.makedirs(data_dir, exist_ok=True)
rng = np.random.default_rng(sample_seed)


if __name__ == "__main__":
    simulator = "c"
    problem = "po"
    p = 1
    qubit_pool = [10]
    seed_pool = range(1)
    # resolutions = [128, 32]
    # bounds = [(-2.2, -0.6), (0.9, 1.3)]
    # bounds = [(-4, 0), (0, 1)]
    resolutions = [64, 64]
    bounds = [(-pi, pi), (-pi, pi)]

    for i, (n, seed) in enumerate(itertools.product(qubit_pool, seed_pool)):
        filename = f"data/{problem}/landscapes/{p=}/{n=}/{problem}-{p=}-{n=}-{seed=}-{bounds}-{resolutions}"
        landscape = Landscape(resolutions, bounds)
        try:
            instance, precomputed_energies = load_problem(problem, n, seed)
            # instances.append((instance, precomputed_energies))
            # if problem == "po":
            #     sense = 1
            #     minval, maxval = instance["feasible_min"], instance["feasible_max"]
            # else:
            #     sense = -1
            #     minval, maxval = np.min(precomputed_energies), np.max(precomputed_energies)

            executor = CustomExecutor(
                get_evaluate_energy(
                    instance,
                    precomputed_energies,
                    p,
                    objective="expectation",
                    simulator=simulator,
                )
            )
            # landscape.sample_and_run(executor, 1 / 16)
            # landscape.reconstruct(BPDNReconstructor())
            landscape.run_all(executor)
            # landscape.interpolate(fill_value=np.max(landscape.reconstructed_landscape))
            landscape.save(filename + ".pckl")
        except Exception as e:
            print(e)

