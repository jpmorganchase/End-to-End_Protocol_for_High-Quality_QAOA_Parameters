import itertools
from math import pi

import numpy as np
from oscar import CustomExecutor, Landscape
from qokit.parameter_utils import get_fixed_gamma_beta, get_sk_gamma_beta

from evaluate_energy import get_evaluate_energy, load_problem

if __name__ == "__main__":
    simulator = "auto"
    problem = "po"
    p = 1
    qubit_pool = [12]
    seed_pool = range(60)
    resolutions = [128] * 2 if p ==1 else [60] * 2 * p
    gamma_shift = pi / 4
    beta_shift = pi / 4

    for i, (n, seed) in enumerate(itertools.product(qubit_pool, seed_pool)):
        instance, precomputed_energies = load_problem(problem, n, seed)
        if problem == "po":
            sense = 1
            beta_scaling = -8
            gamma, beta = get_sk_gamma_beta(p)
            minval, maxval = instance["feasible_min"], instance["feasible_max"]
        elif problem == "skmodel":
            sense = -1
            beta_scaling = 4
            gamma, beta = get_sk_gamma_beta(p)
            minval, maxval = np.min(precomputed_energies), np.max(precomputed_energies)
        else:
            sense = -1
            beta_scaling = 4
            gamma, beta, ar = get_fixed_gamma_beta(3, p, True)
            gamma, beta = np.array(gamma), np.array(beta)
            minval, maxval = np.min(precomputed_energies), np.max(precomputed_energies)
        beta *= beta_scaling
        initial_point = np.concatenate((gamma, beta))

        bounds = [(g - gamma_shift, g + gamma_shift) for g in gamma] + [
            (b - beta_shift, b + beta_shift) for b in beta
        ]

        filename = f"data/{problem}/landscapes/{p=}/{n=}/{problem}-{p=}-{n=}-{seed=}-({2*gamma_shift:.2f}, {2*beta_shift:.2f})-{resolutions}"
        landscape = Landscape(resolutions, bounds)
        for objective in [
            "expectation",
            "std",
        ]:
            executor = CustomExecutor(
                get_evaluate_energy(
                    instance,
                    precomputed_energies,
                    p,
                    objective=objective,
                    simulator=simulator,
                )
            )
            landscape.run_all(executor)
            landscape.interpolate(bounds_error=True)
            landscape.save(filename + f"-{objective}.pckl")
