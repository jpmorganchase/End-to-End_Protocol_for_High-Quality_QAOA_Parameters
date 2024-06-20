import argparse
import itertools
import os
import pickle
from functools import partial
from math import pi

import numpy as np
from oscar import (CustomExecutor, HyperparameterGrid, HyperparameterTuner,
                   Landscape, NLoptOptimizer, PDFOOptimizer, QiskitOptimizer,
                   SciPyOptimizer)
from qokit.parameter_utils import get_fixed_gamma_beta, get_sk_gamma_beta

from evaluate_energy import load_problem

sample_seed = 42
rng = np.random.default_rng(sample_seed)


def shotted_measurement(params, landscape, landscape_std, sense, shots=None):
    energy = landscape.interpolator(np.asarray(params)).item()
    if shots is None:
        return sense * energy
    energy_std = landscape_std.interpolator(np.asarray(params)).item() / np.sqrt(shots)
    return sense * rng.normal(energy, energy_std)


def eval_point(point, eval_func, optimal_metric, sense):
    result = eval_func(point)
    return (
        sense
        * (result - optimal_metric[int((sense + 1) / 2)])
        / (optimal_metric[0] - optimal_metric[1])
    )


def process_results(i, j, trace, result, eval_func, optimal_metric, sense):
    return [
        eval_point(
            trace.params_trace[np.argmin(trace.value_trace[:k])],
            eval_func,
            optimal_metric,
            sense,
        )
        for k in range(1, 21)
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str, default="maxcut")
    parser.add_argument("-n", type=int, default=12)
    parser.add_argument("-p", type=int, default=1)
    parser.add_argument("-s", "--seed", type=int, default=100)
    parser.add_argument("-r", "--reps", type=int, default=10)

    args = parser.parse_args()
    print(args)

    problem = args.problem
    p = args.p
    if p == 1:
        if problem == "po":
            resolutions = [128, 128]
            gamma_shift = pi / 4
            beta_shift = pi / 4
        elif problem == "maxcut":
            resolutions = [256, 128]
            gamma_shift = pi / 2
            beta_shift = -pi / 4
    else:
        resolutions = [60] * 2 * p
        gamma_shift = pi / 4
        beta_shift = pi / 4
    seed_pool = list(range(args.seed))
    qubit_pool = list(range(args.n, args.n + 1, 2))
    budget = 10000
    maxfev_pool = list(range(2 * p + 2, 21))
    shots_pool = budget // np.array(maxfev_pool)
    rhobeg_pool = np.linspace(0.05, 0.5, 10).tolist()

    configs = [
        # HyperparameterGrid(
        #     SciPyOptimizer,
        #     optimizer=[
        #         "COBYLA",
        #     ],
        #     options={"maxiter": maxfev_pool, "rhobeg": rhobeg_pool},
        # ),
        HyperparameterGrid(
            NLoptOptimizer,
            optimizer=[
                "LN_COBYLA",
                "LN_BOBYQA",
                # "LN_NEWUOA",
                "LN_NELDERMEAD",
                "GN_ESCH",
                "GN_DIRECT_L",
                "GN_CRS2_LM",
            ],
            maxeval=[maxfev_pool[-1]],
            initial_step=rhobeg_pool,
            xtol_rel=[1e-5],
            ftol_rel=[1e-5],
        ),
        HyperparameterGrid(
            QiskitOptimizer,
            optimizer=[
                "SPSA",
                "GSLS",
                "IMFIL",
                # "ESCH",
                # "NFT",
                # "SNOBFIT",
            ],
            maxiter=[maxfev_pool[-1]],
            max_evals=[maxfev_pool[-1]],
            max_fev=[maxfev_pool[-1]],
        ),
        # HyperparameterGrid(
        #     PDFOOptimizer, optimizer=["bobyqa", "cobyla"], rhobeg=rhobeg_pool
        # ),
    ]

    results = []
    for i, (n, seed) in enumerate(itertools.product(qubit_pool, seed_pool)):
        print(f"{n=}, {seed=}")
        filename = f"data/{problem}/landscapes/{p=}/{n=}/{problem}-{p=}-{n=}-{seed=}-({2*gamma_shift:.2f}, {2*beta_shift:.2f})-{resolutions}"
        landscape = Landscape.load(filename + "-expectation.pckl")
        landscape.interpolate(fill_value=landscape.max())
        landscape_std = Landscape.load(filename + "-std.pckl")
        landscape_std.interpolate(fill_value=0)
        results.append([])
        instance, precomputed_energies = load_problem(problem, n, seed, True)
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

        eval_func = partial(
            shotted_measurement,
            landscape=landscape,
            landscape_std=landscape_std,
            sense=sense,
        )

        for j in range(args.reps):
            shotted_executor = CustomExecutor(eval_func)
            run_configs = HyperparameterGrid(
                executor=[shotted_executor],
                initial_point=[initial_point.copy()],
                shots=shots_pool,
                bounds=[
                    [(g - 0.7, g + 0.7) for g in gamma]
                    + [(b - 0.7, b + 0.7) for b in beta]
                ],
                # bounds=[[landscape.param_bounds[0], landscape.param_bounds[1][::-1]]],
            )
            tuner = HyperparameterTuner(configs)
            tuner.run(run_configs)

            result = tuner.process_results(
                partial(
                    process_results,
                    eval_func=eval_func,
                    optimal_metric=(minval, maxval),
                    sense=sense,
                )
            )
            results[-1].append(
                np.concatenate(
                    [
                        np.max(
                            np.diagonal(
                                np.asarray(res).reshape(
                                    len(configs[k]["optimizer"]),
                                    -1,
                                    len(maxfev_pool),
                                    maxfev_pool[-1],
                                )[..., maxfev_pool[0] :],
                                axis1=2,
                                axis2=3,
                            ),
                            axis=(1, 2),
                        ).reshape(-1)
                        for k, res in enumerate(result)
                    ]
                )
            )
    results = np.einsum("ijk->kij", results)
    print(np.mean(results, axis=(1, 2)), np.std(results, axis=(1, 2)))
    savepath = f"data/{problem}/optimizer/p{p}-s{seed_pool[0]}-{seed_pool[-1]}-r={args.reps}.pckl"
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    pickle.dump(
        {
            "config": configs,
            "result": results,
        },
        open(savepath, "wb"),
    )
