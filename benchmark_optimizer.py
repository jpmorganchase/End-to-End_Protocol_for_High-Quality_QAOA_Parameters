import argparse
import itertools
import pickle
from functools import partial
from pprint import pprint

import numpy as np
from oscar import (CustomExecutor, HyperparameterGrid, HyperparameterTuner,
                   InterpolatedLandscapeExecutor, NLoptOptimizer,
                   QiskitOptimizer, plot_2d_landscape)
from qokit.parameter_utils import get_fixed_gamma_beta, get_sk_gamma_beta

from evaluate_energy import get_evaluate_energy, load_problem
from restarting_cobyla import RECOBYLA

sample_seed = 42
rng = np.random.default_rng(sample_seed)


def shotted_measurement(params, function, shots, sense, fix_beta=None):
    if fix_beta is not None:
        params = np.concatenate([params, fix_beta])
    mean, std = function(params)
    if shots is None:
        return sense * mean
    return sense * rng.normal(mean, std / np.sqrt(shots))


def eval_point(point, eval_func, optimal_metric, sense, fix_beta=None):
    if fix_beta is not None:
        point = np.concatenate([point, fix_beta])
    ar = (
        sense
        * (eval_func(point) - optimal_metric[int((sense + 1) / 2)])
        / (optimal_metric[0] - optimal_metric[1])
    )
    return ar


def process_results(
    method, i, trace, result, eval_func, optimal_metric, sense, fix_beta=None
):
    return eval_point(trace.optimal_params, eval_func, optimal_metric, sense, fix_beta)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", type=str, default="max_ar")
    parser.add_argument("--problem", type=str, default="maxcut")
    parser.add_argument("-n", type=int, default=12)
    parser.add_argument("-p", type=int, default=1)
    parser.add_argument("-s", "--seed", type=int, default=1000)
    parser.add_argument("-b", "--batch", type=int, default=0)
    parser.add_argument("--cpu", default=False, action="store_true")
    parser.add_argument("--fix-beta", default=False, action="store_true")

    args = parser.parse_args()
    print(args)

    problem = args.problem
    p = args.p
    seed_pool = list(range(args.batch * args.seed, (args.batch + 1) * args.seed))
    simulator = "c" if args.cpu else "auto"
    qubit_pool = list(range(args.n, args.n + 1, 2))
    budget = 10000
    target = args.target
    if target == "max_ar":
        maxfev_pool = [200]
        shots_pool = [None]
    elif target == "budget":
        if args.fix_beta:
            target += "-fix-beta"
            maxfev_pool = list(range(p + 2, 20)) + list(range(20, 51, 5))
        else:
            maxfev_pool = list(range(2 * p + 2, 20)) + list(range(20, 51, 5))
        # shots_pool = list(range(500, 2501, 100))
        shots_pool = budget // np.array(maxfev_pool)
    else:
        raise NotImplementedError()
    # reps = 2
    rhobeg_pool = np.linspace(0.01, 0.3, 30).tolist()
    rhobeg_pool = [0.1]
    xtol_pool = [0.045]
    xtol_pool = np.linspace(0.01, 0.0, 4).tolist()
    scaling = [2]
    scaling = np.linspace(1.4, 3.2, 10).tolist()

    for i, n in enumerate(qubit_pool):
        instances = []
        results, optimal_params = {}, {}
        initial_ar = []
        for j, seed in enumerate(seed_pool):
            instance, precomputed_energies = load_problem(problem, n, seed, True)
            instances.append((instance, precomputed_energies))
            if problem == "po":
                sense = 1
                beta_scaling = -8
                # initial_point = [-1.24727193, 1.04931211 * 8]
                gamma, beta = get_sk_gamma_beta(p)
                minval, maxval = instance["feasible_min"], instance["feasible_max"]
            elif problem == "skmodel":
                sense = -1
                beta_scaling = 4
                gamma, beta = get_sk_gamma_beta(p)
                minval, maxval = np.min(precomputed_energies), np.max(
                    precomputed_energies
                )
            else:
                sense = -1
                beta_scaling = 4
                gamma, beta, ar = get_fixed_gamma_beta(3, p, True)
                gamma, beta = np.array(gamma), np.array(beta)
                minval, maxval = np.min(precomputed_energies), np.max(
                    precomputed_energies
                )
            beta *= beta_scaling
            initial_point = np.concatenate((gamma, beta))

            configs = [
                # HyperparameterGrid(
                #     RECOBYLA(),
                #     initial_point=[initial_point],
                #     budget=budget,
                #     rhobeg=rhobeg_pool,
                #     xtol_abs=xtol_pool,
                #     shots=shots_pool,
                #     scaling=scaling,
                # ),
                HyperparameterGrid(
                    NLoptOptimizer("LN_COBYLA"),
                    initial_point=[initial_point],
                    maxeval=maxfev_pool,
                    initial_step=rhobeg_pool,
                    ftol_rel=[1e-13],
                    executor_kwargs={"shots": shots_pool},
                ),
            ]

            eval_func = get_evaluate_energy(
                instance,
                precomputed_energies,
                p,
                objective="expectation",
                simulator=simulator,
            )
            initial_ar.append(
                eval_point(
                    initial_point,
                    eval_func,
                    (minval, maxval),
                    sense,
                    beta if args.fix_beta else None,
                )
            )
            print(f"{p=} {n=} {seed=} initial_ar={initial_ar[-1]}", flush=True)

            tuner = HyperparameterTuner(configs)
            shotted_executor = CustomExecutor(
                partial(
                    shotted_measurement,
                    function=get_evaluate_energy(
                        instance,
                        precomputed_energies,
                        p,
                        objective=("expectation", "std"),
                        simulator=simulator,
                    ),
                    sense=sense,
                    fix_beta=beta if args.fix_beta else None,
                )
            )
            tuner.run(shotted_executor)

            result = tuner.process_results(
                partial(
                    process_results,
                    eval_func=eval_func,
                    optimal_metric=(minval, maxval),
                    sense=sense,
                    fix_beta=beta if args.fix_beta else None,
                )
            )
            params = tuner.process_results(
                lambda method, ind, trace, res: trace.optimal_params
            )

            for key, val in result.items():
                if key not in results:
                    results[key] = np.empty((len(seed_pool),) + val.shape)
                results[key][j] = val

            for key, val in params.items():
                if key not in optimal_params:
                    optimal_params[key] = np.empty((len(seed_pool),) + val.shape)
                optimal_params[key][j] = val

        for config in configs:
            method = config.method
            pickle.dump(
                {
                    "config": config,
                    "result": results[method],
                    "initial_ar": initial_ar,
                    "optimal_params": optimal_params[method],
                },
                open(
                    f"data/{problem}/configs/{target}/{method}-p{p}-q{n}-s{seed_pool[0]}-{seed_pool[-1]}.pckl",
                    "wb",
                ),
            )

        print("Mean initial AR:", np.mean(initial_ar))
        for i, (key, val) in enumerate(results.items()):
            mean = np.mean(val, axis=0)
            indices = np.argsort(mean.flat)[-1:-100:-1]
            for r, c in zip(mean.flatten()[indices], configs[i].interpret(indices)):
                print(r, c)
