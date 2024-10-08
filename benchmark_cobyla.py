import argparse
import os
import pickle
from functools import partial

import numpy as np
from oscar import (CustomExecutor, HyperparameterGrid, HyperparameterTuner,
                   NLoptOptimizer)
from qokit.parameter_utils import get_fixed_gamma_beta, get_sk_gamma_beta

from evaluate_energy import get_evaluate_energy, load_problem

sample_seed = 42
rng = np.random.default_rng(sample_seed)


def shotted_measurement(params, function, shots, sense, fix_beta=None):
    if fix_beta is not None:
        params = np.concatenate([params, fix_beta])
    if shots is None:
        mean = function(params)
        return sense * mean
    mean, std = function(params)
    return sense * rng.normal(mean, std / np.sqrt(shots))


def eval_point(point, eval_func, optimal_metric, sense, fix_beta=None):
    if fix_beta is not None:
        point = np.concatenate([point, fix_beta])
    result = eval_func(point)
    if isinstance(result, tuple):
        result = result[0]
    return (
        sense
        * (sense * result - optimal_metric[int((sense + 1) / 2)])
        / (optimal_metric[0] - optimal_metric[1])
    )


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
    parser.add_argument(
        "--no-precompute", dest="precompute", default=False, action="store_true"
    )

    args = parser.parse_args()
    print(args)

    problem = args.problem
    p = args.p
    seed_pool = list(range(args.batch * args.seed, (args.batch + 1) * args.seed))
    simulator = "c" if args.cpu else "auto"
    qubit_pool = list(range(args.n, args.n + 1, 2))
    budget = 10000
    target = args.target
    shots_pool = [None]
    rhobeg_pool = [0.1]
    if target == "max_ar":
        maxfev_pool = [1000]
    elif target == "budget":
        if args.fix_beta:
            target += "-fix-beta"
            maxfev_pool = list(range(p + 2, 20)) + list(range(20, 51, 5))
        else:
            maxfev_pool = list(range(2 * p + 2, 20)) + list(range(20, 51, 5))
        # shots_pool = list(range(500, 2501, 100))
        shots_pool = budget // np.array(maxfev_pool)
    elif target == "rhobeg":
        maxfev_pool = list(range(2 * p + 2, ((p + 1) // 2 + 1) * 5)) + list(
            range(((p + 1) // 2 + 1) * 5, 51, 5)
        )
        rhobeg_pool = np.linspace(0.01, 1, 20).tolist()
    elif target == "opt2steps":
        maxfev_pool = [2 * p + 3]
        shots_pool = budget // np.array(maxfev_pool)
    else:
        raise NotImplementedError()

    for i, n in enumerate(qubit_pool):
        results, optimal_params = {}, {}
        initial_ar = []
        for j, seed in enumerate(seed_pool):
            instance, precomputed_energies = load_problem(
                problem, n, seed, not args.precompute
            )
            if problem == "po":
                sense = 1
                beta_scaling = -8
                # initial_point = [-1.24727193, 1.04931211 * 8]
                gamma, beta = get_sk_gamma_beta(p)
                minval, maxval = instance["feasible_min"], instance["feasible_max"]
                if target != "rhobeg":
                    rhobeg_pool = [0.5]
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
                objective=(
                    "expectation" if None in shots_pool else ("expectation", "std")
                ),
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
                    function=eval_func,
                    sense=1,
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
            del eval_func, shotted_executor, tuner, precomputed_energies

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
            savepath = f"data/{problem}/configs/{target}/{method}-p{p}-q{n}-s{seed_pool[0]}-{seed_pool[-1]}.pckl"
            os.makedirs(os.path.dirname(savepath), exist_ok=True)
            pickle.dump(
                {
                    "config": config,
                    "result": results[method],
                    "initial_ar": initial_ar,
                    "optimal_params": optimal_params[method],
                },
                open(savepath, "wb"),
            )

        print("Mean initial AR:", np.mean(initial_ar))
        for i, (key, val) in enumerate(results.items()):
            mean = np.mean(val, axis=0)
            indices = np.argsort(mean.flat)[-1:-100:-1]
            for r, c in zip(mean.flatten()[indices], configs[i].interpret(indices)):
                print(r, c)
