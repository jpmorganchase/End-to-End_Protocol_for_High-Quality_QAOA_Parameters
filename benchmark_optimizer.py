import itertools
import pickle
from functools import partial

import numpy as np
from oscar import (
    CustomExecutor,
    InterpolatedLandscapeExecutor,
    NLoptOptimizer,
    QiskitOptimizer,
    plot_2d_landscape,
)

sample_seed = 42
rng = np.random.default_rng(sample_seed)


def sample_energy(params, landscape, landscape_std, shots):
    energy = landscape.interpolator(params)[0]
    if shots is None:
        return energy
    energy_std = landscape_std.interpolator(params)[0] / np.sqrt(shots)
    return rng.normal(energy, energy_std)


def calculate_ar(po_problem, energy):
    min_constrained = po_problem["feasible_min"]
    max_constrained = po_problem["feasible_max"]
    return (energy - max_constrained) / (min_constrained - max_constrained)


p = 1
qubit_pool = [14, 16, 18]
seed_pool = list(range(10))
resolutions = [128, 32]
bounds = [(-2.2, -0.6), (0.9, 1.3)]
shots_pool = range(50, 2001, 500)
maxfev_pool = list(range(4, 101))
initial_point = [-1.24727193, 1.04931211]
# initial_point = [-2.2, 0.9]
reps = 2
method_pool = ["RECOBYLA"]
# budget = 10000
rhobeg_pool = np.linspace(0.1, 0.3, 2).tolist()
# rhobeg_pool = [0.2]
xtol_pool = np.linspace(0, 0.1, 2).tolist()
# xtol_pool = [0.045]
scaling = 2

configs = []
for method, shots, rhobeg, xtol in itertools.product(
    method_pool, shots_pool, rhobeg_pool, xtol_pool
):
    configs.append((method, shots, rhobeg, xtol))
mean_ar = np.zeros(len(configs) * len(maxfev_pool), dtype=float)
solved_ratio = np.zeros(len(configs) * len(maxfev_pool), dtype=float)
num_instances = len(qubit_pool) * len(seed_pool)
print(f"Trying {len(configs)} configs on {num_instances} instances for {reps} trials each...")

grid_best_ar, initial_point_ar = [], []
for n, seed in itertools.product(qubit_pool, seed_pool):
    print(f"Landscape {p=} {n=} {seed=}")
    filename = f"data/landscapes/{p=}/{n=}/{p=}-{n=}-{seed=}-{bounds}-{resolutions}"
    landscape = np.load(filename + ".pckl", allow_pickle=True)
    print("Grid minima: ", landscape.optimal_params().flatten(), landscape.optimal_value())
    landscape_std = np.load(filename + "-std.pckl", allow_pickle=True)
    po_problem = pickle.load(
        open(f"data/random/po_problem_rule_{n}_{n//2}_0.5_seed{seed}.pckl", "rb")
    )
    true_executor = InterpolatedLandscapeExecutor(landscape)
    grid_best_ar.append(calculate_ar(po_problem, landscape.optimal_value()))
    initial_point_ar.append(calculate_ar(po_problem, true_executor.run(initial_point)))

    for _ in range(reps):
        energies, solved_counts = [], []
        for method, shots, rhobeg, xtol in configs:
            budget = shots * maxfev_pool[-1]
            itpl_executor = CustomExecutor(
                partial(
                    sample_energy,
                    landscape=landscape,
                    landscape_std=landscape_std,
                    shots=shots,
                )
            )
            if method == "RECOBYLA":
                from restarting_cobyla import RECOBYLA

                trace, result = RECOBYLA().run(
                    CustomExecutor(
                        partial(
                            sample_energy,
                            landscape=landscape,
                            landscape_std=landscape_std,
                        )
                    ),
                    initial_point,
                    budget,
                    bounds,
                    rhobeg,
                    xtol,
                    shots,
                    scaling,
                    p,
                )
            elif method in ["COBYLA"]:
                from nlopt import LN_COBYLA, opt

                optimizer = opt(LN_COBYLA, 2 * p)
                optimizer.set_lower_bounds(np.array(bounds).T[0])
                optimizer.set_upper_bounds(np.array(bounds).T[1])
                optimizer.set_initial_step(rhobeg)
                # optimizer.set_xtol_abs(xtol)
                optimizer.set_maxeval(maxfev_pool[-1])
                trace, result = NLoptOptimizer(optimizer).run(
                    itpl_executor, initial_point=initial_point
                )
            else:
                from qiskit.algorithms import optimizers as qiskit_optimizers

                trace, _ = QiskitOptimizer(
                    getattr(qiskit_optimizers, method)(maxiter=maxfev, rhobeg=rhobeg)
                ).run(itpl_executor, initial_point=initial_point, bounds=bounds)
            true_trace = true_executor.run_with_trace(trace).value_trace
            best_val, opt_vals = np.inf, []
            for i, val in enumerate(trace.value_trace):
                if val < best_val:
                    opt_vals.append(true_trace[i])
                    best_val = val
                else:
                    opt_vals.append(opt_vals[-1])
            for maxfev in maxfev_pool:
                energies.append(opt_vals[min(maxfev, len(opt_vals)) - 1])
            # print(method, maxfev, shots, trace.num_fun_evals, rhobeg, trace.num_iters)
        ar = calculate_ar(po_problem, np.array(energies))
        mean_ar = mean_ar + ar
        solved_counts = (grid_best_ar[-1] - ar) <= grid_best_ar[-1] * 0.01
        solved_ratio += np.array(solved_counts, dtype=float)

mean_ar /= num_instances * reps
solved_ratio /= num_instances * reps

for i, method in enumerate(method_pool):
    pickle.dump(
        {
            "mean_ar": mean_ar.reshape(len(method_pool), -1)[i],
            "solved_ratio": solved_ratio.reshape(len(method_pool), -1)[i],
            "configs": configs,
            "method": method,
            "maxfev_pool": maxfev_pool,
            "shots_pool": shots_pool,
            "rhobeg_pool": rhobeg_pool,
            "p": p,
            "qubit_pool": qubit_pool,
            "seed_pool": seed_pool,
            "resolutions": resolutions,
            "bounds": bounds,
            "initial_point": initial_point,
            "reps": reps,
            # "budget": budget,
            "scaling": scaling,
            "xtol_pool": xtol_pool,
        },
        open(
            f"data/configs/{method}-p{p}-q{qubit_pool}-s{len(seed_pool)}-b{bounds}-r{resolutions}-i{initial_point}.pckl",
            "wb",
        ),
    )
