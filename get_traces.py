import os

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["NUMBA_NUM_THREADS"] = "8"

import numpy as np
from functools import partial
from itertools import starmap, product
import pickle
from pathlib import Path
import itertools

from utils import (
    get_problem,
    precompute_energies_parallel,
    get_adjusted_state,
    get_problem_H,
    scale_map,
    generate_dicke_state_fast,
    get_sk_ini,
)
from circuit_utils import (
    get_configuration_cost_slow,
    get_configuration_cost,
    get_configuration_cost_kw,
    apply_mixer_Txy,
    apply_mixer_Txy_yue,
    get_qaoa_circuit,
    measure_circuit,
)
from optimizer import circuit_measurement_function
from fur import QAOAFURXYRingSimulatorC

from tqdm import tqdm
import matplotlib.pyplot as plt
import nlopt
import time
from skquant.opt import minimize


optimizer = "snofit"
budget = 30


def minimize_skquant(f, X0, bounds, budget, method):
    result, history = minimize(f, X0, bounds, budget=budget, method=method)
    return result.optpar, result.optval


def minimize_nlopt(f, X0, rhobeg=None, p=None):
    def nlopt_wrapper(x, grad):
        if grad.size > 0:
            raise RuntimeError("Shouldn't be calling a gradient!")
        return f(x)

    opt = nlopt.opt(nlopt.LN_BOBYQA, 2 * p)
    opt.set_min_objective(nlopt_wrapper)

    opt.set_xtol_rel(1e-8)
    opt.set_ftol_rel(1e-8)
    opt.set_initial_step(rhobeg)

    xstar = opt.optimize(X0)
    minf = opt.last_optimum_value()

    return xstar, minf


def get_trace(N, K, X0, precomputed_energies, factor=1):
    trace = []
    p = int(len(X0) // 2)

    def f(theta):
        gamma, beta = theta[:p], theta[p:]
        sim = QAOAFURXYRingSimulatorC(N, factor * precomputed_energies)
        sv = sim.simulate_qaoa(gamma, beta, sv0=generate_dicke_state_fast(N, K))
        en = sv.get_norm_squared().dot(precomputed_energies).real
        trace.append(en)
        return en

    # res = minimize_skquant(
    #     f,
    #     X0,
    #     bounds=np.array([[-1.4, -0.8], [0.9, 1.3]], dtype=float),
    #     budget=budget,
    #     method=optimizer,
    # )
    res = minimize_nlopt(f, X0, p=p, rhobeg=0.01 / p)

    return trace, res


def kbits(n, k):
    for bits in itertools.combinations(range(n), k):
        s = [0] * n
        for bit in bits:
            s[bit] = 1
        yield np.array(s)


if __name__ == "__main__":
    data_dir = "data/random"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    out_dir = "traces"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for N in [14]:  # ,11,15
        K = int(N / 2)
        q = 0.5
        for seed in range(10):
            po_path = f"{data_dir}/po_problem_rule_{N}_{K}_{q}_seed{seed}.pckl"
            energy_path = f"{data_dir}/precomputed_energies_rule_{N}_{K}_{q}_seed{seed}.npy"
            if Path(po_path).exists() and Path(energy_path).exists():
                precomputed_energies = np.load(energy_path)
                po_problem = pickle.load(open(po_path, "rb"))
                min_constrained = po_problem["feasible_min"]
                max_constrained = po_problem["feasible_max"]
            else:
                po_problem_unscaled = get_problem(N, K, q, seed=seed, pre=1)
                means_in_spins = np.array(
                    [
                        po_problem_unscaled["means"][i]
                        - po_problem_unscaled["q"] * np.sum(po_problem_unscaled["cov"][i, :])
                        for i in range(len(po_problem_unscaled["means"]))
                    ]
                )
                scale = 1 / (
                    np.sqrt(
                        np.mean(
                            (
                                (po_problem_unscaled["q"] * po_problem_unscaled["cov"]) ** 2
                            ).flatten()
                        )
                    )
                    + np.sqrt(np.mean((means_in_spins**2).flatten()))
                )
                po_problem = get_problem(N=N, K=K, q=q, seed=seed, pre=scale)

                min_constrained = float("inf")
                max_constrained = float("-inf")
                mean_constrained = 0
                total_constrained = 0
                po_obj = partial(get_configuration_cost_kw, po_problem=po_problem)
                for x in tqdm(kbits(N, K)):
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
                precomputed_energies = get_adjusted_state(
                    precompute_energies_parallel(po_obj, N, 1)
                )

                np.save(energy_path, precomputed_energies, allow_pickle=False)
                pickle.dump(po_problem, open(po_path, "wb"))
            ########################
            for p in [1]:
                outpath = f"{out_dir}/{N}_{K}_{q}_{seed}_p{p}_{optimizer.lower()}_{budget}.pickle"
                # if not Path(outpath).exists():
                if True:
                    np.random.seed(seed)
                    sk_ini = True
                    if sk_ini is True:
                        gamma, beta = get_sk_ini(p)
                        X0 = np.concatenate((gamma, beta), axis=0)
                        X0 = np.array([-1, 1], dtype=float)
                    else:
                        X0 = np.random.rand(p * 2)

                    ##########################
                    orig_time0 = time.time()
                    trace_orig, res_orig = get_trace(N, K, X0, precomputed_energies, factor=1)
                    orig_time = time.time() - orig_time0
                    rescale_time0 = time.time()
                    trace_rescaled, res_rescaled = get_trace(
                        N,
                        K,
                        X0,
                        precomputed_energies,
                        factor=po_problem["scale"]
                        # 1 / (0.5*(np.sqrt(np.mean((po_problem['cov']**2).flatten()))+np.sqrt(np.mean((po_problem['means']**2).flatten())))),
                        # precomputed_energies / (np.sqrt(np.mean((po_problem['cov']**2).flatten()))),
                        # precomputed_energies / (np.sqrt(np.mean((po_problem['means']**2).flatten()))),
                    )
                    rescale_time = time.time() - rescale_time0
                    orig_AR = (res_orig[1] - max_constrained) / (min_constrained - max_constrained)
                    rescale_AR = (res_rescaled[1] - max_constrained) / (
                        min_constrained - max_constrained
                    )
                    res = {
                        "N": N,
                        "K": K,
                        "seed": seed,
                        "po_problem": po_problem,
                        "trace_rescaled": trace_rescaled,
                        "res_rescaled": res_rescaled,
                        "res_orig": res_orig,
                        "trace_orig": trace_orig,
                        "rescale_time": rescale_time,
                        "orig_time": orig_time,
                        "rescale_ar": rescale_AR,
                        "orig_ar": orig_AR,
                    }
                    print(
                        f"Completed N={N}, seed={seed}, p={p}, res_AR={rescale_AR:.4f}, X0={X0}, func_call={len(trace_rescaled)}, opt_x={res_rescaled[0]}, res_energy = {res_rescaled[1]:.4f}, time = {rescale_time:.2f}\n"
                    )
                    pickle.dump(res, open(outpath, "wb"))