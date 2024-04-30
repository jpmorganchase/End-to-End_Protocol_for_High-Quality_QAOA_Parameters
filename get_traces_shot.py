import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

import itertools
import pickle
import time
from functools import partial
from itertools import product, starmap
from pathlib import Path

import matplotlib.pyplot as plt
import nlopt
import numpy as np
from tqdm import tqdm

from circuit_utils import (
    apply_mixer_Txy,
    apply_mixer_Txy_yue,
    get_configuration_cost,
    get_configuration_cost_kw,
    get_configuration_cost_slow,
    get_qaoa_circuit,
    measure_circuit,
)
from fur import QAOAFURXYRingSimulatorC
from get_traces import minimize_skquant
from optimizer import circuit_measurement_function
from utils import (
    generate_dicke_state_fast,
    get_adjusted_state,
    get_problem,
    get_problem_H,
    get_sk_ini,
    precompute_energies_parallel,
    scale_map,
)

n_shot_pool = [None]
qubit_pool = range(22, 24, 2)
depth_pool = [1]
seed_pool = range(0, 100)
optimizer = "bobyqa"
budget = 100000
bounds = np.array([[-1.8, -0.6], [0.8, 1.4]], dtype=float)


def minimize_nlopt(f, X0, rhobeg=None, p=None):
    all_solution = []

    def nlopt_wrapper(x, grad):
        if grad.size > 0:
            raise RuntimeError("Shouldn't be calling a gradient!")
        all_solution.append(f(x))
        return f(x)

    opt = nlopt.opt(nlopt.LN_BOBYQA, 2 * p)
    opt.set_min_objective(nlopt_wrapper)
    # if n_shot is False:
    #     opt.set_xtol_rel(1e-4)
    #     opt.set_ftol_rel(1e-4)
    # else:
    #     opt.set_xtol_rel(1e-4/np.sqrt(n_shot)) #####
    #     opt.set_ftol_rel(1e-4/np.sqrt(n_shot)) #####
    opt.set_xtol_rel(1e-4)
    opt.set_ftol_rel(1e-4)
    opt.set_initial_step(rhobeg)

    xstar = opt.optimize(X0)
    minf = opt.last_optimum_value()
    return xstar, minf, all_solution


def get_trace(N, K, X0, precomputed_energies, factor=1, n_shot=None):
    trace, trace_sv = [], []
    p = int(len(X0) // 2)

    def f(theta):
        gamma, beta = theta[:p], theta[p:]
        sim = QAOAFURXYRingSimulatorC(N, factor * precomputed_energies)
        sv = sim.simulate_qaoa(gamma, beta, sv0=generate_dicke_state_fast(N, K))
        energy_mean = sv.get_norm_squared().dot(precomputed_energies).real
        if n_shot is None:
            # print(energy_mean)
            trace.append(energy_mean)
            trace_sv.append(energy_mean)
            return energy_mean
        else:
            energy_var = sv.get_norm_squared().dot(precomputed_energies**2) - energy_mean**2
            # if len(trace) > 50:
            #     energy_std = np.sqrt(energy_var.real)/np.sqrt(n_shot*10)
            # else:
            energy_std = np.sqrt(energy_var.real) / np.sqrt(n_shot)

            en = np.random.normal(energy_mean, energy_std)
            # print(energy_mean,energy_std,en)
            trace.append(en)
            trace_sv.append(energy_mean)
            return en

    # res = minimize_skquant(
    #     f,
    #     X0,
    #     bounds=bounds,
    #     budget=budget // (100 if n_shot is None else n_shot),
    #     method=optimizer,
    # )
    res = minimize_nlopt(f, X0, p=p, rhobeg=0.01 / p)

    return trace, trace_sv, res


def get_trace_qiskit(N, K, X0, po_problem, factor=1):
    trace = []
    p = int(len(X0) // 2)

    def f(theta):
        obj = circuit_measurement_function(
            po_problem=po_problem,
            p=p,
            ini="dicke",
            mixer="t-xy",
            T=1,
            ini_state=None,
            save_state=False,
            n_trials=100,  # number of shots if save_state is False
            minus=False,
        )
        gammas = theta[:p] / 2
        betas = theta[p:] / 4
        ini_x = np.concatenate((gammas, betas), axis=0)
        en = obj(ini_x)
        trace.append(en)
        return en

    res = minimize_nlopt(f, X0, p=p, rhobeg=0.01 / p)
    return trace, res


def kbits(n, k):
    for bits in itertools.combinations(range(n), k):
        s = [0] * n
        for bit in bits:
            s[bit] = 1
        yield np.array(s)


data_dir = "data/random"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
out_dir = f"data/traces/{optimizer}"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

for n_shot in n_shot_pool:
    for N in qubit_pool:
        K = int(N / 2)
        # K = 5
        q = 0.5
        for seed in seed_pool:
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
            for p in depth_pool:
                outpath = f"{out_dir}/{N}_{seed}_p{p}_{optimizer.lower()}_{budget}eval_{n_shot}shot.pickle"
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
                    # orig_time0 = time.time()
                    # trace_orig, res_orig = get_trace(N, K, X0, precomputed_energies, factor=1)
                    # orig_time = time.time()-orig_time0
                    rescale_time0 = time.time()
                    # trace_rescaled, res_rescaled = get_trace_qiskit(
                    #     N,
                    #     K,
                    #     X0,
                    #     po_problem,
                    #     factor = po_problem['scale']
                    # )
                    trace_rescaled, trace_rescaled_sv, res_rescaled = get_trace(
                        N,
                        K,
                        X0,
                        precomputed_energies,
                        factor=po_problem["scale"],
                        n_shot=n_shot,
                    )
                    rescale_time = time.time() - rescale_time0
                    # orig_AR = (res_orig[1]-max_constrained)/(min_constrained-max_constrained)
                    rescale_AR = (res_rescaled[1] - max_constrained) / (
                        min_constrained - max_constrained
                    )
                    sv_AR = (res_rescaled[1] - max_constrained) / (
                        min_constrained - max_constrained
                    )
                    res = {
                        "N": N,
                        "K": K,
                        "seed": seed,
                        "po_problem": po_problem,
                        "trace_rescaled": trace_rescaled,
                        "trace_sv": trace_rescaled_sv,
                        "res_rescaled": res_rescaled,
                        # 'res_orig':res_orig,
                        # 'trace_orig':tdrace_orig,
                        "rescale_time": rescale_time,
                        # 'orig_time':orig_time,
                        "rescale_ar": rescale_AR,
                        # 'orig_ar':orig_AR,
                        "opt_params": res_rescaled[0],
                    }
                    print(
                        f"Completed {N=} {seed=} {p=} res_AR={rescale_AR:.4f} {n_shot=}, func_call={len(trace_rescaled)}, norm_diffx={np.linalg.norm(res_rescaled[0]-X0)/np.linalg.norm(X0):.2f}, time={rescale_time:.2f}\n"
                    )  # X0={X0}, opt_x={res_rescaled[0]}, res_energy = {res_rescaled[1]:.4f}
                    pickle.dump(res, open(outpath, "wb"))