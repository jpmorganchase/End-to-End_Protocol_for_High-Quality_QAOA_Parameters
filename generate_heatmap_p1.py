import os
os.environ["OMP_NUM_THREADS"] = "25"
os.environ["NUMBA_NUM_THREADS"] = "25"

import numpy as np
from functools import partial
import itertools
from itertools import starmap, product
from pathos.multiprocessing import Pool
import time
import pickle
from pathlib import Path
from tqdm import tqdm

from utils import (
    get_problem,
    precompute_energies_parallel,
    get_adjusted_state,
    get_problem_H,
    scale_map,
    get_real_problem,
    generata_dicke_state,
    generate_dicke_state_fast,
    binary_array_to_decimal,
    exact_fidelity
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
def kbits(n, k):
    for bits in itertools.combinations(range(n), k):
        s = [0] * n
        for bit in bits:
            s[bit] = 1
        yield np.array(s)
        
for N in [14]:
    # for K in [1,2,3,4]:
    K = N//2 #, 5
    q = 0.5
    # scale = 100 

    for seed in range(10):
        po_path = f'data/po_problem_rule_{N}_{K}_{q}_seed{seed}.pckl'
        energy_path = f'data/precomputed_energies_rule_{N}_{K}_{q}_seed{seed}.npy'
        if Path(po_path).exists() and Path(energy_path).exists():
            precomputed_energies = np.load(energy_path)
            po_problem = pickle.load(open(po_path, 'rb'))
            scale = po_problem["scale"]
        else:
            po_problem_unscaled=get_problem(N, K, q, seed, pre=1)
            means_in_spins = np.array([po_problem_unscaled['means'][i] - po_problem_unscaled['q'] * np.sum(po_problem_unscaled['cov'][i, :]) for i in range(len(po_problem_unscaled['means']))])
            scale = 1 / (np.sqrt(np.mean((( po_problem_unscaled['q']*po_problem_unscaled['cov'])**2).flatten()))+np.sqrt(np.mean((means_in_spins**2).flatten())))
            print(f'scale = {scale}')
            po_problem = get_problem(N=N,K=K,q=q,seed=seed,pre=scale)
           
            min_constrained = float('inf')
            max_constrained = float('-inf')
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
            po_problem['feasible_min'] = min_constrained
            po_problem['feasible_max'] = max_constrained
            po_problem['feasible_min_x'] = min_x
            po_problem['feasible_max_x'] = max_x
            po_problem['feasible_mean'] = mean_constrained
            precomputed_energies = get_adjusted_state(precompute_energies_parallel(po_obj, N, 1))

            np.save(energy_path, precomputed_energies, allow_pickle=False)
            pickle.dump(po_problem, open(po_path, 'wb'))
        ################################################

        index = binary_array_to_decimal(po_problem['feasible_min_x'])
        gs = np.zeros(2**N)
        gs[index] = 1 
        gs = 1 / np.sqrt(np.sum(gs)) * gs
        ################
        sim = QAOAFURXYRingSimulatorC(N, scale*precomputed_energies)

        def f(gamma, beta):
            start_time=time.time()
            sv = sim.simulate_qaoa([gamma], [beta], sv0=generate_dicke_state_fast(N, K))
            overlap = exact_fidelity(gs,sv.real + 1j * sv.imag)
            print(f'gamma-{gamma:.2f}, beta-{beta:.2f}, time-{time.time()-start_time:.2f}',flush=True)
            return [sv.get_norm_squared().dot(precomputed_energies), overlap.real]

        npoints = 100
        min_gamma = -1*np.pi
        max_gamma = 1*np.pi
        min_beta = 0
        max_beta = 2*np.pi
        
        betas = np.linspace(min_beta, max_beta, 25 )
        gammas = np.linspace(min_gamma, max_gamma, npoints)

        # hm = list(starmap(f, product(gammas, betas)))
        with Pool(processes=6) as pool:
            results = pool.starmap(f, product(gammas, betas))
            results = np.array(results).reshape(-1,2)
        hm = np.array(results[:,0]).reshape(npoints, 25).T
        overlap_hm = np.array(results[:,1]).reshape(npoints, 25).T

        np.save(f'data/heatmap_{N}_{K}_{q}_p1_seed{seed}_rand_rule.npy', (hm,overlap_hm), allow_pickle=False)
