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
    exact_fidelity,
    get_sk_ini
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

def random_orthogonal_vector(v):
    if np.all(v == 0):
        raise ValueError("The zero vector has no orthogonal vectors.")
    
    w = 2*np.random.rand(v.shape[0])-1
    w[-1] = 0

    if v[-1] == 0:
        w[0] = 0
        w[-1] = -(np.dot(v, w) / v[0])
    else:
        w[-1] = -(np.dot(v, w) / v[-1])

    return w

for N in [10]:
    K = int(N/2) 
    q = 0.5
    # for seed in range(3):
        po_path = f'data/po_problem_rule_{N}_{K}_{q}_seed{seed}.pckl'
        energy_path = f'data/precomputed_energies_rule_{N}_{K}_{q}_seed{seed}.npy'
        if Path(po_path).exists() and Path(energy_path).exists():
            precomputed_energies = np.load(energy_path)
            po_problem = pickle.load(open(po_path, 'rb'))
        else:
            po_problem_unscaled=get_problem(N, K, q, seed, pre=1)
            means_in_spins = np.array([po_problem_unscaled['means'][i] - po_problem_unscaled['q'] * np.sum(po_problem_unscaled['cov'][i, :]) for i in range(len(po_problem_unscaled['means']))])
            scale = 1 / (np.sqrt(np.mean((( po_problem_unscaled['q']*po_problem_unscaled['cov'])**2).flatten()))+np.sqrt(np.mean((means_in_spins**2).flatten())))
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
        sim = QAOAFURXYRingSimulatorC(N, po_problem['scale']*precomputed_energies)

        for p in [12,14]:
            ori_lst = ['optpath'] #['rand', 'opt', 'sk', 'optpath', 'rand_optpath'] 
            for ori in ori_lst:
                for rand_dim in range(20):
                    if ori == 'opt':
                        ori_x = pickle.load(open(f'traces/{N}_{K}_{q}_{seed}_p{p+1}_SK_init.pickle', 'rb'))['res_rescaled'][0]
                        dim1 = 2*np.random.rand(2*p)-1
                        dim1 = dim1/np.linalg.norm(dim1)
                        dim2 = random_orthogonal_vector(dim1)
                        dim2 = dim2/np.linalg.norm(dim2)
                    elif ori == 'sk':
                        ori_x = get_sk_ini(p)
                        dim1 = 2*np.random.rand(2*p)-1
                        dim1 = dim1/np.linalg.norm(dim1)
                        dim2 = random_orthogonal_vector(dim1)
                        dim2 = dim2/np.linalg.norm(dim2)
                    elif ori == 'rand':
                        ori_x = np.random.rand(p*2)
                        dim1 = 2*np.random.rand(2*p)-1
                        dim1 = dim1/np.linalg.norm(dim1)
                        dim2 = random_orthogonal_vector(dim1)
                        dim2 = dim2/np.linalg.norm(dim2)
                    elif ori == 'optpath':
                        ori_x = get_sk_ini(p)
                        opt_x = pickle.load(open(f'traces/{N}_{K}_{q}_{seed}_p{p+1}_SK_init.pickle', 'rb'))['res_rescaled'][0]
                        dim1 = opt_x-ori_x
                        dim2 = random_orthogonal_vector(dim1)
                        dim2 = dim2/np.linalg.norm(dim2)*np.linalg.norm(dim1)
                    elif ori == 'rand_optpath':
                        ori_x = np.random.rand(p*2)
                        opt_x = pickle.load(open(f'traces/{N}_{K}_{q}_{seed}_p{p+1}_SK_init.pickle', 'rb'))['res_rescaled'][0]
                        dim1 = opt_x-ori_x
                        dim2 = random_orthogonal_vector(dim1)
                        dim2 = dim2/np.linalg.norm(dim2)*np.linalg.norm(dim1)
                    ###################################
                    len_grid = 150
                    outpath = f'data/heatmap_{N}_{K}_{q}_seed{seed}_p{p}_{ori}_randdim{rand_dim}_grid{len_grid}.npy'
                    if not Path(outpath).exists():
                    # if True:
                        grid1 = np.linspace(-0.25, 1.25, len_grid)
                        grid2 = np.linspace(-1, 1, 100)
                        
                        # min_gamma = -1*np.pi
                        # max_gamma = 1*np.pi
                        # min_beta = 0
                        # max_beta = 2*np.pi
                        # betas = np.linspace(min_beta, max_beta, 25)
                        # gammas = np.linspace(min_gamma, max_gamma, npoints)

                        def f(para1, para2):
                            start_time=time.time()
                            update_x = ori_x + para1*dim1 + para2*dim2
                            gamma = update_x[:p] 
                            beta = update_x[p:] 
                            sv = sim.simulate_qaoa(gamma, beta, sv0 = generate_dicke_state_fast(N, K))
                            overlap = exact_fidelity(gs, sv.real + 1j * sv.imag)
                            print(f'para1-{para1:.2f}, para2-{para2:.2f}, time-{time.time()-start_time:.2f}',flush=True)
                            energy_mean = sv.get_norm_squared().dot(precomputed_energies)
                            energy_var = sv.get_norm_squared().dot(precomputed_energies**2) - energy_mean ** 2
                            energy_std = np.sqrt(energy_var.real)
                            return [energy_mean, energy_std, overlap.real]

                        with Pool(processes=6) as pool:
                            results = pool.starmap(f, product(grid1, grid2))
                            results = np.array(results).reshape(-1,3)
                        hm = np.array(results[:,0]).reshape(len_grid, 100).T
                        ar_hm = (hm-po_problem['feasible_max'])/(po_problem['feasible_min']-po_problem['feasible_max'])
                        overlap_hm = np.array(results[:,2]).reshape(len_grid, 100).T
                        hm_std = np.array(results[:,1]).reshape(len_grid, 100).T
                        ar_hm_std = hm_std/(po_problem['feasible_max']-po_problem['feasible_min'])
                        
                        np.save(outpath, (hm,overlap_hm,ar_hm,hm_std,ar_hm_std), allow_pickle=False)
