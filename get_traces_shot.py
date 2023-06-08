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

n_shot = 1e3
def minimize_nlopt(f, X0, rhobeg=None, p=None):
    all_solution = []
    def nlopt_wrapper(x, grad):
        if grad.size > 0:
            sys.exit("Shouldn't be calling a gradient!")
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

def get_trace(N, K, X0, precomputed_energies, factor=1):
    trace = []
    p = int(len(X0)//2)
    
    def f(theta):
        gamma, beta = theta[:p], theta[p:]
        sim = QAOAFURXYRingSimulatorC(N, factor*precomputed_energies)
        sv = sim.simulate_qaoa(gamma, beta, sv0=generate_dicke_state_fast(N, K))
        energy_mean = sv.get_norm_squared().dot(precomputed_energies).real
        if n_shot is False:
            # print(energy_mean)
            trace.append(energy_mean)
            return energy_mean
        else:
            energy_var = sv.get_norm_squared().dot(precomputed_energies**2) - energy_mean ** 2
            # if len(trace) > 50:
            #     energy_std = np.sqrt(energy_var.real)/np.sqrt(n_shot*10)
            # else:
            energy_std = np.sqrt(energy_var.real)/np.sqrt(n_shot)
            
            en = np.random.normal(energy_mean, energy_std)
            # print(energy_mean,energy_std,en)
            trace.append(en)
            return en
        
    if n_shot is False:
        res = minimize_nlopt(f, X0, p=p, rhobeg=0.01/p)
    else:
        res = minimize_nlopt(f, X0, p=p, rhobeg=100/p) #rhobeg=10/np.sqrt(n_shot)*p
    return trace, res

def get_trace_qiskit(N, K, X0, po_problem, factor=1):
    trace = []
    p = int(len(X0)//2)
    
    def f(theta):
        obj = circuit_measurement_function(
                po_problem = po_problem,
                p=p,
                ini='dicke',
                mixer='t-xy',
                T=1,
                ini_state=None, 
                save_state=False,
                n_trials=100, # number of shots if save_state is False
                minus=False,
            )
        gammas = theta[:p]/2
        betas = theta[p:]/4
        ini_x = np.concatenate((gammas,betas),axis=0)
        en = obj(ini_x)
        trace.append(en)
        return en

    res = minimize_nlopt(f, X0, p=p, rhobeg=0.01/p)
    return trace, res

def kbits(n, k):
    for bits in itertools.combinations(range(n), k):
        s = [0] * n
        for bit in bits:
            s[bit] = 1
        yield np.array(s)
        

for N in [10]: #,11,15
    K = int(N/2)
    q = 0.5
    for seed in range(10):
        po_path = f'data/random/po_problem_rule_{N}_{K}_{q}_seed{seed}.pckl'
        energy_path = f'data/random/precomputed_energies_rule_{N}_{K}_{q}_seed{seed}.npy'
        if Path(po_path).exists() and Path(energy_path).exists():
            precomputed_energies = np.load(energy_path)
            po_problem = pickle.load(open(po_path, 'rb'))
            min_constrained = po_problem['feasible_min'] 
            max_constrained = po_problem['feasible_max'] 
        else:
            po_problem_unscaled=get_problem(N, K, q, seed=seed, pre=1)
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
         ########################
        for p in range(1,13):
            outpath = f'traces_shot/{N}_{K}_{q}_{seed}_p{p+1}_shot{int(n_shot)}_SK_init.pickle'
            # if not Path(outpath).exists():
            if True:
                np.random.seed(seed)
                sk_ini = True
                gamma_scale, beta_scale = -2, 2
                if sk_ini is True:
                    if p == 1:
                        gamma = gamma_scale*np.array([0.5])
                        beta = beta_scale*np.array([np.pi/8])
                        # gamma = np.array([1.9675])
                        # beta = np.array([1.0472])
                        X0 = np.concatenate((gamma,beta),axis=0)
                    elif p == 2:
                        gamma = gamma_scale*np.array([0.3817, 0.6655])
                        beta = beta_scale*np.array([0.4960, 0.2690])
                        X0 = np.concatenate((gamma,beta),axis=0)
                    elif p == 3:
                        gamma = gamma_scale*np.array([0.3297, 0.5688, 0.6406])
                        beta = beta_scale*np.array([0.5500, 0.3675, 0.2109])
                        X0 = np.concatenate((gamma,beta),axis=0)
                    elif p == 4:
                        gamma = gamma_scale*np.array([0.2949, 0.5144, 0.5586, 0.6429 ])
                        beta = beta_scale*np.array([0.5710, 0.4176, 0.3028, 0.1729])
                        X0 = np.concatenate((gamma,beta),axis=0)
                    elif p == 5: 
                        gamma = gamma_scale*np.array([0.2705, 0.4804, 0.5074, 0.5646, 0.6397])
                        beta = beta_scale*np.array([0.5899, 0.4492, 0.3559, 0.2643, 0.1486])
                        X0 = np.concatenate((gamma,beta),axis=0)
                    elif p == 6:
                        gamma = gamma_scale*np.array([0.2528, 0.4531, 0.4750, 0.5146, 0.5650, 0.6392 ])
                        beta = beta_scale*np.array([0.6004, 0.4670, 0.3880, 0.3176, 0.2325, 0.1291])
                        X0 = np.concatenate((gamma,beta),axis=0)
                    elif p == 7:
                        gamma = gamma_scale*np.array([0.2383, 0.4327, 0.4516, 0.4830, 0.5147, 0.5686, 0.6393])
                        beta = beta_scale*np.array([0.6085, 0.4810, 0.4090, 0.3534, 0.2857, 0.2080, 0.1146])
                        X0 = np.concatenate((gamma,beta),axis=0)
                    elif p == 8:
                        gamma = gamma_scale*np.array([0.2268, 0.4162, 0.4332, 0.4608, 0.4818, 0.5179, 0.5717, 0.6393])
                        beta = beta_scale*np.array([0.6151, 0.4906, 0.4244, 0.3780, 0.3224, 0.2606, 0.1884, 0.1030])
                        X0 = np.concatenate((gamma,beta),axis=0)
                    elif p == 9:
                        gamma = gamma_scale*np.array([0.2172, 0.4020, 0.4187, 0.4438, 0.4592, 0.4838, 0.5212, 0.5754, 0.6398])
                        beta = beta_scale*np.array([0.6196, 0.4973, 0.4354, 0.3956, 0.3481, 0.2973, 0.2390, 0.1717, 0.0934])
                        X0 = np.concatenate((gamma,beta),axis=0)
                    elif p == 10:
                        gamma = gamma_scale*np.array([0.2089, 0.3902, 0.4066, 0.4305, 0.4423, 0.4604, 0.4858, 0.5256, 0.5789, 0.6402])
                        beta = beta_scale*np.array([0.6235, 0.5029, 0.4437, 0.4092, 0.3673, 0.3246, 0.2758, 0.2208, 0.1578, 0.0855])
                        X0 = np.concatenate((gamma,beta),axis=0)
                    elif p == 11:
                        gamma = gamma_scale*np.array([0.2019, 0.3799, 0.3963, 0.4196, 0.4291,0.4431, 0.4611, 0.4895, 0.5299, 0.5821,0.6406])
                        beta = beta_scale*np.array([0.6268, 0.5070, 0.4502, 0.4195, 0.3822,0.3451, 0.3036, 0.2571, 0.2051, 0.1459,0.0788])
                        X0 = np.concatenate((gamma,beta),axis=0)
                    elif p == 12:
                        gamma = gamma_scale*np.array([0.1958, 0.3708, 0.3875, 0.4103, 0.4185,0.4297, 0.4430, 0.4639, 0.4933, 0.5343,0.5851, 0.6410 ])
                        beta = beta_scale*np.array([0.6293, 0.5103, 0.4553, 0.4275, 0.3937,0.3612, 0.3248, 0.2849, 0.2406, 0.1913,0.1356, 0.0731])
                        X0 = np.concatenate((gamma,beta),axis=0)
                    elif p == 13:
                        gamma = gamma_scale*np.array([0.1903, 0.3627, 0.3797, 0.4024, 0.4096,0.4191, 0.4290, 0.4450, 0.4668, 0.4975,0.5385, 0.5878, 0.6414])
                        beta = beta_scale*np.array([0.6315, 0.5130, 0.4593, 0.4340, 0.4028,0.3740, 0.3417, 0.3068, 0.2684, 0.2260,0.1792, 0.1266, 0.0681])
                        X0 = np.concatenate((gamma,beta),axis=0)
                    elif p == 14:
                        gamma = gamma_scale*np.array([0.1855, 0.3555, 0.3728, 0.3954, 0.4020,0.4103, 0.4179, 0.4304, 0.4471, 0.4703,0.5017, 0.5425, 0.5902, 0.6418])
                        beta = beta_scale*np.array([0.6334, 0.5152, 0.4627, 0.4392, 0.4103,0.3843, 0.3554, 0.3243, 0.2906, 0.2535,0.2131, 0.1685, 0.1188, 0.0638])
                        X0 = np.concatenate((gamma,beta),axis=0)
                    elif p == 15:
                        gamma = gamma_scale*np.array([0.1811, 0.3489, 0.3667, 0.3893, 0.3954,0.4028, 0.4088, 0.4189, 0.4318, 0.4501,0.4740, 0.5058, 0.5462, 0.5924, 0.6422])
                        beta = beta_scale*np.array([0.6349, 0.5169, 0.4655, 0.4434, 0.4163, 0.3927, 0.3664, 0.3387, 0.3086, 0.2758, 0.2402, 0.2015, 0.1589, 0.1118, 0.0600])
                        X0 = np.concatenate((gamma,beta),axis=0)
                    elif p == 16:
                        gamma = gamma_scale*np.array([0.1771, 0.3430, 0.3612, 0.3838, 0.3896,0.3964, 0.4011, 0.4095, 0.4197, 0.4343,0.4532, 0.4778, 0.5099, 0.5497, 0.5944,0.6425 ])
                        beta = beta_scale*np.array([0.6363, 0.5184, 0.4678, 0.4469, 0.4213,0.3996, 0.3756, 0.3505, 0.3234, 0.2940,0.2624, 0.2281, 0.1910, 0.1504, 0.1056,0.0566])
                        X0 = np.concatenate((gamma,beta),axis=0)
                    elif p == 17:
                        gamma = gamma_scale*np.array([0.1735, 0.3376, 0.3562, 0.3789, 0.3844,0.3907, 0.3946, 0.4016, 0.4099, 0.4217,0.4370, 0.4565, 0.4816, 0.5138, 0.5530,0.5962, 0.6429 ])
                        beta = beta_scale*np.array([0.6375, 0.5197, 0.4697, 0.4499, 0.4255,0.4054, 0.3832, 0.3603, 0.3358, 0.3092,0.2807, 0.2501, 0.2171, 0.1816, 0.1426,0.1001, 0.0536])
                        X0 = np.concatenate((gamma,beta),axis=0)
                else:
                    X0 = np.random.rand(p*2)
                # X0 = [np.random.uniform(-np.pi, 0), np.random.uniform(0, np.pi)]
                
                ##########################
                orig_time0 = time.time()
                # trace_orig, res_orig = get_trace_qiskit(N, K, X0, po_problem, factor=1)
                trace_orig, res_orig = get_trace(N, K, X0, precomputed_energies, factor=1)
                orig_time = time.time()-orig_time0
                rescale_time0 = time.time()
                # trace_rescaled, res_rescaled = get_trace_qiskit(
                #     N, 
                #     K, 
                #     X0,
                #     po_problem,
                #     factor = po_problem['scale']
                # )
                trace_rescaled, res_rescaled = get_trace(
                    N, 
                    K, 
                    X0,
                    precomputed_energies,
                    factor = po_problem['scale']
                )
                rescale_time = time.time()-rescale_time0
                orig_AR = (res_orig[1]-max_constrained)/(min_constrained-max_constrained)
                rescale_AR = (res_rescaled[1]-max_constrained)/(min_constrained-max_constrained)
                res = {
                    'N':N,
                    'K':K,
                    'seed':seed,
                    'po_problem':po_problem,
                    'trace_rescaled':trace_rescaled,
                    'res_rescaled':res_rescaled,
                    'res_orig':res_orig, 
                    'trace_orig':trace_orig,
                    'rescale_time':rescale_time,
                    'orig_time':orig_time,
                    'rescale_ar':rescale_AR,
                    'orig_ar':orig_AR
                }
                print(f'Completed N={N}, seed={seed}, p={p}, res_AR={rescale_AR:.4f}, n_shot={int(n_shot)}, func_call={len(trace_rescaled)}, norm_diffx={np.linalg.norm(res_rescaled[0]-X0)/np.linalg.norm(X0):.2f}, time = {rescale_time:.2f}\n')  # X0={X0}, opt_x={res_rescaled[0]}, res_energy = {res_rescaled[1]:.4f}
                pickle.dump(res, open(outpath, 'wb'))