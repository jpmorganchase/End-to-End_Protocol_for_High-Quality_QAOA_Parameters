from utils import *
from circuit_utils import *
from optimizer import *
import pickle
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import math
from torch.autograd import grad
from torch.autograd.functional import hessian
from pathlib import Path
from fur import QAOAFURXYRingSimulatorC, QAOAFURXYRingSimulator, QAOAFURXYRingSimulator_torch
torch.set_printoptions(precision=6)
from tqdm import tqdm
import itertools
def kbits(n, k):
    for bits in itertools.combinations(range(n), k):
        s = [0] * n
        for bit in bits:
            s[bit] = 1
        yield np.array(s)

nshot = 10000
num_iterations = 50
for N in [10]: #,11,15
    K = int(N/2)
    q = 0.5
    for seed in range(3):
        po_path = f'data/new_scale/po_problem_rule_{N}_{K}_{q}_seed{seed}.pckl'
        energy_path = f'data/new_scale/precomputed_energies_rule_{N}_{K}_{q}_seed{seed}.npy'
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
            
        # if Path(po_path).exists() and Path(energy_path).exists():
        #     precomputed_energies = np.load(energy_path)
        #     precomputed_energies = torch.from_numpy(precomputed_energies.real)
        #     po_problem = pickle.load(open(po_path, 'rb'))
        #     min_constrained = po_problem['feasible_min'] 
        #     max_constrained = po_problem['feasible_max'] 

        factor = po_problem['scale']
        precomputed_energies = torch.from_numpy(precomputed_energies.real)
        sim = QAOAFURXYRingSimulator_torch(N, factor*precomputed_energies)            
        def f_torch(inputs,nshot=False):
            gammas, betas = inputs[:len(inputs)//2], inputs[len(inputs)//2:]
            sv = sim.simulate_qaoa(gammas, betas, sv0=torch.from_numpy(generate_dicke_state_fast(N, K)))
            sv = torch.abs(sv)**2
            if nshot is False:
                print(input, torch.dot(sv.real, precomputed_energies))
                return torch.dot(sv.real, precomputed_energies)
            else:
                energy_mean = torch.dot(sv.real, precomputed_energies)
                energy_var = torch.dot(sv.real, precomputed_energies**2) - energy_mean ** 2
                energy_std = torch.sqrt(energy_var/nshot)
                # print(energy_mean.requires_grad, energy_mean.grad, 1)
                # print(energy_std.requires_grad, energy_std.grad, 2)
                # print(torch.normal(energy_mean, energy_std).requires_grad, torch.normal(energy_mean, energy_std).grad, 3)
                epsilon = torch.randn_like(energy_std)
                return energy_mean + epsilon*energy_std
                
        for p in range(1,13):
            if nshot is False:
                outpath = f'traces_GD/{N}_{K}_{q}_{seed}_p{p+1}_iter{num_iterations}_SK_init.pickle'
            else:
                outpath = f'traces_GD/{N}_{K}_{q}_{seed}_p{p+1}_iter{num_iterations}_shot{int(nshot)}_SK_init.pickle'
            # if not Path(outpath).exists():
            if True:
                np.random.seed(seed)
                sk_ini = True
                if sk_ini is True:
                    gamma, beta = get_sk_ini(p)
                    X0 = np.concatenate((gamma,beta),axis=0)
                else:
                    X0 = np.random.rand(p*2)
                start_time = time.time()
                x0 = torch.tensor(X0, dtype=torch.float32, requires_grad=True)
                ############## Set up the optimizer
                lr = 0.5  # Learning rate
                # optimizer = optim.SGD([x], lr=lr)
                optimizer = optim.Adam([x0], lr=lr)

                ############## Optimization loop
                x_value = []
                x_grad = []
                ar_value = []
                for i in range(num_iterations):
                    # Zero the gradients from the previous iteration
                    optimizer.zero_grad()

                    # Compute the function value and gradients
                    y = f_torch(x0,nshot=nshot)
                    y.backward()

                    # Update the optimization variables
                    optimizer.step()
                    # print(x0)
                    x_value.append(x0.detach().clone())
                    x_grad.append(x0.grad.detach().clone())
                    ar_value.append((y.item()-max_constrained)/(min_constrained-max_constrained))
                    if i % 5 == 0: # or i == num_iterations-1
                        print(f"iter {i}: AR(x) = {(y.item()-max_constrained)/(min_constrained-max_constrained)}, x = {x0.data}, grad = {x0.grad.data}")
                total_time = time.time()-start_time
                res = {
                    'N':N,
                    'K':K,
                    'seed':seed,
                    'po_problem':po_problem,
                    'x_value':x_value,
                    'x_grad':x_grad,
                    'ar_value':ar_value, 
                }
                print(f'Completed N={N}, seed={seed}, p={p}, shot-{nshot}, res_AR={ar_value[-1]:.4f}, func_call={num_iterations}, norm_diffx={np.linalg.norm(x_value[-1]-X0)/np.linalg.norm(X0):.2f}, time = {total_time:.2f}\n') 
                pickle.dump(res, open(outpath, 'wb'))
