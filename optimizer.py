import numpy as np
from numpy.random import default_rng
from circuit_utils import (
    measure_circuit,
    get_energy_expectation,
    get_energy_expectation_sv,
    get_energy_std_sv,
    get_qaoa_circuit,
    get_configuration_cost,
)
from utils import state_to_ampl_counts
import multiprocessing
from scipy.optimize import minimize, LinearConstraint
import time
import os, sys

def brute_force_search(po_problem, all_config=False):
    import itertools
    """Determine the optimal solutions by brute force."""
    N = po_problem["N"]
    K = po_problem["K"]

    results = {}
    best_solutions = None
    # Search the space of feasible solutions
    index = [0, 1]
    keys = list(itertools.product(index, repeat=N))

    # filter out non-feasable solutions
    feasible = []

    for key in keys:
        z = np.array(key)
        hamming_weight = np.sum(z)
        if hamming_weight == K:
            feasible.append(z)

    feasible = np.array(feasible)
    config_costs = np.zeros(len(feasible))

    # Find the best solutions and also the worst solutions
    for k in range(len(feasible)):
        config = feasible[k]
        config_costs[k] = get_configuration_cost(po_problem, config)

    ## if there are degenerate ground states
    min_cost_indx = np.argmin(config_costs)
    max_cost_indx = np.argmax(config_costs)
    min_cost_indx_list = []
    for i in range(len(config_costs)):
        cost = config_costs[i]
        if cost == config_costs[min_cost_indx]:
            min_cost_indx_list.append(i)

    if all_config == True:
        results["config_cost"] = config_costs
    results["minimum_cost_states"] = feasible[min_cost_indx_list]
    results["minimum_cost"] = min(config_costs)
    results["maximum_cost_states"] = feasible[max_cost_indx]
    results["maximum_cost"] = max(config_costs)

    return results


def circuit_measurement_function(
    po_problem,
    p,
    ini="dicke",
    mixer="1-xy",
    T=None,
    ini_state=None,
    n_trials=1024,
    save_state=True,
    minus=False,
):
    """Helper function to define the objective function to optimize"""
    def f(x):
        gammas = x[0:p]
        betas = x[p:]
        circuit = get_qaoa_circuit(
            po_problem,
            ini=ini,
            mixer=mixer,
            T=T,
            ini_state=ini_state,
            gammas=gammas,
            betas=betas,
            depth=p,
            save_state=save_state,
            minus=minus,
        )
        samples = measure_circuit(circuit, n_trials=n_trials, save_state=save_state)
        if save_state is False:
            energy_expectation_value = get_energy_expectation(po_problem, samples)
        else:
            energy_expectation_value = get_energy_expectation_sv(po_problem, samples)
        return energy_expectation_value

    return f

def circuit_measurement_function_std(
    po_problem,
    p,
    ini="dicke",
    mixer="1-xy",
    T=None,
    ini_state=None,
    n_trials=1024,
    minus=False,
):
    """Helper function to define the objective function to optimize"""
    def f(x):
        gammas = x[0:p]
        betas = x[p:]
        circuit = get_qaoa_circuit(
            po_problem,
            ini=ini,
            mixer=mixer,
            T=T,
            ini_state=ini_state,
            gammas=gammas,
            betas=betas,
            depth=p,
            save_state=True,
            minus=minus,
        )
        samples = measure_circuit(circuit, n_trials=n_trials, save_state=True)
        std_value = get_energy_std_sv(po_problem, samples)
        return std_value

    return f

def optimize_circuit_seed(
    po_problem,
    p,
    N_seed=1,
    ini="dicke",
    mixer="1-xy",
    T=None,
    x0=None,
    noise_level=0.01,
    n_trials=1024,
    maxiter=300,
    disp=True,
    save_state=True,
    minus=False,
):
    """ Unparallelled version to optimize QAOA with multiple initial """
    rng = default_rng()
    if x0 is None:
        x0 = 2 * np.pi * rng.random((N_seed, p)) #gamma [0, 2pi]
        x0 = np.concatenate(
            (x0, np.pi * 0.5 * (2 * rng.random((N_seed, p)) - 1)), axis=1 #beta [-0.5pi, 0.5pi]
        )
    else:
        # fine tuning
        noise = 2 * np.pi * rng.random((N_seed, p))
        noise = np.concatenate((noise, np.pi * 0.5 * (2 * rng.random((N_seed, p)) - 1)), axis=1)
        x0 = x0 + noise_level * noise
    cur_best = 1e5
    for i_seed in range(N_seed):
        results = optimize_circuit(
            po_problem,
            p,
            ini=ini,
            mixer=mixer,
            T=T,
            x0=x0[i_seed],
            n_trials=n_trials,
            maxiter=maxiter,
            disp=disp,
            save_state=save_state,
            minus=minus,
        )
        if results["optimal_energy_measurement"] < cur_best:
            cur_best = results["optimal_energy_measurement"]
            cur_result = results
    ###
    bf_result = brute_force_search(po_problem)
    cur_result["approx_ratio"] = (
        cur_result["optimal_energy_measurement"] - bf_result["maximum_cost"]
    ) / (bf_result["minimum_cost"] - bf_result["maximum_cost"])
    return cur_result


def optimize_circuit(
    po_problem,
    p,
    ini="dicke",
    mixer="1-xy",
    T=None,
    x0=None,
    n_trials=1024,
    maxiter=50,
    disp=True,
    save_state=True,
    minus=False,
):
    """ Unparallelled version to optimize QAOA with single initial """
    rng = default_rng()
    if x0 is None:
        x0 = 2 * np.pi * rng.random(p) #gamma [0, 2pi]
        x0 = np.concatenate(
            (x0, np.pi * 0.5 * (2 * rng.random(p) - 1)) #beta [-0.5pi, 0.5pi]
        )  

    results = {}
    results["x0"] = x0
    obj = circuit_measurement_function(
        po_problem,
        p=p,
        ini=ini,
        mixer=mixer,
        T=T,
        n_trials=n_trials,
        save_state=save_state,
        minus=minus,
    )
    
    res = minimize(
        obj,
        x0,
        method="BFGS",  
        options={"maxiter": maxiter, "disp": False},  
    )
    if disp is True:
        print(
            f"optimal cost: {res.fun}, best gamma: {res.x[0:p]}, best beta: {res.x[p:]}"
        )

    gammas = res.x[0:p]
    betas = res.x[p:]
    circuit = get_qaoa_circuit(
        po_problem,
        ini=ini,
        mixer=mixer,
        T=T,
        gammas=gammas,
        betas=betas,
        depth=p,
        save_state=save_state,
        minus=minus,
    )
    samples = measure_circuit(circuit, n_trials=n_trials, save_state=save_state)
    if save_state is False:
        energy_expectation_value = get_energy_expectation(po_problem, samples)
    else:
        energy_expectation_value = get_energy_expectation_sv(po_problem, samples)

    results["p"] = p
    results["optimal_gammas"] = gammas
    results["optimal_betas"] = betas
    results["optimal_energy_measurement"] = energy_expectation_value
    return results


def single_optimize_circuit(
    x0,
    po_problem,
    p,
    ini,
    mixer,
    T,
    ini_state,
    n_trials,
    maxiter,
    disp,
    save_state,
    minus,
):
    """ Helper function to optimize QAOA with single initial in parallelled version """
    results = {}
    results["x0"] = x0
    obj = circuit_measurement_function(
        po_problem,
        p=p,
        ini=ini,
        mixer=mixer,
        T=T,
        ini_state=ini_state,
        n_trials=n_trials,
        save_state=save_state,
        minus=minus,
    )
    res = minimize(
        obj,
        x0,
        method="BFGS",  
        options={"maxiter": maxiter, "disp": False},  
    )
    if disp is True:
        print(
            f"optimal cost: {res.fun}, best gamma: {res.x[0:p]}, best beta: {res.x[p:]}, x0: {x0}", flush=True
        )
    
    # get the other results
    gammas = res.x[0:p]
    betas = res.x[p:]
    circuit = get_qaoa_circuit(
        po_problem,
        ini=ini,
        mixer=mixer,
        T=T,
        ini_state=ini_state,
        gammas=gammas,
        betas=betas,
        depth=p,
        save_state=save_state,
        minus=minus,
    )
    samples = measure_circuit(circuit, n_trials=n_trials, save_state=save_state)
    if save_state is False:
        energy_expectation_value = get_energy_expectation(po_problem, samples)
    else:
        energy_expectation_value = get_energy_expectation_sv(po_problem, samples)
    
    results["p"] = p
    results["seed"] = po_problem['seed']
    results["T"] = T
    results["mixer"] = mixer
    results["optimal_gammas"] = gammas
    results["optimal_betas"] = betas
    results["optimal_energy_measurement"] = energy_expectation_value
    
    return results

def positive_gamma(x):
    for i in range(len(x)):
        if x[i] < 0 or x[i] > 11:
            return False
    return True

def parallel_optimize_circuit_seed(
    po_problem,
    p,
    N_seed=10,
    ini="dicke",
    mixer="1-xy",
    T=None,
    ini_state=None,
    x0=None,
    noise_level=0.01,
    n_trials=1024,
    maxiter=300,
    disp=True,
    save_state=True,
    minus=False,
):
    """ Parallelled version to optimize QAOA with multiple initial """
    rng = default_rng()
    if x0 is None:
        x0 = 2 * np.pi * rng.random((N_seed, p)) #gamma [0, 2pi]
        x0 = np.concatenate(
            (x0, np.pi * 0.5 * (2 * rng.random((N_seed, p)) - 1)), axis=1 #beta [-0.5pi, 0.5pi]
        )
    else:
        # fine tuning
        noise = np.pi * rng.random((N_seed, p))
        noise = np.concatenate((noise, np.pi * rng.random((N_seed, p))), axis=1)
        x0 = x0 + noise_level * noise

    try:
        per_task_num = int(os.environ['SLURM_CPUS_PER_TASK'])
    except:
        per_task_num = 1
    print(f'CPUS_PER_TASK: {per_task_num}')
    if per_task_num > 1:
        with multiprocessing.Pool(per_task_num-1) as pool:
            opt_results = pool.starmap(
                single_optimize_circuit,
                (
                    (
                        x0_i,
                        po_problem,
                        p,
                        ini,
                        mixer,
                        T,
                        ini_state,
                        n_trials,
                        maxiter,
                        disp,
                        save_state,
                        minus,
                    )
                    for x0_i in x0
                ),
            )
    else:
        assert per_task_num == 1
        opt_results = []
        for x0_i in x0:
            temp_result = single_optimize_circuit(
                    x0_i,
                    po_problem,
                    p,
                    ini,
                    mixer,
                    T,
                    ini_state,
                    n_trials,
                    maxiter,
                    disp,
                    save_state,
                    minus,
                )
            opt_results.append(temp_result)
        
    best_res = None
    for res in opt_results:
        # Add a condition to keep gamma positive
        if (best_res is None) or (res["optimal_energy_measurement"] < best_res) and positive_gamma(res["optimal_gammas"]):
            best_res = res["optimal_energy_measurement"]
            cur_result = res
    
    #############################
    bf_result = brute_force_search(po_problem)
    cur_result["approx_ratio"] = (
        cur_result["optimal_energy_measurement"] - bf_result["maximum_cost"]
    ) / (bf_result["minimum_cost"] - bf_result["maximum_cost"])
    return cur_result


############################# Optimize delta for phase diagram #############################
def opt_phase_function(
    po_problem,
    p,
    ini="dicke",
    mixer="1-xy",
    ub=1,
    T=1,
    ini_state=None,
    n_trials=1024,
    save_state=True,
    minus=False,
):
    """ Helper function to define the object to optimize """
    gammas = np.zeros(p)
    betas = np.zeros(p)
    N = po_problem["N"]
    results={}    
    def f(delta):
        for i in range(p):
            f0 = (i + 1) / (p + 1) * ub
            gammas[i] = delta * f0
            betas[i] = delta * (ub - f0)
            
        circuit = get_qaoa_circuit(
            po_problem,
            ini=ini,
            mixer=mixer,
            T=T,
            ini_state=ini_state,
            gammas=gammas,
            betas=betas,
            depth=p,
            save_state=save_state,
            minus=minus,
        )
        samples = measure_circuit(circuit, n_trials=n_trials, save_state=save_state)
        if save_state is False:
            energy_expectation_value = get_energy_expectation(po_problem, samples)
        else:
            energy_expectation_value = get_energy_expectation_sv(po_problem, samples)
        
        return energy_expectation_value
    
    return f

def opt_circuit_phase_single(
    po_problem,
    p,
    delta0=None,
    ub=1,
    minus=False,
    T=None,
    ini="dicke",
    ini_state=None,
    mixer="1-xy",
    n_trials=1024,
    disp=True,
    save_state=True,
    maxiter=300, 
):
    """ Helper function to optimize phase diagram QAOA with single initial in parallelled version """
    if delta0 is None:
        delta0 = np.random.rand(1) #delta [0,1]
    obj = opt_phase_function(
        po_problem,
        p=p,
        ini=ini,
        mixer=mixer,
        ub=ub,
        T=T,
        ini_state=ini_state,
        n_trials=n_trials,
        save_state=save_state,
        minus=minus,
    )
    res = minimize(
        obj,
        delta0,
        method="BFGS",  
        options={"maxiter": maxiter, "disp": False},  
    )
    if disp is True:
        print(
            f"optimal cost: {res.fun}, best delta: {res.x}, delta_0: {delta0}", flush=True
        )
    return res

def delta_boundary(x, b):
    if x[i] < 0 or x[i] > b:
        return False
    else:
        return True

def opt_circuit_phase_parallel(
    po_problem,
    delta0=None,
    N_seed=10,
    p=100,
    ub=1,
    minus=False,
    T=None,
    ini="dicke",
    ini_state=None,
    mixer="1-xy",
    n_trials=1024,
    disp=True,
    save_state=True,
    maxiter=300, 
):
    """ Parallelled version to optimize phase diagram QAOA with multiple initial """
    rng = default_rng()
    if delta0 is None:
        delta0 = rng.random(N_seed)
    try:
        per_task_num = int(os.environ['SLURM_CPUS_PER_TASK'])
    except:
        per_task_num = 1
    print(f'CPUS_PER_TASK: {per_task_num}')
    if per_task_num > 1:
        with multiprocessing.Pool(per_task_num-1) as pool:
            opt_results = pool.starmap(
                opt_circuit_phase_single,
                (
                    (
                        po_problem,
                        p,
                        delta0_i,
                        ub,
                        minus,
                        T,
                        ini,
                        ini_state,
                        mixer,
                        n_trials,
                        disp,
                        save_state,
                        maxiter, 
                    )
                    for delta0_i in delta0
                ),
            )
    else:
        assert per_task_num == 1
        opt_results = []
        for delta0_i in delta0:
            temp_result = opt_circuit_phase_single(
                        po_problem,
                        p,
                        delta0_i,
                        ub,
                        minus,
                        T,
                        ini,
                        ini_state,
                        mixer,
                        n_trials,
                        disp,
                        save_state,
                        maxiter, 
                    )
            opt_results.append(temp_result)
    boundary = 1
    if mixer == 'exact-complete':
        boundary = 0.6
    best_res = None
    for res in opt_results:
        if (best_res is None) or (res.fun < best_res.fun) and delta_boundary(res.x,boundary):
            best_res = res
            
    cur_result = {}
    cur_result["optimal_energy_measurement"] = best_res.fun
    cur_result['p'] = p
    cur_result['T'] = T
    cur_result['mixer'] = mixer
    cur_result['delta'] = best_res.x
    
    bf_result = brute_force_search(po_problem)
    cur_result["approx_ratio"] = (
        cur_result["optimal_energy_measurement"] - bf_result["maximum_cost"]
    ) / (bf_result["minimum_cost"] - bf_result["maximum_cost"]) 
    return cur_result

#############################
def get_problem_eigenvalue_gap(po_problem):
    """ Get the difference of maximum and minimum eigenvalue """
    bf_result = brute_force_search(po_problem)
    delta_w = bf_result["maximum_cost"] - bf_result["minimum_cost"]
    # from utils import get_problem_H
    # H_all = get_problem_H(po_problem)
    # w, v = LA.eigh(H_all)
    # delta_w = np.max(w)-np.min(w)
    return delta_w
    
def determine_best_solution_from_trials(po_problem, samples):
    """Determine the best solutions from measurement samples"""
    # Initialize the energy
    energies = np.zeros(len(samples))

    # Determine the energy of these states and find the best solutions among the subset
    k = 0
    N_total = 0
    for (config, count) in samples.items():
        energies[k] = get_configuration_cost(po_problem, config)
        k += 1
        N_total += count

    # Get the minimum energy of these experiments
    E_min = min(energies)
    E_min_indx = np.argmin(energies)
    min_states = list(samples.keys())[E_min_indx]
    min_states_probabilities = list(samples.values())[E_min_indx] / N_total

    results = {}
    results["minimum_cost"] = E_min
    results["minimum_cost_states"] = min_states
    results["minimum_cost_probabilities"] = min_states_probabilities

    return results


def determine_best_solution_from_sv(po_problem, samples):
    """Determine the best solutions from state vector"""
    samples = state_to_ampl_counts(samples)
    # Initialize the energy
    energies = np.zeros(len(samples))

    # Determine the energy of these states and find the best solutions among the subset
    k = 0
    N_total = 0
    for (config, wf) in samples.items():
        energies[k] = (np.abs(wf) ** 2) * get_configuration_cost(po_problem, config)
        k += 1

    # Get the minimum energy of these experiments
    E_min = min(energies)
    E_min_indx = np.argmin(energies)
    min_states = list(samples.keys())[E_min_indx]
    min_states_probabilities = np.abs(list(samples.values())[E_min_indx]) ** 2

    results = {}
    results["minimum_cost"] = E_min
    results["minimum_cost_states"] = min_states
    results["minimum_cost_probabilities"] = min_states_probabilities

    return results