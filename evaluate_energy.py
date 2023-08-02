from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal
from functools import partial
import itertools

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from qokit.maxcut import maxcut_obj, get_adjacency_matrix
from qokit.qaoa_objective_maxcut import get_qaoa_maxcut_objective
from qokit.qaoa_objective_portfolio import get_qaoa_portfolio_objective
from qokit.utils import brute_force, precompute_energies
from tqdm import tqdm

from circuit_utils import get_configuration_cost_kw
from utils import get_problem, get_real_problem, get_adjusted_state, precompute_energies_parallel


def load_problem(
    problem: Literal["maxcut", "po"], n: int, seed: int
) -> tuple[dict[str, Any] | nx.Graph, NDArray[np.float_]]:
    if problem == "maxcut":
        return load_maxcut_problem(n, seed)
    if problem == "po":
        return load_po_problem(n, seed)
    raise ValueError(f"Problem {problem} not recognized")


def kbits(n, k):
    for bits in itertools.combinations(range(n), k):
        s = [0] * n
        for bit in bits:
            s[bit] = 1
        yield np.array(s)


def sample_gaussian_mixture(
    num_samples: int, components: Sequence[dict[str, float]], seed: int
) -> NDArray[np.float_]:
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(num_samples):
        component = rng.choice(components, p=[c["weight"] for c in components])
        sample = rng.normal(component["mean"], component["std_dev"])
        samples.append(sample)
    return np.array(samples)


def load_maxcut_problem(n: int, seed: int) -> tuple[nx.Graph, NDArray[np.float_]]:
    g = nx.random_regular_graph(3, n, seed)

    # Define the parameters for the Gaussian components
    component1 = {"mean": 0, "std_dev": 1, "weight": 0.5}
    component2 = {"mean": 5, "std_dev": 2, "weight": 0.3}
    component3 = {"mean": 10, "std_dev": 1, "weight": 0.2}
    components = [component1, component2, component3]
    weights = sample_gaussian_mixture(3 * n // 2, components, seed)
    # generate random weights
    # weights = np.random.uniform(0, 10, g.number_of_edges())
    # rescale following the rule in Eq. 6 of https://arxiv.org/pdf/2305.15201.pdf
    weights = weights / np.sqrt(np.mean(weights**2))

    for i, (w, v) in enumerate(g.edges):
        g.edges[w, v]["weight"] = weights[i]

    return g, precompute_energies(partial(maxcut_obj, w=get_adjacency_matrix(g)), n)


def load_po_problem(n, seed):
    k = n // 2
    po_problem = get_real_problem(n, k, 0.5, seed, pre=1)
    means_in_spins = np.array(
        [
            po_problem["means"][i] - po_problem["q"] * np.sum(po_problem["cov"][i, :])
            for i in range(len(po_problem["means"]))
        ]
    )
    scale = 1 / (
        np.sqrt(np.mean(((po_problem["q"] * po_problem["cov"]) ** 2).flatten()))
        + np.sqrt(np.mean((means_in_spins**2).flatten()))
    )
    po_problem["scale"] = scale
    po_problem["means"] = scale * po_problem["means"]
    po_problem["cov"] = scale * po_problem["cov"]

    min_constrained = float("inf")
    max_constrained = float("-inf")
    mean_constrained = 0
    total_constrained = 0
    po_obj = partial(get_configuration_cost_kw, po_problem=po_problem)
    for x in tqdm(kbits(n, k)):
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
        precompute_energies_parallel(po_obj, n, 1)
    ).real

    return po_problem, precomputed_energies


def get_evaluate_energy(
    problem: dict[str, Any] | nx.Graph,
    precomputed_energies: NDArray[np.float_],
    p: int,
    objective: str = "expectation",
    simulator: str = "auto",
) -> Callable:
    if isinstance(problem, nx.Graph):
        beta_scaling = 1 / 4
        func = get_qaoa_maxcut_objective(
            problem.number_of_nodes(),
            p,
            problem,
            precomputed_energies,
            objective=objective,
            simulator=simulator,
        )
    else:
        beta_scaling = 1 / 8
        func = get_qaoa_portfolio_objective(
            problem,
            p,
            precomputed_energies=precomputed_energies,
            objective=objective,
            simulator=simulator,
        )

    def f(params):
        params = np.array(params)
        params[len(params)//2:] *= beta_scaling
        return func(params)

    return f
