from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

import numpy as np
from nlopt import LN_COBYLA, opt, RoundoffLimited
from numpy.typing import NDArray
from oscar import CustomOptimizer


def RECOBYLA() -> CustomOptimizer:
    def opt_fun(
        f: Callable,
        initial_point: NDArray[np.float_],
        budget: int,
        bounds: list[tuple[float, float]],
        rhobeg: float,
        xtol_abs: float,
        shots: int,
        scaling: float = 2.0,
        p: int = 1,
        callback: Callable = None,
    ) -> dict[str, Any]:
        def f_wrapper(x, grad, shots):
            if grad.size > 0:
                raise ValueError("Gradient shouldn't be requested")
            return f(x, shots=shots)

        if budget < (2 * p + 2) * shots:
            raise ValueError(f"Not enough budget for a minimal requirement of 2 * (p + 1) * shots for {p = } and {shots = }.")

        bounds = np.array(bounds)
        used_budget, iteration, num_evals = 0, 0, 0

        while used_budget + (2 * p + 2) * shots <= budget:
            optimizer = opt(LN_COBYLA, 2 * p)
            optimizer.set_min_objective(partial(f_wrapper, shots=shots))
            optimizer.set_lower_bounds(bounds.T[0])
            optimizer.set_upper_bounds(bounds.T[1])
            optimizer.set_initial_step(rhobeg)
            optimizer.set_xtol_abs(xtol_abs)
            optimizer.set_maxeval((budget - used_budget) // shots)
            try:
                initial_point = optimizer.optimize(initial_point)
            except RoundoffLimited:
                break
            finally:
                if callback is not None:
                    callback(optimizer)
                num_evals += optimizer.get_numevals()
                used_budget += optimizer.get_numevals() * shots
                rhobeg /= scaling
                xtol_abs /= scaling
                shots *= scaling
                iteration += 1

        return {
            "optimal_params": initial_point,
            "optimal_value": optimizer.last_optimum_value(),
            "num_iters": iteration,
            "num_fun_evals": num_evals,
            "used_budget": used_budget,
        }

    return CustomOptimizer(opt_fun)
