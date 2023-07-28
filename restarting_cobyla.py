from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from typing import Any
import warnings

import numpy as np
from nlopt import LN_COBYLA, opt, RoundoffLimited
from numpy.typing import NDArray
from oscar import CustomOptimizer


def RECOBYLA() -> CustomOptimizer:
    return CustomOptimizer(minimize, "RECOBYLA")

def minimize(
        f: Callable,
        initial_point: NDArray[np.float_],
        budget: int,
        shots: int,
        bounds: Sequence[tuple[float, float]] | None = None,
        rhobeg: float | None = None,
        xtol_abs: float | None = None,
        scaling: float = 2.0,
        callback: Callable = None,
    ) -> dict[str, Any]:
        def f_wrapper(x, grad, shots):
            if grad.size > 0:
                raise ValueError("Gradient shouldn't be requested")
            return f(x, shots=shots)

        num_params = len(initial_point)
        if budget < (num_params + 2) * shots:
            raise ValueError(
                "Not enough budget for a minimal requirement of (num_params + 2) * shots for "
                f"{num_params = } and {shots = }."
            )

        if bounds is not None:
            bounds = np.array(bounds).T
        used_budget, iteration, num_evals = 0, 0, 0

        while used_budget + (num_params + 2) * shots <= budget:
            optimizer = opt(LN_COBYLA, num_params)
            optimizer.set_min_objective(partial(f_wrapper, shots=shots))
            if bounds is not None:
                optimizer.set_lower_bounds(bounds[0])
                optimizer.set_upper_bounds(bounds[1])
            if rhobeg is not None:
                optimizer.set_initial_step(rhobeg)
            if xtol_abs is not None:
                optimizer.set_xtol_abs(xtol_abs)
            optimizer.set_maxeval(int((budget - used_budget) / shots))
            try:
                initial_point = optimizer.optimize(initial_point)
            except RoundoffLimited as err:
                warnings.warn(f"NLopt encounters roundoff limited error: {err}")
            finally:
                if callback is not None:
                    callback(optimizer)
                num_evals += optimizer.get_numevals()
                used_budget += optimizer.get_numevals() * shots
                rhobeg /= scaling
                xtol_abs /= scaling
                shots = int(scaling * shots)
                iteration += 1

        return {
            "optimal_params": initial_point,
            "optimal_value": optimizer.last_optimum_value(),
            "num_iters": iteration,
            "num_fun_evals": num_evals,
            "used_budget": used_budget,
        }