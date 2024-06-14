# Shot-frugal Optimization
This is the repo accompanying the paper [End-to-End Protocol for High-Quality QAOA Parameters with Few Shots](TBA).

## Usage
`reproduce_figures/` contains a notebook that can reproduce all the figures used in the paper. It relies on files in `data/`, which can be obtained by executing the following scripts:

- `run_landscape.py` and `benchmark_optimizer` for optimizer benchmarking
- `benchmark_cobyla.py` for COBYLA hyperparameter search and shot budget allocation study

The configurations of `benchmark_optimizer` and `benchmark_cobyla` are controlled by command-line arguments:

- `--problem` specifies the problem, can be "maxcut", "po", or "skmodel".
- `-n` specifies the number of variables/qubits of the problem instance.
- `-p` specifies the number of QAOA layers.
- `-s` `--seed` controls the problem instance generation.
- `-b` `--batch` groups many instances (seeds) into batches (useful for parallel execution).
- `--cpu` if provided, use the cpu implementation of the `qokit` simulator. Default uses GPU.
- `--precompute` controls whether the energies are precomputed for the simulator. In most of the cases, it is necessary (for computing the AR), but for extreme-scale simulations, it might be favorable to let a GPU implementation do the computation.
- `--fix-beta` if provided, fix the beta parameters and only optimize for gamma.

Additionally, 
- `benchmark_cobyla.py` has `-t` `--target` that specifies its behavior.
    - "rhobeg" grid searches the initial step size with exact simulation.
    - "max_ar" optimizes for the highest achievable AR for each instance with exact simulation.
    - "budget" grid searches budget allocation strategies (combinations of numbers of evaluations and numbers of shots per evaluation, given a total shot budget) with shot-based simulation.
- `benchmark_optimizer.py` has `-r` `--reps` that controls how many times an optimization configuration is executed.

The configurations of landscapes in `run_landscape.py` and `benchmark_optimizer.py` can be specified with in-file variables.

## Dependencies
### Pip-installable packages
#### For array operation
- `numpy<2.0`

#### For plotting
- `matplotlib`

#### For problem generation
- `networkx`
- `yfinance`

#### For optimization
- `scipy`
- `nlopt`
- `scikit-quant`
- `pdfo`

### Pip+git-installable packages
#### For QAOA simulation
- `qokit`: [https://github.com/jpmorganchase/QOKit/tree/std](https://github.com/jpmorganchase/QOKit/tree/std) (The `std` branch is needed for computing the standard deviation of the energy)

#### For grid search
- `oscar`: [https://github.com/QUEST-UWMadison/OSCAR](https://github.com/QUEST-UWMadison/OSCAR) (`benchmark_optimizer.py`, `run_landscape.py`, and `reproduce_figures.ipynb` need version `1.0.x`; `benchmark_cobyla` needs version `0.4.x` (commit `4f65fc3`))
