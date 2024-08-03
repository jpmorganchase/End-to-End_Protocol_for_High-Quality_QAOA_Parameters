# Shot-frugal Optimization
This is the repo accompanying the paper [End-to-End Protocol for High-Quality QAOA Parameters with Few Shots](https://arxiv.org/abs/2408.00557). The data is available at [https://doi.org/10.5281/zenodo.12209739](https://doi.org/10.5281/zenodo.12209739).

## Usage
`reproduce_figures/` contains notebooks that can reproduce all the figures used in the paper. They rely on files in `data/` (moved to [Zenodo](https://doi.org/10.5281/zenodo.12209739) to shrink the size of the repo), which can be obtained by executing the following scripts:

- `run_landscape.py` and `benchmark_optimizer` for optimizer benchmarking
- `benchmark_cobyla.py` for COBYLA hyperparameter search and shot budget allocation study

The configurations of `benchmark_optimizer` and `benchmark_cobyla` are controlled by command-line arguments:

- `--problem` specifies the problem, can be "maxcut", "po", or "skmodel".
- `-n` specifies the number of variables/qubits of the problem instance.
- `-p` specifies the number of QAOA layers.
- `-s` `--seed` controls the number of problem instances generated.

Additionally, 
- `benchmark_optimizer.py` has `-r` `--reps` that controls how many times an optimization configuration is executed.
- `benchmark_cobyla.py` has 
    - `-t` `--target` that specifies its behavior.
        - "rhobeg" grid searches the initial step size with exact simulation.
        - "max_ar" optimizes for the highest achievable AR for each instance with exact simulation.
        - "budget" grid searches budget allocation strategies (combinations of numbers of evaluations and numbers of shots per evaluation, given a total shot budget) with shot-based simulation.
        - "opt2steps" optimizes for 2 iterations after the initial $2p+1$ evaluations.
    - `-b` `--batch` groups many instances (seeds) into batches (useful for parallel execution).
    - `--cpu` if provided, use the cpu implementation of the `qokit` simulator. Default uses GPU.
    - `--no-precompute` controls whether the energies are precomputed for the simulator. In most of the cases, this flag should not be specified (so that we have min and max values to compute the AR), but for extreme-scale simulations, it might be favorable to let a GPU implementation do the computation.
    - `--fix-beta` if provided, fix the beta parameters and only optimize for gamma.

The configurations of landscapes in `run_landscape.py` and `benchmark_optimizer.py` can be specified with in-file variables.

## Dependencies
`pip install -r requirements.txt` will install all required packages. See a list of descriptions below.

### Pip-installable packages
#### For array operation
- `numpy`

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
- `oscar`: [https://github.com/QUEST-UWMadison/OSCAR](https://github.com/QUEST-UWMadison/OSCAR) (`benchmark_optimizer.py`, `run_landscape.py`, and `reproduce_figures.ipynb` need version `1.0.x`; `benchmark_cobyla` needs version `0.4.x` (commit `4f65fc3`); both are included in `requirements.txt`)
