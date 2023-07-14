from functools import partial
from itertools import product
from multiprocessing import Pool
from typing import Optional, Union

import numpy as np
import pandas as pd
import yfinance as yf
from numpy import linalg as LA
from qiskit.opflow import I, X, Y
from scipy.linalg import expm
from tqdm import tqdm


def create_portfolio_instance(
    start_date: str = "2015-01-01",
    end_date: str = "2019-12-31",
    num_assets: int = 0,
    return_dtype: str = "numpy",
    log_returns: bool = False,
    tickers: Optional[list] = None,
    sort_by_avg_volume: bool = False,
    seed: int = 42,
) -> Union[tuple[np.ndarray, np.ndarray], tuple[pd.core.frame.DataFrame, pd.core.frame.DataFrame]]:
    """
    Parameters
    ------------------
        start_date (str): The starting date
        end_date (str): The ending date
        num_assets (int): The number of assets


        Optional:
        return_dtype (str): The return datatype for the correlation matrix
                     This is either numpy or panda

        tickers (list): The list of ticker symbols. If this is not provided, you will get
        the targets and correlation for the random tickers and you can read the
        symbols via accessing the correlation as a pandas dataframe

        log_returns: If we want the log returns instead of just returns

    -------------------
    Returns
    ---------
    List(returns, correlation) : type List(np.array, np.array or pd.dataframe)

    """

    if (
        return_dtype == "numpy"
        or return_dtype == "np"
        or return_dtype == "pd"
        or return_dtype == "pandas"
    ):
        pass
    else:
        # print(return_dtype)
        raise ValueError("The return dtype should be either (numpy) or (np) or pandas or (pd)")

    if tickers is None:
        # If the user doesn't provide the ticker symbols, we sort by average trading volume and just
        # give the data for the number of assets provided

        # Download data for the time period
        sp500_tickers = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[
            0
        ]

        ticker_symbols = sp500_tickers["Symbol"].tolist()

        if num_assets > 0 and not sort_by_avg_volume:
            rng = np.random.default_rng(seed)
            random_tickers = rng.choice(len(ticker_symbols), size=num_assets, replace=False)

            top_N_ticker_symbols = np.array(ticker_symbols)[random_tickers].tolist()

        else:
            top_N_ticker_symbols = ticker_symbols

        # Retrieve historical price data for each ticker symbol
        stock_data = {}

        invalid_ticker_symbols = []

        for symbol in top_N_ticker_symbols:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)

            if data.empty:
                ### Removing the ticker symbols that are invalid
                invalid_ticker_symbols.append(symbol)
            else:
                stock_data[symbol] = data

        if sort_by_avg_volume or num_assets == 0:
            # Calculate average trading volume for each stock
            average_volumes = {}
            for symbol, data in stock_data.items():
                average_volume = data["Volume"].mean()
                average_volumes[symbol] = average_volume

            # Sort the stocks based on average trading volume
            sorted_stocks = sorted(average_volumes.items(), key=lambda x: x[1], reverse=True)

            # Extract the ticker symbols from the sorted list
            ticker_list = [tick[0] for tick in sorted_stocks]

            if num_assets > 0:
                top_N_ticker_symbols = ticker_list[:num_assets]

    ### This is if we are given a set of tickers
    ### Then just download data for these, and compute correlation matrices and return vector

    else:
        num_assets = len(tickers)
        # Retrieve historical price data for each ticker symbol
        stock_data = {}

        invalid_ticker_symbols = []

        for symbol in tickers:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)

            if data.empty:
                invalid_ticker_symbols.append(symbol)
            else:
                stock_data[symbol] = data

        top_N_ticker_symbols = tickers

    top_N_ticker_symbols = list(stock_data.keys())

    ### List of dataframes with the closing prices
    list_df_close = [stock_data[symbol]["Close"] for symbol in top_N_ticker_symbols]

    df_close = pd.concat(list_df_close, axis=1, keys=top_N_ticker_symbols)

    ### Drop the ones with rows with NaN values
    df_close = df_close.dropna(axis=1)
    df_close = df_close.dropna(axis=0)

    if not log_returns:
        df_returns = df_close.pct_change()  # .dropna( )

        #### Drop the first row since we have NaN's
        df_returns = df_returns.iloc[1:]

        # print(df_returns.shape)
        df_mean_returns = df_returns.mean(axis=0)

        correlation_matrix = df_returns.corr()

        if return_dtype in ("pandas", "pd"):
            return [df_mean_returns, correlation_matrix]
        else:
            return [df_mean_returns.to_numpy(), correlation_matrix.to_numpy()]

    elif log_returns:
        df_logreturns = np.log(df_close / df_close.shift(1)).dropna()
        df_meanlog_returns = df_logreturns.mean(axis=0)

        correlation_matrix = df_logreturns.corr()

        if return_dtype in ("pandas", "pd"):
            return [df_meanlog_returns, correlation_matrix]

        else:
            return [df_meanlog_returns.to_numpy(), correlation_matrix.to_numpy()]


def get_data(N, seed=1, real=False):
    """
    load portofolio data from qiskit-finance (Yahoo)
    https://github.com/Qiskit/qiskit-finance/blob/main/docs/tutorials/11_time_series.ipynb
    """
    import datetime

    from qiskit_finance.data_providers import RandomDataProvider, YahooDataProvider

    tickers = []
    for i in range(N):
        tickers.append("t" + str(i))
    if real is False:
        data = RandomDataProvider(
            tickers=tickers,
            start=datetime.datetime(2016, 1, 1),
            end=datetime.datetime(2016, 1, 30),
            seed=seed,
        )
    else:
        stock_symbols = [
            "AAPL",
            "GOOGL",
            "AMZN",
            "MSFT",
            "TSLA",
            "NFLX",
            "NVDA",
            "JPM",
            "V",
            "JNJ",
            "WMT",
            "PG",
            "MA",
            "UNH",
            "HD",
            "DIS",
            "BRK-B",
            "VZ",
            "KO",
            "MRK",
            "INTC",
            "CMCSA",
            "PEP",
            "PFE",
            "CSCO",
            "XOM",
            "BA",
            "MCD",
            "ABBV",
            "IBM",
            "GE",
            "MMM",
        ]

        # switch to Atithi's implementation
        rng = np.random.default_rng(seed)
        date = rng.integers(0, 60)
        year = 2015 + date // 12
        month = date % 12 + 1
        start_date = f"{year}-{month}-01"
        end_date = f"{year}-{month}-28"
        return create_portfolio_instance(
            start_date,
            end_date,
            0,
            log_returns=True,
            seed=seed,
            tickers=[
                stock_symbols[i] for i in rng.choice(len(stock_symbols), size=N, replace=False)
            ],
        )

        # data = YahooDataProvider(
        #     tickers=stock_symbols[:N],
        #     start=datetime.datetime(2020, 1, 1),
        #     end=datetime.datetime(2020, 1, 30),
        #     # end=datetime.datetime(2021, 1, 1),
        # )

    data.run()
    # use get_period_return_mean_vector & get_period_return_covariance_matrix to get return!
    # https://github.com/Qiskit/qiskit-finance/blob/main/docs/tutorials/01_portfolio_optimization.ipynb
    means = data.get_period_return_mean_vector()
    cov = data.get_period_return_covariance_matrix()
    return means, cov


def get_real_problem(N, K, q, seed=1, pre=False):
    po_problem = {}
    po_problem["N"] = N
    po_problem["K"] = K
    po_problem["q"] = q
    po_problem["real"] = True
    po_problem["means"], po_problem["cov"] = get_data(N, seed, real=True)
    po_problem["pre"] = pre
    scale = 1
    if pre == "constant":
        scale = abs(1 / sum(po_problem["means"]))
    elif np.isscalar(pre):
        scale = pre

    po_problem["scale"] = scale
    po_problem["means"] = scale * po_problem["means"]
    po_problem["cov"] = scale * po_problem["cov"]

    return po_problem


def get_problem(N, K, q, seed=1, pre=False):
    po_problem = {}
    po_problem["N"] = N
    po_problem["K"] = K
    po_problem["q"] = q
    po_problem["seed"] = seed
    po_problem["means"], po_problem["cov"] = get_data(N, seed=seed)
    po_problem["pre"] = pre
    scale = 1
    if pre == "constant":
        scale = abs(1 / sum(po_problem["means"]))
    elif np.isscalar(pre):
        scale = pre

    po_problem["scale"] = scale
    po_problem["means"] = scale * po_problem["means"]
    po_problem["cov"] = scale * po_problem["cov"]

    return po_problem


def print_problem(po_problem):
    from qiskit_finance.applications.optimization import PortfolioOptimization

    portfolio = PortfolioOptimization(
        expected_returns=po_problem["means"],
        covariances=po_problem["cov"],
        risk_factor=po_problem["q"],
        budget=po_problem["K"],
    )
    qp = portfolio.to_quadratic_program()
    print(qp)


def get_problem_H(po_problem):
    """
    Problem Hamiltonian in the matrix form
    0.5 q \sum_{i=1}^{n-1} \sum_{j=i+1}^n \sigma_{ij}Z_i Z_j + 0.5 \sum_i (-q\sum_{j=1}^n{\sigma_ij} + \mu_i) Z_i +
    0.5 q (\sum_{i}\sum_{j=i}^N \sigma[i,j] - \sum_i \mu_i) I
    """
    N = po_problem["N"]
    scale = po_problem["scale"]
    means = po_problem["means"] / scale
    cov = po_problem["cov"] / scale
    q = po_problem["q"]

    H_all = np.zeros((2**N, 2**N), dtype=complex)
    Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    I = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
    for k1 in range(N - 1):
        for k2 in range(k1 + 1, N):
            H = 1
            for i in range(N):
                if i == k1 or i == k2:
                    H = np.kron(H, Z)
                else:
                    H = np.kron(H, I)
            H = 0.5 * q * cov[k1, k2] * H
            H_all += H
    for k1 in range(N):
        H = 1
        for i in range(N):
            if i == k1:
                H = np.kron(H, Z)  # the order is important!
            else:
                H = np.kron(H, I)
        H = 0.5 * (means[k1] - q * np.sum(cov[k1, :])) * H  #
        H_all += H

    constant = 0
    for k1 in range(N):
        constant += q * np.sum(cov[k1, k1:]) - means[k1]
    H_all = H_all + 0.5 * constant * np.eye(2**N)

    ########## For sanity check ###########
    S = (I - Z) / 2
    H_all2 = np.zeros((2**N, 2**N), dtype=complex)
    for k1 in range(N):
        for k2 in range(N):
            H = 1
            for i in range(N):
                if i == k1 or i == k2:
                    H = np.kron(H, S)
                else:
                    H = np.kron(H, I)
            H = q * cov[k1, k2] * H
            H_all2 += H
    for k1 in range(N):
        H = 1
        for i in range(N):
            if i == k1:
                H = np.kron(H, S)  # the order is important!
            else:
                H = np.kron(H, I)
        H = -means[k1] * H
        H_all2 += H
    assert np.allclose(H_all, H_all2)
    return H_all2


def get_problem_GS(po_problem):
    """GS and energy for unconstrained problem
    Results match brute force search
    """
    H_all = get_problem_H(po_problem)
    w, v = LA.eigh(H_all)
    min_energy_indx = np.argmin(w)
    GS = []
    for i in range(len(w)):
        if abs(w[i] - w[min_energy_indx]) < 1e-8:
            GS.append(v[:, i])
    GS_energy = w[min_energy_indx]

    return GS_energy, GS


def get_problem_GS_K(po_problem):
    """GS and energy for equality constrained problem"""
    N = po_problem["N"]
    H_all = get_problem_H(po_problem)
    K = po_problem["K"]
    w, v = LA.eigh(H_all)
    while 1:
        min_energy_indx = np.argmin(w)
        min_energy = np.min(w)
        w_list = []
        eigen_vector = []
        for i in range(len(w)):
            if abs(w[i] - w[min_energy_indx]) < 1e-5:
                eigen_vector.append(v[:, i])
                w_list.append(i)
        nonzero_idx = []
        nonzero_config = []
        for eig_vec in eigen_vector:
            nonzero_idx.append([i for i, v in enumerate(np.abs(eig_vec) ** 2) if v > 1e-8])
            nonzero_config.append([f"{i:0{N}b}" for i in nonzero_idx[-1]])
        rm = False
        for ind, eig_vec in enumerate(nonzero_config):
            for i in range(len(eig_vec)):
                config = np.zeros(N)
                for j in range(N):
                    config[j] = eig_vec[i][j]
                if int(np.sum(config)) != K:
                    rm = True
            if rm == True:
                w = np.delete(w, w_list[ind])
                v = np.delete(v, w_list[ind], axis=1)
        if rm == False:
            break

    if len(eigen_vector) == 1:
        print("Eigenvector is unique")
        nonzero_idx = [i for i, v in enumerate(np.abs(eigen_vector[0]) ** 2) if v > 1e-8]
        nonzero_config = [f"{i:0{N}b}" for i in nonzero_idx]
        print(f"Nonzero config = {nonzero_config}")
    else:
        print("Eigenvector is not unique")
        nonzero_idx = []
        nonzero_config = []
        for eig_vec in eigen_vector:
            nonzero_idx.append([i for i, v in enumerate(np.abs(eig_vec) ** 2) if v > 1e-8])
            nonzero_config.append([f"{i:0{N}b}" for i in nonzero_idx[-1]])
        print(f"Nonzero config = {nonzero_config}")

    return min_energy, eigen_vector


def invert_counts(counts):
    """convert qubit order for measurement samples"""
    return {k[::-1]: v for k, v in counts.items()}


def convert_bitstring_to_spin(config):
    """make configuration to spin. not used"""
    N = len(config)
    z = np.zeros(N).astype(int)
    for i in range(len(config)):
        z[i] = 1 - 2 * int(config[i])
    return z


def convert_bitstring_to_int(config):
    """make configuration iterable"""
    N = len(config)
    z = np.zeros(N).astype(int)
    for i in range(len(config)):
        z[i] = int(config[i])
    return z


def exact_fidelity(psi, phi):
    """
    Calculates the mod square overlap of two normalized state vectors.
    """
    return np.abs(np.vdot(psi, phi)) ** 2


def is_unitary(m):
    return np.allclose(np.eye(m.shape[0]), m.T.conjugate() @ m)


def get_feasible_ratio(po_problem, samples):
    """Samples are dictionary from measurement samples: (config, count)"""

    N_feasiable = 0
    N_total = 0
    for config, count in samples.items():
        config = convert_bitstring_to_int(config)
        if sum(config) == po_problem["K"]:
            N_feasiable += count
        N_total += count
    feasible_ratio = N_feasiable / N_total
    return feasible_ratio


def get_feasible_ratio_sv(po_problem, samples):
    """Samples: full state vector"""
    samples = state_to_ampl_counts(samples)
    feasible_ratio = 0
    for config, wf in samples.items():
        config = convert_bitstring_to_int(config)
        if sum(config) == po_problem["K"]:
            feasible_ratio += np.abs(wf) ** 2
    return feasible_ratio


def state_num2str(basis_state_as_num, nqubits):
    return "{0:b}".format(basis_state_as_num).zfill(nqubits)


def state_str2num(basis_state_as_str):
    return int(basis_state_as_str, 2)


def state_reverse(basis_state_as_num, nqubits):
    basis_state_as_str = state_num2str(basis_state_as_num, nqubits)
    new_str = basis_state_as_str[::-1]
    return state_str2num(new_str)


def get_adjusted_state(state):
    """
    Convert qubit ordering invert for state vector
    https://github.com/rsln-s/QAOA_tutorial/blob/main/Hands-on.ipynb
    """
    nqubits = np.log2(state.shape[0])
    if nqubits % 1:
        raise ValueError("Input vector is not a valid statevector for qubits.")
    nqubits = int(nqubits)

    adjusted_state = np.zeros(2**nqubits, dtype=complex)
    for basis_state in range(2**nqubits):
        adjusted_state[state_reverse(basis_state, nqubits)] = state[basis_state]
    return adjusted_state


def state_to_ampl_counts(vec, eps=1e-15):
    """
    Converts a statevector to a dictionary
    of bitstrings and corresponding amplitudes
    """
    qubit_dims = np.log2(vec.shape[0])
    if qubit_dims % 1:
        raise ValueError("Input vector is not a valid statevector for qubits.")
    qubit_dims = int(qubit_dims)
    counts = {}
    str_format = "0{}b".format(qubit_dims)
    for kk in range(vec.shape[0]):
        val = vec[kk]
        if val.real**2 + val.imag**2 > eps:
            counts[format(kk, str_format)] = val
    return counts


def exact_partial_graph_H(N, chains):
    """get exact H of a XY model built with chains, H = 0.5 \sum_{ij \in chain} XiXj + YiYj
    example of chains: [[0, 1, 5, 2, 4, 3], [1, 2, 0, 3, 5, 4], [2, 3, 1, 4, 0, 5]] with N=6
    """
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)

    H = np.zeros((2**N, 2**N), dtype=complex)
    n_chain = len(chains)
    for i_chain in range(n_chain):
        chain = chains[i_chain]
        for i in range(N):
            if i != N - 1:
                edge = [chain[i], chain[i + 1]]
                Hx, Hy = 1, 1
                for j in range(N):
                    if j == edge[0] or j == edge[1]:
                        Hx = np.kron(Hx, X)
                        Hy = np.kron(Hy, Y)
                    else:
                        Hx = np.kron(Hx, I)
                        Hy = np.kron(Hy, I)
                H = H + 0.5 * (Hx + Hy)
                assert np.allclose(H, H.T.conj())
    return H


def get_ring_xy_mixer(N, ring=True):
    """Implementation of exact ring-XY Hamiltonian: H = 0.5*\sum_i\sum_{j=i+1} XiXj + YiYj"""
    XY = 0.5 * ((X ^ X) + (Y ^ Y))

    terms = []
    for idx_xy in range(N - 1):
        if idx_xy == 0:
            term = XY
            for idx in range(2, N):
                term = term ^ I
        else:
            term = I
            for idx in range(1, idx_xy):
                term = term ^ I
            term = term ^ XY
            for idx in range(idx_xy + 2, N):
                term = term ^ I
        terms.append(term)

    # add boundary condition
    if ring == True:
        term1 = 0.5 * X
        term2 = 0.5 * Y
        for idx in range(1, N - 1):
            term1 = term1 ^ I
            term2 = term2 ^ I
        term1 = term1 ^ X
        term2 = term2 ^ Y
        term = term1 + term2
        terms.append(term)

    return sum(terms).to_matrix_op()._primitive._data


def get_trottered_ring_xy_mixer(N, T, beta=1, ring=True):
    """Implementation of trotterized ring-XY Hamiltonian in unitary:
    expm(-i*beta* 0.5*\sum_i\sum_{j=i+1} XiXj + YiYj)
    Match the qiskit "get_mixer_Txy" unitary, notice the order of applying gate
    """
    XY = 0.5 * ((X ^ X) + (Y ^ Y))
    terms = []
    for idx_xy in range(N - 1):
        if idx_xy == 0:
            term = XY
            for idx in range(2, N):
                term = term ^ I
        else:
            term = I
            for idx in range(1, idx_xy):
                term = term ^ I
            term = term ^ XY
            for idx in range(idx_xy + 2, N):
                term = term ^ I
        terms.append(term)

    terms_even = [terms[i] for i in range(0, N - 1, 2)]  # different from complete-xy
    terms_odd = [terms[i] for i in range(1, N - 1, 2)]

    # add boundary condition
    if ring == True:
        term1 = 0.5 * X
        term2 = 0.5 * Y
        for idx in range(1, N - 1):
            term1 = term1 ^ I
            term2 = term2 ^ I
        term1 = term1 ^ X
        term2 = term2 ^ Y
        term = term1 + term2
    terms_odd.append(term)

    exp_term = np.diag(np.ones(2**N))
    for _ in range(T):
        for term in terms_even:
            exp_term = expm(-beta * 1j * term.to_matrix_op()._primitive._data) @ exp_term
        for term in terms_odd:
            exp_term = expm(-beta * 1j * term.to_matrix_op()._primitive._data) @ exp_term

    return exp_term


def get_complete_xy_mixer(N):
    """Implementation of complete-XY Hamiltonian: H = 0.5*\sum_i\sum_j XiXj + YiYj"""
    terms = []
    for k1 in range(N):
        for k2 in range(k1 + 1, N):
            for ind in range(N):
                if ind == k1 or ind == k2:
                    if ind != 0:
                        term1 = term1 ^ X
                        term2 = term2 ^ Y
                    else:
                        term1 = X
                        term2 = Y
                else:
                    if ind != 0:
                        term1 = term1 ^ I
                        term2 = term2 ^ I
                    else:
                        term1 = I
                        term2 = I

            terms.append(0.5 * (term1 + term2))
    return sum(terms).to_matrix_op()._primitive._data


def get_trottered_complete_xy_mixer(N, T, beta=1):
    """Implementation of trotterized complete-XY Hamiltonian in unitary:
    expm(-i*beta* 0.5*\sum_i\sum_j XiXj + YiYj)
    """
    chains = []
    chain = [0, 1]
    for i in range(N // 2 - 1):
        chain.append(N - 1 - i)
        chain.append(2 + i)
    chains.append(chain)
    for i in range(N // 2 - 1):
        chain = [i + 1 for i in chains[-1]]
        for idx, item in enumerate(chain):
            if item > N - 1:
                chain[idx] = chain[idx] - N
        chains.append(chain)

    exp_term = np.diag(np.ones(2**N))
    for _ in range(T):
        for chain in chains:
            terms = []
            for k in range(N - 1):
                k1 = chain[k]
                k2 = chain[k + 1]
                for ind in range(N):
                    if ind == k1 or ind == k2:
                        if ind != 0:
                            term1 = X ^ term1
                            term2 = Y ^ term2
                        else:
                            term1 = X
                            term2 = Y
                    else:
                        if ind != 0:
                            term1 = I ^ term1
                            term2 = I ^ term2
                        else:
                            term1 = I
                            term2 = I
                terms.append(0.5 * (term1 + term2))
            terms_even = [terms[i] for i in range(0, N - 1, 2)]
            terms_odd = [terms[i] for i in range(1, N - 1, 2)]
            for term in terms_even:
                exp_term = expm(-beta * 1j * term.to_matrix_op()._primitive._data) @ exp_term
            for term in terms_odd:
                exp_term = expm(-beta * 1j * term.to_matrix_op()._primitive._data) @ exp_term

    return exp_term


def get_trottered_complete_xy_mixer_double(N, T, beta):
    """
    Implementation of trotterized complete-XY Hamiltonian in unitary
    Double means trotterize in both inner chains and among chains:
    {Chain decomposition * {chain trotterization}^T}^T
    """

    beta = beta / T
    chains = []
    chain = [0, 1]
    for i in range(N // 2 - 1):
        chain.append(N - 1 - i)
        chain.append(2 + i)
    chains.append(chain)
    for i in range(N // 2 - 1):
        chain = [i + 1 for i in chains[-1]]
        for idx, item in enumerate(chain):
            if item > N - 1:
                chain[idx] = chain[idx] - N
        chains.append(chain)

    exp_term = np.diag(np.ones(2**N))
    for _ in range(T):
        for chain in chains:
            terms = []
            for k in range(N - 1):
                k1 = chain[k]
                k2 = chain[k + 1]
                for ind in range(N):
                    if ind == k1 or ind == k2:
                        if ind != 0:
                            term1 = X ^ term1
                            term2 = Y ^ term2
                        else:
                            term1 = X
                            term2 = Y
                    else:
                        if ind != 0:
                            term1 = I ^ term1
                            term2 = I ^ term2
                        else:
                            term1 = I
                            term2 = I
                terms.append(0.5 * (term1 + term2))
            terms_even = [terms[i] for i in range(0, N - 1, 2)]
            terms_odd = [terms[i] for i in range(1, N - 1, 2)]
            for _ in range(T):
                for term in terms_even:
                    exp_term = expm(-beta * 1j * term.to_matrix_op()._primitive._data) @ exp_term
                for term in terms_odd:
                    exp_term = expm(-beta * 1j * term.to_matrix_op()._primitive._data) @ exp_term

    return exp_term


def get_RX_mixer(N):
    """Implementation of RX Hamiltonian: H = \sum_i X_i"""
    terms = []
    for idx_xx in range(N - 1):
        if idx_xx == 0:
            term = X
            for idx in range(1, N):
                term = term ^ I
        else:
            term = I
            for idx in range(1, idx_xx):
                term = term ^ I
            term = term ^ X
            for idx in range(idx_xx + 1, N):
                term = term ^ I
        terms.append(term)

    return sum(terms).to_matrix_op()._primitive._data


def get_all_constrained_eigenpairs(H, N, K):
    """Find all eigenpairs of H that statisfy the Hamming weight == K"""
    w, v = LA.eig(H)

    assert N == int(np.log2(H.shape[0]))
    nonzero_idx_lst = []
    nonzero_config_lst = []
    for ii in range(2**N):
        eig_vec = v[:, ii]
        nonzero_idx = [i for i, v in enumerate(np.abs(eig_vec) ** 2) if v > 1e-8]
        nonzero_config = [f"{i:0{N}b}" for i in nonzero_idx]
        nonzero_idx_lst.append(nonzero_idx)
        nonzero_config_lst.append(nonzero_config)

    indexes_to_delete = []
    for ind, eig_vec in enumerate(nonzero_config_lst):
        # print(ind,eig_vec)
        i = 0
        del_lst = []
        while i < len(eig_vec):
            config = np.zeros(N)
            for j in range(N):
                config[j] = eig_vec[i][j]
            if int(np.sum(config)) != K:
                indexes_to_delete.append(ind)
                i += 1
                break
            i += 1
    indexes_to_delete.sort(reverse=True)
    v = np.delete(v, indexes_to_delete, axis=1)
    w = np.delete(w, indexes_to_delete)
    return w, v


def get_constrained_eigenpair(H, N, K):
    """
    Find eigenpairs of H with minimum eigenvalue under Hamming weight == K
    TODO: Need to double check for corner cases, e.g., N=5, K=2
    """
    w, v = LA.eigh(H)
    assert N == int(np.log2(H.shape[0]))
    while w.size > 0:
        min_energy_indx = np.argmin(w)
        min_energy = np.min(w)
        w_list = []
        eigen_vector = []
        for i in range(len(w)):
            if abs(w[i] - w[min_energy_indx]) < 1e-8:
                eigen_vector.append(v[:, i])
                w_list.append(i)
        nonzero_idx = []
        nonzero_config = []
        for eig_vec in eigen_vector:
            nonzero_idx.append([i for i, v in enumerate(np.abs(eig_vec) ** 2) if v > 1e-8])
            nonzero_config.append([f"{i:0{N}b}" for i in nonzero_idx[-1]])
        rm = False
        for ind, eig_vec in enumerate(nonzero_config):
            for i in range(len(eig_vec)):
                config = np.zeros(N)
                for j in range(N):
                    config[j] = eig_vec[i][j]
                if int(np.sum(config)) != K:
                    rm = True
            if rm == True:
                w = np.delete(w, w_list[ind])
                v = np.delete(v, w_list[ind], axis=1)
        if rm == False:
            break

    if len(eigen_vector) == 1:
        # print('Eigenvector is unique')
        nonzero_idx = [i for i, v in enumerate(np.abs(eigen_vector[0]) ** 2) if v > 1e-8]
        nonzero_config = [f"{i:0{N}b}" for i in nonzero_idx]
        # print(f'Nonzero config = {nonzero_config}')
    else:
        # print('Eigenvector is not unique')
        nonzero_idx = []
        nonzero_config = []
        for eig_vec in eigen_vector:
            nonzero_idx.append([i for i, v in enumerate(np.abs(eig_vec) ** 2) if v > 1e-8])
            nonzero_config.append([f"{i:0{N}b}" for i in nonzero_idx[-1]])
        # print(f'Nonzero config = {nonzero_config}')
    return min_energy, eigen_vector


def generata_dicke_state(N, K):
    """Generate dicke state in numpy array"""
    import itertools

    index = [0, 1]
    keys = list(itertools.product(index, repeat=N))
    feasible = []
    for key in keys:
        z = np.array(key)
        sum_ones = np.sum(z)
        if sum_ones == K:
            feasible.append(z)

    feasible = np.array(feasible)
    gs = np.zeros(2**N, dtype=complex)
    for config_lst in feasible:
        temp = 0
        for i, val in enumerate(config_lst):
            if val == 1:
                temp += 2 ** (N - 1 - i)
        gs[temp] = 1
    gs = 1 / np.sqrt(np.sum(gs)) * gs
    return gs


def scale_map(seed):
    ## hard instance
    if seed == 2038:
        scale = 45
    if seed == 2043:
        scale = 60
    if seed == 2053:
        scale = 10
    if seed == 2058:
        scale = 50
    if seed == 6031:
        scale = 35
    ## completehard instance
    if seed == 2072:
        scale = 35
    if seed == 2075:
        scale = 20
    if seed == 2082:
        scale = 65
    if seed == 2083:
        scale = 45
    if seed == 2085:
        scale = 30
    return scale


def yield_all_bitstrings(nbits):
    """
    Helper function to avoid having to store all bitstrings in memory
    """
    for x in product([0, 1], repeat=nbits):
        yield np.array(x[::-1])


def dec_to_bin(x, nbits):
    """
    Index to
    ```
    for idx, x in enumerate(yield_all_bitstrings(10)):
        assert np.allclose(dec_to_bin(idx, 10), x)
    ```
    """
    return np.array([int(y) for y in bin(x)[2:].zfill(nbits)[::-1]])


def wrap_function_to_take_index(x, obj_f=None, nbits=None):
    """
    Helper to faster feed the precomputation
    """
    return obj_f(dec_to_bin(x, nbits))


def precompute_energies_parallel(obj_f, nbits, num_processes, chunksize=None):
    """
    Precomputed a vector of objective function values
    that accelerates the energy computation in obj_from_statevector

    Parameters
    ----------
    obj_f : callable
        Objective function to precompute
    nbits : int
        Number of parameters obj_f takes
    num_processes : int
        Number of processes to use in pathos.Pool
    postfix : list
        the last k bits

    Returns
    -------
    energies : np.array
        vector of energies such that E = energies.dot(amplitudes)
        where amplitudes are absolute values squared of qiskit statevector
    """
    if chunksize is None:
        chunksize = min(10000, max(1, int(2**nbits / num_processes)))
    obj_wrapped = partial(wrap_function_to_take_index, obj_f=obj_f, nbits=nbits)
    with Pool(num_processes) as pool:
        ens = np.array(
            list(
                tqdm(
                    pool.imap(obj_wrapped, range(2**nbits), chunksize=chunksize),
                    total=2**nbits,
                )
            )
        )
    return ens


def yield_all_bitstrings_cosntrained(nbits, K):
    """
    Helper function to avoid having to store all bitstrings in memory
    """
    for x in product([0, 1], repeat=nbits):
        if np.sum(x) == K:
            yield np.array(x[::-1])


def precompute_energies_parallel_constrained(obj_f, nbits, K, num_processes, postfix=[]):
    """
    Precomputed a vector of objective function values
    that accelerates the energy computation in obj_from_statevector

    Parameters
    ----------
    obj_f : callable
        Objective function to precompute
    nbits : int
        Number of parameters obj_f takes
    num_processes : int
        Number of processes to use in pathos.Pool
    postfix : list
        the last k bits

    Returns
    -------
    energies : np.array
        vector of energies such that E = energies.dot(amplitudes)
        where amplitudes are absolute values squared of qiskit statevector
    """
    bit_strings = (
        np.hstack([x, postfix]) for x in yield_all_bitstrings_cosntrained(nbits - len(postfix), K)
    )
    with Pool(num_processes) as pool:
        ens = np.array(pool.map(obj_f, bit_strings))
    return ens


def hamming_weight(index):
    """e.g. 107 == 1101011 --> 5"""
    binary = bin(index)[2:]
    return binary.count("1")


def yield_all_indices_cosntrained(N, K):
    """
    Helper function to avoid having to store all indices in memory
    """
    for ind in range(2**N):
        if hamming_weight(ind) == K:
            yield ind


def generate_dicke_state_fast(N, K):
    index = yield_all_indices_cosntrained(N, K)
    s = np.zeros(2**N)
    for i in index:
        s[i] = 1
    s = 1 / np.sqrt(np.sum(s)) * s
    return s


def binary_array_to_decimal(binary_array):
    """
    from a array of binary number to decimal number
    e.g. [1, 0, 0] -> 4
    order to match precomputed_energy
    """
    decimal_array = int("".join(map(str, binary_array)), 2)
    return decimal_array


def get_sk_ini(p):
    gamma_scale, beta_scale = -2, 2
    if p == 1:
        gamma = gamma_scale * np.array([0.5])
        beta = beta_scale * np.array([np.pi / 8])
        X0 = np.concatenate((gamma, beta), axis=0)
    elif p == 2:
        gamma = gamma_scale * np.array([0.3817, 0.6655])
        beta = beta_scale * np.array([0.4960, 0.2690])
        X0 = np.concatenate((gamma, beta), axis=0)
    elif p == 3:
        gamma = gamma_scale * np.array([0.3297, 0.5688, 0.6406])
        beta = beta_scale * np.array([0.5500, 0.3675, 0.2109])
        X0 = np.concatenate((gamma, beta), axis=0)
    elif p == 4:
        gamma = gamma_scale * np.array([0.2949, 0.5144, 0.5586, 0.6429])
        beta = beta_scale * np.array([0.5710, 0.4176, 0.3028, 0.1729])
        X0 = np.concatenate((gamma, beta), axis=0)
    elif p == 5:
        gamma = gamma_scale * np.array([0.2705, 0.4804, 0.5074, 0.5646, 0.6397])
        beta = beta_scale * np.array([0.5899, 0.4492, 0.3559, 0.2643, 0.1486])
        X0 = np.concatenate((gamma, beta), axis=0)
    elif p == 6:
        gamma = gamma_scale * np.array([0.2528, 0.4531, 0.4750, 0.5146, 0.5650, 0.6392])
        beta = beta_scale * np.array([0.6004, 0.4670, 0.3880, 0.3176, 0.2325, 0.1291])
        X0 = np.concatenate((gamma, beta), axis=0)
    elif p == 7:
        gamma = gamma_scale * np.array([0.2383, 0.4327, 0.4516, 0.4830, 0.5147, 0.5686, 0.6393])
        beta = beta_scale * np.array([0.6085, 0.4810, 0.4090, 0.3534, 0.2857, 0.2080, 0.1146])
        X0 = np.concatenate((gamma, beta), axis=0)
    elif p == 8:
        gamma = gamma_scale * np.array(
            [0.2268, 0.4162, 0.4332, 0.4608, 0.4818, 0.5179, 0.5717, 0.6393]
        )
        beta = beta_scale * np.array(
            [0.6151, 0.4906, 0.4244, 0.3780, 0.3224, 0.2606, 0.1884, 0.1030]
        )
        X0 = np.concatenate((gamma, beta), axis=0)
    elif p == 9:
        gamma = gamma_scale * np.array(
            [0.2172, 0.4020, 0.4187, 0.4438, 0.4592, 0.4838, 0.5212, 0.5754, 0.6398]
        )
        beta = beta_scale * np.array(
            [0.6196, 0.4973, 0.4354, 0.3956, 0.3481, 0.2973, 0.2390, 0.1717, 0.0934]
        )
        X0 = np.concatenate((gamma, beta), axis=0)
    elif p == 10:
        gamma = gamma_scale * np.array(
            [0.2089, 0.3902, 0.4066, 0.4305, 0.4423, 0.4604, 0.4858, 0.5256, 0.5789, 0.6402]
        )
        beta = beta_scale * np.array(
            [0.6235, 0.5029, 0.4437, 0.4092, 0.3673, 0.3246, 0.2758, 0.2208, 0.1578, 0.0855]
        )
        X0 = np.concatenate((gamma, beta), axis=0)
    elif p == 11:
        gamma = gamma_scale * np.array(
            [
                0.2019,
                0.3799,
                0.3963,
                0.4196,
                0.4291,
                0.4431,
                0.4611,
                0.4895,
                0.5299,
                0.5821,
                0.6406,
            ]
        )
        beta = beta_scale * np.array(
            [
                0.6268,
                0.5070,
                0.4502,
                0.4195,
                0.3822,
                0.3451,
                0.3036,
                0.2571,
                0.2051,
                0.1459,
                0.0788,
            ]
        )
        X0 = np.concatenate((gamma, beta), axis=0)
    elif p == 12:
        gamma = gamma_scale * np.array(
            [
                0.1958,
                0.3708,
                0.3875,
                0.4103,
                0.4185,
                0.4297,
                0.4430,
                0.4639,
                0.4933,
                0.5343,
                0.5851,
                0.6410,
            ]
        )
        beta = beta_scale * np.array(
            [
                0.6293,
                0.5103,
                0.4553,
                0.4275,
                0.3937,
                0.3612,
                0.3248,
                0.2849,
                0.2406,
                0.1913,
                0.1356,
                0.0731,
            ]
        )
        X0 = np.concatenate((gamma, beta), axis=0)
    elif p == 13:
        gamma = gamma_scale * np.array(
            [
                0.1903,
                0.3627,
                0.3797,
                0.4024,
                0.4096,
                0.4191,
                0.4290,
                0.4450,
                0.4668,
                0.4975,
                0.5385,
                0.5878,
                0.6414,
            ]
        )
        beta = beta_scale * np.array(
            [
                0.6315,
                0.5130,
                0.4593,
                0.4340,
                0.4028,
                0.3740,
                0.3417,
                0.3068,
                0.2684,
                0.2260,
                0.1792,
                0.1266,
                0.0681,
            ]
        )
        X0 = np.concatenate((gamma, beta), axis=0)
    elif p == 14:
        gamma = gamma_scale * np.array(
            [
                0.1855,
                0.3555,
                0.3728,
                0.3954,
                0.4020,
                0.4103,
                0.4179,
                0.4304,
                0.4471,
                0.4703,
                0.5017,
                0.5425,
                0.5902,
                0.6418,
            ]
        )
        beta = beta_scale * np.array(
            [
                0.6334,
                0.5152,
                0.4627,
                0.4392,
                0.4103,
                0.3843,
                0.3554,
                0.3243,
                0.2906,
                0.2535,
                0.2131,
                0.1685,
                0.1188,
                0.0638,
            ]
        )
        X0 = np.concatenate((gamma, beta), axis=0)
    elif p == 15:
        gamma = gamma_scale * np.array(
            [
                0.1811,
                0.3489,
                0.3667,
                0.3893,
                0.3954,
                0.4028,
                0.4088,
                0.4189,
                0.4318,
                0.4501,
                0.4740,
                0.5058,
                0.5462,
                0.5924,
                0.6422,
            ]
        )
        beta = beta_scale * np.array(
            [
                0.6349,
                0.5169,
                0.4655,
                0.4434,
                0.4163,
                0.3927,
                0.3664,
                0.3387,
                0.3086,
                0.2758,
                0.2402,
                0.2015,
                0.1589,
                0.1118,
                0.0600,
            ]
        )
        X0 = np.concatenate((gamma, beta), axis=0)
    elif p == 16:
        gamma = gamma_scale * np.array(
            [
                0.1771,
                0.3430,
                0.3612,
                0.3838,
                0.3896,
                0.3964,
                0.4011,
                0.4095,
                0.4197,
                0.4343,
                0.4532,
                0.4778,
                0.5099,
                0.5497,
                0.5944,
                0.6425,
            ]
        )
        beta = beta_scale * np.array(
            [
                0.6363,
                0.5184,
                0.4678,
                0.4469,
                0.4213,
                0.3996,
                0.3756,
                0.3505,
                0.3234,
                0.2940,
                0.2624,
                0.2281,
                0.1910,
                0.1504,
                0.1056,
                0.0566,
            ]
        )
        X0 = np.concatenate((gamma, beta), axis=0)
    elif p == 17:
        gamma = gamma_scale * np.array(
            [
                0.1735,
                0.3376,
                0.3562,
                0.3789,
                0.3844,
                0.3907,
                0.3946,
                0.4016,
                0.4099,
                0.4217,
                0.4370,
                0.4565,
                0.4816,
                0.5138,
                0.5530,
                0.5962,
                0.6429,
            ]
        )
        beta = beta_scale * np.array(
            [
                0.6375,
                0.5197,
                0.4697,
                0.4499,
                0.4255,
                0.4054,
                0.3832,
                0.3603,
                0.3358,
                0.3092,
                0.2807,
                0.2501,
                0.2171,
                0.1816,
                0.1426,
                0.1001,
                0.0536,
            ]
        )
        X0 = np.concatenate((gamma, beta), axis=0)

    return X0
