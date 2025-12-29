import numpy as np
import scipy
from itertools import product
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import common_functions as cf
import neuron_data_functions as ndf


def steady_state_covariance(A, D):
    """解Lyapunov方程 AΣ + ΣAᵀ = -D 得到稳态协方差Σ"""
    return scipy.linalg.solve_continuous_lyapunov(A, -D)


def ccovf_single_lag_from_A(A, D, tau):
    """通过矩阵指数计算时滞τ的协方差函数"""
    Sigma = steady_state_covariance(A, D)
    if tau >= 0:
        return scipy.linalg.expm(A * tau) @ Sigma  # 正向传播
    else:
        return Sigma @ scipy.linalg.expm(A.T * (-tau))  # 反向传播


def steady_state_covariance_from_eigendecomposition(V, eigenvalues, D):
    """通过特征分解计算稳态协方差"""
    Vinv = np.linalg.inv(V)
    D_tilde = Vinv @ D @ Vinv.T  # 变换到特征空间
    lam_sum = eigenvalues[:, None] + eigenvalues[None, :]  # λ_i + λ_j
    Sigma_tilde = -D_tilde / lam_sum  # 特征空间的解
    return V @ Sigma_tilde @ V.T  # 变换回原空间


def ccovf_single_lag_from_eigendecomposition(V, eigenvalues, D, tau):
    """通过特征分解计算时滞τ的协方差函数"""
    Vinv = np.linalg.inv(V)
    D_tilde = Vinv @ D @ Vinv.T
    lam_sum = eigenvalues[:, None] + eigenvalues[None, :]
    Sigma_tilde = -D_tilde / lam_sum
    
    if tau >= 0:
        decay = np.exp(eigenvalues * tau)  # 正向衰减因子
        return V @ (decay[:, None] * Sigma_tilde) @ V.T
    else:
        decay = np.exp(eigenvalues * (-tau))  # 反向衰减因子
        return V @ (Sigma_tilde * decay[None, :]) @ V.T


def ccovf_single_lag_via_spectrum(A, D, tau):
    """通过特征分解计算协方差函数"""
    eigenvalues, V = np.linalg.eig(A)  # 计算A的特征分解
    return ccovf_single_lag_from_eigendecomposition(V, eigenvalues, D, tau)


def ccovf_multiple_lags_from_A(A, D, taus):
    """
    通过矩阵指数计算多个时滞τ的协方差函数
    """
    results = []
    for tau in taus:
        cov_matrix = ccovf_single_lag_from_A(A, D, tau)
        results.append(cov_matrix)
    
    n = A.shape[0]
    processed_results = {}
    for i, j in product(range(n), range(n)):
        processed_results[(i, j)] = [results[k][i, j] for k in range(len(taus))]
    return processed_results


def ccovf_multiple_lags_via_spectrum(A, D, taus):
    """
    通过特征分解计算多个时滞τ的协方差函数
    """
    results = []
    for tau in taus:
        cov_matrix = ccovf_single_lag_via_spectrum(A, D, tau)
        results.append(cov_matrix)
    
    n = A.shape[0]
    processed_results = {}
    for i, j in product(range(n), range(n)):
        processed_results[(i, j)] = [results[k][i, j] for k in range(len(taus))]
    return processed_results


def ccovf_multiple_lags_from_eigendecomposition(V, eigenvalues, D, taus):
    """
    通过给定的特征分解计算多个时滞τ的协方差函数
    """
    results = []
    for tau in taus:
        cov_matrix = ccovf_single_lag_from_eigendecomposition(V, eigenvalues, D, tau)
        results.append(cov_matrix)
    
    n = V.shape[0]
    processed_results = {}
    for i, j in product(range(n), range(n)):
        processed_results[(i, j)] = [results[k][i, j] for k in range(len(taus))]
    return processed_results


def get_acovf_and_fit_from_A_D(A, D, dt, nlags, fit_method='auto', **kwargs):
    lag_times = np.arange(nlags+1) * dt
    ccovf_dict = ccovf_multiple_lags_from_A(A, D, lag_times)
    n = A.shape[0]
    fit_results = {}
    for i in range(n):
        acovf_series = np.array(ccovf_dict[(i, i)])
        if fit_method == 'auto':
            results = ndf.select_exp_fit(lag_times, acovf_series, **kwargs)
        elif fit_method == 'single':
            results = ndf.single_exp_fit(lag_times, acovf_series, **kwargs)
        elif fit_method == 'double':
            results = ndf.double_exp_fit(lag_times, acovf_series, **kwargs)
        else:
            raise ValueError(f"Unknown fit_method: {fit_method}")
        results['acovf'] = acovf_series
        results['lag_times'] = lag_times
        fit_results[i] = results
    return fit_results


def eigen_decompose_linear_dynamics(A, tolerance=1e-12):
    """
    Performs spectral decomposition with comprehensive sorting and projection analysis.

    SORTING BEHAVIOR:
    -----------------
    1. Global Lists (eigenvalues_all): Sorted Algebraically (Real part: -Inf to +Inf).
    2. Dynamic Modes (decay/growth): Filtered and sorted by Slow -> Fast.
       - Index 0: Largest Timescale.
       - Index -1: Smallest Timescale.

    Args:
        A (np.ndarray): The Jacobian matrix (NxN).
        tolerance (float): Threshold for timescale calculations, if the real part of an eigenvalue is within
                           ±tolerance, it is not considered for decay/growth modes.

    Returns:
        dict: exhaustive dictionary containing:
            - Global eigen-info (algebraic sort)
            - Global projections
            - Decay-specific info (timescale sort, including projections)
            - Growth-specific info (timescale sort, including projections)
            - Stability and math info
    """
    n_nodes = A.shape[0]
    
    # 1. Eigen decomposition
    evals, evecs = np.linalg.eig(A)
    
    # --- GLOBAL SORTING (Algebraic: Most Negative -> Most Positive) ---
    # Useful for mathematical analysis of the spectrum spectrum
    real_parts_all = np.real(evals)
    global_sort_idx = np.argsort(real_parts_all)
    
    sorted_evals_complex = evals[global_sort_idx]
    sorted_evecs = evecs[:, global_sort_idx]
    
    # --- GLOBAL PROJECTIONS ---
    # Magnitude of eigenvectors: Rows=Nodes, Cols=Modes
    global_projection_matrix = np.abs(sorted_evecs)
    
    global_node_projections = {}
    for i in range(n_nodes):
        global_node_projections[i] = global_projection_matrix[i, :]
    
    # --- HELPER: FILTERING, SORTING & PROJECTION ---
    def process_modes(condition_mask):
        """
        Extracts modes, calculates timescales, sorts by slowness, 
        and computes specific projection matrices.
        """
        # 1. Extract raw candidates
        subset_evals = sorted_evals_complex[condition_mask]
        subset_evecs = sorted_evecs[:, condition_mask]
        
        # Handle empty case (e.g., no unstable modes)
        if len(subset_evals) == 0:
            return {
                'timescales': np.array([]),
                'eigenvalues': np.array([]),
                'eigenvectors': np.empty((n_nodes, 0)),
                'projection_matrix': np.empty((n_nodes, 0)),
                'node_projections': {i: np.array([]) for i in range(n_nodes)}
            }

        # 2. Calculate Timescales: Tau = 1 / |Re(lambda)|
        timescales = 1.0 / np.abs(np.real(subset_evals))
        
        # 3. Sort by Timescale: Descending (Largest Tau/Slowest -> Smallest Tau/Fastest)
        sort_idx = np.argsort(timescales)[::-1]
        
        # Apply sorting
        final_timescales = timescales[sort_idx]
        final_evals = subset_evals[sort_idx]
        final_evecs = subset_evecs[:, sort_idx]
        
        # 4. Compute Projections for this specific subset
        # This tells you node participation specifically in these modes (ordered by slowness)
        proj_matrix = np.abs(final_evecs)
        
        node_projs = {}
        for i in range(n_nodes):
            node_projs[i] = proj_matrix[i, :]
            
        return {
            'timescales': final_timescales,
            'eigenvalues': final_evals,
            'eigenvectors': final_evecs,
            'projection_matrix': proj_matrix,
            'node_projections': node_projs
        }

    # Identify masks based on algebraic real part
    real_parts_sorted = np.real(sorted_evals_complex)
    
    decay_mask = real_parts_sorted < -tolerance
    growth_mask = real_parts_sorted > tolerance
    
    # Process the specific lists
    decay_info = process_modes(decay_mask)
    growth_info = process_modes(growth_mask)

    # 4. Characteristic polynomial
    poly_coeffs = np.poly(evals)
    poly_coeffs_dict = {}
    for i, coeff in enumerate(poly_coeffs[::-1]):
        poly_coeffs_dict[i] = coeff

    return {
        # 1. GLOBAL INFO (Sorted Algebraically: -Re to +Re)
        'eigenvalues_all': sorted_evals_complex,
        'eigenvectors_all': sorted_evecs,
        'projection_matrix_all': global_projection_matrix,
        'node_projections_all': global_node_projections,
        
        # 2. DECAY MODES (Stable)
        # Sorted: Slowest -> Fastest
        'decay_modes_timescales': decay_info['timescales'],
        'decay_modes_eigenvalues': decay_info['eigenvalues'], 
        'decay_modes_eigenvectors': decay_info['eigenvectors'],
        'decay_modes_projection_matrix': decay_info['projection_matrix'],
        'decay_modes_node_projections': decay_info['node_projections'], # node_projections[node_index][mode_index]
        
        # 3. GROWTH MODES (Unstable)
        # Sorted: Slowest -> Fastest
        'growth_modes_timescales': growth_info['timescales'],
        'growth_modes_eigenvalues': growth_info['eigenvalues'],
        'growth_modes_eigenvectors': growth_info['eigenvectors'],
        'growth_modes_projection_matrix': growth_info['projection_matrix'],
        'growth_modes_node_projections': growth_info['node_projections'], # node_projections[node_index][mode_index]
        
        # 4. SUMMARY STATS
        'is_stable_system': len(growth_info['timescales']) == 0,
        'characteristic_polynomial_coefficients': poly_coeffs_dict
    }


def test_ccovf():
    # 生成随机稳定矩阵A（特征值实部为负）
    n = 3
    A = np.random.randn(n, n) - 5 * np.eye(n)
    
    # 生成对称正定矩阵D
    B = np.random.randn(n, n)
    D = B @ B.T
    
    # 测试稳态协方差
    Sigma1 = steady_state_covariance(A, D)
    eigenvalues, V = np.linalg.eig(A)
    Sigma2 = steady_state_covariance_from_eigendecomposition(V, eigenvalues, D)
    
    assert np.allclose(Sigma1, Sigma2)
    
    # 测试单个时滞
    tau = 0.5
    cov1 = ccovf_single_lag_from_A(A, D, tau)
    cov2 = ccovf_single_lag_from_eigendecomposition(V, eigenvalues, D, tau)
    cov3 = ccovf_single_lag_via_spectrum(A, D, tau)
    
    assert np.allclose(cov1, cov2)
    assert np.allclose(cov1, cov3)
    
    # 测试多个时滞
    taus = [-1.0, 0.0, 0.5, 1.0]
    result1 = ccovf_multiple_lags_from_A(A, D, taus)
    result2 = ccovf_multiple_lags_via_spectrum(A, D, taus)
    result3 = ccovf_multiple_lags_from_eigendecomposition(V, eigenvalues, D, taus)
    
    for key in result1:
        assert np.allclose(result1[key], result2[key])
        assert np.allclose(result1[key], result3[key])
    
    # 测试对称性
    assert np.allclose(ccovf_single_lag_from_A(A, D, tau), 
                       ccovf_single_lag_from_A(A, D, -tau).T)
    
    # 模拟一个ODE,并验证自协方差函数
    x = np.zeros((n, 100000))
    dt = 0.01
    C = np.array([[2.0, 0.5, 0.2],
                  [0.5, 5.0, 0.3],
                  [0.2, 0.3, 10.0]])
    for t in range(1, x.shape[1]):
        dx = A @ x[:, t-1] * dt + C @ np.random.multivariate_normal(np.zeros(n), D * dt)
        x[:, t] = x[:, t-1] + dx
    nlags = 1000
    lag_times, empirical_ccovf = cf.get_multi_ccovf(x, x, T=dt, nlags=nlags)
    theoretical_ccovf = ccovf_multiple_lags_from_A(A, C@D@C.T, lag_times)
    fit_results = get_acovf_and_fit_from_A_D(A, C@D@C.T, dt, nlags, fit_method='auto')
    
    if n > 4:
        raise ValueError("Too many variables to plot.")
    fig, ax = cf.gfa(ncols=n, nrows=n)
    for i, j in product(range(n), range(n)):
        ax[i, j].plot(lag_times, empirical_ccovf[(i, j)], label='Empirical', color='blue')
        ax[i, j].plot(lag_times, theoretical_ccovf[(i, j)], label='Theoretical', color='red', linestyle='--')
        if i == j:
            ax[i, j].plot(lag_times, fit_results[i]['fitted_curve'], label='Fitted', color='green', linestyle=':')
            ax[i, j].set_title(f'Auto-covariance of variable {i}')
        else:
            ax[i, j].set_title(f'Cross-covariance of variables {i} and {j}')
        ax[i, j].legend()
    return True