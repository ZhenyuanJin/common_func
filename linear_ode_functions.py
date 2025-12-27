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
    Performs spectral decomposition on the linear stability matrix A.
    Sorts eigenmodes from slowest to fastest.
    
    Args:
        A (np.ndarray): The Jacobian or linear stability matrix.
        tolerance (float): Threshold below which a decay rate is considered zero.
                           Rates < tolerance will result in NaN timescales.
    
    Returns:
        dict: Contains sorted eigenvalues, mode decay times, eigenvectors,
              and node-specific projection weights.
    """
    # 1. Eigen decomposition
    evals, evecs = np.linalg.eig(A)
    
    # 2. Sort by decay rate (absolute real part) in ascending order
    # Smallest rate = Slowest decay = Largest time constant
    decay_rates = np.abs(np.real(evals))
    sorted_indices = np.argsort(decay_rates)
    
    # 3. Reorder arrays based on the sorted indices
    sorted_evals = evals[sorted_indices]
    sorted_evecs = evecs[:, sorted_indices]
    sorted_rates = decay_rates[sorted_indices]
    
    # 4. Calculate characteristic decay times for each mode (Tau = 1 / |Re(lambda)|)
    # logic: if rate is close to 0, set to NaN to represent infinite/undefined timescale
    
    # Initialize an array full of NaNs
    mode_decay_times = np.full_like(sorted_rates, np.nan)
    
    # Create a mask for values that are effectively non-zero
    valid_mask = sorted_rates > tolerance
    
    # Only perform division where the rate is valid
    # This prevents RuntimeWarning: divide by zero encountered in true_divide
    np.divide(1.0, sorted_rates, out=mode_decay_times, where=valid_mask)
    
    # 5. Calculate projection weights matrix (magnitude of eigenvectors)
    # Shape: (N_nodes, N_modes)
    projection_matrix = np.abs(sorted_evecs)
    
    # 6. Organize projection weights by node index for easy retrieval
    node_projections = {}
    n_nodes = A.shape[0]
    for i in range(n_nodes):
        node_projections[i] = projection_matrix[i, :]
    
    # 7. characteristic polynomial coefficients
    poly_coeffs = np.poly(evals)  # Returns [a_n, a_{n-1}, ..., a_0], where a_n=1
    
    # Create a dictionary format of characteristic polynomial coefficients
    poly_coeffs_dict = {}
    n = len(poly_coeffs) - 1
    for i, coeff in enumerate(poly_coeffs):
        power = n - i
        poly_coeffs_dict[power] = coeff
            
    return {
        'sorted_eigenvalues': sorted_evals,
        'mode_decay_times': mode_decay_times,    # Array: [Slowest_Tau (or NaN), ..., Fastest_Tau]
        'sorted_eigenvectors': sorted_evecs,     # Matrix: Columns are eigenvectors
        'node_projections': node_projections,    # Dict: access node i's k-th mode weight via node_projections[i][k]
        'projection_matrix': projection_matrix,  # Matrix: For heatmap visualization
        'characteristic_polynomial_coefficients': poly_coeffs_dict  # Dict: {power: coefficient}
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