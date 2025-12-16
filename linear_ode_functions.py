import numpy as np
import scipy


def steady_state_covariance(A, D):
    """解Lyapunov方程 AΣ + ΣAᵀ = -D 得到稳态协方差Σ"""
    return scipy.linalg.solve_continuous_lyapunov(A, -D)


def acovf_single_lag_from_A(A, D, tau):
    """通过矩阵指数计算时滞τ的自协方差函数"""
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


def acovf_single_lag_from_eigendecomposition(V, eigenvalues, D, tau):
    """通过特征分解计算时滞τ的自协方差函数"""
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


def acovf_single_lag_via_spectrum(A, D, tau):
    """通过特征分解计算自协方差函数"""
    eigenvalues, V = np.linalg.eig(A)  # 计算A的特征分解
    return acovf_single_lag_from_eigendecomposition(V, eigenvalues, D, tau)


def acovf_multiple_lags_from_A(A, D, taus):
    """
    通过矩阵指数计算多个时滞τ的自协方差函数
    """
    Sigma = steady_state_covariance(A, D)
    
    results = []
    for tau in taus:
        if tau >= 0:
            cov_matrix = scipy.linalg.expm(A * tau) @ Sigma
        else:
            cov_matrix = Sigma @ scipy.linalg.expm(A.T * (-tau))
        results.append(cov_matrix)
    
    n = Sigma.shape[0]
    processed_results = {}
    for i, j in zip(range(n), range(n)):
        processed_results[(i, j)] = [results[k][i, j] for k in range(len(taus))]
    return processed_results


def acovf_multiple_lags_via_spectrum(A, D, taus):
    """
    通过特征分解计算多个时滞τ的自协方差函数
    """
    eigenvalues, V = np.linalg.eig(A)
    Vinv = np.linalg.inv(V)
    
    D_tilde = Vinv @ D @ Vinv.T
    lam_sum = eigenvalues[:, None] + eigenvalues[None, :]
    Sigma_tilde = -D_tilde / lam_sum
    
    results = []
    for tau in taus:
        if tau >= 0:
            decay = np.exp(eigenvalues * tau)
            cov_matrix = V @ (decay[:, None] * Sigma_tilde) @ V.T
        else:
            decay = np.exp(eigenvalues * (-tau))
            cov_matrix = V @ (Sigma_tilde * decay[None, :]) @ V.T
        results.append(cov_matrix)
    
    n = len(Sigma_tilde)
    processed_results = {}
    for i, j in zip(range(n), range(n)):
        processed_results[(i, j)] = [results[k][i, j] for k in range(len(taus))]
    return processed_results


def acovf_multiple_lags_from_eigendecomposition(V, eigenvalues, D, taus):
    """
    通过给定的特征分解计算多个时滞τ的自协方差函数
    """
    Vinv = np.linalg.inv(V)
    D_tilde = Vinv @ D @ Vinv.T
    lam_sum = eigenvalues[:, None] + eigenvalues[None, :]
    Sigma_tilde = -D_tilde / lam_sum
    
    results = []
    for tau in taus:
        if tau >= 0:
            decay = np.exp(eigenvalues * tau)
            cov_matrix = V @ (decay[:, None] * Sigma_tilde) @ V.T
        else:
            decay = np.exp(eigenvalues * (-tau))
            cov_matrix = V @ (Sigma_tilde * decay[None, :]) @ V.T
        results.append(cov_matrix)
    
    n = len(Sigma_tilde)
    processed_results = {}
    for i, j in zip(range(n), range(n)):
        processed_results[(i, j)] = [results[k][i, j] for k in range(len(taus))]
    return processed_results


def test_acovf():
    # 生成随机稳定矩阵A（特征值实部为负）
    n = 3
    A = np.random.randn(n, n) - 2 * np.eye(n)
    
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
    cov1 = acovf_single_lag_from_A(A, D, tau)
    cov2 = acovf_single_lag_from_eigendecomposition(V, eigenvalues, D, tau)
    cov3 = acovf_single_lag_via_spectrum(A, D, tau)
    
    assert np.allclose(cov1, cov2)
    assert np.allclose(cov1, cov3)
    
    # 测试多个时滞
    taus = [-1.0, 0.0, 0.5, 1.0]
    result1 = acovf_multiple_lags_from_A(A, D, taus)
    result2 = acovf_multiple_lags_via_spectrum(A, D, taus)
    result3 = acovf_multiple_lags_from_eigendecomposition(V, eigenvalues, D, taus)
    
    for key in result1:
        assert np.allclose(result1[key], result2[key])
        assert np.allclose(result1[key], result3[key])
    
    # 测试对称性
    assert np.allclose(acovf_single_lag_from_A(A, D, tau), 
                       acovf_single_lag_from_A(A, D, -tau).T)
    
    return True