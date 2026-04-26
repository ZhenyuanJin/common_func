import numpy as np
import scipy.optimize
import os
import sys
import tqdm
import itertools
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import common_functions as cf


def get_unique_roots(candidates, repeat_tol=1e-6):
    unique_roots = []
    for candidate in candidates:
        is_unique = True
        for existing in unique_roots:
            if np.linalg.norm(candidate - existing) < repeat_tol:
                is_unique = False
                break
        if is_unique:
            unique_roots.append(candidate)
    
    return np.array(unique_roots)


def find_roots_one_dim(f, grid_points, repeat_tol=1e-6):
    roots = []
    for i in range(len(grid_points) - 1):
        # bisect
        left = grid_points[i]
        right = grid_points[i + 1]

        if f(left) * f(right) < 0:
            root = scipy.optimize.bisect(f, left, right)
            roots.append(root)
    
        # hybr, in case the root is not found by bisect (e.g. multiple roots, the grid points are roots)
        root = scipy.optimize.root(f, (left + right) / 2, method='hybr')
        roots.append(root.x)
    return get_unique_roots(roots, repeat_tol)


def generate_initial_points(bounds, num):
    '''
    Generate initial points for root finding
    bounds: list of tuples, each tuple contains the lower and upper bounds of the variable. Example: [(0, 1), (0, 1)]
    num: number of initial points
    '''
    initial_points = []
    for _ in range(num):
        point = np.array([np.random.uniform(bound[0], bound[1]) for bound in bounds])
        initial_points.append(point)
    return initial_points


def find_roots_multi_dim(func, initial_guesses, repeat_tol=1e-6, method='hybr', **kwargs):
    '''
    repeat_tol: tolerance for repeating roots
    '''
    candidates = []
    
    for x0 in initial_guesses:
        result = scipy.optimize.root(func, x0, method=method, **kwargs)
        
        if result.success:
            candidates.append(result.x)
    return get_unique_roots(candidates, repeat_tol)


def get_closest_root(roots, point, keep_shape=False):
    """
    从根列表中找到最接近指定点的根
    
    参数:
    roots: numpy数组,形状为(n, d),其中n是根的数量,d是维度
    point: numpy数组,形状为(d,),指定点的坐标
    keep_shape: 是否保持roots为二维
    
    返回:
    closest_root: 最接近指定点的根
    """
    if len(roots) == 0:
        raise ValueError("根列表为空")
    
    # 计算每个根到指定点的距离
    distances = np.linalg.norm(roots - point, axis=1)
    
    # 找到最小距离的索引
    min_idx = np.argmin(distances)
    
    # 返回最接近的根
    if keep_shape:
        return roots[min_idx:min_idx+1]
    else:
        return roots[min_idx]


def test_root_finding():
    """测试函数正确性"""
    
    # 测试一维情况
    f1 = lambda x: x**2
    grid = np.linspace(-5, 5, 21)
    print(grid)
    roots_1d = find_roots_one_dim(f1, grid)
    print(f"一维根: {roots_1d}")
    
    # 测试多维情况
    def f2(x):
        return [x[0]**2 + x[1]**2 - 1, x[0] - x[1]]
    
    bounds = [(-2, 2), (-2, 2)]
    samples = generate_initial_points(bounds, 100)
    roots_2d = find_roots_multi_dim(f2, samples, tol=1e-8)
    print(f"多维根:\n{roots_2d}")
    
    # 验证结果
    for root in roots_2d:
        residual = np.linalg.norm(f2(root))
        assert residual < 1e-6, f"残差过大: {residual}"
    
    return roots_1d, roots_2d


def find_monotonic_intervals(func, domain=(-1e6, 1e6), num_points=10000, min_interval_length=0.0):
    """
    找到给定函数在指定域上的单调区间
    
    参数:
    func: 输入函数
    domain: 定义域
    num_points: 采样点数量
    min_interval_length: 最小区间长度,小于此长度的区间将被忽略
    
    返回:
    单调区间列表

    注意:
    有可能比实际的单调区间切分的更细,但是保证输出的每个区间内函数都是单调的
    """
    lower, upper = domain
    x_vals = np.linspace(lower, upper, num_points)
    y_vals = func(x_vals)
    
    monotonic_intervals = []
    start_idx = 0
    
    for i in range(1, num_points - 1):
        dy_prev = y_vals[i] - y_vals[i-1]
        dy_next = y_vals[i+1] - y_vals[i]
        
        sign_change = (dy_prev * dy_next < 0) or (dy_prev == 0 and dy_next != 0) or (dy_prev != 0 and dy_next == 0)
        
        if sign_change:
            interval_length = x_vals[i] - x_vals[start_idx]
            if interval_length >= min_interval_length:
                if x_vals[start_idx] != x_vals[i]:
                    monotonic_intervals.append((x_vals[start_idx], x_vals[i]))
            start_idx = i
    
    if x_vals[start_idx] != x_vals[-1]:
        interval_length = x_vals[-1] - x_vals[start_idx]
        if interval_length >= min_interval_length:
            monotonic_intervals.append((x_vals[start_idx], x_vals[-1]))
    
    return monotonic_intervals


def estimate_derivative(x, y):
    n = len(x)
    if n != len(y):
        raise ValueError("x and y must have the same length")
    if n == 0:
        return [], []
    if n == 1:
        return [x[0]], [float('nan')]
    
    d = [0.0] * n
    d[0] = (y[1] - y[0]) / (x[1] - x[0])
    d[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
    
    for i in range(1, n - 1):
        d[i] = (y[i + 1] - y[i - 1]) / (x[i + 1] - x[i - 1])
    
    return d


def get_numerical_derivative_f(f, h=1e-5, method='central'):
    """
    数值求导函数
    
    参数:
    f: 要求导的函数
    h: 步长
    method: 求导方法，可选 'forward', 'backward', 'central'
    
    返回:
    f在x处的导数
    """
    if method == 'forward':
        def df(x):
            return (f(x + h) - f(x)) / h
    elif method == 'backward':
        def df(x):
            return (f(x) - f(x - h)) / h
    elif method == 'central':
        def df(x):
            return (f(x + h) - f(x - h)) / (2 * h)
    else:
        raise ValueError("method必须是'forward', 'backward'或'central'")
    return df


def cubic_spline_fit(x, y):
    """
    三次样条拟合函数
    
    参数:
    x: 自变量数据点
    y: 因变量数据点
    
    返回:
    spline: 一个CubicSpline对象,可以像函数一样调用来进行插值计算
            基本用法示例:
                spline = cubic_spline_fit(x_data, y_data)
                y_new = spline(x_new)  # 计算新x值对应的y值
                
            对象属性:
                - spline.c: 样条系数
                - spline.x: 节点位置
                - spline.derivative(): 返回样条的导数函数
                - spline.antiderivative(): 返回样条的原函数(积分)
                
            注意:
                由于设置了extrapolate=True,可以在数据范围外进行外推
                但外推结果可能不可靠，特别是在远离数据点的区域
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    if len(x) != len(y):
        raise ValueError("x和y的长度必须相同")
    
    if len(x) < 2:
        raise ValueError("至少需要2个点才能进行插值")
    
    spline = scipy.interpolate.CubicSpline(x, y, extrapolate=True)
    
    return spline


def piecewise_linear_interpolate(x, y):
    """
    线性分段插值函数
    
    参数:
    x: 自变量数据点
    y: 因变量数据点
    
    返回:
    interpolant: 可调用函数,接收x_new参数返回插值结果
                基本用法示例:
                    f = piecewise_linear_interp(x_data, y_data)
                    y_new = f(x_new)
                    
                内部属性:
                    - f.x_nodes: 节点位置
                    - f.y_nodes: 节点值
                    
                注意:
                    - 支持单点输入和数组输入
                    - 输入超出数据范围时使用最近邻外推
    """
    x_nodes = np.asarray(x, dtype=np.float64)
    y_nodes = np.asarray(y, dtype=np.float64)
    
    if len(x_nodes) != len(y_nodes):
        raise ValueError("x和y的长度必须相同")
    
    if len(x_nodes) < 2:
        raise ValueError("至少需要2个点才能进行插值")
    
    # 确保节点按升序排列
    sort_idx = np.argsort(x_nodes)
    x_sorted = x_nodes[sort_idx]
    y_sorted = y_nodes[sort_idx]
    
    def interpolate(x_new):
        x_new_arr = np.asarray(x_new, dtype=np.float64)
        is_scalar = x_new_arr.ndim == 0
        
        if is_scalar:
            x_new_arr = np.array([x_new_arr])
        
        # 查找每个x_new对应的区间索引
        idx = np.searchsorted(x_sorted, x_new_arr, side='right') - 1
        
        # 处理边界外的情况
        idx = np.clip(idx, 0, len(x_sorted) - 2)
        
        # 计算插值权重
        x_left = x_sorted[idx]
        x_right = x_sorted[idx + 1]
        y_left = y_sorted[idx]
        y_right = y_sorted[idx + 1]
        
        # 避免除零（处理重复节点）
        with np.errstate(divide='ignore', invalid='ignore'):
            slope = (y_right - y_left) / (x_right - x_left)
            result = y_left + slope * (x_new_arr - x_left)
        
        # 对于重复节点，使用左侧值
        mask_duplicate = x_right == x_left
        if mask_duplicate.any():
            result = np.where(mask_duplicate, y_left, result)
        
        return result[0] if is_scalar else result
    
    # 添加属性便于访问
    interpolate.x_nodes = x_sorted
    interpolate.y_nodes = y_sorted
    
    return interpolate


def inverse_function(f, domain, n_points=1000, tol=1e-10):
    """
    数值求解函数的反函数
    
    参数:
    f: 原函数
    domain: 定义域区间(a, b)
    n_points: 用于构建反函数的点数
    tol: 容差
    
    返回:
    反函数
    """
    a, b = domain
    
    x_test = np.linspace(a, b, 100)
    y_test = f(x_test)
    
    diff = np.diff(y_test)
    is_increasing = np.all(diff >= -tol)
    is_decreasing = np.all(diff <= tol)
    
    if not (is_increasing or is_decreasing):
        print(f"警告:函数在给定区间内可能不是单调的,反函数可能不准确,递增的点数为: {np.sum(is_increasing)}, 递减的点数为: {np.sum(is_decreasing)}")
    
    x_dense = np.linspace(a, b, n_points)
    y_dense = f(x_dense)
    
    if is_increasing:
        y_min, y_max = y_dense[0], y_dense[-1]
    else:
        y_min, y_max = y_dense[-1], y_dense[0]
    
    def inverse_f(y):
        if np.isscalar(y):
            y = float(y)
            
            if y < y_min - tol or y > y_max + tol:
                raise ValueError(f"y={y}不在值域[{y_min:.4f}, {y_max:.4f}]内")
            
            left, right = a, b
            for _ in range(100):
                mid = (left + right) / 2
                f_mid = f(mid)
                
                if abs(f_mid - y) < tol:
                    return mid
                
                if (is_increasing and f_mid < y) or (is_decreasing and f_mid > y):
                    left = mid
                else:
                    right = mid
            
            return (left + right) / 2
        else:
            y_arr = np.asarray(y)
            result = np.zeros_like(y_arr)
            
            for i, yi in enumerate(y_arr):
                if yi < y_min - tol or yi > y_max + tol:
                    raise ValueError(f"y[{i}]={yi}不在值域[{y_min:.4f}, {y_max:.4f}]内")
                
                if is_increasing:
                    idx = np.searchsorted(y_dense, yi)
                else:
                    idx = np.searchsorted(-y_dense, -yi)
                
                if idx == 0:
                    ratio = 0.0
                elif idx == len(y_dense):
                    ratio = 1.0
                else:
                    if is_increasing:
                        ratio = (yi - y_dense[idx-1]) / (y_dense[idx] - y_dense[idx-1])
                    else:
                        ratio = (y_dense[idx-1] - yi) / (y_dense[idx-1] - y_dense[idx])
                
                x_val = x_dense[idx-1] + ratio * (x_dense[idx] - x_dense[idx-1])
                result[i] = x_val
            
            return result
    
    return inverse_f


def transform_pdf(original_pdf, transform_func, find_monotonic_kwargs=None, inverse_function_kwargs=None, numerical_derivative_kwargs=None):
    """
    计算转换后的随机变量的概率密度函数
    
    参数:
    original_pdf: 原始随机变量的概率密度函数
    transform_func: 转换函数 Y = g(X)
    
    返回:
    转换后随机变量的概率密度函数
    """
    if find_monotonic_kwargs is None:
        find_monotonic_kwargs = {}
    if inverse_function_kwargs is None:
        inverse_function_kwargs = {}
    if numerical_derivative_kwargs is None:
        numerical_derivative_kwargs = {}

    monotonic_intervals = find_monotonic_intervals(transform_func, **find_monotonic_kwargs)
    print(f'单调区间为: {monotonic_intervals}')

    inverse_funcs = []
    interval_ranges = []
    
    for interval in monotonic_intervals:
        try:
            inv_func = inverse_function(transform_func, interval, **inverse_function_kwargs)
            inverse_funcs.append(inv_func)
            
            left, right = interval
            y_left = transform_func(left)
            y_right = transform_func(right)
            y_min = min(y_left, y_right)
            y_max = max(y_left, y_right)
            interval_ranges.append((y_min, y_max))
            
        except Exception as e:
            print(f"警告: 在区间 {interval} 上计算反函数时出错: {e}")
            continue
    
    inverse_derivatives = []
    for inv_func in inverse_funcs:
        try:
            inv_deriv = get_numerical_derivative_f(inv_func, **numerical_derivative_kwargs)
            inverse_derivatives.append(inv_deriv)
        except Exception as e:
            print(f"警告: 计算反函数的导数时出错: {e}")
            inverse_derivatives.append(None)
    
    def transformed_pdf(y):
        """
        转换后的概率密度函数
        
        参数:
        y: 转换后随机变量的值
        
        返回:
        转换后随机变量在y处的概率密度
        """
        if isinstance(y, (int, float)):
            y = float(y)
            result = 0.0
            
            for (y_min, y_max), inv_func, inv_deriv in zip(interval_ranges, inverse_funcs, inverse_derivatives):
                if y_min <= y <= y_max and inv_deriv is not None:
                    try:
                        x = inv_func(y)
                        pdf_val = original_pdf(x)
                        jacobian = abs(inv_deriv(y))
                        contribution = pdf_val * jacobian
                        result += contribution
                    except Exception as e:
                        continue
            
            return result
        
        else:
            y_arr = np.asarray(y)
            result = np.zeros_like(y_arr)
            
            for i, yi in enumerate(y_arr):
                result[i] = transformed_pdf(yi)
                
            return result
    
    return transformed_pdf


def find_local_extreme(x, y):
    '''
    找到局部极值点,返回局部极小值点和局部极大值点的坐标数组
    '''
    x = np.array(x)
    y = np.array(y)
    
    min_list = []
    max_list = []
    
    for i in range(1, len(y)-1):
        if y[i] < y[i-1] and y[i] < y[i+1]:
            min_list.append([x[i], y[i]])
            
    for i in range(1, len(y)-1):
        if y[i] > y[i-1] and y[i] > y[i+1]:
            max_list.append([x[i], y[i]])
    
    return min_list, max_list


def matrix_norm(A, mode):
    '''
    '1', '2', 'inf', 'fro' 都可用于估计特征值模长上界
    '''
    if mode == '1':
        return np.max(np.sum(np.abs(A), axis=0))
    
    elif mode == '2':
        s = np.linalg.svd(A, compute_uv=False)
        return np.max(s) if s.size > 0 else 0.0
    
    elif mode == 'inf':
        return np.max(np.sum(np.abs(A), axis=1))
    
    elif mode == 'fro':
        return np.linalg.norm(A, 'fro')
    
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def get_refined_grid(start, end, coarse_num, keypoints, near_num, refine_factor):
    '''
    refine_factor: 关键点附近细化倍数
    '''
    coarse_pts = np.linspace(start, end, coarse_num + 1)
    fine_spacing = (end - start) / coarse_num / refine_factor
    fine_points = []
    
    for kp in keypoints:
        left_idx = max(0, int((kp - start - near_num * fine_spacing) / fine_spacing))
        right_idx = int((kp - start + near_num * fine_spacing) / fine_spacing) + 1
        left_bound = start + left_idx * fine_spacing
        right_bound = start + right_idx * fine_spacing
        
        fine_segment = np.linspace(left_bound, right_bound, right_idx - left_idx + 1)
        fine_points.append(fine_segment)
    
    if fine_points:
        fine_points = np.concatenate(fine_points)
        all_points = np.sort(np.unique(np.concatenate([coarse_pts, fine_points])))
    else:
        all_points = coarse_pts
    
    return all_points


# region timeseries
def find_steady_state(timeseries, threshold, mode='median'):
    """
    Find steady state characteristics in a time series.
    
    Args:
        time_series: 输入时间序列数据
        threshold: 判定稳定的阈值
    
    Returns:
        steady_value: 稳态典型值(使用中位数)
        steady_phase: 稳定阶段的所有数据点(从稳定起点到结尾)
        steady_start_idx: 稳定阶段的起始索引
    """
    # 计算稳态典型值
    if mode == 'median':
        steady_value = np.median(timeseries)
    elif mode == 'mean':
        steady_value = np.mean(timeseries)
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    
    # 找出所有接近稳态值的候选点
    near_steady_mask = np.abs(timeseries - steady_value) < threshold
    near_steady_indices = np.where(near_steady_mask)[0]
    
    # 处理空候选情况(说明threshold过小)
    if len(near_steady_indices) == 0:
        return find_steady_state(timeseries, threshold=threshold * 2, mode=mode)
    
    # 确定稳定起始点(取候选点前20%位置)
    steady_start_idx = near_steady_indices[int(len(near_steady_indices) * 0.2)]
    
    # 提取稳定阶段数据
    steady_timeseries = timeseries[steady_start_idx:]
    
    return steady_value, steady_timeseries, steady_start_idx


def get_unified_steady_state(multi_timeseries, threshold, mode='median'):
    """
    获取多个节点统一的稳态起始点和稳态时间序列。
    
    对每个节点计算其稳态起始索引，取所有节点中的最大值作为统一的起始索引，
    然后从这个统一索引开始截取所有节点的稳态时间序列。
    
    Args:
        multi_timeseries: 形状为 (n_nodes, n_timesteps) 的时间序列数组
        threshold: 判定稳定的阈值
        mode: 计算典型值的模式，'median' 或 'mean'
    
    Returns:
        unified_start_idx: 统一的稳态起始索引
        steady_timeseries: 形状为 (n_nodes, n_timesteps) 的稳态时间序列
    """
    n_nodes = multi_timeseries.shape[0]
    all_steady_start_idxs = np.zeros(n_nodes, dtype=int)
    
    for node in range(n_nodes):
        _, _, start_idx = find_steady_state(multi_timeseries[node, :], threshold, mode)
        all_steady_start_idxs[node] = start_idx
    
    unified_start_idx = np.max(all_steady_start_idxs)
    steady_timeseries = multi_timeseries[:, unified_start_idx:]
    
    return unified_start_idx, steady_timeseries
# endregion


# region eigen
def sort_eigenvalues_eigenvectors(eigenvalues, eigenvectors):
    '''
    sort eigenvalues and eigenvectors by real part and imaginary part, both in ascending order
    '''
    sort_key = np.lexsort((eigenvalues.imag, eigenvalues.real))
    
    sorted_eigenvalues = eigenvalues[sort_key]
    sorted_eigenvectors = eigenvectors[:, sort_key]
    
    return sorted_eigenvalues, sorted_eigenvectors


def get_sorted_eigenvalues_eigenvectors(A):
    eigenvalues, eigenvectors = np.linalg.eig(A)
    sorted_eigenvalues, sorted_eigenvectors = sort_eigenvalues_eigenvectors(eigenvalues, eigenvectors)
    return sorted_eigenvalues, sorted_eigenvectors
# endregion


# region custom matrix
def get_custom_matrix_shape(d):
    '''
    d: {(i,j): val}, i: row index, j: column index
    返回矩阵维数 (row_dim, col_dim)
    '''
    row_dim = max(i for (i, j) in d.keys()) + 1
    col_dim = max(j for (i, j) in d.keys()) + 1
    return (row_dim, col_dim)


def fill_custom_matrix(d, fill_val, row_dim, col_dim):
    '''
    d: {(i,j): val}, i: row index, j: column index
    fill_val: 填充值
    返回填充后的矩阵字典
    '''
    filled_dict = {}
    # 先填充全量默认值
    for i in range(row_dim):
        for j in range(col_dim):
            filled_dict[(i, j)] = fill_val
    # 用原始数据覆盖
    filled_dict.update(d)
    return filled_dict


def np_to_custom_matrix(np_mat):
    '''
    np_mat: numpy 矩阵
    返回矩阵字典 {(i,j): val}, i: row index, j: column index
    '''
    d = {}
    assert len(np_mat.shape) == 2, "输入必须是二维矩阵"
    rows, cols = np_mat.shape
    for i in range(rows):
        for j in range(cols):
            d[(i, j)] = np_mat[i, j]
    return d


def custom_matrix_mul(d1, d2, mul_func, add_func, zero_val=None, d1_row_dim=None, d1_col_dim=None, d2_row_dim=None, d2_col_dim=None, fill_zero=True):
    '''
    d: {(i,j): val}, i: row index, j: column index
    mul_func: function to multiply two values
    add_func: function to add two values
    '''
    # 计算维数
    if d1_row_dim is None or d1_col_dim is None:
        d1_row_dim, d1_col_dim = get_custom_matrix_shape(d1)
    if d2_row_dim is None or d2_col_dim is None:
        d2_row_dim, d2_col_dim = get_custom_matrix_shape(d2)
    if d1_col_dim != d2_row_dim:
        raise ValueError("矩阵维数不匹配,无法相乘")

    # 预处理 d1: 按行组织 {i: {j: val}}
    row_d1 = {}
    for (i, j), val in d1.items():
        if i not in row_d1:
            row_d1[i] = {}
        row_d1[i][j] = val
    
    # 预处理 d2: 按行组织 {j: {k: val}}(注意: d2 的行索引 j 对应乘法中的中间维度)
    row_d2 = {}
    for (j, k), val in d2.items():
        if j not in row_d2:
            row_d2[j] = {}
        row_d2[j][k] = val
    
    # 计算结果字典
    result = {}
    for i, row1 in row_d1.items():
        for j, val1 in row1.items():
            # 仅当 d2 包含行 j 时继续(避免 KeyError 且提升效率)
            if j in row_d2:
                for k, val2 in row_d2[j].items():
                    product = mul_func(val1, val2)
                    key = (i, k)
                    if key in result:
                        result[key] = add_func(result[key], product)
                    else:
                        result[key] = product
    if fill_zero:
        result = fill_custom_matrix(result, zero_val, d1_row_dim, d2_col_dim)
    return result
# endregion


# region poly
def get_poly_root(coef):
    '''
    coef: 多项式系数,升序 (index 0 is s^0)
    '''
    coef_rev = coef[::-1]
    roots = np.roots(coef_rev)
    return roots


def get_slowest_root(roots):
    '''
    获取实部最接近零的负实部根,代表系统的慢动态.
    '''
    negative_roots = [r for r in roots if np.real(r) < 0]
    if not negative_roots:
        raise ValueError("没有负实部根")
    if len(negative_roots) != len(roots):
        print("警告: 存在非负实部根,已忽略这些根")
    slowest_root = min(negative_roots, key=lambda r: abs(np.real(r)))
    return slowest_root


def poly_mul(p1_coef, p2_coef):
    '''
    多项式乘法辅助函数.
    输入 p1_coef, p2_coef 为升序系数 (index 0 is s^0).
    np.convolve 在这种定义下直接对应多项式乘法,结果也是升序.
    '''
    return np.convolve(p1_coef, p2_coef)


def poly_div(numerator_coef, denominator_coef):
    '''
    多项式除法辅助函数.
    输入 numerator_coef, denominator_coef 为升序系数 (index 0 is s^0).
    np.polydiv 需要降序系数,所以先反转.
    返回 商 和 余数,均为升序系数.
    即: numerator_coef = poly_mul(denominator_coef, q) + r
    '''
    num_rev = numerator_coef[::-1]
    den_rev = denominator_coef[::-1]
    q_rev, r_rev = np.polydiv(num_rev, den_rev)
    q = q_rev[::-1]
    r = r_rev[::-1]
    return q, r


def poly_linear_comb(coef1, coef2, a, b):
    '''
    多项式线性组合辅助函数.
    输入 coef1, coef2 为升序系数 (index 0 is s^0).
    '''
    len1 = len(coef1)
    len2 = len(coef2)
    if len1 < len2:
        coef1 = np.pad(coef1, (0, len2 - len1), 'constant')
    elif len2 < len1:
        coef2 = np.pad(coef2, (0, len1 - len2), 'constant')
    return a * coef1 + b * coef2


def poly_add(coef1, coef2):
    '''
    多项式加法辅助函数.
    输入 coef1, coef2 为升序系数 (index 0 is s^0).
    '''
    return poly_linear_comb(coef1, coef2, 1.0, 1.0)


def constant_to_rational(constant):
    '''
    将常数转换为有理函数形式的系数表示.
    返回 (numerator_coef, denominator_coef)
    '''
    numerator_coef = np.array([constant])
    denominator_coef = np.array([1.0])
    return numerator_coef, denominator_coef


def constant_matrix_to_rational(constant_matrix, fill_zero=True, tol=1e-12):
    '''
    将常数矩阵转换为有理函数矩阵形式的系数表示.
    constant_matrix: numpy 矩阵
    fill_zero: 是否填充零元素
    tol: 零元素的阈值
    返回 {(i,j): (numerator_coef, denominator_coef)}, i: row index, j: column index
    '''
    rational_coef_dict = {}
    rows, cols = constant_matrix.shape
    for i in range(rows):
        for j in range(cols):
            if fill_zero or abs(constant_matrix[i, j]) > tol:
                numerator_coef, denominator_coef = constant_to_rational(constant_matrix[i, j])
                rational_coef_dict[(i, j)] = (numerator_coef, denominator_coef)
            else:
                pass
    return rational_coef_dict


def rational_mul(numerator1_coef, denominator1_coef, numerator2_coef, denominator2_coef, simplify_rational=False, simplify_mode='gcd', simplify_tol=1e-8):
    '''
    计算两个有理函数的乘积: (numerator1_coef/denominator1_coef) * (numerator2_coef/denominator2_coef)
    '''
    numerator = poly_mul(numerator1_coef, numerator2_coef)
    denominator = poly_mul(denominator1_coef, denominator2_coef)
    if np.array_equal(numerator1_coef, np.array([0.0])) or np.array_equal(numerator2_coef, np.array([0.0])):
        numerator = np.array([0.0])
        denominator = np.array([1.0])
    if simplify_rational:
        numerator, denominator = rational_simplify(numerator, denominator, tol=simplify_tol, mode=simplify_mode)
    return numerator, denominator


def rational_mul_matrix(rational_coef_dict1, rational_coef_dict2, simplify_rational=False, simplify_mode='gcd', simplify_tol=1e-8, fill_zero=True):
    '''
    计算两个有理函数矩阵的乘积: {(i,j): (numerator_coef, denominator_coef)}, i: row index, j: column index
    返回结果也是 {(i,j): (numerator_coef, denominator_coef)}
    '''
    def mul_func(val1, val2):
        num1, den1 = val1
        num2, den2 = val2
        num_res, den_res = rational_mul(num1, den1, num2, den2, simplify_rational, simplify_mode, simplify_tol)
        return (num_res, den_res)
    def add_func(val1, val2):
        num1, den1 = val1
        num2, den2 = val2
        num_res, den_res = rational_add(num1, den1, num2, den2, simplify_rational=simplify_rational, simplify_mode=simplify_mode, simplify_tol=simplify_tol)
        return (num_res, den_res)
    result_dict = custom_matrix_mul(rational_coef_dict1, rational_coef_dict2, mul_func, add_func, zero_val=(np.array([0.0]), np.array([1.0])), fill_zero=fill_zero)
    return result_dict


def rational_linear_comb(numerator1_coef, denominator1_coef, numerator2_coef, denominator2_coef, a, b, simplify_rational=False, simplify_mode='gcd', simplify_tol=1e-8):
    '''
    计算两个有理函数的线性组合: a*(numerator1_coef/denominator1_coef) + b*(numerator2_coef/denominator2_coef)
    '''
    term1 = poly_mul(numerator1_coef, denominator2_coef)
    term2 = poly_mul(numerator2_coef, denominator1_coef)
    numerator = poly_linear_comb(term1, term2, a, b)
    denominator = poly_mul(denominator1_coef, denominator2_coef)
    if simplify_rational:
        numerator, denominator = rational_simplify(numerator, denominator, tol=simplify_tol, mode=simplify_mode)
    return numerator, denominator


def rational_linear_comb_matrix(rational_coef_dict1, rational_coef_dict2, a, b, simplify_rational=False, simplify_mode='gcd', simplify_tol=1e-8):
    rational_coef_dict = {}
    for (i, j) in rational_coef_dict1.keys():
        numerator, denominator = rational_linear_comb(rational_coef_dict1[(i, j)][0], rational_coef_dict1[(i, j)][1], rational_coef_dict2[(i, j)][0], rational_coef_dict2[(i, j)][1], a, b, simplify_rational, simplify_mode, simplify_tol)
        rational_coef_dict[(i, j)] = (numerator, denominator)
    return rational_coef_dict


def rational_add(numerator1_coef, denominator1_coef, numerator2_coef, denominator2_coef, **kwargs):
    '''
    计算两个有理函数的和: (numerator1_coef/denominator1_coef) + (numerator2_coef/denominator2_coef)
    '''
    return rational_linear_comb(numerator1_coef, denominator1_coef, numerator2_coef, denominator2_coef, 1.0, 1.0, **kwargs)


def poly_val(coef, s):
    '''
    多项式求值辅助函数.
    输入 coef 为升序系数 (index 0 is s^0).
    np.polyval 需要降序系数,所以先反转.
    '''
    coef_rev = coef[::-1]
    return np.polyval(coef_rev, s)


def rational_val(numerator_coef, denominator_coef, s, simplify_rational=False, simplify_mode='gcd', simplify_tol=1e-8):
    '''
    Evaluate rational function numerator(s) / denominator(s) at point(s) s.
    Although we enable simplification before evaluation, it's better to simplify the rational function once and reuse it for multiple evaluations.
    '''
    if simplify_rational:
        numerator_coef, denominator_coef = rational_simplify(numerator_coef, denominator_coef, tol=simplify_tol, mode=simplify_mode)
    num_val = poly_val(numerator_coef, s)
    den_val = poly_val(denominator_coef, s)
    return num_val / den_val


def rational_val_matrix(rational_coef_dict, s, simplify_rational=False, simplify_mode='gcd', simplify_tol=1e-8):
    '''
    Evaluate rational function matrix at point(s) s.
    rational_coef_dict: {(i,j): (numerator_coef, denominator_coef)}, i: row index, j: column index
    Returns a matrix of evaluated values.
    '''
    row_dim, col_dim = get_custom_matrix_shape(rational_coef_dict)
    result_matrix = np.zeros((row_dim, col_dim), dtype=np.complex128)
    
    for (i, j), (numerator_coef, denominator_coef) in rational_coef_dict.items():
        val = rational_val(numerator_coef, denominator_coef, s, simplify_rational, simplify_mode, simplify_tol)
        result_matrix[i, j] = val
    
    return result_matrix


def poly_gcd(p1_coef, p2_coef, tol=1e-8):
    '''
    计算两个多项式的最大公约数 (GCD).
    输入 p1_coef, p2_coef 为升序系数 (index 0 is s^0).
    返回 GCD 的升序系数.
    '''
    a = p1_coef
    b = p2_coef
    
    while np.any(np.abs(b) > tol):
        q, r = poly_div(a, b)
        r[np.abs(r) < tol] = 0.0
        a, b = b, r
    
    leading_coeff = a[-1]
    if abs(leading_coeff) > tol:
        a = a / leading_coeff
    
    return a


def rational_simplify(numerator_coef, denominator_coef, tol=1e-8, mode='gcd'):
    '''
    有理函数约分辅助函数.
    输入 numerator_coef, denominator_coef 为升序系数 (index 0 is s^0).
    mode: 'gcd' 使用多项式GCD约分, 'zeros' 使用根的比较约分.
    '''
    if mode == 'gcd':
        gcd_coef = poly_gcd(numerator_coef, denominator_coef, tol)
        q_num, _ = poly_div(numerator_coef, gcd_coef)
        q_den, _ = poly_div(denominator_coef, gcd_coef)
        return q_num, q_den
    elif mode == 'zeros':
        zeros = get_poly_root(numerator_coef)
        poles = get_poly_root(denominator_coef)
        zeros_simplified, poles_simplified = cancel_poles_zeros(zeros, poles, tol)
        if len(zeros_simplified) == 0:
            new_numerator_coef = np.array([1.0]) * numerator_coef[-1]
        else:
            new_numerator_coef = np.poly(zeros_simplified)[::-1] * numerator_coef[-1]
        if len(poles_simplified) == 0:
            new_denominator_coef = np.array([1.0]) * denominator_coef[-1]
        else:
            new_denominator_coef = np.poly(poles_simplified)[::-1] * denominator_coef[-1]
        
        return new_numerator_coef, new_denominator_coef
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def cancel_poles_zeros(zeros, poles, tol=1e-5):
    '''
    对比分子根(Zeros)和分母根(Poles),如果足够接近则约去.
    返回约分后的 Zeros 和 Poles.
    '''
    final_poles = list(poles)
    final_zeros = []
    
    for z in zeros:
        best_idx = -1
        min_dist = float('inf')
        
        for idx, p in enumerate(final_poles):
            dist = np.abs(z - p)
            if dist < min_dist:
                min_dist = dist
                best_idx = idx
        
        if min_dist < tol:
            final_poles.pop(best_idx)
        else:
            final_zeros.append(z)
            
    return np.array(final_zeros), np.array(final_poles)


def rational_residue(numerator_coef, denominator_coef, simplify_rational=False, simplify_mode='gcd', simplify_tol=1e-8):
    '''
    return: residues, poles, direct_term
    residues: residues of the partial fraction expansion, as a numpy array
    poles: poles of the partial fraction expansion, as a numpy array
    direct_term: coefficients of the direct polynomial term, as a numpy array, from lowest degree to highest degree
    '''
    if simplify_rational:
        numerator_coef, denominator_coef = rational_simplify(numerator_coef, denominator_coef, tol=simplify_tol, mode=simplify_mode)
    numerator_rev = numerator_coef[::-1]
    denominator_rev = denominator_coef[::-1]
    r, p, k = scipy.signal.residue(numerator_rev, denominator_rev)
    k = k[::-1]
    return r, p, k


def visualize_rational_function_on_real_axis(ax, numerator_coef, denominator_coef, linthresh=1e-3, cancel_mode=True, cancel_tol=1e-5):
    zeros = get_poly_root(numerator_coef)
    poles = get_poly_root(denominator_coef)
    if cancel_mode:
        zeros_processed, poles_processed = cancel_poles_zeros(zeros, poles, cancel_tol)
    else:
        zeros_processed, poles_processed = zeros, poles
    def f(x):
        return poly_val(numerator_coef, x) / poly_val(denominator_coef, x)

    zeros_real_min = np.min(zeros_processed.real)
    zeros_real_max = np.max(zeros_processed.real)

    poles_real_min = np.min(poles_processed.real)
    poles_real_max = np.max(poles_processed.real)

    real_min = min(zeros_real_min, poles_real_min)
    real_max = max(zeros_real_max, poles_real_max)
    
    x_range = cf.scale_range(real_min, real_max, prop=2)
    
    x_grid = get_refined_grid(x_range[0], x_range[1], coarse_num=1000, keypoints=np.concatenate([zeros.real, poles.real]), near_num=50, refine_factor=10)
    
    f_values = f(x_grid)
    magnitude = np.abs(f_values)
    
    cf.plt_line(ax, x_grid, magnitude)
    
    for zero in zeros_processed:
        cf.add_vline(ax, zero.real, color='green', linestyle='--', label=f'Zero: {zero.real:.3f}, timescale: { -1/zero.real:.2f}')
    for pole in poles_processed:
        cf.add_vline(ax, pole.real, color='red', linestyle='--', label=f'Pole: {pole.real:.3f}, timescale: { -1/pole.real:.2f}')
    
    cf.set_ax(ax, xlabel='Real', ylabel='|f|', ylog=True)
    cf.set_symlog_scale(ax, axis='x', linthresh=linthresh)
# endregion


# region Laplace
def inverse_laplace(F, t, sigma=1.0, max_omega=1000, n_points=10000):
    """
    数值计算拉普拉斯逆变换
    
    Args:
        F: 拉普拉斯变换函数 F(s)，接受复数 s
        t: 时间点（可标量或数组）
        sigma: Bromwich 积分路径实部
        max_omega: 积分上限 ω
        n_points: 积分点数
        
    Returns:
        拉普拉斯逆变换结果 f(t)
    """
    if np.isscalar(t):
        t = np.array([t])
    
    def integrand(omega, t_val):
        s = sigma + 1j * omega
        return (np.exp(s * t_val) * F(s)).real
    
    result = np.zeros_like(t, dtype=float)
    
    for i, t_val in enumerate(t):
        if t_val <= 0:
            result[i] = 0.0
        else:
            omega_vals = np.linspace(-max_omega, max_omega, n_points)
            f_vals = integrand(omega_vals, t_val)
            result[i] = (np.exp(sigma * t_val) / np.pi) * np.trapz(f_vals, omega_vals)
    
    return result[0] if len(result) == 1 else result
# endregion


# region stochastic process
def get_OU_process(tau, mu, sigma, dt, n_steps, x0=None, seed=12345):
    '''
    t: shape (n_steps,)
    x: shape (n_steps,)
    '''
    if x0 is None:
        x0 = mu
    t = np.arange(n_steps) * dt
    x = np.zeros(n_steps)
    x[0] = x0
    rng = cf.get_local_rng(seed)
    noise = rng.normal(size=n_steps-1)  # 一次性生成所有噪声
    for i in range(1, n_steps):
        dx_drift = -(x[i-1] - mu) / tau * dt
        dx_diffusion = sigma * np.sqrt(dt) * noise[i-1]
        x[i] = x[i-1] + dx_drift + dx_diffusion
    return t, x


def get_multi_independent_OU_process(tau_vec, mu_vec, sigma_vec, dt, n_steps, x0_vec=None, seed=12345):
    '''
    t: shape (n_steps,)
    x: shape (n_dim, n_steps)
    '''
    tau_vec = np.array(tau_vec)
    mu_vec = np.array(mu_vec)
    sigma_vec = np.array(sigma_vec)
    n_dim = len(tau_vec)
    if x0_vec is None:
        x0_vec = mu_vec
    t = np.arange(n_steps) * dt
    x = np.zeros((n_dim, n_steps))
    x[:, 0] = x0_vec
    rng = cf.get_local_rng(seed)
    noise = rng.normal(size=(n_dim, n_steps-1))  # 一次性生成所有噪声
    for i in range(1, n_steps):
        dx_drift = -(x[:, i-1] - mu_vec) / tau_vec * dt
        dx_diffusion = sigma_vec * np.sqrt(dt) * noise[:, i-1]
        x[:, i] = x[:, i-1] + dx_drift + dx_diffusion
    return t, x
# endregion


# region fit distribution (power-law)
def fit_powerlaw_mle(data, xmin=None, mode='continuous'):
    if isinstance(data, list):
        data = np.array(data)
    if xmin is None:
        xmin = np.min(data)
    
    data_filtered = data[data >= xmin]
    n = len(data_filtered)
    if n == 0:
        raise ValueError("No data points within the specified range [xmin, inf)")

    if mode == 'continuous':
        alpha = 1 + n / np.sum(np.log(data_filtered / xmin))
    elif mode == 'discrete':
        alpha = 1 + n / np.sum(np.log(data_filtered / (xmin - 0.5)))
    else:
        raise ValueError("mode must be 'continuous' or 'discrete'")
        
    return alpha


def fit_powerlaw_scatter(x, y):
    '''
    return: alpha (positive), C, where y = C * x^(-alpha)
    '''
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(y, list):
        y = np.array(y)

    mask = (x > 0) & (y > 0)

    x_fit = x[mask]
    y_fit = y[mask]

    if len(x_fit) < 2:
        raise ValueError("Not enough data points to perform fitting.")

    logx = np.log10(x_fit)
    logy = np.log10(y_fit)

    slope, intercept = np.polyfit(logx, logy, 1)

    alpha = -slope
    C = 10**intercept

    return alpha, C


def plot_powerlaw_pdf_line(ax, alpha, xmin, xmax, C=None, mode='continuous'):
    x_range = np.array([xmin, xmax])
    if C is None:
        if mode == 'continuous':
            C = (alpha - 1) * xmin**(alpha - 1)
        elif mode == 'discrete':
            C = (alpha - 1) * (xmin - 0.5)**(alpha - 1)
        else:
            raise ValueError("mode must be 'continuous' or 'discrete'")
            
    y_range = C * x_range**(-alpha)
    ax.plot(x_range, y_range, 'k--', label=f'slope={-alpha:.2f}')


def get_log_bin_pdf(data, n_bins):
    if isinstance(data, list):
        data = np.array(data)
    data = data[data > 0]
    if len(data) == 0:
        raise ValueError("Data must contain positive values for logarithmic binning.")
    
    min_val = np.min(data)
    max_val = np.max(data)
    bins = np.logspace(np.log10(min_val), np.log10(max_val), n_bins)
    
    counts, _ = np.histogram(data, bins=bins)
    bin_widths = np.diff(bins)
    pdf = counts / (np.sum(counts) * bin_widths)
    
    bin_centers = np.sqrt(bins[:-1] * bins[1:])
    
    mask = pdf > 0
    return bin_centers[mask], pdf[mask]


def get_powerlaw_deviation_index_D(sizes, n_bins, mode='discrete'):
    """
    Calculates the normalized distance D to the best-fitting power-law distribution
    using logarithmic bins and a least-squares fit on the log-log histogram.

    D = sum(S * |P(S) - P_fit(S)|) / sum(S * P(S))

    Parameters
    ----------
    sizes : array_like
        Array of avalanche sizes.

    Returns
    -------
    float
        The normalized distance D.
    """
    assert mode == 'discrete', "Currently only 'discrete' mode is supported."
    sizes = np.array(sizes)
    bin_centers, pdf = get_log_bin_pdf(sizes, n_bins=n_bins)
    
    if len(bin_centers) < 2:
        raise ValueError("Not enough data points to perform fitting.")

    log_s = np.log10(bin_centers)
    log_p = np.log10(pdf)

    b1, b0 = np.polyfit(log_s, log_p, 1)

    p_fit = 10**(b0 + b1 * log_s)

    numerator = np.sum(bin_centers * np.abs(pdf - p_fit))
    denominator = np.sum(bin_centers * pdf)

    return numerator / denominator


def fit_powerlaw_mle_doubly_truncated(data, xmin, xmax, mode='continuous'):
    """
    Estimates the power-law exponent alpha for a doubly truncated distribution 
    (xmin <= x <= xmax) using Maximum Likelihood Estimation (MLE).
    """
    data = np.array(data)
    data = data[(data >= xmin) & (data <= xmax)]
    n = len(data)
    
    if n == 0:
        raise ValueError("No data points within the specified range [xmin, xmax]")

    sum_log_data = np.sum(np.log(data))

    def neg_log_likelihood(alpha):
        if alpha <= 0: 
            raise ValueError("alpha must be positive")
        
        if mode == 'discrete':
            term_range = np.arange(int(xmin), int(xmax) + 1)
            normalization = np.sum(term_range**(-alpha))
        elif mode == 'continuous':
            if np.isclose(alpha, 1):
                normalization = np.log(xmax / xmin)
            else:
                normalization = (xmax**(1 - alpha) - xmin**(1 - alpha)) / (1 - alpha)
        else:
            raise ValueError("mode must be 'continuous' or 'discrete'")

        ll = -alpha * sum_log_data - n * np.log(normalization)
        return -ll

    res = scipy.optimize.minimize_scalar(neg_log_likelihood, bounds=(0.1, 5.0), method='bounded')
    return res.x


def get_powerlaw_ks_statistic_truncated(data, alpha, xmin, xmax, mode='continuous'):
    """
    Calculates the Kolmogorov-Smirnov (KS) statistic for a doubly truncated 
    power-law distribution within the range [xmin, xmax].
    """
    data = np.array(data)
    data = data[(data >= xmin) & (data <= xmax)]
    n = len(data)
    
    if n == 0:
        raise ValueError("No data points within the specified range [xmin, xmax]")

    data_sorted = np.sort(data)
    empirical_cdf = np.arange(1, n + 1) / n
    
    if mode == 'discrete':
        x_values = np.arange(int(xmin), int(xmax) + 1)
        pdf_theoretical = x_values**(-alpha)
        pdf_theoretical /= np.sum(pdf_theoretical)
        
        cdf_theoretical_lookup = np.cumsum(pdf_theoretical)
        
        indices = (data_sorted - xmin).astype(int)
        theoretical_cdf_vals = cdf_theoretical_lookup[indices]
        
    elif mode == 'continuous':
        if np.isclose(alpha, 1):
            num = np.log(data_sorted / xmin)
            den = np.log(xmax / xmin)
        else:
            num = data_sorted**(1 - alpha) - xmin**(1 - alpha)
            den = xmax**(1 - alpha) - xmin**(1 - alpha)
        theoretical_cdf_vals = num / den
        
    else:
        raise ValueError("mode must be 'continuous' or 'discrete'")

    d_plus = np.abs(empirical_cdf - theoretical_cdf_vals)
    d_minus = np.abs(theoretical_cdf_vals - (np.arange(0, n) / n))
    
    return np.max(np.concatenate((d_plus, d_minus)))


def find_optimal_powerlaw_truncated_range(data, n_sims=100, mode='continuous', step=1, min_width_ratio=1/3):
    """
    Identifies the largest valid doubly truncated data range [xmin, xmax] according to 
    the KS-statistic criteria described Liang et al 2020 Frontiers, with adaptations.

    This function performs a search over possible start (xmin) and end (xmax) points 
    to find the widest range (on a logarithmic scale) where the data consistently 
    follows a power-law distribution.

    The validity of a range is determined by two conditions derived from the text:
    1. The logarithmic width of the range is at least min_width_ratio of the total 
       data range.
    2. The Goodness-of-Fit p-value is the best.

    The p-value is calculated using a Monte Carlo (parametric bootstrap) approach:
    - The KS statistic (D) is computed for the empirical data against the fitted model.
    - 'n_sims' synthetic datasets are generated using the estimated parameters.
    - For each synthetic set, parameters are re-fitted and a new KS statistic (D_sim) is computed.
    - The p-value is the fraction of synthetic sets where D_sim > D (i.e., the model 
      generates a worse fit than the empirical data).

    Parameters
    ----------
    data : array-like
        The input dataset containing avalanche sizes or durations.
    n_sims : int, optional
        The number of Monte Carlo simulations to run for p-value estimation. 
        Higher values provide more precision but increase computation time. 
        Default is 100.
    mode : str, optional
        The nature of the data, either 'continuous' or 'discrete'. 
        Default is 'continuous'.
    step : int, optional
        The step size for iterating over unique data values when selecting 
        potential xmin and xmax. Larger steps reduce computation time but may 
        miss optimal ranges. Default is 1.

    Returns
    -------
    dict or None
        A dictionary containing the optimal boundary parameters if found, otherwise None.
        Keys:
            - 'xmin': Lower bound of the truncated range.
            - 'xmax': Upper bound of the truncated range.
            - 'alpha': The MLE estimated exponent for this range.
            - 'p_value': The bootstrap p-value.
            - 'ks_stat': The KS statistic for the empirical data.
            - 'log_width': The width of the range in log10 scale.
    """
    data = np.sort(np.array(data))
    data = data[data > 0] 
    
    unique_vals = np.unique(data)
    n_unique = len(unique_vals)
    
    if n_unique < 2:
        raise ValueError("Data must contain at least two unique positive values.")
        
    global_log_min = np.log10(unique_vals[0])
    global_log_max = np.log10(unique_vals[-1])
    total_log_width = global_log_max - global_log_min
    min_width_threshold = total_log_width * min_width_ratio

    best_result = {'p_value': -np.inf}

    # for i in range(0, n_unique, step):
    for i in tqdm.tqdm(range(0, n_unique, step), desc="Searching for optimal power-law range", total=(n_unique + step - 1) // step):
        xmin = unique_vals[i]
        
        remaining_log_width = global_log_max - np.log10(xmin)
        if remaining_log_width < min_width_threshold:
            break

        for j in range(n_unique - 1, i, -step):
            xmax = unique_vals[j]
            
            current_log_width = np.log10(xmax) - np.log10(xmin)
            if current_log_width < min_width_threshold:
                break

            subset = data[(data >= xmin) & (data <= xmax)]
            n = len(subset)
            if n < 10:
                continue

            with cf.FlexibleTry():
                alpha = fit_powerlaw_mle_doubly_truncated(subset, xmin, xmax, mode=mode)
                ks_real = get_powerlaw_ks_statistic_truncated(subset, alpha, xmin, xmax, mode=mode)

            ks_greater_count = 0
            valid_sims = 0

            for _ in range(n_sims):
                if np.isclose(alpha, 1):
                    rand_vals = np.random.random(n)
                    sim_data = xmin * (xmax / xmin) ** rand_vals
                else:
                    rand_vals = np.random.random(n)
                    term1 = xmax ** (1 - alpha)
                    term2 = xmin ** (1 - alpha)
                    sim_data = ((term1 - term2) * rand_vals + term2) ** (1 / (1 - alpha))
                
                if mode == 'discrete':
                    sim_data = np.floor(sim_data)
                    sim_data = sim_data[(sim_data >= xmin) & (sim_data <= xmax)]
                    if len(sim_data) == 0:
                        continue

                with cf.FlexibleTry():
                    alpha_sim = fit_powerlaw_mle_doubly_truncated(sim_data, xmin, xmax, mode=mode)
                    ks_sim = get_powerlaw_ks_statistic_truncated(sim_data, alpha_sim, xmin, xmax, mode=mode)
                    valid_sims += 1
                    if ks_sim > ks_real:
                        ks_greater_count += 1

            if valid_sims == 0:
                continue

            p_value = ks_greater_count / valid_sims

            if p_value > best_result['p_value']:
                C = (1 - alpha) / (xmax ** (1 - alpha) - xmin ** (1 - alpha))
                best_result = {
                    'xmin': xmin,
                    'xmax': xmax,
                    'alpha': alpha,
                    'C': C,
                    'p_value': p_value,
                    'ks_stat': ks_real,
                    'log_width': current_log_width,
                }

    return best_result
# endregion


# region NCC toolbox python version
def plmle(x, mode, xmin=None, xmax=None, exponent_range=(1.0, 5.0), precision=1e-3):
    """
    Maximum likelihood estimation for truncated power-law exponent.

    Parameters
    ----------
    x : array_like
        Data values. Must be positive.
    mode : {'continuous', 'discrete'}
        Data type:
        - 'continuous': real-valued data from a continuous power law.
        - 'discrete': integer data from a discrete power law.
    xmin : float, optional
        Lower truncation bound. If None, set to min(x).
    xmax : float, optional
        Upper truncation bound. If None, set to max(x).
    exponent_range : (float, float), optional
        Search interval for the exponent. Default (1.0, 5.0).
    precision : float, optional
        Desired precision of the estimate. Must be a power of ten (e.g., 1e-3).
        Default 1e-3.

    Returns
    -------
    dict
        A dictionary with keys:
        - 'exponent' : float, estimated exponent.
        - 'xmin' : float, lower bound used.
        - 'xmax' : float, upper bound used.
        - 'log_likelihood' : float, maximum log-likelihood value.
        - 'mode' : str, data type used.

    Raises
    ------
    ValueError
        If inputs are invalid (e.g., xmin <= 0, precision not a power of ten,
        no data points in [xmin, xmax]).
    """
    # ----------------------------------------------------------------------
    # Input validation
    # ----------------------------------------------------------------------
    if mode not in ('continuous', 'discrete'):
        raise ValueError("mode must be either 'continuous' or 'discrete'")

    x = np.asarray(x).ravel()
    if np.any(x <= 0):
        raise ValueError("All data values must be positive (power law requires x > 0).")

    # Check that precision is a power of ten
    log10_prec = np.log10(precision)
    if not np.isclose(log10_prec, round(log10_prec)):
        raise ValueError("precision must be a power of ten (e.g., 1e-2, 1e-3, ...)")

    # Set bounds if not provided
    if xmin is None:
        xmin = x.min()
    if xmax is None:
        xmax = x.max()

    if xmin >= xmax:
        raise ValueError("xmin must be strictly less than xmax.")
    if xmin <= 0:
        raise ValueError("xmin must be positive (power law requires x >= xmin > 0).")

    if exponent_range[0] >= exponent_range[1]:
        raise ValueError("exponent_range[0] must be less than exponent_range[1].")

    # ----------------------------------------------------------------------
    # Data preparation
    # ----------------------------------------------------------------------
    # For discrete mode, ensure integer boundaries and round data
    if mode == 'discrete':
        # Check that data are effectively integers
        if not np.allclose(x, np.round(x)):
            raise ValueError("Data contains non-integer values but mode='discrete'.")
        x = np.round(x).astype(int)
        xmin = int(np.round(xmin))
        xmax = int(np.round(xmax))
        # Ensure xmin <= xmax after rounding
        if xmin > xmax:
            xmin, xmax = xmax, xmin

    # Truncate data to [xmin, xmax]
    mask = (x >= xmin) & (x <= xmax)
    z = x[mask]
    n = len(z)
    if n == 0:
        raise ValueError("No data points lie in the interval [xmin, xmax].")

    # Pre-compute sum of logarithms (used in both continuous and discrete)
    logz_sum = np.sum(np.log(z))

    # ----------------------------------------------------------------------
    # Pre-compute support for discrete case (all integer values in [xmin, xmax])
    # ----------------------------------------------------------------------
    if mode == 'discrete':
        # Note: xmin and xmax are already integers
        all_vals = np.arange(xmin, xmax + 1)
        # Remove any zero or negative values (should not happen due to earlier check)
        all_vals = all_vals[all_vals > 0]
        if len(all_vals) == 0:
            raise ValueError("The interval [xmin, xmax] contains no positive integers.")
        # For vectorized computation, we will use all_vals as a column vector
        # and exponents as a row vector later.
    else:
        r = xmin / xmax  # ratio used in continuous formula
        # Handle r == 1 (xmin == xmax) separately to avoid division by zero
        if np.isclose(r, 1.0):
            # Degenerate case: only one distinct value; cannot estimate exponent reliably.
            # Fallback: treat as continuous with xmin == xmax, which makes the distribution
            # essentially a single point. The continuous formula is not defined.
            # We'll raise an error (or could return NaN).
            raise ValueError("xmin and xmax are equal; cannot estimate exponent.")

    # ----------------------------------------------------------------------
    # Iterative refinement search for exponent
    # ----------------------------------------------------------------------
    n_iter = int(-np.log10(precision))          # number of refinement steps
    exp_low, exp_high = exponent_range

    # We'll store the log-likelihood values for the final iteration
    L_vals_final = None
    exp_candidates_final = None
    best_exp = None

    for it in range(n_iter):
        # Step size for this iteration: 10^(-it-1)
        spacing = 10.0 ** (-it - 1)

        # Build candidate exponents
        if it == 0:
            # First iteration: full range
            # Use np.arange carefully to avoid floating point overshoot
            exp_candidates = np.arange(exp_low, exp_high + spacing * 0.5, spacing)
        else:
            # Shrink interval around the best exponent from previous iteration
            idx_best = np.argmax(L_vals)
            if idx_best == 0:
                # best is leftmost, expand to right only
                new_low = exp_candidates[0]
                new_high = exp_candidates[1]
            elif idx_best == len(exp_candidates) - 1:
                # best is rightmost, expand to left only
                new_low = exp_candidates[-2]
                new_high = exp_candidates[-1]
            else:
                # best is interior
                new_low = exp_candidates[idx_best - 1]
                new_high = exp_candidates[idx_best + 1]
            exp_candidates = np.arange(new_low, new_high + spacing * 0.5, spacing)

        # Compute log-likelihood for each candidate
        if mode == 'continuous':
            # Continuous truncated power law (Deluca & Corral 2013)
            # L(exponent) = log((exponent-1) / (1 - r^(exponent-1))) - exponent * mean(log(z)) - (1-exponent)*log(xmin)
            # Special case exponent = 1
            exponent = exp_candidates
            is_one = np.isclose(exponent, 1.0)

            # For exponent != 1, compute normally
            exponent_safe = exponent.copy()
            exponent_safe[is_one] = 1.0 + 1e-12  # avoid division by zero later

            term1 = np.log((exponent_safe - 1) / (1 - r ** (exponent_safe - 1)))
            # For exponent == 1, the limit is -log(log(1/r))
            term1[is_one] = -np.log(np.log(1 / r))

            L_vals = term1 - exponent * (logz_sum / n) - (1 - exponent) * np.log(xmin)

        else:  # mode == 'discrete'
            # Discrete power law: L(exponent) = -log( Z(exponent) ) - exponent * mean(log(z))
            # where Z(exponent) = sum_{x = xmin}^{xmax} x^{-exponent}
            # Use broadcasting: all_vals (M) x exponent (K) -> M x K matrix
            all_vals_col = all_vals[:, np.newaxis]   # shape (M, 1)
            exponent_row = exp_candidates[np.newaxis, :]  # shape (1, K)
            # Compute x^(-exponent) for all x and all exponent
            # Note: avoid overflow for large exponents by working in log space?
            # Direct computation should be fine if exponents not too large (range up to ~10).
            powers = all_vals_col ** (-exponent_row)      # M x K
            Z = np.sum(powers, axis=0)                # shape (K,)
            L_vals = -np.log(Z) - exp_candidates * (logz_sum / n)

        # Find best exponent for this iteration
        best_idx = np.argmax(L_vals)
        best_exp = exp_candidates[best_idx]
        L_vals_final = L_vals
        exp_candidates_final = exp_candidates

    # ----------------------------------------------------------------------
    # Prepare output
    # ----------------------------------------------------------------------
    return {
        'exponent': best_exp,
        'xmin': xmin,
        'xmax': xmax,
        'log_likelihood': L_vals_final[best_idx],
        'mode': mode
    }


def pvcalc(x, exponent, xmin=None, xmax=None, samples=500, threshold=1.0, likelihood=1e-3):
    """
    Compute p-value for a truncated power-law fit using Monte Carlo simulations.

    Parameters
    ----------
    x : array_like
        Empirical data. Must be positive.
    exponent : float
        Estimated exponent from plmle (or another MLE routine).
    xmin : float, optional
        Lower truncation bound. If None, set to min(x).
    xmax : float, optional
        Upper truncation bound. If None, set to max(x).
    samples : int, optional
        Number of Monte Carlo samples to draw. Default 500.
    threshold : float, optional
        Critical p-value threshold for early stopping. If threshold=1,
        full computation is performed. Default 1.
    likelihood : float, optional
        Likelihood threshold for the binomial early-stopping rule.
        Default 1e-3.

    Returns
    -------
    dict
        A dictionary with keys:
        - 'p' : float, estimated p-value.
        - 'empirical_ks' : float, KS statistic between empirical data and model.
        - 'sample_ks' : ndarray, KS statistics for each drawn sample (length = samples).
        - 'sigma_exponent' : float or NaN, standard deviation of exponent estimates from samples
          (only meaningful if threshold == 1 and all samples were used).
        - 'mode' : str, inferred data type ('continuous' or 'discrete').

    Raises
    ------
    ValueError
        If inputs are invalid (e.g., no data in [xmin, xmax], xmin >= xmax,
        exponent <= 0, etc.)
    """
    # ----------------------------------------------------------------------
    # Input validation and preparation
    # ----------------------------------------------------------------------
    x = np.asarray(x).ravel()
    if np.any(x <= 0):
        raise ValueError("All data values must be positive (power law requires x > 0).")

    if exponent <= 0:
        raise ValueError("Exponent exponent must be positive.")

    # Set bounds if not provided
    if xmin is None:
        xmin = x.min()
    if xmax is None:
        xmax = x.max()

    if xmin >= xmax:
        raise ValueError("xmin must be strictly less than xmax.")
    if xmin <= 0:
        raise ValueError("xmin must be positive (power law requires x >= xmin > 0).")

    # Truncate data to [xmin, xmax] for analysis
    mask = (x >= xmin) & (x <= xmax)
    x_trunc = x[mask]
    n = len(x_trunc)
    if n == 0:
        raise ValueError("No data points lie in the interval [xmin, xmax].")

    # ----------------------------------------------------------------------
    # Infer data type (continuous / discrete) from the truncated data
    # ----------------------------------------------------------------------
    if np.allclose(x_trunc, np.round(x_trunc)):
        mode = 'discrete'
        # Round to integers and ensure bounds are integers
        x_trunc = np.round(x_trunc).astype(int)
        xmin_int = int(np.round(xmin))
        xmax_int = int(np.round(xmax))
        if xmin_int > xmax_int:
            xmin_int, xmax_int = xmax_int, xmin_int
        xmin = xmin_int
        xmax = xmax_int
        support = np.arange(xmin, xmax + 1)
        # Remove any zero or negative values (should not happen due to earlier check)
        support = support[support > 0]
        if len(support) == 0:
            raise ValueError("The interval [xmin, xmax] contains no positive integers.")
    else:
        mode = 'continuous'
        # No further modification needed

    # ----------------------------------------------------------------------
    # Compute empirical KS statistic
    # ----------------------------------------------------------------------
    if mode == 'discrete':
        # Empirical pdf over the integer support
        counts = np.bincount(x_trunc - xmin, minlength=len(support))
        pdf_emp = counts / n
        cdf_emp = np.cumsum(pdf_emp)

        # Theoretical pdf and cdf
        pdf_theo = support ** (-exponent)
        pdf_theo /= pdf_theo.sum()
        cdf_theo = np.cumsum(pdf_theo)

        empirical_ks = np.max(np.abs(cdf_emp - cdf_theo))

    else:  # continuous
        # Sort truncated data
        x_sorted = np.sort(x_trunc)
        # Empirical CDF
        cdf_emp = np.arange(1, n + 1) / n

        # Theoretical CDF at the sorted data points
        # F(x) = (x^(1-exponent) - xmin^(1-exponent)) / (xmax^(1-exponent) - xmin^(1-exponent))
        # Avoid exponent == 1 by a small perturbation if necessary
        if np.isclose(exponent, 1.0):
            exponent = 1.0 + 1e-12
        term_x = x_sorted ** (1 - exponent)
        term_min = xmin ** (1 - exponent)
        term_max = xmax ** (1 - exponent)
        cdf_theo = (term_x - term_min) / (term_max - term_min)

        empirical_ks = np.max(np.abs(cdf_emp - cdf_theo))

    # ----------------------------------------------------------------------
    # Prepare for Monte Carlo simulation
    # ----------------------------------------------------------------------
    # Pre‑compute theoretical probabilities (used for sample generation)
    if mode == 'discrete':
        # Probabilities for each integer in the support
        p_theo = support ** (-exponent)
        p_theo /= p_theo.sum()
    else:
        # For continuous, we will use inverse transform sampling inside the loop
        pass

    critical_threshold = samples * threshold
    success = np.zeros(samples, dtype=bool)
    sample_ks = np.zeros(samples)
    sample_exponent = np.zeros(samples) if threshold == 1.0 else None

    n_successes = 0
    i = 0
    early_stop = False
    binom_cutoff = likelihood

    # ----------------------------------------------------------------------
    # Monte Carlo loop with early stopping
    # ----------------------------------------------------------------------
    while i < samples and not early_stop:
        # Generate a synthetic sample of size n from the fitted model
        if mode == 'discrete':
            # Use np.random.choice (equivalent to multinomial + rldecode)
            x_sample = np.random.choice(support, size=n, p=p_theo)
            # Empirical pdf and cdf of the sample
            counts_sample = np.bincount(x_sample - xmin, minlength=len(support))
            pdf_sample = counts_sample / n
            cdf_sample = np.cumsum(pdf_sample)
            # Theoretical cdf (same as before)
            ks_sample = np.max(np.abs(cdf_sample - cdf_theo))
        else:  # continuous
            # Inverse transform sampling
            u = np.random.uniform(0, 1, n)
            # x = [ u*(xmax^(1-exponent)-xmin^(1-exponent)) + xmin^(1-exponent) ]^(1/(1-exponent))
            base = u * (term_max - term_min) + term_min
            x_sample = base ** (1.0 / (1.0 - exponent))
            x_sample = np.sort(x_sample)
            # Empirical CDF of the sample
            cdf_sample = np.arange(1, n + 1) / n
            # Theoretical CDF at the sample points
            term_x = x_sample ** (1 - exponent)
            cdf_theo_sample = (term_x - term_min) / (term_max - term_min)
            ks_sample = np.max(np.abs(cdf_sample - cdf_theo_sample))

        sample_ks[i] = ks_sample

        # Fit exponent to the sample if requested (full computation)
        if threshold == 1.0:
            # Use plmle with the same mode and fixed bounds
            res = plmle(x_sample, mode, xmin=xmin, xmax=xmax,
                        exponent_range=(exponent/2, exponent*2), precision=1e-3)
            sample_exponent[i] = res['exponent']

        # Check if this sample KS exceeds the empirical KS
        if ks_sample >= empirical_ks:
            success[i] = True
            n_successes += 1

        # Early stopping conditions
        if n_successes >= critical_threshold:
            early_stop = True

        if threshold != 1.0:
            # Probability of achieving at least (critical_threshold - n_successes)
            # successes in the remaining trials, assuming true p = threshold.
            remaining = samples - (i + 1)
            need = critical_threshold - n_successes
            if need > 0 and remaining > 0:
                # P(X >= need) = 1 - P(X <= need-1)
                prob = 1.0 - scipy.stats.binom.cdf(need - 1, remaining, threshold)
                if prob <= binom_cutoff:
                    early_stop = True

        i += 1

    # ----------------------------------------------------------------------
    # Assemble results
    # ----------------------------------------------------------------------
    p_value = np.sum(success) / samples   # total successes divided by requested samples

    if threshold == 1.0 and i == samples:
        sigma_exponent = np.std(sample_exponent, ddof=1)
    else:
        sigma_exponent = np.nan

    return {
        'p': p_value,
        'empirical_ks': empirical_ks,
        'sample_ks': sample_ks,
        'sigma_exponent': sigma_exponent,
        'mode': mode
    }


def plparams(x, samples=500, p_value_threshold=0.1, likelihood=1e-3):
    """
    Automated estimation of power-law parameters with support search.

    Performs a greedy search over possible (xmin, xmax) pairs, ranks them by
    the normalized log range log(xmax/xmin), and selects the first pair for
    which the p-value of the fit is at least the critical threshold. After
    the search, a full Monte Carlo p-value and the uncertainty of the exponent
    are computed.

    Parameters
    ----------
    x : array_like
        Data values. Must be positive.
    samples : int, optional
        Number of Monte Carlo samples used for p-value computation.
        Default 500.
    p_value_threshold : float, optional
        Critical p-value for early stopping during the support search.
        If the p-value for a candidate support pair reaches or exceeds this
        value, the search stops. Default 0.1.
    likelihood : float, optional
        Likelihood threshold for the binomial early-stopping rule used inside
        `pvcalc` during the search. Default 1e-3.

    Returns
    -------
    dict
        A dictionary with the following keys:
        - 'exponent' : float, estimated exponent.
        - 'xmin' : float, lower bound of the selected support.
        - 'xmax' : float, upper bound of the selected support.
        - 'sigma_exponent' : float, standard deviation of exponent estimates from the
          full Monte Carlo samples (NaN if the final p-value was not computed
          with threshold=1).
        - 'p' : float, final p-value (full computation).
        - 'p_crit' : float, the critical threshold used during the search.
        - 'ks' : float, Kolmogorov-Smirnov statistic between empirical data
          and the fitted model.
        - 'mode' : str, data type inferred ('continuous' or 'discrete').

    Raises
    ------
    ValueError
        If inputs are invalid (e.g., no data points, all x identical, etc.).
    """
    # ----------------------------------------------------------------------
    # Input validation and preparation
    # ----------------------------------------------------------------------
    x = np.asarray(x).ravel()
    if x.size == 0:
        raise ValueError("Input data x must not be empty.")
    if np.any(x <= 0):
        raise ValueError("All data values must be positive (power law requires x > 0).")

    # Infer data type (continuous / discrete) from the original data
    # (same logic as in pvcalc)
    if np.allclose(x, np.round(x)):
        mode = 'discrete'
        # For discrete data we will work with rounded values later,
        # but plmle and pvcalc handle that internally.
    else:
        mode = 'continuous'

    # ----------------------------------------------------------------------
    # Build candidate support pairs (xmin, xmax) from unique values
    # ----------------------------------------------------------------------
    unq = np.unique(x)                     # sorted unique values
    if unq.size < 2:
        raise ValueError("Need at least two distinct data values to form a support interval.")

    # All combinations of two distinct unique values
    support = np.array(list(itertools.combinations(unq, 2)))   # shape (n_pairs, 2)

    # ----------------------------------------------------------------------
    # Remove the first occurrence of each unique value as xmin
    # (this mimics the MATLAB loop that deletes one row per unique value)
    # ----------------------------------------------------------------------
    keep = np.ones(support.shape[0], dtype=bool)
    for u in unq:
        idx = np.where(support[:, 0] == u)[0]
        if len(idx) > 0:
            # Remove the first such row (lowest index)
            keep[idx[0]] = False
    support = support[keep]
    if support.shape[0] == 0:
        # Fallback: if removal wiped everything, use all pairs (should rarely happen)
        support = np.array(list(itertools.combinations(unq, 2)))

    # ----------------------------------------------------------------------
    # Rank support pairs by normalized log range (xmax/xmin)
    # ----------------------------------------------------------------------
    r_inv = support[:, 1] / support[:, 0]               # xmax / xmin (≥ 1)
    log_r_inv = np.log(r_inv)
    max_log = np.max(log_r_inv)
    if max_log == 0:          # all ratios are 1 – impossible here, but safe
        norm_log = np.zeros_like(log_r_inv)
    else:
        norm_log = log_r_inv / max_log
    rank = norm_log ** 2

    # Sort by descending rank
    order = np.argsort(rank)[::-1]
    support = support[order]

    # ----------------------------------------------------------------------
    # Greedy search: try support pairs in ranked order until p ≥ p_value_threshold
    # ----------------------------------------------------------------------
    best_exponent = None
    best_xmin = None
    best_xmax = None
    p_value_search = None

    for (xmin_cand, xmax_cand) in support:
        # Estimate exponent for this support
        res_mle = plmle(x, mode, xmin=xmin_cand, xmax=xmax_cand,
                        exponent_range=(1.0, 5.0), precision=1e-3)
        exponent_cand = res_mle['exponent']

        # Compute p‑value with early stopping (threshold = p_value_threshold)
        res_pv = pvcalc(x, exponent_cand, xmin=xmin_cand, xmax=xmax_cand,
                        samples=samples, threshold=p_value_threshold,
                        likelihood=likelihood)
        p_cand = res_pv['p']

        # Check stopping criterion
        if p_cand >= p_value_threshold:
            best_exponent = exponent_cand
            best_xmin = xmin_cand
            best_xmax = xmax_cand
            p_value_search = p_cand
            break

    # If no pair satisfied p ≥ threshold, use the last evaluated pair
    if best_exponent is None:
        # Reached the end of the loop without a break – use the final candidate
        # (the last pair in the loop is already stored in exponent_cand etc.)
        best_exponent = exponent_cand
        best_xmin = xmin_cand
        best_xmax = xmax_cand
        p_value_search = p_cand

    # ----------------------------------------------------------------------
    # Final full computation with threshold = 1 to obtain sigma_exponent and ks
    # ----------------------------------------------------------------------
    res_final = pvcalc(x, best_exponent, xmin=best_xmin, xmax=best_xmax,
                       samples=samples, threshold=1.0, likelihood=likelihood)

    # ----------------------------------------------------------------------
    # Assemble output dictionary
    # ----------------------------------------------------------------------
    return {
        'exponent': best_exponent,
        'xmin': best_xmin,
        'xmax': best_xmax,
        'sigma_exponent': res_final['sigma_exponent'],
        'p': res_final['p'],
        'p_crit': p_value_threshold,
        'ks': res_final['empirical_ks'],
        'mode': mode
    }


def pldist(n_samples, exponent, exp_decay_rate, min_value=1, max_value=None, truncate_min=None, truncate_max=None):
    """
    Generate discrete power-law distributed data with optional exponential tails.

    The data are integer values drawn from a probability mass function that
    is a power law over a central interval and exponential outside that
    interval (if truncation points are given). The exponential tails follow
    the form exp(-exp_decay_rate * x). The distribution is discrete.

    Parameters
    ----------
    n_samples : int
        Approximate number of data points to generate. The actual number may
        differ slightly due to rounding of the probability mass.
    exponent : float
        Power-law exponent. Must be positive; typical
        values are > 1.
    min_value : int, optional
        Absolute minimum value of the data. Must be >= 1. Default 1.
    max_value : int, optional
        Absolute maximum value of the data. If None, it is automatically set
        to 10^(ceil(log10(n_samples)) + 1) to cover a reasonable range.
    exp_decay_rate : float, optional
        Exponential decay rate (lambda) for the non-power-law tails. Must be positive.
        The exponential factor is exp(-lambda x). Larger values cause faster decay.
    truncate_min : int, optional
        Lower truncation point. Values below this are exponentially distributed.
        Must be > min_value if provided.
    truncate_max : int, optional
        Upper truncation point. Values above this are exponentially distributed.
        Must be < max_value if provided.

    Returns
    -------
    x : numpy.ndarray
        1D array of integers representing the generated data. The data lie in
        the interval [min_value, max_value].

    Raises
    ------
    ValueError
        If any input violates the required constraints:
        - n_samples <= 0
        - exponent <= 0
        - min_value < 1
        - min_value >= max_value
        - exp_decay_rate <= 0
        - truncate_min <= min_value (if truncate_min provided)
        - truncate_max >= max_value (if truncate_max provided)
        - truncate_min >= truncate_max (if both provided)
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if exponent <= 0:
        raise ValueError("exponent must be positive (typically > 1).")
    if min_value < 1:
        raise ValueError("min_value must be >= 1 (power law requires x > 0).")
    if exp_decay_rate <= 0:
        raise ValueError("exp_decay_rate must be positive.")

    if max_value is None:
        max_value = int(10 ** (np.ceil(np.log10(n_samples)) + 1))
    else:
        max_value = int(max_value)

    if min_value >= max_value:
        raise ValueError("min_value must be strictly less than max_value.")

    if truncate_min is not None and truncate_max is not None:
        if truncate_min <= min_value:
            raise ValueError("truncate_min must be > min_value.")
        if truncate_max >= max_value:
            raise ValueError("truncate_max must be < max_value.")
        if truncate_min >= truncate_max:
            raise ValueError("truncate_min must be < truncate_max.")
    elif truncate_min is not None:
        if truncate_min <= min_value:
            raise ValueError("truncate_min must be > min_value.")
    elif truncate_max is not None:
        if truncate_max >= max_value:
            raise ValueError("truncate_max must be < max_value.")

    # Build probability mass function (PDF for discrete values)
    if truncate_min is None and truncate_max is None:
        values = np.arange(min_value, max_value + 1)
        pmf = np.round(n_samples * values ** (-exponent))

    elif truncate_min is not None and truncate_max is None:
        v_left = np.arange(min_value, truncate_min)
        v_right = np.arange(truncate_min, max_value + 1)
        # Coefficients chosen to make the distribution continuous at truncate_min
        c_left = n_samples * (truncate_min ** (-exponent) /
                              np.exp(-exp_decay_rate * truncate_min))
        c_right = n_samples
        pmf_left = np.round(c_left * np.exp(-exp_decay_rate * v_left))
        pmf_right = np.round(c_right * v_right ** (-exponent))
        values = np.concatenate([v_left, v_right])
        pmf = np.concatenate([pmf_left, pmf_right])

    elif truncate_min is None and truncate_max is not None:
        v_left = np.arange(min_value, truncate_max + 1)
        v_right = np.arange(truncate_max + 1, max_value + 1)
        c_left = n_samples
        c_right = n_samples * (truncate_max ** (-exponent) /
                               np.exp(-exp_decay_rate * truncate_max))
        pmf_left = np.round(c_left * v_left ** (-exponent))
        pmf_right = np.round(c_right * np.exp(-exp_decay_rate * v_right))
        values = np.concatenate([v_left, v_right])
        pmf = np.concatenate([pmf_left, pmf_right])

    else:  # both truncation points provided
        v_left = np.arange(min_value, truncate_min)
        v_mid = np.arange(truncate_min, truncate_max + 1)
        v_right = np.arange(truncate_max + 1, max_value + 1)

        c_left = n_samples * (truncate_min ** (-exponent) /
                              np.exp(-exp_decay_rate * truncate_min))
        c_mid = n_samples
        c_right = n_samples * (truncate_max ** (-exponent) /
                               np.exp(-exp_decay_rate * truncate_max))

        pmf_left = np.round(c_left * np.exp(-exp_decay_rate * v_left))
        pmf_mid = np.round(c_mid * v_mid ** (-exponent))
        pmf_right = np.round(c_right * np.exp(-exp_decay_rate * v_right))

        values = np.concatenate([v_left, v_mid, v_right])
        pmf = np.concatenate([pmf_left, pmf_mid, pmf_right])

    pmf = np.maximum(pmf, 0).astype(int)
    x = np.repeat(values, pmf)
    return x
# endregion