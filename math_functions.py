import numpy as np
import scipy.optimize


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
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    if len(x) != len(y):
        raise ValueError("x和y的长度必须相同")
    
    if len(x) < 2:
        raise ValueError("至少需要2个点才能进行插值")
    
    spline = scipy.interpolate.CubicSpline(x, y, extrapolate=True)
    
    return spline


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