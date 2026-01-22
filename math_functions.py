import numpy as np
import scipy.optimize
import os
import sys
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


# region poly
def get_poly_root(coef):
    '''
    coef: 多项式系数,升序 (index 0 is s^0)
    '''
    coef_rev = coef[::-1]
    roots = np.roots(coef_rev)
    return roots


def poly_mul(p1_coef, p2_coef):
    '''
    多项式乘法辅助函数.
    输入 p1_coef, p2_coef 为升序系数 (index 0 is s^0).
    np.convolve 在这种定义下直接对应多项式乘法,结果也是升序.
    '''
    return np.convolve(p1_coef, p2_coef)


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


def poly_val(coef, s):
    '''
    多项式求值辅助函数.
    输入 coef 为升序系数 (index 0 is s^0).
    np.polyval 需要降序系数,所以先反转.
    '''
    coef_rev = coef[::-1]
    return np.polyval(coef_rev, s)


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


def visualize_rational_function_on_real_axis(ax, numerator_coef, denominator_coef):
    zeros = get_poly_root(numerator_coef)
    poles = get_poly_root(denominator_coef)
    def f(x):
        return poly_val(numerator_coef, x) / poly_val(denominator_coef, x)

    zeros_real_min = np.min(zeros.real)
    zeros_real_max = np.max(zeros.real)

    poles_real_min = np.min(poles.real)
    poles_real_max = np.max(poles.real)

    real_min = min(zeros_real_min, poles_real_min)
    real_max = max(zeros_real_max, poles_real_max)
    
    x_range = cf.scale_range(real_min, real_max, prop=2)
    
    x_grid = get_refined_grid(x_range[0], x_range[1], coarse_num=1000, keypoints=np.concatenate([zeros.real, poles.real]), near_num=50, refine_factor=10)
    
    f_values = f(x_grid)
    magnitude = np.abs(f_values)
    
    cf.plt_line(ax, x_grid, magnitude)
    
    for zero in zeros:
        cf.add_vline(ax, zero.real, color='green', linestyle='--', label=f'Zero, x={zero.real:.3f}')
    for pole in poles:
        cf.add_vline(ax, pole.real, color='red', linestyle='--', label=f'Pole, x={pole.real:.3f}')
    
    cf.set_ax(ax, xlabel='Real', ylabel='|f|', ylog=True)
    cf.set_symlog_scale(ax, axis='x', linthresh=1e-2)
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