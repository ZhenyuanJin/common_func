import numpy as np
import scipy


def find_input_for_target_with_max_iter(func, target_value, bounds=None, x0=None, max_iter=100, method='auto', weights=None):
    """
    找到使 func(x) = target_value 的输入 x(支持多维输入输出)
    
    参数:
    - func: 目标函数,输入x,输出可以是多维
    - target_value: 目标输出值,可以是标量或多维数组
    - bounds: 变量的边界 [(min, max), ...]
    - x0: 初始猜测
    - max_iter: 最大迭代次数
    - method: 优化方法,'auto'自动选择,'local'局部优化,'global'全局优化
    - weights: 用于多维输出的加权数组,与target_value形状相同
    """
    
    # 确保是numpy数组以便进行向量运算
    target_array = np.array(target_value)
    weights_array = np.array(weights) if weights is not None else None
    
    # 定义优化目标
    def objective(x):
        output = func(x)
        diff = np.array(output) - target_array
        if weights is not None:
            diff = diff * weights_array
        return np.sum(diff**2)  # 计算平方和
    
    # 自动选择方法
    if method == 'auto':
        if x0 is not None:
            method = 'local'
        elif bounds is not None:
            method = 'global'
        else:
            raise ValueError("需要提供初始猜测x0或变量边界bounds")
    
    # 局部优化
    if method == 'local' and x0 is not None:
        x0_array = np.array(x0)
        
        # 如果有边界,使用约束优化
        if bounds is not None:
            constraints = []
            result = scipy.optimize.minimize(objective, x0_array, method='L-BFGS-B', 
                            bounds=bounds, options={'maxiter': max_iter})
        else:
            result = scipy.optimize.minimize(objective, x0_array, method='Nelder-Mead',
                            options={'maxiter': max_iter})
        
        return result.x, result.fun
    
    # 全局优化
    elif method == 'global' and bounds is not None:
        result = scipy.optimize.differential_evolution(objective, bounds, 
                                      maxiter=max_iter, popsize=10)
        return result.x, result.fun
    
    else:
        raise ValueError("参数不匹配,请检查x0和bounds的提供情况")