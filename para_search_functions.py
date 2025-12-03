import numpy as np
import scipy


def find_input_for_target_with_max_iter(func, target_value, bounds=None, x0=None, max_iter=100):
    """
    找到使 func(x) = target_value 的输入 x
    
    参数:
    - func: 目标函数
    - target_value: 目标输出值
    - bounds: 变量的边界 [(min, max), ...]
    - x0: 初始猜测（可选）
    - max_iter: 最大迭代次数
    """
    def objective(x):
        return (func(x) - target_value)**2
    
    # 如果有初始猜测,使用局部优化
    if x0 is not None:
        result = scipy.optimize.minimize(objective, x0, method='Nelder-Mead', 
                         options={'maxiter': max_iter})
        return result.x
    
    # 如果没有初始猜测但有限制范围,使用全局优化
    elif bounds is not None:
        result = scipy.optimize.differential_evolution(objective, bounds, 
                                       maxiter=max_iter, popsize=10)
        return result.x
    
    else:
        raise ValueError("需要提供初始猜测x0或变量边界bounds")