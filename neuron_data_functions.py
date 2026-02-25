# region 标准库导入
import os
import sys
import json
import pickle
import random
import shutil
import time
import warnings
from math import ceil
from multiprocessing import Process
from pathlib import Path
from typing import Union, Sequence, Callable, Optional
import abc
from collections import defaultdict
from tqdm import tqdm
from functools import partial


# 数学和科学计算库
import numpy as np
import scipy
import scipy.stats as st
from scipy.stats import gaussian_kde, zscore
from scipy.fft import fft, fftfreq
from scipy.integrate import quad
from scipy.sparse import coo_matrix, csr_matrix
import scipy.sparse as sps
from sklearn.decomposition import PCA
# import jax
# import jax.numpy as jnp


# 数据处理和可视化库
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, ScalarFormatter
from matplotlib.colors import BoundaryNorm, Normalize
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable


# 神经网络和脑模型库
# import brainpy as bp
# import brainpy.math as bm
# from brainpy._src.dyn.base import SynDyn
# from brainpy._src.mixin import AlignPost, ReturnInfo
# from brainpy._src import connect, initialize as init
# from brainpy._src.integrators.ode.generic import odeint
# from brainpy._src.integrators.joint_eq import JointEq
# from brainpy._src.initialize import parameter
# from brainpy._src.context import share
# from brainpy.types import ArrayType
# from brainpy._src.dynsys import DynamicalSystem, DynView
# from brainpy._src.math.object_transform.base import StateLoadResult
# 脑区处理库
# import networkx as nx
# import pointpats
# import shapely.geometry


# 自定义库
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import common_functions as cf
import math_functions
# endregion


# region 定义默认参数
E_COLOR = cf.RED
I_COLOR = cf.BLUE
TIME_AXIS = 0
NEURON_AXIS = 1
# endregion


# region 利用idx提取数据
def neuron_idx_data(data, indices=None, keep_size=False):
    '''
    从spike或者V中提取指定索引的神经元数据。indices可以是slice对象或单个值。
    data: 二维矩阵，其中行表示时间点，列表示神经元。(与brainpy的输出一致)
    indices: 要提取的神经元索引列表或单个值。
    keep_size: 是否保持返回数据的二维形状。
    '''
    if indices is None:
        return data
    if keep_size and isinstance(indices, int):  # 单个索引时保持二维形状
        return data[:, [indices]]
    else:
        return data[:, indices]


def get_neuron_num_from_data(data):
    return data.shape[1]


def time_idx_data(data, indices=None, keep_size=False):
    '''
    从spike或者V中提取指定时间点的数据。indices可以是slice对象或单个值。
    data: 二维矩阵，其中行表示时间点，列表示神经元。(与brainpy的输出一致)
    indices: 要提取的时间点索引列表或单个值。
    keep_size: 是否保持返回数据的二维形状。
    '''
    if indices is None:
        return data
    if keep_size and isinstance(indices, int):  # 单个索引时保持二维形状
        return data[[indices], :]
    else:
        return data[indices, :]


def get_time_point_from_data(data):
    return data.shape[0]
# endregion


# region 神经元数据预处理
def spike_times_to_array(spike_times, start_time, end_time, dt, mode='binary'):
    '''
    Convert spike times to 2D spike train array.
    
    Args:
        spike_times: list of lists, spike times for each neuron
        start_time: start time of the recording window
        end_time: end time of the recording window
        dt: time bin width
        mode: 'binary' (0/1) or 'num' (count multiple spikes)
    
    Returns:
        np.ndarray: shape (n_bins, n_neurons) spike train array
    '''
    n_bins = int(np.floor((end_time - start_time) / dt))
    n_neurons = len(spike_times)
    spike_array = np.zeros((n_bins, n_neurons), dtype=int)
    
    for neuron_idx, times in enumerate(spike_times):
        if not times:
            continue
        
        indices = ((np.array(times) - start_time) / dt).astype(int)
        valid_indices = indices[(indices >= 0) & (indices < n_bins)]
        
        if mode == 'binary':
            spike_array[valid_indices, neuron_idx] = 1
        elif mode == 'num':
            unique_indices, counts = np.unique(valid_indices, return_counts=True)
            spike_array[unique_indices, neuron_idx] = counts
        else:
            raise ValueError(f"Unsupported mode: {mode}. Must be 'binary' or 'num'")
    
    return spike_array


def spike_array_to_times(spike_array, start_time, dt, position='left'):
    '''
    Convert spike train array to spike times.
    
    Args:
        spike_array: np.ndarray, shape (n_bins, n_neurons) spike train array
        start_time: start time of the recording window
        dt: time bin width
        position: 'left', 'center', or 'right' - where within the bin to place the spike time
    
    Returns:
        list of lists: spike times for each neuron
    '''
    n_bins, n_neurons = spike_array.shape
    spike_times = [[] for _ in range(n_neurons)]
    
    if np.any(spike_array > 1):
        raise ValueError("Multiple spikes found in a bin. This function only supports binary spike arrays.")
    
    # 计算偏移量
    if position == 'left':
        offset = 0.0
    elif position == 'center':
        offset = dt / 2.0
    elif position == 'right':
        offset = dt
    else:
        raise ValueError("position must be 'left', 'center', or 'right'")
    
    for neuron_idx in range(n_neurons):
        neuron_spikes = spike_array[:, neuron_idx]
        spike_indices = np.where(neuron_spikes == 1)[0]
        
        # 向量化计算所有时间点
        times = start_time + (spike_indices * dt) + offset
        spike_times[neuron_idx].extend(times.tolist())
    
    return spike_times


def get_start_end_time_and_dt(spike_times, interval_ratio=0.9):
    '''
    Args:
        spike_times: list of lists, spike times for each neuron
        interval_ratio: dt is set to the minimum inter-spike interval multiplied by this ratio
    
    Returns:
        tuple: (start_time, end_time, dt)
    '''
    all_intervals = []
    all_spikes = []
    
    for neuron_spikes in spike_times:
        if neuron_spikes:
            all_spikes.extend(neuron_spikes)
            if len(neuron_spikes) > 1:
                neuron_spikes_sorted = sorted(neuron_spikes)
                intervals = np.diff(neuron_spikes_sorted)
                all_intervals.extend(intervals)
    
    if not all_intervals:
        raise ValueError("No inter-spike intervals found. Cannot determine dt.")
    else:
        all_intervals = np.array(all_intervals)
        min_interval = np.min(all_intervals)
        dt = min_interval * interval_ratio
    
    if not all_spikes:
        raise ValueError("No spike times found. Cannot determine start and end time.")
    else:
        all_spikes_array = np.array(all_spikes)
        start_time = np.min(all_spikes_array)
        end_time = np.max(all_spikes_array)
    
    return start_time, end_time, dt


def sort_neuron_data(neuron_data, sort_measure, ascending=True):
    '''
    对神经元数据进行排序
    '''
    if ascending:
        return neuron_data[:, np.argsort(sort_measure)]
    else:
        return neuron_data[:, np.argsort(sort_measure)[::-1]]


def get_spike_sort_measure_by_corr_and_fr(spike, fr, fr_threshold=0.1, fallback_value=-10.0):
    '''
    计算用于排序神经元的 sort_measure,根据
    - 单个神经元的 spike 与 firing rate 的相关性
    - 神经元平均 firing rate 是否大于阈值
    
    参数：
    - spike: ndarray, shape (T, N),T为时间点,N为神经元数
    - fr: ndarray, shape (T,),对应时间段的整体 firing rate
    - fr_threshold: float,低于此阈值的神经元会被惩罚
    - fallback_value: float,对于无法计算相关性(nan)的神经元赋值为该值

    返回：
    - sort_measure: ndarray, shape (N,),每个神经元的排序度量值
    '''
    N = spike.shape[1]

    # 计算每个神经元与 firing rate 的相关性
    spike_fr_corr = np.array([
        np.corrcoef(spike[:, i], fr)[0, 1]
        for i in range(N)
    ])

    # 计算平均 firing rate 是否超过阈值
    fr_mask = (np.mean(spike, axis=0) > fr_threshold).astype(float)

    # 构造排序度量
    sort_measure = (spike_fr_corr + 1) * fr_mask

    # 将 NaN 替换为 fallback 值
    sort_measure[np.isnan(sort_measure)] = fallback_value

    return sort_measure
# endregion


# region 神经元放电性质计算
def _bp_measure_firing_rate(spikes, width, dt):
    '''
    adapted from bp.measure.firing_rate
    '''
    width1 = int(width / 2 / dt) * 2 + 1
    window = np.ones(width1) * 1000 / width
    return np.convolve(np.mean(spikes, axis=1), window, mode='same')


def spike_to_fr(spike, width, dt, neuron_idx=None, **kwargs):
    '''
    修改bp.measure.firing_rate使得一维数组的spike也能够计算firing rate(但是使用方式是设定neuron_idx而不是直接传入一个一维的spike)

    注意:
    如果需要对一维的spike算,先np.reshape
    adjusted_spike = np.reshape(spike, (spike.shape[0], -1))
    '''
    partial_spike = neuron_idx_data(spike, neuron_idx, keep_size=True)
    return _bp_measure_firing_rate(partial_spike, width, dt, **kwargs)


def get_neuron_data_acf(neuron_data, dt, nlags, neuron_idx=None, process_num=1, **kwargs):
    '''
    计算单个神经元级别的自相关函数

    由于brainpy的数据格式是(T, N),所以需要转置再输入到acf函数中
    对于multi_acf,其shape是(N, nlags)
    '''
    partial_neuron_data = neuron_idx_data(neuron_data, neuron_idx, keep_size=True)
    lag_times, multi_acf = cf.get_multi_acf(partial_neuron_data.T, T=dt, nlags=nlags, process_num=process_num, **kwargs)
    return lag_times, multi_acf


def get_neuron_data_acovf(neuron_data, dt, nlags, neuron_idx=None, process_num=1, **kwargs):
    '''
    计算单个神经元级别的自协方差函数
    '''
    partial_neuron_data = neuron_idx_data(neuron_data, neuron_idx, keep_size=True)
    lag_times, multi_acovf = cf.get_multi_acovf(partial_neuron_data.T, T=dt, nlags=nlags, process_num=process_num, **kwargs)
    return lag_times, multi_acovf


def get_neuron_data_ccf(neuron_data_x, neuron_data_y, dt, nlags, neuron_idx_x=None, neuron_idx_y=None, process_num=1, **kwargs):
    '''
    计算神经元数据的互相关函数
    '''
    partial_neuron_data_x = neuron_idx_data(neuron_data_x, neuron_idx_x, keep_size=True)
    partial_neuron_data_y = neuron_idx_data(neuron_data_y, neuron_idx_y, keep_size=True)
    lag_times, multi_ccf = cf.get_multi_ccf(partial_neuron_data_x.T, partial_neuron_data_y.T, T=dt, nlags=nlags, process_num=process_num, **kwargs)
    return lag_times, multi_ccf


def get_neuron_data_ccovf(neuron_data_x, neuron_data_y, dt, nlags, neuron_idx_x=None, neuron_idx_y=None, process_num=1, **kwargs):
    '''
    计算神经元数据的互协方差函数
    '''
    partial_neuron_data_x = neuron_idx_data(neuron_data_x, neuron_idx_x, keep_size=True)
    partial_neuron_data_y = neuron_idx_data(neuron_data_y, neuron_idx_y, keep_size=True)
    lag_times, multi_ccovf = cf.get_multi_ccovf(partial_neuron_data_x.T, partial_neuron_data_y.T, T=dt, nlags=nlags, process_num=process_num, **kwargs)
    return lag_times, multi_ccovf


def get_neuron_data_ccf_auto_and_cross(neuron_data, dt, nlags, neuron_idx=None, process_num=1, return_complete=False, **kwargs):
    '''
    计算神经元数据的自相关和互相关函数
    '''
    partial_neuron_data = neuron_idx_data(neuron_data, neuron_idx, keep_size=True)
    return cf.get_multi_ccf_auto_and_cross(partial_neuron_data.T, T=dt, nlags=nlags, process_num=process_num, return_complete=return_complete, **kwargs)


def get_neuron_data_ccovf_auto_and_cross(neuron_data, dt, nlags, neuron_idx=None, process_num=1, return_complete=False, **kwargs):
    '''
    计算神经元数据的自协方差和互协方差函数
    '''
    partial_neuron_data = neuron_idx_data(neuron_data, neuron_idx, keep_size=True)
    return cf.get_multi_ccovf_auto_and_cross(partial_neuron_data.T, T=dt, nlags=nlags, process_num=process_num, return_complete=return_complete, **kwargs)


def spike_to_fr_acf(spike, width, dt, nlags, neuron_idx=None, spike_to_fr_kwargs=None, **kwargs):
    '''
    计算spike的firing rate的自相关函数,注意,计算fr的过程自动平均了neuron_idx中的神经元。
    '''
    if spike_to_fr_kwargs is None:
        spike_to_fr_kwargs = {}
    fr = spike_to_fr(spike, width, dt, neuron_idx, **spike_to_fr_kwargs)
    return cf.get_acf(fr, T=dt, nlags=nlags, **kwargs)


def _get_fr_each_neuron(spike, width, dt):
    '''
    对每个神经元的spike进行firing rate计算
    '''
    width1 = int(width / 2 / dt) * 2 + 1
    window = np.ones(width1) * 1000 / width
    return np.apply_along_axis(lambda m: np.convolve(m, window, mode='same'), axis=0, arr=spike)


def get_fr_each_neuron(spike, width, dt, process_num=1):
    spike_list = cf.split_array(spike, axis=1, n=process_num)
    r = cf.multi_process_list_for(process_num=process_num, func=_get_fr_each_neuron, kwargs={'width': width, 'dt': dt}, for_list=spike_list, for_idx_name='spike')
    r = np.concatenate(r, axis=1)
    return r


def get_ISI(spike, dt, neuron_idx=None, **kwargs):
    '''
    计算spike的ISI
    '''
    partial_spike = neuron_idx_data(spike, neuron_idx, keep_size=True)
    ISI = []
    for i in range(partial_spike.shape[1]):
        spike_times = np.where(partial_spike[:, i])[0] * dt
        if len(spike_times) < 2:
            continue
        ISI.append(list(np.diff(spike_times)))
    if len(ISI) == 0:
        return []
    else:
        ISI = cf.flatten_list(ISI)
    return ISI


def get_ISI_mean(spike, dt, neuron_idx=None, **kwargs):
    '''
    计算spike的ISI均值
    '''
    partial_spike = neuron_idx_data(spike, neuron_idx, keep_size=True)
    ISI_mean = []
    for i in range(partial_spike.shape[1]):
        spike_times = np.where(partial_spike[:, i])[0] * dt
        if len(spike_times) < 2:
            continue
        ISI_mean.append(np.mean(np.diff(spike_times)))
    return ISI_mean


def get_ISI_CV(spike, dt, neuron_idx=None, **kwargs):
    '''
    计算spike的ISI CV
    '''
    partial_spike = neuron_idx_data(spike, neuron_idx, keep_size=True)
    CV = []
    for i in range(partial_spike.shape[1]):
        spike_times = np.where(partial_spike[:, i])[0] * dt
        if len(spike_times) < 2:
            continue
        CV.append(cf.get_CV(np.diff(spike_times)))
    return CV


def get_spike_FF(spike, dt, timebin_list, neuron_idx=None, **kwargs):
    '''
    计算spike的FF(对每个神经元的整个spike序列计算FF,spike被方波卷积)
    '''
    partial_spike = neuron_idx_data(spike, neuron_idx, keep_size=True)

    FF = []
    for timebin in timebin_list:
        FF_partial = []
        kernel = np.ones(timebin) / timebin
        bin_spike = cf.convolve_multi_timeseries(partial_spike.T, kernel).T
        bin_spike = bin_spike / (timebin * dt)
        for i in range(bin_spike.shape[1]):
            FF_partial.append(cf.get_FF(bin_spike[:, i]))
        FF.append(FF_partial)
    return FF


def get_avalanche(timeseries, dt, bin_size, neuron_idx=None, threshold=0, timeseries_mode='discrete', **kwargs):
    partial_timeseries = neuron_idx_data(timeseries, neuron_idx, keep_size=True)
    if timeseries_mode == 'discrete':
        timeseries_sum = np.sum(partial_timeseries, axis=1)
        bin_timeseries = cf.bin_timeseries(timeseries_sum, bin_size, mode='sum')

        is_active = bin_timeseries > threshold
        is_active_padded = np.pad(is_active, (1, 1), mode='constant', constant_values=False)
        diff_active = np.diff(is_active_padded.astype(int))

        non_zero_starts = np.where(diff_active == 1)[0]
        non_zero_ends = np.where(diff_active == -1)[0] - 1

        avalanche_size = []
        avalanche_duration = []
        for start, end in zip(non_zero_starts, non_zero_ends):
            avalanche_size.append(np.sum(bin_timeseries[start:end+1]))
            avalanche_duration.append((end - start + 1) * bin_size * dt)

        results = {}
        results['avalanche_size'] = np.array(avalanche_size)
        results['avalanche_duration'] = np.array(avalanche_duration)
        return results
    elif timeseries_mode == 'continuous':
        partial_timeseries = partial_timeseries[partial_timeseries > 0]
        raise NotImplementedError('continuous mode is not implemented yet.')


def get_average_size_per_duration(sizes, durations, n_bins):
    '''
    calculate average avalanche size for each duration bin using logarithmic binning.
    '''
    if isinstance(durations, list):
        durations = np.array(durations)
    if isinstance(sizes, list):
        sizes = np.array(sizes)
    if len(durations) == 0:
        return np.array([]), np.array([])
    
    min_val = np.min(durations)
    max_val = np.max(durations)
    bins = np.logspace(np.log10(min_val), np.log10(max_val), n_bins)
    
    bin_centers = np.sqrt(bins[:-1] * bins[1:])
    digitized = np.digitize(durations, bins)
    
    avg_sizes = []
    valid_centers = []
    
    for i in range(1, len(bins)):
        mask = digitized == i
        if np.any(mask):
            avg_sizes.append(np.mean(sizes[mask]))
            valid_centers.append(bin_centers[i-1])
            
    return np.array(valid_centers), np.array(avg_sizes)


def fit_scaling_law_weighted(sizes, durations, min_duration, max_duration):
    """
    Fits the scaling relation S(T) ~ T^(1/sigma*nu*z) using a weighted 
    least squares method for avalanches with durations in the specified range.
    
    The weights are based on the number of samples for each duration.

    Parameters
    ----------
    sizes : array_like
        Array of avalanche sizes.
    durations : array_like
        Array of avalanche durations (corresponding to sizes).
    min_duration : float
        Lower bound of the truncated duration range.
    max_duration : float
        Upper bound of the truncated duration range.

    Returns
    -------
    float
        The estimated scaling exponent (gamma).
    """
    sizes = np.asarray(sizes)
    durations = np.asarray(durations)

    mask = (durations >= min_duration) & (durations <= max_duration)
    sizes = sizes[mask]
    durations = durations[mask]

    unique_durations = np.unique(durations)

    mean_sizes = []
    counts = []

    for T in unique_durations:
        s_T = sizes[durations == T]
        mean_sizes.append(np.mean(s_T))
        counts.append(len(s_T))

    mean_sizes = np.asarray(mean_sizes)
    counts = np.asarray(counts)

    log_T = np.log(unique_durations)
    log_mean_S = np.log(mean_sizes)

    weights = counts

    coeffs = np.polyfit(log_T, log_mean_S, 1, w=weights)
    gamma = coeffs[0]
    gamma_C = np.exp(coeffs[1])
    return gamma, gamma_C


def check_criticality(tau, alpha, gamma):
    '''
    tau: avalanche size exponent (from P(S) ~ S^(-tau))
    alpha: avalanche duration exponent (from P(T) ~ T^(-alpha))
    gamma: scaling exponent relating size and duration (from <S> ~ T^gamma)
    '''
    if tau < 0:
        raise ValueError('tau should be positive.')
    if alpha < 0:
        raise ValueError('alpha should be positive.')
    if gamma < 0:
        raise ValueError('gamma should be positive.')
    predicted_gamma = (alpha - 1) / (tau - 1)
    difference = abs(predicted_gamma - gamma)
    ratio = gamma / predicted_gamma if predicted_gamma != 0 else np.nan
    return predicted_gamma, difference, ratio


class AvalancheToolbox:
    def __init__(self, spikes, dt, n_bins, neuron_idx=None, get_avalanche_kwargs=None, use_ISI_bin_size=True, doubly_truncate=True, step=1, fit_scaling_law_by_weight=False, truncate_min_prop=None, truncate_max_prop=None, use_injected_truncate=False):
        if get_avalanche_kwargs is None:
            local_get_avalanche_kwargs = {}
        else:
            local_get_avalanche_kwargs = get_avalanche_kwargs.copy()
        
        spikes_sum = np.sum(neuron_idx_data(spikes, neuron_idx, keep_size=True), axis=1, keepdims=True)
        ISI_mean = get_ISI_mean(spikes_sum, dt, neuron_idx=neuron_idx)
        if use_ISI_bin_size:
            bin_size = int(ISI_mean[0] / dt)
        else:
            bin_size = local_get_avalanche_kwargs.pop('bin_size')
        self.avalanche_results = get_avalanche(spikes, dt, bin_size=bin_size, neuron_idx=neuron_idx, **local_get_avalanche_kwargs)
    
        size_bin_centers, size_pdf = math_functions.get_log_bin_pdf(self.avalanche_results['avalanche_size'], n_bins=n_bins)
        if doubly_truncate:
            if use_injected_truncate:
                size_truncate_min, size_truncate_max = self.log_shrink(min_val=np.min(size_bin_centers), max_val=np.max(size_bin_centers), left_prop=truncate_min_prop, right_prop=truncate_max_prop)
                tau, tau_C = math_functions.fit_powerlaw_scatter(
                    size_bin_centers[(size_bin_centers >= size_truncate_min) & (size_bin_centers <= size_truncate_max)],
                    size_pdf[(size_bin_centers >= size_truncate_min) & (size_bin_centers <= size_truncate_max)]
                )
            else:
                size_truncate_result = math_functions.find_optimal_powerlaw_truncated_range(self.avalanche_results['avalanche_size'], n_sims=100, mode='discrete', step=step)
                size_truncate_min = size_truncate_result['xmin']
                size_truncate_max = size_truncate_result['xmax']
                tau = size_truncate_result['alpha']
                tau_C = size_truncate_result['C']
        else:
            size_truncate_min = np.min(self.avalanche_results['avalanche_size'])
            size_truncate_max = np.max(self.avalanche_results['avalanche_size'])
            tau, tau_C = math_functions.fit_powerlaw_scatter(size_bin_centers, size_pdf)
        
        duration_bin_centers, duration_pdf = math_functions.get_log_bin_pdf(self.avalanche_results['avalanche_duration'], n_bins=n_bins)
        if doubly_truncate:
            if use_injected_truncate:
                duration_truncate_min, duration_truncate_max = self.log_shrink(min_val=np.min(duration_bin_centers), max_val=np.max(duration_bin_centers), left_prop=truncate_min_prop, right_prop=truncate_max_prop)
                alpha, alpha_C = math_functions.fit_powerlaw_scatter(
                    duration_bin_centers[(duration_bin_centers >= duration_truncate_min) & (duration_bin_centers <= duration_truncate_max)],
                    duration_pdf[(duration_bin_centers >= duration_truncate_min) & (duration_bin_centers <= duration_truncate_max)]
                )
            else:
                duration_truncate_result = math_functions.find_optimal_powerlaw_truncated_range(self.avalanche_results['avalanche_duration'], n_sims=100, mode='discrete', step=step)
                duration_truncate_min = duration_truncate_result['xmin']
                duration_truncate_max = duration_truncate_result['xmax']
                alpha = duration_truncate_result['alpha']
                alpha_C = duration_truncate_result['C']
        else:
            duration_truncate_min = np.min(self.avalanche_results['avalanche_duration'])
            duration_truncate_max = np.max(self.avalanche_results['avalanche_duration'])
            alpha, alpha_C = math_functions.fit_powerlaw_scatter(duration_bin_centers, duration_pdf)
        
        # for not doubly_truncate case, the truncate min and max are set above as the data min and max, thus we do not need to do if else here
        if fit_scaling_law_by_weight:
            gamma, gamma_C = fit_scaling_law_weighted(
                self.avalanche_results['avalanche_size'],
                self.avalanche_results['avalanche_duration'],
                duration_truncate_min,
                duration_truncate_max
            )
        else:
            duration_centers_for_size_duration_relation, size_centers_for_size_duration_relation = get_average_size_per_duration(
                self.avalanche_results['avalanche_size'],
                self.avalanche_results['avalanche_duration'],
                n_bins
            )
            mask = (duration_centers_for_size_duration_relation >= duration_truncate_min) & (duration_centers_for_size_duration_relation <= duration_truncate_max)
            filtered_duration_centers = duration_centers_for_size_duration_relation[mask]
            filtered_size_centers = size_centers_for_size_duration_relation[mask]
            gamma, gamma_C = math_functions.fit_powerlaw_scatter(filtered_duration_centers, filtered_size_centers)
        gamma = -gamma
        
        predicted_gamma, difference, ratio = check_criticality(tau, alpha, gamma)

        self.collected_results = {
            'size_truncate_min': size_truncate_min,
            'size_truncate_max': size_truncate_max,
            'size_bin_centers': size_bin_centers,
            'size_pdf': size_pdf,
            'tau': tau,
            'tau_C': tau_C,
            'duration_truncate_min': duration_truncate_min,
            'duration_truncate_max': duration_truncate_max,
            'duration_bin_centers': duration_bin_centers,
            'duration_pdf': duration_pdf,
            'alpha': alpha,
            'alpha_C': alpha_C,
            'duration_centers_for_size_duration_relation': duration_centers_for_size_duration_relation,
            'size_centers_for_size_duration_relation': size_centers_for_size_duration_relation,
            'gamma': gamma,
            'gamma_C': gamma_C,
            'predicted_gamma': predicted_gamma,
            'difference': difference,
            'ratio': ratio
        }

    def log_shrink(self, min_val, max_val, left_prop, right_prop):
        lmin = np.log(min_val)
        lmax = np.log(max_val)
        span = lmax - lmin
        new_min = np.exp(lmin + left_prop * span)
        new_max = np.exp(lmax - right_prop * span)
        return new_min, new_max

    def visualize(self, fig=None, axes=None, unit='ms'):
        if fig is None or axes is None:
            fig, axes = cf.gfa(ncols=3)
        
        size_truncate_min = self.collected_results['size_truncate_min']
        size_truncate_max = self.collected_results['size_truncate_max']
        size_bin_centers = self.collected_results['size_bin_centers']
        size_pdf = self.collected_results['size_pdf']
        tau = self.collected_results['tau']
        tau_C = self.collected_results['tau_C']
        duration_truncate_min = self.collected_results['duration_truncate_min']
        duration_truncate_max = self.collected_results['duration_truncate_max']
        duration_bin_centers = self.collected_results['duration_bin_centers']
        duration_pdf = self.collected_results['duration_pdf']
        alpha = self.collected_results['alpha']
        alpha_C = self.collected_results['alpha_C']
        duration_centers_for_size_duration_relation = self.collected_results['duration_centers_for_size_duration_relation']
        size_centers_for_size_duration_relation = self.collected_results['size_centers_for_size_duration_relation']
        gamma = self.collected_results['gamma']
        gamma_C = self.collected_results['gamma_C']
        ratio = self.collected_results['ratio']

        ax = axes[0]
        ax.plot(size_bin_centers, size_pdf)
        math_functions.plot_powerlaw_pdf_line(ax, tau, size_truncate_min, size_truncate_max, C=tau_C)
        cf.set_ax(ax, xlog=True, ylog=True, xlabel='Avalanche Size')
        
        ax = axes[1]
        ax.plot(duration_bin_centers, duration_pdf)
        math_functions.plot_powerlaw_pdf_line(ax, alpha, duration_truncate_min, duration_truncate_max, C=alpha_C)
        cf.set_ax(ax, xlog=True, ylog=True, xlabel=f'Avalanche Duration ({unit})')
        
        ax = axes[2]
        ax.scatter(duration_centers_for_size_duration_relation, size_centers_for_size_duration_relation, s=50)
        if not np.isnan(gamma):
            y_fit = gamma_C * (duration_centers_for_size_duration_relation ** gamma)
            ax.plot(duration_centers_for_size_duration_relation, y_fit, 'k--', label=f'slope={gamma:.2f}')
        cf.set_ax(ax, xlog=True, ylog=True, xlabel='Duration', ylabel='Avg Size')

        cf.add_axes_title(axes, f"Scaling relation value = {ratio:.2f} (theory: 1)")
        return fig, axes


def single_exp(x, amp, tau):
    return amp * np.exp(-x / tau)


def single_exp_fit(lag_times, acf, fix_amp_value=None):
    with cf.FlexibleTry() as ft:
        if fix_amp_value is None:
            f = single_exp
        else:
            def f(x, tau):
                return single_exp(x, fix_amp_value, tau)
        single_popt, single_pcov, single_error = cf.get_curvefit(lag_times, acf, f)
        results = {}
        if fix_amp_value is not None:
            single_popt = [fix_amp_value, single_popt[0]]
        results['amp'] = single_popt[0]
        results['tau'] = single_popt[1]
        results['error'] = single_error
        results['cov'] = single_pcov
        results['fitted_curve'] = single_exp(lag_times, results['amp'], results['tau'])
    if not ft.success:
        results = {}
        results['amp'] = np.nan
        results['tau'] = np.nan
        results['error'] = np.nan
        results['cov'] = np.nan
        results['fitted_curve'] = np.nan * lag_times
    return results


def double_exp(x, tau1, tau2, amp1, amp2):
    return amp1 * np.exp(-x / tau1) + amp2 * np.exp(-x / tau2)


def double_exp_fit(lag_times, acf):
    '''
    tau1: shorter timescale
    tau2: longer timescale
    amp1: amplitude for shorter timescale
    amp2: amplitude for longer timescale
    '''
    with cf.FlexibleTry() as ft:
        f = double_exp
        
        single_results = single_exp_fit(lag_times, acf)
        p0 = [single_results['tau'], single_results['tau'] / 10, single_results['amp'], 0]
        if p0[0] <= 0:
            p0[0] = 1.0
        if p0[1] <= 0:
            p0[1] = 1.0 / 10
        bounds = ([0, 0, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf])
        
        double_popt, double_pcov, double_error = cf.get_curvefit(
            lag_times, acf, f, p0=p0, bounds=bounds, maxfev=5000
        )
        
        results = {}
        params_list = list(double_popt)
        
        results['tau1'] = params_list[0]
        results['tau2'] = params_list[1]
        results['amp1'] = params_list[2]
        results['amp2'] = params_list[3]

        results['tau1'], results['tau2'] = min(results['tau1'], results['tau2']), max(results['tau1'], results['tau2'])
        results['amp1'], results['amp2'] = (results['amp1'], results['amp2']) if results['tau1'] == double_popt[0] else (results['amp2'], results['amp1'])

        results['tau'] = (results['amp1'] * results['tau1'] + results['amp2'] * results['tau2']) / (results['amp1'] + results['amp2'])
        results['error'] = double_error
        results['cov'] = double_pcov
        results['fitted_curve'] = double_exp(lag_times, **{k: results[k] for k in ['tau1', 'tau2', 'amp1', 'amp2']})
    
    if not ft.success:
        results = {}
        results['tau1'] = np.nan
        results['tau2'] = np.nan
        results['amp1'] = np.nan
        results['amp2'] = np.nan
        results['tau'] = np.nan
        results['error'] = np.nan
        results['cov'] = np.nan
        results['fitted_curve'] = np.nan * lag_times
    
    return results


def select_exp_fit(lag_times, acf, threshold=8.0, fix_amp_value=None):
    single_results = single_exp_fit(lag_times, acf, fix_amp_value=fix_amp_value)
    double_results = double_exp_fit(lag_times, acf)
    
    if np.isfinite(single_results['error']) and np.isfinite(double_results['error']):
        error_ratio = single_results['error'] / double_results['error']
        
        if error_ratio > threshold:
            selected_model = 'double'
            selected_results = double_results
            selected_tau = double_results['tau']
        else:
            selected_model = 'single'
            selected_results = single_results
            selected_tau = single_results['tau']
    else:
        error_ratio = np.nan
        selected_model = 'single' if np.isfinite(single_results['tau']) else 'none'
        selected_results = single_results if selected_model == 'single' else {}
        selected_tau = single_results.get('tau', np.nan)
    
    results = {
        'selected_model': selected_model,
        'error_ratio': error_ratio,
        'single_results': single_results,
        'double_results': double_results,
        'selected_results': selected_results,
        'selected_tau': selected_tau,
        'tau': selected_tau, # 维持一致性,重复存储tau
        'fitted_curve': selected_results['fitted_curve'] # 维持一致性,重复存储fitted_curve
    }
    
    return results


def get_timescale_by_area_under_curve(lag_times, acf):
    processed_acf = acf / acf[0]
    timescale = 0.0
    for i in range(1, len(lag_times)):
        width = lag_times[i] - lag_times[i-1]
        avg_height = (processed_acf[i] + processed_acf[i-1]) / 2
        timescale += width * avg_height
    return timescale
# endregion


# region 神经元放电性质计算(基于spike_times)
def get_combined_spike_times(spike_times, keep_size=False):
    """
    合并所有神经元的放电时间并排序
    
    参数:
    spike_times: 列表的列表,每个内层列表包含一个神经元的放电时间
    keep_size: 如果为True,则返回的形式类似于一个整合后的有效神经元
    """
    # 合并所有神经元的放电时间
    combined = []
    for neuron_times in spike_times:
        combined.extend(neuron_times)
    
    # 排序合并后的时间
    combined.sort()
    
    if  keep_size:
        return [combined] 
    else:
        return combined


def get_ISI_from_spike_times(spike_times):
    """
    计算每个神经元的放电间隔
    
    参数:
    spike_times: 列表的列表,每个内层列表包含一个神经元的放电时间
    
    返回:
    ISI_list: 列表的列表,每个内层列表包含对应神经元的ISI序列
    """
    ISI_list = []
    for neuron_spikes in spike_times:
        if len(neuron_spikes) > 1:
            ISI = np.diff(neuron_spikes).tolist()
            ISI_list.append(ISI)
        else:
            ISI_list.append([])
    return ISI_list
# endregion


# region 作图函数
def visualize_one_second(ax, x_start, y=None, fontsize=cf.LEGEND_SIZE, color=cf.BLACK):
    if y is None:
        y = ax.get_ylim()[1] * 0.9
    one_second_x = [x_start, x_start + 1000.]
    one_second_y = [y, y]
    cf.plt_line(ax, one_second_x, one_second_y, color=color)
    cf.add_text(ax, '1s', x=np.mean(one_second_x), y=np.mean(one_second_y), fontsize=fontsize, color=color, va='bottom', ha='center', transform=ax.transData)


def _get_indices_xy_for_spike_raster(ts, spike):
    # Get the indices of the spikes
    if spike.ndim == 2:
        spike_timestep, neuron_indices = np.where(spike)
    else:
        print(f'Error: spike should be 2D array, but got {spike.ndim}D array.')

    results = {}
    results['spike_timestep'] = spike_timestep
    results['neuron_indices'] = neuron_indices
    results['x'] = ts[spike_timestep]
    results['y'] = neuron_indices
    return results


def raster_plot(ax, ts, spike, color=cf.BLUE, xlabel='time (ms)', ylabel='neuron index', label=None, xlim=None, ylim=None, scatter_kwargs=None, set_ax_kwargs=None):
    if ylim is None:
        # spike 的 ylim 要设置的严格才会比较好看
        ylim = [0, spike.shape[1]]
    scatter_kwargs = cf.update_dict({'s': cf.MARKER_SIZE / 2}, scatter_kwargs)
    set_ax_kwargs = cf.update_dict({'adjust_tick_size': False}, set_ax_kwargs)
    
    r = _get_indices_xy_for_spike_raster(ts, spike)

    cf.plt_scatter(ax, r['x'], r['y'], color=color, label=label, clip_on=False, xlim=xlim, ylim=ylim, **scatter_kwargs)

    cf.set_ax(ax, xlabel, ylabel, xlim=xlim, ylim=ylim, **set_ax_kwargs)


def fr_scale_raster_plot(ax, ts, spike, fr, cmap=cf.DENSITY_CMAP, xlabel='time (ms)', ylabel='neuron index', label=None, xlim=None, ylim=None, scatter_kwargs=None, set_ax_kwargs=None):
    scatter_kwargs = cf.update_dict({'s': cf.MARKER_SIZE / 2}, scatter_kwargs)
    scatter_kwargs = cf.update_dict(scatter_kwargs, {'xlim': xlim, 'ylim': ylim})
    set_ax_kwargs = cf.update_dict({'adjust_tick_size': False}, set_ax_kwargs)
    
    r = _get_indices_xy_for_spike_raster(ts, spike, xlim=xlim, ylim=ylim)
    
    # Get the color based on firing rate
    c = fr[r['spike_timestep']]

    cf.plt_colorful_scatter(ax, r['x'], r['y'], c, cmap=cmap, label=label, scatter_kwargs={'clip_on': False}, **scatter_kwargs)

    cf.set_ax(ax, xlabel, ylabel, xlim=xlim, ylim=ylim, **set_ax_kwargs)


def EI_raster_plot(ax, ts, E_spike, I_spike, E_color=E_COLOR, I_color=I_COLOR, xlabel='time (ms)', ylabel='', E_label=None, I_label=None, E_xlim=None, E_ylim=None, I_xlim=None, I_ylim=None, split_ax_kwargs=None, scatter_kwargs=None, set_ax_kwargs=None):
    split_ax_kwargs = cf.update_dict({'nrows': 2, 'sharex': True, 'hspace': cf.SIDE_PAD*3, 'height_ratios': [E_spike.shape[1], I_spike.shape[1]]}, split_ax_kwargs)
    
    ax_E, ax_I = cf.split_ax(ax, **split_ax_kwargs)
    raster_plot(ax_E, ts, E_spike, color=E_color, xlabel='', ylabel='E '+ylabel, label=E_label, xlim=E_xlim, ylim=E_ylim, scatter_kwargs=scatter_kwargs, set_ax_kwargs=set_ax_kwargs)
    raster_plot(ax_I, ts, I_spike, color=I_color, xlabel=xlabel, ylabel='I '+ylabel, label=I_label,  xlim=I_xlim, ylim=I_ylim, scatter_kwargs=scatter_kwargs, set_ax_kwargs=set_ax_kwargs)

    cf.rm_ax_spine(ax_E, 'bottom')
    cf.rm_ax_tick(ax_E, 'x')
    cf.rm_ax_ticklabel(ax_E, 'x')
    cf.align_label([ax_E, ax_I], 'y')
    return ax_E, ax_I


def template_line_plot(ax, x, y, color=cf.BLUE, xlabel='x', ylabel='y', label=None, line_kwargs=None, set_ax_kwargs=None):
    if line_kwargs is None:
        line_kwargs = {}
    if set_ax_kwargs is None:
        set_ax_kwargs = {}

    cf.plt_line(ax, x, y, color=color, label=label, **line_kwargs)
    cf.set_ax(ax, xlabel, ylabel, **set_ax_kwargs)


def template_EI_line_plot(ax, x, E_y, I_y, xlabel='x', ylabel='y', E_label='E', I_label='I', line_kwargs=None, set_ax_kwargs=None):
    template_line_plot(ax, x, E_y, color=E_COLOR, xlabel=xlabel, ylabel='E '+ylabel, label=E_label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)
    template_line_plot(ax, x, I_y, color=I_COLOR, xlabel=xlabel, ylabel='I '+ylabel, label=I_label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)


def input_current_plot(ax, ts, current, color=cf.BLUE, xlabel='time (ms)', ylabel='input current (nA)', label=None, line_kwargs=None, set_ax_kwargs=None):
    template_line_plot(ax, ts, current, color=color, xlabel=xlabel, ylabel=ylabel, label=label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)


def seperate_ext_input_current_plot(ax, ts, internal_current, external_current, internal_color=cf.BLUE, external_color=cf.GREEN, total_color=cf.BLACK, xlabel='time (ms)', ylabel='input current (nA)', internal_label='internal', external_label='external', total_label='total', line_kwargs=None, set_ax_kwargs=None):
    input_current_plot(ax, ts, internal_current, color=internal_color, xlabel=xlabel, ylabel=ylabel, label=internal_label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)
    input_current_plot(ax, ts, external_current, color=external_color, xlabel=xlabel, ylabel=ylabel, label=external_label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)
    input_current_plot(ax, ts, internal_current + external_current, color=total_color, xlabel=xlabel, ylabel=ylabel, label=total_label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)


def seperate_EI_input_current_plot(ax, ts, E_current, I_current, E_color=E_COLOR, I_color=I_COLOR, total_color=cf.BLACK, xlabel='time (ms)', ylabel='input current (nA)', E_label='E', I_label='I', total_label='total', set_ax_kwargs=None, line_kwargs=None):
    input_current_plot(ax, ts, E_current, color=E_color, xlabel=xlabel, ylabel=ylabel, label=E_label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)
    input_current_plot(ax, ts, I_current, color=I_color, xlabel=xlabel, ylabel=ylabel, label=I_label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)
    input_current_plot(ax, ts, E_current + I_current, color=total_color, xlabel=xlabel, ylabel=ylabel, label=total_label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)

@cf.deprecated
def seperate_EI_ext_input_current_plot(ax, ts, E_current, I_current, external_current, E_color=E_COLOR, I_color=I_COLOR, external_color=cf.GREEN, total_color=cf.BLACK, xlabel='time (ms)', ylabel='input current (nA)', E_label='E', I_label='I', external_label='external', total_label='total', set_ax_kwargs=None, line_kwargs=None):
    input_current_plot(ax, ts, E_current, color=E_color, xlabel=xlabel, ylabel=ylabel, label=E_label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)
    input_current_plot(ax, ts, I_current, color=I_color, xlabel=xlabel, ylabel=ylabel, label=I_label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)
    input_current_plot(ax, ts, external_current, color=external_color, xlabel=xlabel, ylabel=ylabel, label=external_label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)
    input_current_plot(ax, ts, E_current + I_current + external_current, color=total_color, xlabel=xlabel, ylabel=ylabel, label=total_label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)


def fr_plot(ax, ts, fr, color=cf.BLUE, xlabel='time (ms)', ylabel='firing rate (Hz)', label=None, line_kwargs=None, set_ax_kwargs=None):
    template_line_plot(ax, ts, fr, color=color, xlabel=xlabel, ylabel=ylabel, label=label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)


def EI_fr_plot(ax, ts, E_fr, I_fr, E_color=E_COLOR, I_color=I_COLOR, xlabel='time (ms)', ylabel='firing rate (Hz)', E_label='E', I_label='I', set_ax_kwargs=None, line_kwargs=None):
    fr_plot(ax, ts, E_fr, color=E_color, xlabel=xlabel, ylabel=ylabel, label=E_label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)
    fr_plot(ax, ts, I_fr, color=I_color, xlabel=xlabel, ylabel=ylabel, label=I_label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)


def voltage_plot(ax, ts, V, threshold, color=cf.BLUE, threshold_color=cf.ORANGE, xlabel='time (ms)', ylabel='membrane potential (mV)', label=None, line_kwargs=None, set_ax_kwargs=None):
    template_line_plot(ax, ts, V, color=color, xlabel=xlabel, ylabel=ylabel, label=label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)
    cf.add_hline(ax, threshold, color=threshold_color, linestyle='--', label='threshold')


def LFP_plot(ax, ts, LFP, color=cf.BLUE, xlabel='time (ms)', ylabel='LFP (mV)', label=None, line_kwargs=None, set_ax_kwargs=None):
    template_line_plot(ax, ts, LFP, color=color, xlabel=xlabel, ylabel=ylabel, label=label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)


def EI_LFP_plot(ax, E_LFP, I_LFP, dt, E_color=E_COLOR, I_color=I_COLOR, xlabel='time (ms)', ylabel='LFP (mV)', E_label='E', I_label='I', set_ax_kwargs=None, line_kwargs=None):
    LFP_plot(ax, E_LFP, dt, color=E_color, xlabel=xlabel, ylabel=ylabel, label=E_label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)
    LFP_plot(ax, I_LFP, dt, color=I_color, xlabel=xlabel, ylabel=ylabel, label=I_label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)


def acf_plot(ax, lag_times, acf, tau=None, amp=None, color=cf.BLUE, xlabel='lag (ms)', ylabel='ACF', label=None, line_kwargs=None, set_ax_kwargs=None, text_x=0.2, text_y=0.9, text_color=cf.BLACK, fontsize=cf.FONT_SIZE*1.6, show_fit_line=False):
    if tau is not None:
        cf.add_text(ax, f'{cf.round_float(tau)} ms', x=text_x, y=text_y, fontsize=fontsize, color=text_color, va='center', ha='center')

    if show_fit_line:
        fit_line = single_exp(lag_times, amp=amp, tau=tau)
        cf.plt_line(ax, lag_times, fit_line, color=color, linestyle='--', label='exp fit')

    template_line_plot(ax, lag_times, acf, color=color, xlabel=xlabel, ylabel=ylabel, label=label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)


class ACFPlotManager:
    '''
    用来方便的让text_y错开
    '''
    def __init__(self, ax, text_x):
        pass


def EI_acf_plot(ax, E_lag_times, E_acf, I_lag_times, I_acf, E_color=E_COLOR, I_color=I_COLOR, xlabel='lag (ms)', ylabel='ACF', E_label='E', I_label='I', set_ax_kwargs=None, line_kwargs=None):
    acf_plot(ax, E_lag_times, E_acf, color=E_color, xlabel=xlabel, ylabel=ylabel, label=E_label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)
    acf_plot(ax, I_lag_times, I_acf, color=I_color, xlabel=xlabel, ylabel=ylabel, label=I_label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)


def freq_plot(ax, freqs, power, color=cf.BLUE, xlabel='frequency (Hz)', ylabel='power', label=None, line_kwargs=None, set_ax_kwargs=None):
    set_ax_kwargs = cf.update_dict({'xlim': [0, 500]}, set_ax_kwargs)
    template_line_plot(ax, freqs, power, color=color, xlabel=xlabel, ylabel=ylabel, label=label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)


def EI_freq_plot(ax, E_freqs, E_power, I_freqs, I_power, E_color=E_COLOR, I_color=I_COLOR, xlabel='frequency (Hz)', ylabel='power', E_label='E', I_label='I', set_ax_kwargs=None, line_kwargs=None):
    freq_plot(ax, E_freqs, E_power, color=E_color, xlabel=xlabel, ylabel=ylabel, label=E_label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)
    freq_plot(ax, I_freqs, I_power, color=I_color, xlabel=xlabel, ylabel=ylabel, label=I_label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)


def FF_timewindow_plot(ax, timebin_list, FF, color=cf.BLUE, xlabel='time window (ms)', ylabel='FF', label=None, line_kwargs=None, set_ax_kwargs=None):
    template_line_plot(ax, timebin_list, FF, color=color, xlabel=xlabel, ylabel=ylabel, label=label, line_kwargs=line_kwargs, set_ax_kwargs=set_ax_kwargs)


def template_hist_plot(ax, data, color=cf.BLUE, xlabel='x', ylabel='probability', label=None, hist_kwargs=None, set_ax_kwargs=None):
    '''
    绘制histogram图
    '''
    if hist_kwargs is None:
        hist_kwargs = {}
    if set_ax_kwargs is None:
        set_ax_kwargs = {}
    cf.plt_hist(ax, data, color=color, label=label, **hist_kwargs)
    cf.set_ax(ax, xlabel, ylabel, **set_ax_kwargs)


def ISI_hist_plot(ax, ISI, color=cf.BLUE, xlabel='ISI (ms)', ylabel='probability', label=None, hist_kwargs=None, set_ax_kwargs=None):
    '''
    绘制ISI分布图
    '''
    if len(ISI) > 0:
        set_ax_kwargs = cf.update_dict({'ylog': True}, set_ax_kwargs)
    else:
        set_ax_kwargs = cf.update_dict({}, set_ax_kwargs)
    template_hist_plot(ax, ISI, color=color, xlabel=xlabel, ylabel=ylabel, label=label, hist_kwargs=hist_kwargs, set_ax_kwargs=set_ax_kwargs)


def ISI_CV_hist_plot(ax, ISI_CV, color=cf.BLUE, xlabel='ISI CV', ylabel='probability', label=None, hist_kwargs=None, set_ax_kwargs=None):
    '''
    绘制ISI CV分布图
    '''
    template_hist_plot(ax, ISI_CV, color=color, xlabel=xlabel, ylabel=ylabel, label=label, hist_kwargs=hist_kwargs, set_ax_kwargs=set_ax_kwargs)


def FF_hist_plot(ax, FF, color=cf.BLUE, xlabel='FF', ylabel='probability', label=None, hist_kwargs=None, set_ax_kwargs=None):
    '''
    绘制FF分布图
    '''
    template_hist_plot(ax, FF, color=color, xlabel=xlabel, ylabel=ylabel, label=label, hist_kwargs=hist_kwargs, set_ax_kwargs=set_ax_kwargs)


def corr_hist_plot(ax, corr, color=cf.BLUE, xlabel='correlation', ylabel='probability', label=None, hist_kwargs=None, set_ax_kwargs=None):
    '''
    绘制correlation分布图
    '''
    template_hist_plot(ax, corr, color=color, xlabel=xlabel, ylabel=ylabel, label=label, hist_kwargs=hist_kwargs, set_ax_kwargs=set_ax_kwargs)


def avalanche_size_hist_plot(ax, avalanche_size, color=cf.BLUE, xlabel='avalanche size', ylabel='probability', label=None, hist_kwargs=None, set_ax_kwargs=None):
    '''
    绘制avalanche size分布图
    '''
    if len(avalanche_size) > 0:
        set_ax_kwargs = cf.update_dict({'xlog': True, 'ylog': True}, set_ax_kwargs)
    else:
        set_ax_kwargs = cf.update_dict({}, set_ax_kwargs)
    template_hist_plot(ax, avalanche_size, color=color, xlabel=xlabel, ylabel=ylabel, label=label, hist_kwargs=hist_kwargs, set_ax_kwargs=set_ax_kwargs)


def avalanche_duration_hist_plot(ax, avalanche_duration, color=cf.BLUE, xlabel='avalanche duration', ylabel='probability', title='avalanche duration distribution', label=None, hist_kwargs=None, set_ax_kwargs=None):
    '''
    绘制avalanche duration分布图
    '''
    if len(avalanche_duration) > 0:
        set_ax_kwargs = cf.update_dict({'xlog': True, 'ylog': True}, set_ax_kwargs)
    else:
        set_ax_kwargs = cf.update_dict({}, set_ax_kwargs)
    template_hist_plot(ax, avalanche_duration, color=color, xlabel=xlabel, ylabel=ylabel, title=title, label=label, hist_kwargs=hist_kwargs, set_ax_kwargs=set_ax_kwargs)


def avalanche_size_duration_plot(ax, avalanche_size, avalanche_duration, scatter_color=cf.BLUE, line_color=cf.RED, xlabel='avalanche size', ylabel='avalanche duration', label=None, linregress_kwargs=None, set_ax_kwargs=None):
    '''
    绘制avalanche size和duration的散点图
    '''
    if linregress_kwargs is None:
        linregress_kwargs = {}
    if set_ax_kwargs is None:
        set_ax_kwargs = {}
    cf.plt_linregress(ax, avalanche_size, avalanche_duration, label=label, scatter_color=scatter_color, line_color=line_color, **linregress_kwargs)
    cf.set_ax(ax, xlabel, ylabel, **set_ax_kwargs)


def get_raster_color(i, pos, spike, faint_num, color, pos_color, pos_alpha):
    scattered_idx = []
    colors = np.zeros((pos.shape[0], 4))  # RGBA
    for previous in range(faint_num + 1):
        if i - previous >= 0:
            alpha = np.linspace(1, 0, faint_num + 1)[previous]
            current_idx = time_idx_data(spike, i - previous) > 0
            scattered_idx.extend(np.where(current_idx)[0])
            colors[current_idx, :3] = color  # Set RGB
            colors[current_idx, 3] = alpha  # Set alpha

    scattered_idx = np.array(scattered_idx)
    unscattered_idx = np.setdiff1d(np.arange(pos.shape[0]), scattered_idx)

    # Set colors for unscattered points
    colors[unscattered_idx, :3] = pos_color
    colors[unscattered_idx, 3] = pos_alpha

    return colors


def spatial_raster_plot(ax, spike, pos, i, dt, faint_num=3, label=None, color=cf.BLUE, scatter_size=(cf.MARKER_SIZE/3)**2, pos_color=cf.RANA, pos_alpha=0.1, scale_prop=1.05, legend_loc='upper left', bbox_to_anchor=(1, 1), set_ax_kwargs=None, scatter_kwargs=None):
    '''
    绘制Raster图
    '''
    if set_ax_kwargs is None:
        set_ax_kwargs = {}
    if scatter_kwargs is None:
        scatter_kwargs = {}
    colors = get_raster_color(i, pos, spike, faint_num, color, pos_color, pos_alpha)
    if pos.shape[1] == 2:
        cf.plt_scatter(ax, pos[:, 0], pos[:, 1], color=colors, s=scatter_size, **scatter_kwargs)
        cf.plt_scatter(ax, [], [], color=color, s=scatter_size, label=label, **scatter_kwargs)
    if pos.shape[1] == 3:
        cf.plt_scatter_3d(ax, pos[:, 0], pos[:, 1], pos[:, 2], color=colors, s=scatter_size, **scatter_kwargs)
        cf.plt_scatter_3d(ax, [], [], [], color=color, s=scatter_size, label=label, **scatter_kwargs)

    expand_xlim_min, expand_xlim_max = cf.scale_range(np.min(pos[:, 0]), np.max(pos[:, 0]), scale_prop)
    expand_ylim_min, expand_ylim_max = cf.scale_range(np.min(pos[:, 1]), np.max(pos[:, 1]), scale_prop)
    ax.set_xlim([expand_xlim_min, expand_xlim_max])
    ax.set_ylim([expand_ylim_min, expand_ylim_max])
    if pos.shape[1] == 3:
        expand_zlim_min, expand_zlim_max = cf.scale_range(np.min(pos[:, 2]), np.max(pos[:, 2]), scale_prop)
        ax.set_zlim([expand_zlim_min, expand_zlim_max])
        cf.set_ax_aspect_3d(ax)
    else:
        cf.set_ax_aspect(ax)
    ax.axis('off')

    title = 't={}'.format(cf.align_decimal(i*dt, dt))
    cf.set_ax(ax, title=title, legend_loc=legend_loc, bbox_to_anchor=bbox_to_anchor, **set_ax_kwargs)


def EI_spatial_raster_plot(ax, E_spike, I_spike, E_pos, I_pos, i, dt, faint_num=3, scatter_size=(cf.MARKER_SIZE/3)**2, pos_color=cf.RANA, pos_alpha=0.1, scale_prop=1.05, legend_loc='upper left', bbox_to_anchor=(1, 1), set_ax_kwargs=None, scatter_kwargs=None):
    '''
    绘制Raster图
    '''
    if set_ax_kwargs is None:
        set_ax_kwargs = {}
    if scatter_kwargs is None:
        scatter_kwargs = {}
    E_colors = get_raster_color(i, E_pos, E_spike, faint_num, E_COLOR, pos_color, pos_alpha)
    I_colors = get_raster_color(i, I_pos, I_spike, faint_num, I_COLOR, pos_color, pos_alpha)
    colors = np.concatenate([E_colors, I_colors], axis=0)
    pos = np.concatenate([E_pos, I_pos], axis=0)
    if pos.shape[1] == 2:
        cf.plt_scatter(ax, pos[:, 0], pos[:, 1], color=colors, s=scatter_size, **scatter_kwargs)
        cf.plt_scatter(ax, [], [], color=E_COLOR, s=scatter_size, label='E', **scatter_kwargs)
        cf.plt_scatter(ax, [], [], color=I_COLOR, s=scatter_size, label='I', **scatter_kwargs)
    if pos.shape[1] == 3:
        cf.plt_scatter_3d(ax, pos[:, 0], pos[:, 1], pos[:, 2], color=colors, s=scatter_size, **scatter_kwargs)
        cf.plt_scatter_3d(ax, [], [], [], color=E_COLOR, s=scatter_size, label='E', **scatter_kwargs)
        cf.plt_scatter_3d(ax, [], [], [], color=I_COLOR, s=scatter_size, label='I', **scatter_kwargs)

    expand_xlim_min, expand_xlim_max = cf.scale_range(np.min(np.concatenate([E_pos[:, 0], I_pos[:, 0]])), np.max(np.concatenate([E_pos[:, 0], I_pos[:, 0]])), scale_prop)
    expand_ylim_min, expand_ylim_max = cf.scale_range(np.min(np.concatenate([E_pos[:, 1], I_pos[:, 1]])), np.max(np.concatenate([E_pos[:, 1], I_pos[:, 1]])), scale_prop)
    ax.set_xlim([expand_xlim_min, expand_xlim_max])
    ax.set_ylim([expand_ylim_min, expand_ylim_max])
    if E_pos.shape[1] == 3:
        expand_zlim_min, expand_zlim_max = cf.scale_range(np.min(np.concatenate([E_pos[:, 2], I_pos[:, 2]])), np.max(np.concatenate([E_pos[:, 2], I_pos[:, 2]])), scale_prop)
        ax.set_zlim([expand_zlim_min, expand_zlim_max])
        cf.set_ax_aspect_3d(ax)
    else:
        cf.set_ax_aspect(ax)
    ax.axis('off')

    title = 't={}'.format(cf.align_decimal(i*dt, dt))
    cf.set_ax(ax, title=title, legend_loc=legend_loc, bbox_to_anchor=bbox_to_anchor, **set_ax_kwargs)


def spatial_V_plot(ax, V, pos, i, dt, label=None, vmin=None, vmax=None, cmap=cf.PINEAPPLE_CMAP, scatter_size=(cf.MARKER_SIZE/3)**2, scale_prop=1.05, legend_loc='upper left', bbox_to_anchor=(1, 1), set_ax_kwargs=None, scatter_kwargs=None):
    '''
    绘制V的空间分布图
    '''
    if set_ax_kwargs is None:
        set_ax_kwargs = {}
    if scatter_kwargs is None:
        scatter_kwargs = {}

    V_now = time_idx_data(V, i)
    
    if pos.shape[1] == 2:
        cf.plt_colorful_scatter(ax, pos[:, 0], pos[:, 1], c=V_now, cmap=cmap, s=scatter_size, vmin=vmin, vmax=vmax, label=label, **scatter_kwargs)
    if pos.shape[1] == 3:
        cf.plt_colorful_scatter_3d(ax, pos[:, 0], pos[:, 1], pos[:, 2], c=V_now, cmap=cmap, s=scatter_size, vmin=vmin, vmax=vmax, label=label, **scatter_kwargs)

    expand_xlim_min, expand_xlim_max = cf.scale_range(np.min(pos[:, 0]), np.max(pos[:, 0]), scale_prop)
    expand_ylim_min, expand_ylim_max = cf.scale_range(np.min(pos[:, 1]), np.max(pos[:, 1]), scale_prop)
    ax.set_xlim([expand_xlim_min, expand_xlim_max])
    ax.set_ylim([expand_ylim_min, expand_ylim_max])
    if pos.shape[1] == 3:
        expand_zlim_min, expand_zlim_max = cf.scale_range(np.min(pos[:, 2]), np.max(pos[:, 2]), scale_prop)
        ax.set_zlim([expand_zlim_min, expand_zlim_max])
        cf.set_ax_aspect_3d(ax)
    else:
        cf.set_ax_aspect(ax)
    ax.axis('off')

    title = 't={}'.format(cf.align_decimal(i*dt, dt))
    cf.set_ax(ax, title=title, legend_loc=legend_loc, bbox_to_anchor=bbox_to_anchor, **set_ax_kwargs)


def EI_spatial_V_plot(ax, E_V, I_V, E_pos, I_pos, i, dt, E_label='E', I_label='I', vmin=None, vmax=None, cmap=cf.PINEAPPLE_CMAP, scatter_size=(cf.MARKER_SIZE/3)**2, scale_prop=1.05, legend_loc='upper left', bbox_to_anchor=(1, 1), set_ax_kwargs=None, scatter_kwargs=None):
    '''
    绘制V的空间分布图
    '''
    if set_ax_kwargs is None:
        set_ax_kwargs = {}
    if scatter_kwargs is None:
        scatter_kwargs = {}
    E_ax = ax[0]
    I_ax = ax[1]
    spatial_V_plot(ax=E_ax, V=E_V, pos=E_pos, i=i, dt=dt, label=E_label, vmin=vmin, vmax=vmax, cmap=cmap, scatter_size=scatter_size, scale_prop=scale_prop, legend_loc=legend_loc, bbox_to_anchor=bbox_to_anchor, set_ax_kwargs=set_ax_kwargs, scatter_kwargs=scatter_kwargs)
    spatial_V_plot(ax=I_ax, V=I_V, pos=I_pos, i=i, dt=dt, label=I_label, vmin=vmin, vmax=vmax, cmap=cmap, scatter_size=scatter_size, scale_prop=scale_prop, legend_loc=legend_loc, bbox_to_anchor=bbox_to_anchor, set_ax_kwargs=set_ax_kwargs, scatter_kwargs=scatter_kwargs)

    E_ax.set_title(cf.concat_str([E_label, 'V']))
    I_ax.set_title(cf.concat_str([I_label, 'V']))
    fig = E_ax.get_figure()
    cf.set_fig_title(fig, 't={}'.format(cf.align_decimal(i*dt, dt)))


def causal_spatial_raster_plot(ax, E_monitored_neuron, I_monitored_neuron, E_spike, I_spike, E_pos, I_pos, E2E_connection, E2I_connection, I2E_connection, I2I_connection, delay_step, i, dt, faint_num=3, E_label='E', I_label='I', scatter_size=(cf.MARKER_SIZE/3)**2, show_pos=True, pos_color=cf.RANA, pos_alpha=0.5, scale_prop=1.05, legend_loc='upper left', bbox_to_anchor=(1, 1), set_ax_kwargs=None, scatter_kwargs=None):
    '''
    绘制Raster图,根据delay_step和连接,画出spike传递的路径
    '''
    if isinstance(E_monitored_neuron, int):
        E_monitored_neuron = [E_monitored_neuron]
    if isinstance(I_monitored_neuron, int):
        I_monitored_neuron = [I_monitored_neuron]
    if set_ax_kwargs is None:
        set_ax_kwargs = {}
    if scatter_kwargs is None:
        scatter_kwargs = {}
    if show_pos:
        if E_pos.shape[1] == 2:
            cf.plt_scatter(ax, E_pos[:, 0], E_pos[:, 1], color=pos_color, s=scatter_size, alpha=pos_alpha, zorder=0, **scatter_kwargs)
            cf.plt_scatter(ax, I_pos[:, 0], I_pos[:, 1], color=pos_color, s=scatter_size, alpha=pos_alpha, zorder=0, **scatter_kwargs)
        if E_pos.shape[1] == 3:
            cf.plt_scatter_3d(ax, E_pos[:, 0], E_pos[:, 1], E_pos[:, 2], color=pos_color, s=scatter_size, alpha=pos_alpha, zorder=0, **scatter_kwargs)
            cf.plt_scatter_3d(ax, I_pos[:, 0], I_pos[:, 1], I_pos[:, 2], color=pos_color, s=scatter_size, alpha=pos_alpha, zorder=0, **scatter_kwargs)
    for previous in range(faint_num+1):
        if i-previous >= 0:
            alpha = np.linspace(1, 0, faint_num+1)[previous]
            if previous == 0:
                local_E_label = E_label
                local_I_label = I_label
            else:
                local_E_label = None
                local_I_label = None
            if E_pos.shape[1] == 2:
                cf.plt_scatter(ax, E_pos[time_idx_data(E_spike, i-previous) > 0, 0], E_pos[time_idx_data(E_spike, i-previous) > 0, 1], color=E_COLOR, s=scatter_size, alpha=alpha, label=local_E_label, **scatter_kwargs)
                cf.plt_scatter(ax, I_pos[time_idx_data(I_spike, i-previous) > 0, 0], I_pos[time_idx_data(I_spike, i-previous) > 0, 1], color=I_COLOR, s=scatter_size, alpha=alpha, label=local_I_label, **scatter_kwargs)
            if E_pos.shape[1] == 3:
                cf.plt_scatter_3d(ax, E_pos[time_idx_data(E_spike, i-previous) > 0, 0], E_pos[time_idx_data(E_spike, i-previous) > 0, 1], E_pos[time_idx_data(E_spike, i-previous) > 0, 2], color=E_COLOR, s=scatter_size, alpha=alpha, label=local_E_label, **scatter_kwargs)
                cf.plt_scatter_3d(ax, I_pos[time_idx_data(I_spike, i-previous) > 0, 0], I_pos[time_idx_data(I_spike, i-previous) > 0, 1], I_pos[time_idx_data(I_spike, i-previous) > 0, 2], color=I_COLOR, s=scatter_size, alpha=alpha, label=local_I_label, **scatter_kwargs)
            for source in ['E', 'I']:
                for target in ['E', 'I']:
                    if source == 'E' and target == 'E':
                        connection = E2E_connection
                        source_pos = E_pos
                        target_pos = E_pos
                        monitored_neuron = E_monitored_neuron
                        color = E_COLOR
                        spike = E_spike
                    if source == 'E' and target == 'I':
                        connection = E2I_connection
                        source_pos = E_pos
                        target_pos = I_pos
                        monitored_neuron = I_monitored_neuron
                        color = E_COLOR
                        spike = E_spike
                    if source == 'I' and target == 'E':
                        connection = I2E_connection
                        source_pos = I_pos
                        target_pos = E_pos
                        monitored_neuron = E_monitored_neuron
                        color = I_COLOR
                        spike = I_spike
                    if source == 'I' and target == 'I':
                        connection = I2I_connection
                        source_pos = I_pos
                        target_pos = I_pos
                        monitored_neuron = I_monitored_neuron
                        color = I_COLOR
                        spike = I_spike
                    for j in range(connection.shape[0]):
                        if monitored_neuron is not None:
                            for k in monitored_neuron:
                                if connection[j, k] > 0 and time_idx_data(spike, i-previous-delay_step)[j] > 0:
                                    if source_pos.shape[1] == 2:
                                        cf.add_mid_arrow(ax, source_pos[j, 0], source_pos[j, 1], target_pos[k, 0], target_pos[k, 1], fc=color, ec=color, linewidth=scatter_size/2, alpha=alpha)
                                    if source_pos.shape[1] == 3:
                                        # ax.plot([source_pos[j, 0], source_pos[k, 0]], [pos[j, 1], pos[k, 1]], [pos[j, 2], pos[k, 2]], color=color, alpha=alpha)
                                        pass
    ax.axis('off')
    title = 't={}'.format(cf.align_decimal(i*dt, dt))
    cf.set_ax(ax, title=title, legend_loc=legend_loc, bbox_to_anchor=bbox_to_anchor, **set_ax_kwargs)
    expand_xlim_min, expand_xlim_max = cf.scale_range(np.min(np.concatenate([E_pos[:, 0], I_pos[:, 0]])), np.max(np.concatenate([E_pos[:, 0], I_pos[:, 0]])), scale_prop)
    expand_ylim_min, expand_ylim_max = cf.scale_range(np.min(np.concatenate([E_pos[:, 1], I_pos[:, 1]])), np.max(np.concatenate([E_pos[:, 1], I_pos[:, 1]])), scale_prop)
    ax.set_xlim([expand_xlim_min, expand_xlim_max])
    ax.set_ylim([expand_ylim_min, expand_ylim_max])
    if E_pos.shape[1] == 3:
        expand_zlim_min, expand_zlim_max = cf.scale_range(np.min(np.concatenate([E_pos[:, 2], I_pos[:, 2]])), np.max(np.concatenate([E_pos[:, 2], I_pos[:, 2]])), scale_prop)
        ax.set_zlim([expand_zlim_min, expand_zlim_max])
        cf.set_ax_aspect_3d(ax)
    else:
        cf.set_ax_aspect(ax)


def neuron_frame(dim, margin, fig_ax_kwargs, plot_func, plot_func_kwargs, elev_list, azim_list, folder, part_figname, save_fig_kwargs, i=None):
    if dim == 2:
        fig, ax = cf.get_fig_ax(margin=margin, **fig_ax_kwargs)
        plot_func(ax=ax, i=i, **plot_func_kwargs)
        figname = cf.concat_str([part_figname, 'i='+str(i)])
        filename = os.path.join(folder, figname)
        cf.save_fig(fig, filename, formats=['png'], pkl=False, dpi=100, **save_fig_kwargs)
        return filename
    if dim == 3:
        fig, ax = cf.get_fig_ax_3d(margin=margin, **fig_ax_kwargs)
        plot_func(ax=ax, i=i, **plot_func_kwargs)
        figname = cf.concat_str([part_figname, 'i='+str(i)])
        filename = os.path.join(folder, figname)
        fig_paths_dict, _ = cf.save_fig_3d(fig, filename, elev_list=elev_list, azim_list=azim_list, generate_video=False, formats=['png'], dpi=100, pkl=False, **save_fig_kwargs)
        return fig_paths_dict


def neuron_video(plot_func, plot_func_kwargs, dim, step_list, folder, video_name, elev_list=None, azim_list=None, margin=None, fig_ax_kwargs=None, set_ax_kwargs=None, part_figname='', save_fig_kwargs=None, video_kwargs=None, process_num=cf.PROCESS_NUM):
    '''
    绘制neuron视频

    参数:
    plot_func: 绘图函数
    plot_func_kwargs: 绘图函数的参数
    dim: 空间维度
    dt: 时间步长
    step_list: 时间步长列表
    folder: 保存文件夹
    video_name: 视频名称
    legend_loc: 图例位置
    bbox_to_anchor: 图例位置
    elev_list: 视角仰角列表
    azim_list: 视角方位角列表
    margin: 图边距
    fig_ax_kwargs: fig和ax的参数
    scatter_kwargs: scatter的参数
    set_ax_kwargs: set_ax的参数
    save_fig_kwargs: save_fig的参数
    video_kwargs: video的参数
    process_num: 进程数
    '''
    cf.mkdir(folder)

    if elev_list is None:
        elev_list = [cf.ELEV]
    if azim_list is None:
        azim_list = [cf.AZIM]
    if margin is None:
        margin = {'left': 0.1, 'right':0.75, 'bottom' : 0.1, 'top': 0.75}
    if fig_ax_kwargs is None:
        fig_ax_kwargs = {}
    if set_ax_kwargs is None:
        set_ax_kwargs = {}
    if save_fig_kwargs is None:
        save_fig_kwargs = {}
    if video_kwargs is None:
        video_kwargs = {}

    if dim == 2:
        local_args = (dim, margin, fig_ax_kwargs, plot_func, plot_func_kwargs, elev_list, azim_list, folder, part_figname, save_fig_kwargs)
        fig_paths = cf.multi_process_list_for(process_num, func=neuron_frame, args=local_args, for_list=step_list, for_idx_name='i')
        cf.fig_to_video(fig_paths, os.path.join(folder, video_name), **video_kwargs)
    if dim == 3:
        fig_paths = {}
        local_args = (dim, margin, fig_ax_kwargs, plot_func, plot_func_kwargs, elev_list, azim_list, folder, part_figname, save_fig_kwargs)
        fig_paths_dict_list = cf.multi_process_list_for(process_num, func=neuron_frame, args=local_args, for_list=step_list, for_idx_name='i')
        for fig_paths_dict in fig_paths_dict_list:
            for elev in elev_list:
                for azim in azim_list:
                    fig_paths[(elev, azim)] = fig_paths.get((elev, azim), []) + fig_paths_dict[(elev, azim)]
        for elev in elev_list:
            for azim in azim_list:
                cf.fig_to_video(fig_paths[(elev, azim)], os.path.join(folder, video_name+'_elev_{}_azim_{}'.format(str(int(elev)), str(int(azim)))), **video_kwargs)


def spike_video(E_spike, I_spike, E_pos, I_pos, dt, step_list, folder, video_name='spike_video', scatter_size=(cf.MARKER_SIZE/3)**2, faint_num=3, legend_loc='upper left', bbox_to_anchor=(1, 1), elev_list=None, azim_list=None, margin=None, fig_ax_kwargs=None, scatter_kwargs=None, set_ax_kwargs=None, part_figname='spike', save_fig_kwargs=None, video_kwargs=None, process_num=cf.PROCESS_NUM):
    margin = cf.update_dict({'left': 0.1, 'right':0.75, 'bottom' : 0.1, 'top': 0.75}, margin)
    EI_spatial_raster_kwargs = {'E_spike': E_spike, 'I_spike': I_spike, 'E_pos': E_pos, 'I_pos': I_pos, 'scatter_size': scatter_size, 'faint_num': faint_num, 'dt': dt, 'legend_loc': legend_loc, 'bbox_to_anchor': bbox_to_anchor, 'set_ax_kwargs': set_ax_kwargs, 'scatter_kwargs': scatter_kwargs}
    neuron_video(EI_spatial_raster_plot, EI_spatial_raster_kwargs, E_pos.shape[1], step_list, folder, video_name, elev_list=elev_list, azim_list=azim_list, margin=margin, fig_ax_kwargs=fig_ax_kwargs, set_ax_kwargs=set_ax_kwargs, part_figname=part_figname, save_fig_kwargs=save_fig_kwargs, video_kwargs=video_kwargs, process_num=process_num)


def V_video(E_V, I_V, E_pos, I_pos, dt, step_list, folder, vmin, vmax, cmap=cf.PINEAPPLE_CMAP, video_name='V_video', scatter_size=(cf.MARKER_SIZE/3)**2, legend_loc='upper left', bbox_to_anchor=(1, 1), elev_list=None, azim_list=None, margin=None, fig_ax_kwargs=None, scatter_kwargs=None, set_ax_kwargs=None, part_figname='V', save_fig_kwargs=None, video_kwargs=None, process_num=cf.PROCESS_NUM):
    margin = cf.update_dict({'left': 0.1, 'right':0.75, 'bottom' : 0.1, 'top': 0.75}, margin)
    fig_ax_kwargs = cf.update_dict({'ncols': 2}, fig_ax_kwargs)
    spatial_V_kwargs = {'E_V': E_V, 'I_V': I_V, 'E_pos': E_pos, 'I_pos': I_pos, 'scatter_size': scatter_size, 'dt': dt, 'vmin': vmin, 'vmax': vmax, 'cmap': cmap, 'legend_loc': legend_loc, 'bbox_to_anchor': bbox_to_anchor, 'set_ax_kwargs': set_ax_kwargs, 'scatter_kwargs': scatter_kwargs}
    neuron_video(EI_spatial_V_plot, spatial_V_kwargs, E_pos.shape[1], step_list, folder, video_name, elev_list=elev_list, azim_list=azim_list, margin=margin, fig_ax_kwargs=fig_ax_kwargs, set_ax_kwargs=set_ax_kwargs, part_figname=part_figname, save_fig_kwargs=save_fig_kwargs, video_kwargs=video_kwargs, process_num=process_num)


def casual_spike_video(E_monitored_neuron, I_monitored_neuron, E_spike, E_pos, I_spike, I_pos, E2E_connection, E2I_connection, I2E_connection, I2I_connection, delay_step, dt, folder, video_name='spike_video', scatter_size=(cf.MARKER_SIZE/3)**2, faint_num=3, legend_loc='upper left', bbox_to_anchor=(1, 1), elev_list=None, azim_list=None, margin=None, fig_ax_kwargs=None, scatter_kwargs=None, set_ax_kwargs=None, save_fig_kwargs=None, video_kwargs=None):
    margin = cf.update_dict({'left': 0.1, 'right':0.75, 'bottom' : 0.1, 'top': 0.75}, margin)
    causal_spatial_raster_kwargs = {'E_monitored_neuron': E_monitored_neuron, 'I_monitored_neuron': I_monitored_neuron, 'E_spike': E_spike, 'I_spike': I_spike, 'E_pos': E_pos, 'I_pos': I_pos, 'E2E_connection': E2E_connection, 'E2I_connection': E2I_connection, 'I2E_connection': I2E_connection, 'I2I_connection': I2I_connection, 'delay_step': delay_step, 'scatter_size': scatter_size, 'faint_num': faint_num, 'dt': dt, 'legend_loc': legend_loc, 'bbox_to_anchor': bbox_to_anchor, 'set_ax_kwargs': set_ax_kwargs, 'scatter_kwargs': scatter_kwargs}
    neuron_video(causal_spatial_raster_plot, causal_spatial_raster_kwargs, E_pos.shape[1], dt, E_spike.shape[0], folder, video_name, legend_loc=legend_loc, bbox_to_anchor=bbox_to_anchor, elev_list=elev_list, azim_list=azim_list, margin=margin, fig_ax_kwargs=fig_ax_kwargs, scatter_kwargs=scatter_kwargs, set_ax_kwargs=set_ax_kwargs, save_fig_kwargs=save_fig_kwargs, video_kwargs=video_kwargs)
# endregion