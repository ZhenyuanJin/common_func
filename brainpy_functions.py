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
import jax
import jax.numpy as jnp


# 数据处理和可视化库
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, ScalarFormatter
from matplotlib.colors import BoundaryNorm, Normalize
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable


# 神经网络和脑模型库
import brainpy as bp
import brainpy.math as bm
# from brainpy._src.dyn.base import SynDyn
# from brainpy._src.mixin import AlignPost, ReturnInfo
from brainpy._src import connect, initialize as init
from brainpy._src.integrators.ode.generic import odeint
from brainpy._src.integrators.joint_eq import JointEq
from brainpy._src.initialize import parameter
from brainpy._src.context import share
from brainpy.types import ArrayType
from brainpy._src.dynsys import DynamicalSystem, DynView
from brainpy._src.math.object_transform.base import StateLoadResult
# 脑区处理库
# import networkx as nx
# import pointpats
# import shapely.geometry


# 自定义库
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import common_functions as cf
import neuron_data_functions as ndf
from neuron_data_functions import *
# endregion


# region 定义默认参数
E_COLOR = cf.RED
I_COLOR = cf.BLUE
cf.print_title('brainpy version: {}'.format(bp.__version__), char='*')
TIME_AXIS = 0
NEURON_AXIS = 1
# endregion


# region brainpy使用说明
def brainpy_data_structure():
    '''
    所有数据按照(T, N)的形式存储,其中T表示时间点的数量,N表示神经元的数量
    当training时,要注意数据维度变为(B, T, N),其中B表示batch size
    '''
    pass


def brainpy_time_axis():
    return 0


def brainpy_neuron_axis():
    return 1


def brainpy_unit(physical_quantity):
    '''
    Note that pA * GOhm = mV, thus consistent with for example \tau * dV/dt = - ( V - V_rest ) + R * I
    '''
    if physical_quantity == 'V':
        return 'mV (10^-3 V)'
    if physical_quantity == 'I':
        return 'pA (10^-12 A)'
    if physical_quantity == 'R':
        return 'GOhm (10^9 Ohm)'
    if physical_quantity == 'g':
        return 'nS (10^-9 S)'
    if physical_quantity == 'tau':
        return 'ms (10^-3 s)'
# endregion


# region gpu设置
def set_to_gpu(pre_allocate=True):
    """
    Set the platform to GPU and enable pre-allocation of GPU memory.
    """
    bm.set_platform('gpu')
    if pre_allocate:
        enable_gpu_memory_preallocation()


def enable_gpu_memory_preallocation(percent=0.95):
    """
    Enable pre-allocating the GPU memory.

    Adapted from https://brainpy.readthedocs.io/en/latest/_modules/brainpy/_src/math/environment.html#enable_gpu_memory_preallocation
    """
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'
    os.environ.pop('XLA_PYTHON_CLIENT_ALLOCATOR', None)
    gpu_memory_preallocation(percent)


def gpu_memory_preallocation(percent: float):
    """GPU memory allocation.

    If preallocation is enabled, this makes JAX preallocate ``percent`` of the total GPU memory,
    instead of the default 75%. Lowering the amount preallocated can fix OOMs that occur when the JAX program starts.
    """
    assert 0. <= percent < 1., f'GPU memory preallocation must be in [0., 1.]. But we got {percent}.'
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(percent)
# endregion


# region 获得常用神经元参数
def get_LifRef_param(paper):
    if paper == 'Joglekar_2018_Neuron':
        E_params = {'V_th': -50.0, 'V_reset': -60.0, 'V_rest':-70.0, 'tau_ref': 2.0, 'R': 50.0, 'tau': 20.0, 'V_initializer': bp.init.Uniform(-70.0, -50.0)} # Initializer is obtained from https://brainpy-examples.readthedocs.io/en/latest/large_scale_modeling/Joglekar_2018_InterAreal_Balanced_Amplification_figure5.html
        I_params = {'V_th': -50.0, 'V_reset': -60.0, 'V_rest':-70.0, 'tau_ref': 2.0, 'R': 50.0, 'tau': 10.0, 'V_initializer': bp.init.Uniform(-70.0, -50.0)}
    if paper == 'Liang_2020_Frontiers':
        E_params = {'V_th': -50.0, 'V_reset': -60.0, 'V_rest':-70.0, 'tau_ref': 2.0, 'R': 50.0, 'tau': 20.0, 'V_initializer': bp.init.Uniform(-70.0, -50.0)}
        I_params = {'V_th': -50.0, 'V_reset': -60.0, 'V_rest':-70.0, 'tau_ref': 1.0, 'R': 50.0, 'tau': 10.0, 'V_initializer': bp.init.Uniform(-70.0, -50.0)}
    if paper == 'Wang_2002_Neuron':
        E_params = {'V_th': -50.0, 'V_reset': -55.0, 'V_rest':-70.0, 'tau_ref': 2.0, 'R': 0.04, 'tau': 20.0, 'V_initializer': bp.init.OneInit(-70.)}
        I_params = {'V_th': -50.0, 'V_reset': -55.0, 'V_rest':-70.0, 'tau_ref': 1.0, 'R': 0.05, 'tau': 10.0, 'V_initializer': bp.init.OneInit(-70.)}
    params = {'E': E_params, 'I': I_params}
    return params


def get_synapse_and_params(mode):
    if mode == 'Liang_2020_Frontiers_Fast':
        synapse = {'E2E': NormalizedDualExponCUBA, 'I2E': NormalizedDualExponCUBA, 'E2I': NormalizedDualExponCUBA, 'I2I': NormalizedDualExponCUBA}
        synapse_params = {'E2E': {'tau_rise': 0.5, 'tau_decay': 2.0, 'delay': 0.0, 'out_label': 'E'}, 'E2I': {'tau_rise': 0.5, 'tau_decay': 2.0, 'delay': 0.0, 'out_label': 'E'}, 'I2E': {'tau_rise': 0.5, 'tau_decay': 1.2, 'delay': 0.0, 'out_label': 'I'}, 'I2I': {'tau_rise': 0.5, 'tau_decay': 1.2, 'delay': 0.0, 'out_label': 'I'}}
    if mode == 'Liang_2020_Frontiers_Critical':
        synapse = {'E2E': NormalizedDualExponCUBA, 'I2E': NormalizedDualExponCUBA, 'E2I': NormalizedDualExponCUBA, 'I2I': NormalizedDualExponCUBA}
        synapse_params = {'E2E': {'tau_rise': 0.5, 'tau_decay': 2.0, 'delay': 0.0, 'out_label': 'E'}, 'E2I': {'tau_rise': 0.5, 'tau_decay': 2.0, 'delay': 0.0, 'out_label': 'E'}, 'I2E': {'tau_rise': 0.5, 'tau_decay': 3.0, 'delay': 0.0, 'out_label': 'I'}, 'I2I': {'tau_rise': 0.5, 'tau_decay': 3.0, 'delay': 0.0, 'out_label': 'I'}}
    if mode == 'Liang_2020_Frontiers_Slow':
        synapse = {'E2E': NormalizedDualExponCUBA, 'I2E': NormalizedDualExponCUBA, 'E2I': NormalizedDualExponCUBA, 'I2I': NormalizedDualExponCUBA}
        synapse_params = {'E2E': {'tau_rise': 0.5, 'tau_decay': 2.0, 'delay': 0.0, 'out_label': 'E'}, 'E2I': {'tau_rise': 0.5, 'tau_decay': 2.0, 'delay': 0.0, 'out_label': 'E'}, 'I2E': {'tau_rise': 0.5, 'tau_decay': 4.3, 'delay': 0.0, 'out_label': 'I'}, 'I2I': {'tau_rise': 0.5, 'tau_decay': 4.3, 'delay': 0.0, 'out_label': 'I'}}
    if mode == 'Liang_2022_PCB_AS':
        synapse = {'E2E': NormalizedExponCOBA, 'I2E': NormalizedExponCOBA, 'E2I': NormalizedExponCOBA, 'I2I': NormalizedExponCOBA}
        synapse_params = {'E2E': {'tau': 4.0, 'delay': 0.0, 'E': 0.0, 'out_label': 'E'}, 'E2I': {'tau': 4.0, 'delay': 0.0, 'E': 0.0, 'out_label': 'E'}, 'I2E': {'tau': 4.0, 'delay': 0.0, 'E': -70.0, 'out_label': 'I'}, 'I2I': {'tau': 4.0, 'delay': 0.0, 'E': -70.0, 'out_label': 'I'}}
    if mode == 'Liang_2022_PCB_Cri':
        synapse = {'E2E': NormalizedExponCOBA, 'I2E': NormalizedExponCOBA, 'E2I': NormalizedExponCOBA, 'I2I': NormalizedExponCOBA}
        synapse_params = {'E2E': {'tau': 4.0, 'delay': 0.0, 'E': 0.0, 'out_label': 'E'}, 'E2I': {'tau': 4.0, 'delay': 0.0, 'E': 0.0, 'out_label': 'E'}, 'I2E': {'tau': 8.0, 'delay': 0.0, 'E': -70.0, 'out_label': 'I'}, 'I2I': {'tau': 8.0, 'delay': 0.0, 'E': -70.0, 'out_label': 'I'}}
    if mode == 'Liang_2022_PCB_SS':
        synapse = {'E2E': NormalizedExponCOBA, 'I2E': NormalizedExponCOBA, 'E2I': NormalizedExponCOBA, 'I2I': NormalizedExponCOBA}
        synapse_params = {'E2E': {'tau': 4.0, 'delay': 0.0, 'E': 0.0, 'out_label': 'E'}, 'E2I': {'tau': 4.0, 'delay': 0.0, 'E': 0.0, 'out_label': 'E'}, 'I2E': {'tau': 11.0, 'delay': 0.0, 'E': -70.0, 'out_label': 'I'}, 'I2I': {'tau': 11.0, 'delay': 0.0, 'E': -70.0, 'out_label': 'I'}}
    if mode == 'Liang_2022_PCB_P':
        synapse = {'E2E': NormalizedExponCOBA, 'I2E': NormalizedExponCOBA, 'E2I': NormalizedExponCOBA, 'I2I': NormalizedExponCOBA}
        synapse_params = {'E2E': {'tau': 4.0, 'delay': 0.0, 'E': 0.0, 'out_label': 'E'}, 'E2I': {'tau': 4.0, 'delay': 0.0, 'E': 0.0, 'out_label': 'E'}, 'I2E': {'tau': 14.0, 'delay': 0.0, 'E': -70.0, 'out_label': 'I'}, 'I2I': {'tau': 14.0, 'delay': 0.0, 'E': -70.0, 'out_label': 'I'}}
    return synapse, synapse_params
# endregion


# region 数据类型转换
def dict_to_np_dict(d):
    '''
    将包含非np数组的dict内的元素全部转换为np数组
    '''
    np_dict = {}
    for k, v in d.items():
        np_dict[k] = np.array(v)
    return np_dict
# endregion


# region neuron
class PoissonGroupWithSeed(bp.dyn.PoissonGroup):
    def __init__(self, size, freqs, keep_size=False, sharding=None, spk_type=None, name=None, mode=None, seed=None):
        super().__init__(size=size, freqs=freqs, keep_size=keep_size, sharding=sharding, spk_type=spk_type, name=name, mode=mode)

        self.rng = bm.random.RandomState(seed_or_key=seed)
    
    def update(self):
        spikes = self.rng.rand_like(self.spike) <= (self.freqs * share['dt'] / 1000.)
        spikes = bm.asarray(spikes, dtype=self.spk_type)
        self.spike.value = spikes
        return spikes
# endregion


# region synapse
class NormalizedExpon(bp.dyn.Expon):
    '''
    不同于brainpy的Expon(https://brainpy.readthedocs.io/en/latest/apis/generated/brainpy.dyn.Expon.html),这里的Expon是使用timescale归一化的,使得整个kernel积分为1
    '''
    def add_current(self, x):
        self.g.value += x / self.tau


class NormalizedDualExponV2(bp.dyn.DualExponV2):
    '''
    调整A的默认值(https://brainpy.readthedocs.io/en/latest/apis/generated/brainpy.dyn.DualExponV2.html),使得整个kernel积分为1

    注意,如果想要获取g的话,要使用这样的语法:
    定义syn
    self.syn = bf.NormalizedDualExponCUBA(self.pre, self.post, delay=None, comm=bp.dnn.CSRLinear(bp.conn.FixedProb(1., pre=self.pre.num, post=self.post.num), 1.), tau_rise=2., tau_decay=20.)
    拿到syn的两个g和a
    (self.syn.proj.refs['syn'].g_decay - self.syn.proj.refs['syn'].g_rise) * self.syn.proj.refs['syn'].a

    相比之下,NormailzedExponCUBA的g可以直接拿到
    '''
    def __init__(
        self,
        size: Union[int, Sequence[int]],
        keep_size: bool = False,
        sharding: Optional[Sequence[str]] = None,
        method: str = 'exp_auto',
        name: Optional[str] = None,
        mode: Optional[bm.Mode] = None,

        # synapse parameters
        tau_decay: Union[float, ArrayType, Callable] = 10.0,
        tau_rise: Union[float, ArrayType, Callable] = 1.,
        A: Optional[Union[float, ArrayType, Callable]] = None,
    ):
        super().__init__(name=name,
                            mode=mode,
                            size=size,
                            keep_size=keep_size,
                            sharding=sharding)

        def _format_dual_exp_A(self, A):
            A = parameter(A, sizes=self.varshape, allow_none=True, sharding=self.sharding)
            if A is None:
                A = 1 / (self.tau_decay - self.tau_rise)
            return A

        # parameters
        self.tau_rise = self.init_param(tau_rise)
        self.tau_decay = self.init_param(tau_decay)
        self.a = _format_dual_exp_A(self, A)

        # integrator
        self.integral = odeint(lambda g, t, tau: -g / tau, method=method)

        self.reset_state(self.mode)


class ExponCUBA(bp.Projection):
    def __init__(self, pre, post, delay, comm, tau, out_label=None, **kwargs):
        super().__init__()
        
        self.proj = bp.dyn.FullProjAlignPostMg(
        pre=pre, 
        delay=delay, 
        comm=comm,
        syn=bp.dyn.Expon.desc(post.num, tau=tau),
        out=bp.dyn.CUBA.desc(),
        post=post,
        out_label=out_label 
        )


class ExponCOBA(bp.Projection):
    def __init__(self, pre, post, delay, comm, tau, E, out_label=None, name=None, **kwargs):
        super().__init__()
        if name is None:
            Expon_name = None
            COBA_name = None
        else:
            Expon_name = cf.cat('Expon', name)
            COBA_name = cf.cat('COBA', name)
        self.proj = bp.dyn.FullProjAlignPostMg(
        pre=pre, 
        delay=delay, 
        comm=comm,
        syn=bp.dyn.Expon.desc(post.num, tau=tau, name=Expon_name),
        out=bp.dyn.COBA.desc(E, name=COBA_name),
        post=post,
        out_label=out_label,
        name=name
        )


class NormalizedExponCUBA(bp.Projection):
    def __init__(self, pre, post, delay, comm, tau, out_label=None, **kwargs):
        super().__init__()
        
        self.proj = bp.dyn.FullProjAlignPostMg(
        pre=pre, 
        delay=delay, 
        comm=comm,
        syn=NormalizedExpon.desc(post.num, tau=tau),
        out=bp.dyn.CUBA.desc(),
        post=post,
        out_label=out_label 
        )


class NormalizedExponCOBA(bp.Projection):
    def __init__(self, pre, post, delay, comm, tau, E, out_label=None, **kwargs):
        super().__init__()
        
        self.proj = bp.dyn.FullProjAlignPostMg(
        pre=pre, 
        delay=delay, 
        comm=comm,
        syn=NormalizedExpon.desc(post.num, tau=tau),
        out=bp.dyn.COBA.desc(E),
        post=post,
        out_label=out_label 
        )


class DualExponCUBA(bp.Projection):
    def __init__(self, pre, post, delay, comm, tau_rise, tau_decay, A=None, out_label=None, **kwargs):
        super().__init__()
        
        self.proj = bp.dyn.FullProjAlignPostMg(
        pre=pre, 
        delay=delay, 
        comm=comm,
        syn=bp.dyn.DualExponV2.desc(post.num, tau_rise=tau_rise, tau_decay=tau_decay, A=A),
        out=bp.dyn.CUBA.desc(),
        post=post,
        out_label=out_label 
        )


class NormalizedDualExponCUBA(bp.Projection):
    def __init__(self, pre, post, delay, comm, tau_rise, tau_decay, out_label=None, **kwargs):
        super().__init__()
        
        self.proj = bp.dyn.FullProjAlignPostMg(
        pre=pre, 
        delay=delay, 
        comm=comm,
        syn=NormalizedDualExponV2.desc(post.num, tau_rise=tau_rise, tau_decay=tau_decay),
        out=bp.dyn.CUBA.desc(),
        post=post,
        out_label=out_label
        )


class NormalizedDualExponCOBA(bp.Projection):
    def __init__(self, pre, post, delay, comm, tau_rise, tau_decay, E, out_label=None, **kwargs):
        super().__init__()
        
        self.proj = bp.dyn.FullProjAlignPostMg(
        pre=pre, 
        delay=delay, 
        comm=comm,
        syn=NormalizedDualExponV2.desc(post.num, tau_rise=tau_rise, tau_decay=tau_decay),
        out=bp.dyn.COBA.desc(E),
        post=post,
        out_label=out_label
        )


class NMDACUBA(bp.Projection):
    def __init__(self, pre, post, delay, comm, tau_rise, tau_decay, a, out_label=None, **kwargs):
        super().__init__()
        
        self.proj = bp.dyn.FullProjAlignPreDSMg(
        pre=pre, 
        delay=delay, 
        comm=comm,
        syn=bp.dyn.NMDA.desc(pre.num, tau_decay=tau_decay, tau_rise=tau_rise, a=a),
        out=bp.dyn.CUBA(),
        post=post,
        out_label=out_label
        )


class NMDACOBA(bp.Projection):
    def __init__(self, pre, post, delay, comm, tau_rise, tau_decay, a, E, out_label=None, **kwargs):
        super().__init__()
        
        self.proj = bp.dyn.FullProjAlignPreDSMg(
        pre=pre, 
        delay=delay, 
        comm=comm,
        syn=bp.dyn.NMDA.desc(pre.num, tau_decay=tau_decay, tau_rise=tau_rise, a=a),
        out=bp.dyn.COBA(E),
        post=post,
        out_label=out_label
        )


class NMDAMgBlock(bp.Projection):
    def __init__(self, pre, post, delay, comm, tau_rise, tau_decay, a, E, cc_Mg, alpha, beta, V_offset, out_label=None, name=None, **kwargs):
        super().__init__()
        if name is None:
            NMDA_name = None
            MgBlock_name = None
        else:
            NMDA_name = cf.cat('NMDA', name)
            MgBlock_name = cf.cat('MgBlock', name)
        self.proj = bp.dyn.FullProjAlignPreDSMg(
        pre=pre, 
        delay=delay, 
        comm=comm,
        syn=bp.dyn.NMDA.desc(pre.num, tau_decay=tau_decay, tau_rise=tau_rise, a=a, name=NMDA_name),
        out=bp.dyn.MgBlock(E=E, cc_Mg=cc_Mg, alpha=alpha, beta=beta, V_offset=V_offset, name=MgBlock_name),
        post=post,
        out_label=out_label,
        name=name
        )
# endregion


# region 神经元连接
def ij_conn(pre, post, pre_size, post_size):
    '''
    利用brainpy的bp.conn.IJConn生成conn
    '''
    conn = bp.conn.IJConn(i=pre, j=post)
    # conn = IJConn(i=pre, j=post)
    conn = conn(pre_size=pre_size, post_size=post_size)
    return conn


def ij_comm(pre, post, pre_size, post_size, weight, mode=None, name=None):
    '''
    利用brainpy的bp.conn.IJConn和bp.dnn.EventCSRLinear生成comm
    '''
    conn = ij_conn(pre, post, pre_size, post_size)
    return bp.dnn.EventCSRLinear(conn, weight, mode=mode, name=name)


class EventCSRLinearWithSingleVar(bp.dnn.EventCSRLinear):
    '''
    只有一个变量来控制整体weight的权重,weight此时不可以被训练
    '''
    def __init__(self, conn, weight, scale_var, sharding=None, mode=None, name=None, transpose=True):
        bp._src.dnn.base.Layer.__init__(self, name=name, mode=mode)
        self.conn = conn
        self.sharding = sharding
        self.transpose = transpose

        # connection
        self.indices, self.indptr = self.conn.require('csr')

        # weight
        weight = init.parameter(weight, (self.indices.size,))
        self.weight = weight
        self.scale_var = scale_var
    
    def update(self, x):
        if x.ndim == 1:
            return bm.event.csrmv(self.weight*self.scale_var, self.indices, self.indptr, x,
                                  shape=(self.conn.pre_num, self.conn.post_num),
                                  transpose=self.transpose)
        elif x.ndim > 1:
            shapes = x.shape[:-1]
            x = bm.flatten(x, end_dim=-2)
            y = jax.vmap(self._batch_csrmv)(x)
            return bm.reshape(y, shapes + (y.shape[-1],))
        else:
            raise ValueError

    def _batch_csrmv(self, x):
        return bm.event.csrmv(self.weight*self.scale_var, self.indices, self.indptr, x,
                              shape=(self.conn.pre_num, self.conn.post_num),
                              transpose=self.transpose)


def ij_comm_with_single_var(pre, post, pre_size, post_size, weight, scale_var, mode=None, name=None):
    '''
    利用brainpy的bp.conn.IJConn和bp.dnn.EventCSRLinearWithSingleVar生成comm
    '''
    conn = ij_conn(pre, post, pre_size, post_size)
    return EventCSRLinearWithSingleVar(conn, weight, scale_var, mode=mode, name=name)


class EventCSRLinearWithEtaAndScale(bp.dnn.EventCSRLinear):
    '''
    利用eta,scale来控制整体weight的权重,weight此时不可以被训练
    
    hier是一个shape为(indices.size,)的变量,用来控制每个连接的权重
    '''
    def __init__(self, conn, weight, eta, hier, scale_var, sharding=None, mode=None, name=None, transpose=True):
        bp._src.dnn.base.Layer.__init__(self, name=name, mode=mode)
        self.conn = conn
        self.sharding = sharding
        self.transpose = transpose

        # connection
        self.indices, self.indptr = self.conn.require('csr')

        # weight
        weight = init.parameter(weight, (self.indices.size,))
        self.weight = weight
        self.scale_var = scale_var
        self.eta = eta
        hier = init.parameter(hier, (self.indices.size,))
        self.hier = hier
    
    def update(self, x):
        if x.ndim == 1:
            return bm.event.csrmv(self.weight*self.scale_var*(1.+self.eta*self.hier), self.indices, self.indptr, x,
                                  shape=(self.conn.pre_num, self.conn.post_num),
                                  transpose=self.transpose)
        elif x.ndim > 1:
            shapes = x.shape[:-1]
            x = bm.flatten(x, end_dim=-2)
            y = jax.vmap(self._batch_csrmv)(x)
            return bm.reshape(y, shapes + (y.shape[-1],))
        else:
            raise ValueError

    def _batch_csrmv(self, x):
        return bm.event.csrmv(self.weight*self.scale_var*(1.+self.eta*self.hier), self.indices, self.indptr, x,
                              shape=(self.conn.pre_num, self.conn.post_num),
                              transpose=self.transpose)


def ij_comm_with_eta_and_scale(pre, post, pre_size, post_size, weight, eta, hier, scale_var, mode=None, name=None):
    '''
    利用brainpy的bp.conn.IJConn和bp.dnn.EventCSRLinearWithEtaAndScale生成comm
    '''
    conn = ij_conn(pre, post, pre_size, post_size)
    return EventCSRLinearWithEtaAndScale(conn, weight, eta, hier, scale_var, mode=mode, name=name)


def _build_block_ids(n, block_slice_list):
    '''
    示例:
    block_slice_list = [slice(0,3), slice(3,7), slice(7,10)]
    block_ids = _build_block_ids(10, block_slice_list)
    block_ids = [0, 0, 0, 1, 1, 1, 1, 2, 2, 2]
    '''
    block_ids = np.zeros(n, dtype=int)
    for i, sl in enumerate(block_slice_list):
        block_ids[sl] = i
    return block_ids


class OneToOneWithBlockVar(bp.dnn.OneToOne):
    '''
    让分块的scale_var可以被训练,但是weight不可以被训练

    block_slice_list: 分块的索引列表,假设共k个
    scale_var: 形状为(k,)的变量,每个分块对应一个变量
    '''
    def __init__(self, num, weight, block_slice_list, scale_var, sharding=None, mode=None, name=None):
        bp._src.dnn.base.Layer.__init__(self, mode=mode, name=name)

        self.num = num
        self.sharding = sharding

        weight = init.parameter(weight, (self.num,), sharding=sharding)
        self.weight = weight
        self.scale_var = scale_var
        self.block_ids = bm.array(_build_block_ids(num, block_slice_list))

    def update(self, pre_val):
        return pre_val * self.weight * self.scale_var[self.block_ids]
# endregion


# region monitor related
def get_neuron_group_size(net, g):
    neuron_group_size = getattr(net, g).size
    return neuron_group_size


class MonitorMixin:
    '''
    注意,对于所有monitor相关的函数,如果真的遇到性能上的问题,还是建议直接进行调用
    '''
    def set_slice_dim(self, bm_mode=bm.nonbatching_mode, batch_size=None):
        if bm_mode == bm.nonbatching_mode:
            self.slice_dim = 0
        elif bm_mode == bm.training_mode:
            self.slice_dim = 1
        else:
            raise ValueError(f'Unknown bm_mode: {bm_mode}')
        self.batch_size = batch_size

    def _process_slice(self, neuron_idx):
        if self.slice_dim == 0:
            return neuron_idx
        elif self.slice_dim == 1:
            return (np.arange(self.batch_size), neuron_idx)
    
    def _process_slice_for_current(self, neuron_idx):
        if isinstance(neuron_idx, slice):
            neuron_idx = cf.slice_to_array(neuron_idx)
        if self.slice_dim == 0:
            return neuron_idx
        elif self.slice_dim == 1:
            return np.ix_(np.arange(self.batch_size), neuron_idx)

    def get_function_for_monitor_V(self, g, neuron_idx=None):
        if neuron_idx is None:
            def f():
                return getattr(self.net, g).V
        else:
            neuron_idx = self._process_slice(neuron_idx)
            def f():
                return getattr(self.net, g).V[neuron_idx]
        return f

    def get_function_for_monitor_V_mean(self, g, neuron_idx=None):
        if neuron_idx is None:
            def f():
                return np.mean(getattr(self.net, g).V)
        else:
            neuron_idx = self._process_slice(neuron_idx)
            def f():
                return np.mean(getattr(self.net, g).V[neuron_idx])
        return f

    def get_function_for_monitor_spike(self, g, neuron_idx=None):
        if neuron_idx is None:
            def f():
                return getattr(self.net, g).spike
        else:
            neuron_idx = self._process_slice(neuron_idx)
            def f():
                return getattr(self.net, g).spike[neuron_idx]
        return f

    def get_function_for_monitor_spike_mean(self, g, neuron_idx=None):
        if neuron_idx is None:
            def f():
                return np.mean(getattr(self.net, g).spike)
        else:
            neuron_idx = self._process_slice(neuron_idx)
            def f():
                return np.mean(getattr(self.net, g).spike[neuron_idx])
        return f

    def get_function_for_monitor_current(self, g, neuron_idx=None, label=None):
        '''
        注意,这里切片是切在得到的array上,根据jax的需求,slice是不可以的
        '''
        if neuron_idx is None:
            def f():
                return getattr(self.net, g).sum_current_inputs(getattr(self.net, g).V, label=label)
        else:
            neuron_idx = self._process_slice_for_current(neuron_idx)
            def f():
                return getattr(self.net, g).sum_current_inputs(getattr(self.net, g).V, label=label)[neuron_idx]
        return f

    def get_function_for_monitor_current_mean(self, g, neuron_idx=None, label=None):
        '''
        注意,这里切片是切在得到的array上,根据jax的需求,slice是不可以的
        '''
        if neuron_idx is None:
            def f():
                return np.mean(getattr(self.net, g).sum_current_inputs(getattr(self.net, g).V, label=label))
        else:
            neuron_idx = self._process_slice_for_current(neuron_idx)
            def f():
                return np.mean(getattr(self.net, g).sum_current_inputs(getattr(self.net, g).V, label=label)[neuron_idx])
        return f
# endregion


# region 神经元网络模型运行
class SNNSimulator(cf.MetaModel):
    '''
    支持分chunk运行,节约内存和显存
    使用方式:
    simulator = SNNSimulator()
    simulator.set_save_mode('chunk')
    simulator.set_chunk_interval(1000.)
    此时simulator.run_time_interval会自动使用run_time_interval_in_chunks
    '''
    def __init__(self):
        self.params = {}
        super().__init__()
        self.extend_ignore_key_list('chunk_interval')
        self.set_bm_mode()
        self.set_save_mode()
        self.total_chunk_num = 0

    # region set
    def set_up(self, basedir, code_file_list, value_dir_key_before=None, both_dir_key_before=None, value_dir_key_after=None, both_dir_key_after=None, ignore_key_list=None, force_run=None):
        super().set_up(params=self.params, basedir=basedir, code_file_list=code_file_list, value_dir_key_before=value_dir_key_before, both_dir_key_before=both_dir_key_before, value_dir_key_after=value_dir_key_after, both_dir_key_after=both_dir_key_after, ignore_key_list=ignore_key_list, force_run=force_run)

    def set_optional_params_default(self):
        '''
        设置一些不强制需要的参数的默认值
        '''
        super().set_optional_params_default()
        self.set_chunk_interval(None)

    def set_simulation_results(self):
        self.simulation_results = defaultdict(list)
        self.simulation_results_type = 'dict'

    def set_random_seed(self, bm_seed=421):
        '''
        设置随机种子(全局,可以的话还是不要使用)
        '''
        bm.random.seed(bm_seed)
        self.params['bm_seed'] = bm_seed

    def set_gpu(self, id, pre_allocate=None):
        '''
        实测发现,这个set_gpu的一系列操作要写在整个py比较靠前的部分,不能对某个模型设置
        '''
        raise ValueError('请不要使用set_gpu,而是将这段代码复制到py文件的最前面')
        super().set_gpu(id)
        bm.set_platform('gpu')
        if pre_allocate is True:
            bm.enable_gpu_memory_preallocation()
        elif pre_allocate is False:
            bm.disable_gpu_memory_preallocation()
        else:
            pass

    def set_cpu(self):
        bm.set_platform('cpu')

    def set_bm_mode(self, bm_mode=None):
        '''
        设置模式
        '''
        if bm_mode is None:
            bm_mode = bm.nonbatching_mode
        bm.set_mode(bm_mode)
        cf.print_title('Set to brainpy {} mode'.format(bm_mode))
        self.bm_mode = bm_mode

    def set_dt(self, dt):
        '''
        设置dt
        '''
        self.dt = dt
        self.params['dt'] = dt
        bm.set_dt(dt)

    def set_total_simulation_time(self, total_simulation_time):
        '''
        设置总的仿真时间
        '''
        self.total_simulation_time = total_simulation_time
        self.params['total_simulation_time'] = total_simulation_time

    def set_chunk_interval(self, chunk_interval):
        '''
        设置分段运行的时间间隔
        '''
        self.chunk_interval = chunk_interval
        self.params['chunk_interval'] = chunk_interval
    
    def set_save_mode(self, save_mode='all'):
        '''
        设置保存模式

        save_mode: 'all', 全部跑完之后储存; 'chunk', 分段储存
        '''
        self.save_mode = save_mode
    # endregion

    # region get
    @abc.abstractmethod
    def get_net(self):
        '''
        获取网络模型(子类需要实现,并且定义为self.net)
        '''
        self.net = None

    @abc.abstractmethod
    def get_monitors(self):
        '''
        获取监测器(子类需要实现,并且定义为self.monitors)
        '''
        self.monitors = None

    @abc.abstractmethod
    def get_runner(self):
        '''
        获取runner(子类需要实现,并且定义为self.runner)
        '''
        self.runner = None
    # endregion

    # region run
    def initialize_model(self):
        self.get_net()
        self.get_monitors()
        self.get_runner()
        super().initialize_model()

    def update_simulation_results_from_runner(self):
        '''
        更新直接结果
        '''
        for k, v in self.runner.mon.items():
            self.simulation_results[k].append(v)

    def organize_simulation_results(self):
        '''
        整理直接结果
        '''
        for k in self.simulation_results.keys():
            self.simulation_results[k] = np.concatenate(self.simulation_results[k], axis=0)

    def organize_chunk_simulation_results(self):
        '''
        把结果重新读取整合好,再保存
        '''
        # 读取第一个chunk的metadata,获取所有的key
        chunk_idx = 0
        metadata = cf.load_pkl(cf.pj(self.outcomes_dir, f'chunk_{chunk_idx}_simulation_results', 'metadata'))

        # 对每个key,读取所有chunk的结果,并且拼接,保存(这种方式比直接读取所有chunk的结果节约内存)
        for dict_k, file_k in metadata.items():
            self.set_simulation_results()
            for chunk_idx in range(self.total_chunk_num):
                part_simulation_results = cf.load_dict_separate(cf.pj(self.outcomes_dir, f'chunk_{chunk_idx}_simulation_results'), key_to_load=[file_k])[dict_k]
                self.simulation_results[dict_k].append(part_simulation_results)
            self.simulation_results[dict_k] = np.concatenate(self.simulation_results[dict_k], axis=0)
            self.save_simulation_results()

        # 删除子chunk的所有文件夹
        for chunk_idx in range(self.total_chunk_num):
            cf.rmdir(cf.pj(self.outcomes_dir, f'chunk_{chunk_idx}_simulation_results'))

    def clear_runner_mon(self):
        '''
        清空runner的监测器
        '''
        if hasattr(self, 'runner'):
            self.runner.mon = None
            self.runner._monitors = None

    def finalize_run_detail(self):
        '''
        运行结束后,整理结果
        '''
        self.organize_simulation_results()
        self.clear_runner_mon()
        self.save_simulation_results()
        bm.clear_buffer_memory()

        if self.save_mode == 'chunk' and (not self.simulation_results_exist):
            self.organize_chunk_simulation_results()

    def log_during_run(self):
        '''
        运行过程中,打印日志
        '''
        pass

    def basic_run_time_interval(self, time_interval):
        '''
        运行模型,并且保存结果
        '''
        self.runner.run(time_interval)
        self.update_simulation_results_from_runner()
        self.log_during_run()

    def finalize_each_chunk(self, chunk_idx):
        if self.save_mode == 'chunk':
            self.organize_simulation_results()
            self.save_simulation_results(filename=f'chunk_{chunk_idx}_simulation_results')
            self.set_simulation_results()

    def run_time_interval_in_chunks(self, time_interval):
        '''
        分段运行模型,以防止内存溢出
        '''
        chunk_num = int(time_interval / self.chunk_interval) + 1
        remaining_time = time_interval
        for chunk_idx in range(chunk_num):
            run_this_chunk = False

            if self.chunk_interval <= remaining_time:
                self.basic_run_time_interval(self.chunk_interval)
                remaining_time -= self.chunk_interval
                self.total_chunk_num += 1
                run_this_chunk = True
            elif remaining_time > 0:
                self.basic_run_time_interval(remaining_time)
                remaining_time = 0
                self.total_chunk_num += 1
                run_this_chunk = True

            if run_this_chunk: # 有可能没有运行
                # 注意这里第一次跑完total_chunk_num=1,所以chunk_idx这边要-1
                self.finalize_each_chunk(chunk_idx=self.total_chunk_num-1)

    def run_time_interval(self, time_interval):
        '''
        运行模型,并且保存结果(自动选择分段运行还是直接运行)
        '''
        if self.chunk_interval is not None:
            self.run_time_interval_in_chunks(time_interval)
        else:
            self.basic_run_time_interval(time_interval)

    def run_detail(self):
        '''
        运行模型,并且保存结果

        注意: 
        当子类有多个阶段,需要重写此方法
        当内存紧张的时候,可以调用run_time_interval,分段运行
        '''
        self.run_time_interval(self.total_simulation_time)
    # endregion


class ComposedSNNSimulator(cf.Simulator):
    '''
    不允许子类修改data_keeper为OrderedDataContainer,因为monitors一定是dict,而结果是按照monitors的key来保存的
    '''
    def __init__(self):
        super().__init__()
        self.total_chunk_num = 0
        self.bm_mode = bm.nonbatching_mode

    def _set_required_key_list(self):
        super()._set_required_key_list()
        self.required_key_list.extend(['total_simulation_time'])

    def _set_optional_key_value_dict(self):
        super()._set_optional_key_value_dict()
        self.optional_key_value_dict.update({
            'dt': 0.1, # ms
            'chunk_interval': None,
        })

    def _config_dir_manager(self):
        super()._config_dir_manager()
        self.dir_manager_kwargs['ignore_key_list'].extend(['chunk_interval'])

    def _set_name(self):
        self.name = 'snn_simulator'

    @abc.abstractmethod
    def _create_net(self):
        self.net = None
    
    @abc.abstractmethod
    def _create_monitors(self):
        self.monitors = None
    
    @abc.abstractmethod
    def _create_runner(self):
        self.runner = bp.DSRunner(self.net, monitors=self.monitors)

    def _update_results_from_runner(self, chunk_idx=None):
        for k, v in self.runner.mon.items():
            if chunk_idx is None:
                processed_k = k
            else:
                processed_k = (chunk_idx, k)  # 添加chunk_idx到key中,以便区分不同chunk的结果
            if processed_k not in self.data_keeper.data:
                self.data_keeper.data[processed_k] = []
            self.data_keeper.data[processed_k].append(v)

    def _organize_results(self):
        for k in self.data_keeper.data.keys():
            self.data_keeper.data[k] = np.concatenate(self.data_keeper.data[k], axis=0)

    def _organize_chunk_results(self):
        '''
        把结果重新读取整合好,再保存
        '''
        key_list = list(self.monitors.keys()).copy()
        key_list.append('ts')
        for k in key_list:
            for chunk_idx in range(self.total_chunk_num):
                part_results = self.data_keeper.get_value(key=(chunk_idx, k))
                if k not in self.data_keeper.data:
                    self.data_keeper.data[k] = []
                self.data_keeper.data[k].append(part_results)
                self.data_keeper.delete(key_to_delete=[(chunk_idx, k)])  # 删除每个chunk在硬盘上的结果,只保留整合后的结果
                del self.data_keeper.data[(chunk_idx, k)]
            self.data_keeper.data[k] = np.concatenate(self.data_keeper.data[k], axis=0)
            self.data_keeper.save()
            self.data_keeper.release_memory()  # 释放内存

    def _clear_runner_mon(self):
        if hasattr(self, 'runner'):
            self.runner.mon = None
            self.runner._monitors = None

    def _log_during_run(self):
        pass

    def _basic_run_time_interval(self, time_interval, chunk_idx=None):
        self.runner.run(time_interval)
        self._update_results_from_runner(chunk_idx=chunk_idx)
        self._log_during_run()

    def _finalize_each_chunk(self):
        if self.chunk_interval is not None:
            self._organize_results() # 这一步是因为每次都是append进去的,哪怕是chunk也需要先整理
            self.data_keeper.save()
            self.data_keeper.release_memory() # 如果分段运行,一定是需要释放内存的

    def _run_time_interval_in_chunks(self, time_interval):
        '''
        注意chunk_idx在输入到其他函数时,要使用self.total_chunk_num,防止多次调用此method时,chunk_idx从0开始
        '''
        chunk_num = int(time_interval / self.chunk_interval) + 1
        remaining_time = time_interval
        for chunk_idx in range(chunk_num):
            if remaining_time > 0:
                current_chunk_duration = min(self.chunk_interval, remaining_time)
                
                self._basic_run_time_interval(current_chunk_duration, chunk_idx=self.total_chunk_num)
                
                self._finalize_each_chunk()
                
                remaining_time -= current_chunk_duration
                self.total_chunk_num += 1

    def _run_time_interval(self, time_interval):
        '''
        运行模型,并且保存结果(自动选择分段运行还是直接运行)
        '''
        if self.chunk_interval is not None:
            self._run_time_interval_in_chunks(time_interval)
        else:
            self._basic_run_time_interval(time_interval)

    def before_run(self):
        super().before_run()
        bm.set_dt(self.dt)
        bm.set_mode(self.bm_mode)
        self._create_net()
        self._create_monitors()
        self._create_runner()

    def run_detail(self):
        '''
        运行模型,并且保存结果

        注意: 
        当子类有多个阶段,需要重写此方法
        当内存紧张的时候,可以调用run_time_interval,分段运行
        '''
        self._run_time_interval(self.total_simulation_time)

    def after_run(self):
        self._organize_results()
        self._clear_runner_mon()
        super().after_run()
        bm.clear_buffer_memory()

        if self.chunk_interval is not None:
            self._organize_chunk_results()


class ComposedSNNAnalyzer(cf.Analyzer):
    def _set_name(self):
        self.name = 'snn_analyzer'


def custom_bp_running_cpu_parallel(func, params_list, num_process=10, mode='ordered'):
    '''
    参数:
    func: 需要并行计算的函数
    params_list: 需要传入的参数列表,例如[(a0, b0), (a1, b1), ...]
    num_process: 进程数
    mode: 运行模式,ordered表示有序运行,unordered表示无序运行

    注意:
    jupyter中使用时,func需要重新import,所以不建议在jupyter中使用
    实际使用时发现,改变import的模块,新的函数会被实时更新(比如在运行过程中修改了common_functions,那么对于还没运行的函数,会使用新的common_functions)
    '''
    bm.set_platform('cpu')
    total_num = len(params_list)
    
    # 将参数列表转换为分块结构(实测这样相比直接运行可以防止mem累积)
    for chunk_idx in range((total_num + num_process - 1) // num_process):
        # 计算当前分片的起止索引
        start_idx = chunk_idx * num_process
        end_idx = min((chunk_idx + 1) * num_process, total_num)
        local_num_process = end_idx - start_idx
        
        # 提取当前分片的参数并转换结构
        chunk_params = params_list[start_idx:end_idx]
        transposed_params = [list(param_chunk) for param_chunk in zip(*chunk_params)]
        
        # 打印调试信息
        cf.print_title(f"Processing chunk {chunk_idx}: [{start_idx}-{end_idx})")
        
        # 执行并行计算
        if mode == 'ordered':
            bp.running.cpu_ordered_parallel(func, transposed_params, num_process=local_num_process)
        else:
            bp.running.cpu_unordered_parallel(func, transposed_params, num_process=local_num_process)
# endregion


# region loss
def bm_ks(points1, points2):
    """
    20240221,Chen Xiaoyu,convert to BrainPy(JAX) style Based on https://github.com/mikesha2/kolmogorov_smirnov_torch/blob/main/ks_test.py
    
    Kolmogorov-Smirnov test for empirical similarity of probability distributions.
    
    Warning: we assume that none of the elements of points1 coincide with points2. 
    The test may gave false negatives if there are coincidences, however the effect
    is small.

    Parameters
    ----------
    points1 : (n1,) 1D array
        Batched set of samples from the first distribution
    points2 : (n2,) 1D array
        Batched set of samples from the second distribution
    """
    n1 = points1.shape[-1]
    n2 = points2.shape[-1]

    comb = bm.concatenate((points1, points2), axis=-1)
    comb_argsort = bm.argsort(comb,axis=-1)

    pdf1 = bm.where(comb_argsort <  n1, 1 / n1, 0)
    pdf2 = bm.where(comb_argsort >= n1, 1 / n2, 0)

    cdf1 = pdf1.cumsum(axis=-1)
    cdf2 = pdf2.cumsum(axis=-1)

    return (cdf1 - cdf2).abs().max(axis=-1)


def bm_spike_mean_to_fr(spike_mean, dt, width):
    '''
    spike_mean: shape=(time_steps, )
    '''
    width1 = int(width / 2 / dt) * 2 + 1
    window = bm.ones(width1) * 1000 / width
    return bm.convolve(spike_mean, window, mode='same')


def bm_spike_mean_to_fr_timescale(spike_mean, dt, width, nlags_list):
    fr = bm_spike_mean_to_fr(spike_mean, dt, width)
    # 掐头去尾,防止没有到稳态和边界效应
    fr = fr[int(fr.shape[0] / 4):int(fr.shape[0] * 3 / 4)]
    timescale = bm_timescale_by_multi_step_estimation(fr, nlags_list, dt)
    return timescale


def bm_multi_spike_mean_to_fr_timescale(multi_spike_mean, dt, width, nlags_list):
    '''
    multi_spike_mean: shape=(time_steps, num)
    '''
    _f = jax.vmap(bm_spike_mean_to_fr_timescale, in_axes=(0, None, None, None))
    result = _f(multi_spike_mean, dt, width, nlags_list)
    return result


def bm_spike_to_mean_freqs(spike, dt):
    '''
    在神经元维度和时间维度都进行平均,计算出频率
    '''
    return bm.mean(spike) * 1000 / dt


def bm_spike_mean_to_freqs(spike_mean, dt):
    '''
    输入一个neuron group的spike_mean,根据dt计算出对应的频率,关键是单位的转换

    spike_mean: shape=(time_steps, 1)
    dt: float, 时间步长,单位为ms

    return: shape=(time_steps, 1),单位为Hz
    '''
    return spike_mean / dt * 1000.  # 转换为Hz


def bm_spike_mean_to_mean_freqs(spike_mean, dt):
    return bm.mean(bm_spike_mean_to_freqs(spike_mean, dt))


def bm_spike_mean_to_median_freqs(spike_mean, dt):
    '''
    在时间维度上取中位数,可以更加稳健,有助于排除没到达稳态的影响
    '''
    return bm.median(bm_spike_mean_to_freqs(spike_mean, dt))


def bm_abs_loss(x, y):
    return bm.abs(x - y)


def bm_corr(x, y):
    shifted_x = x - bm.mean(x)
    shifted_y = y - bm.mean(y)
    return bm.sum(shifted_x * shifted_y) / bm.sqrt(bm.sum(shifted_x * shifted_x) * bm.sum(shifted_y * shifted_y))


def bm_delay_corr_single_step(series_to_delay, series_to_advance, nlags):
    """
    计算延迟相关系数

    series_to_delay: 需要被延迟的序列,shape = (n, )
    series_to_advance: 需要被提前的序列,shape = (n, )
    nlags: 延迟步数

    注意:
    series_to_delay 将被延迟 nlags 步
    series_to_advance 将被提前 nlags 步

    只返回nlags步的相关系数
    """
    n = series_to_delay.shape[0]
    delayed = series_to_delay[nlags:n]
    advanced = series_to_advance[0:n-nlags]
    return bm_corr(delayed, advanced)


def bm_delay_corr_multi_step(series_to_delay, series_to_advance, nlags):
    """
    计算多延迟步数的相关系数（向量化版本）

    series_to_delay: 需要被延迟的序列, shape = (n, )
    series_to_advance: 需要被提前的序列, shape = (n, )
    nlags: 延迟步数(整数)

    返回: 各延迟步数对应的相关系数,shape = (nlags, )
    """
    # 向量化映射：对每个延迟步数并行计算
    nlags_array = bm.arange(nlags + 1)
    results = []
    for lag in nlags_array:
        results.append(bm_delay_corr_single_step(series_to_delay, series_to_advance, lag))
    return bm.stack(results, axis=0)


def bm_delay_corr_single_step_multi_series(multi_series_to_delay, multi_series_to_advance, nlags):
    """
    输入: shape = (n, k),沿列(k维)vmap
    输出: shape = (k,)
    """
    _f = jax.vmap(bm_delay_corr_single_step, in_axes=(1, 1, None))
    return _f(multi_series_to_delay, multi_series_to_advance, nlags)


def bm_delay_corr_multi_step_multi_series(multi_series_to_delay, multi_series_to_advance, nlags):
    """
    输入: shape = (n, k)
    输出: shape = (nlags+1, k)
    """
    _f = jax.vmap(bm_delay_corr_multi_step, in_axes=(1, 1, None))
    result = _f(multi_series_to_delay, multi_series_to_advance, nlags)
    return bm.transpose(result, (1, 0))


def bm_timescale_by_area_under_acf(timeseries, nlags, dt):
    """
    计算时间序列的timescale
    timeseries: shape = (n, ) 或者 (n, k), 其中k为batch
    nlags: 延迟的范围

    由于时间只有积分到无穷的时候,acf下的面积才会是tau,所以使用此函数需要较大的nlags(至少要让nlags*dt大于3*tau)
    """
    local_timeseries = timeseries.reshape(timeseries.shape[0], -1)

    # 计算延迟相关系数
    delay_corr = bm_delay_corr_multi_step_multi_series(local_timeseries, local_timeseries, nlags)
    delay_corr = bm.mean(delay_corr, axis=0)
    
    # 计算时间尺度
    timescale = bm.mean(delay_corr) * local_timeseries.shape[0] * dt
    
    return timescale


def bm_timescale_by_single_step_estimation(timeseries, nlags, dt):
    '''
    只利用nlags处的延迟相关系数来估计时间尺度
    acf = exp(-t/tau)
    acf(nlags*dt) = exp(-nlags*dt/tau)
    tau = - nlags*dt / log(acf(nlags*dt))
    '''
    local_timeseries = timeseries.reshape(timeseries.shape[0], -1)

    # 计算延迟相关系数
    delay_corr = bm_delay_corr_single_step_multi_series(local_timeseries, local_timeseries, nlags)
    delay_corr = bm.mean(delay_corr)

    # 计算时间尺度
    timescale = - nlags * dt / bm.log(bm.abs(delay_corr)) # 加abs防止对负数取对数

    return timescale


def bm_timescale_by_multi_step_estimation(timeseries, nlags_list, dt):
    '''
    利用nlags_list处的延迟相关系数来估计时间尺度
    '''
    results = bm.zeros((len(nlags_list),))
    for i, nlags in enumerate(nlags_list):
        results[i] = bm_timescale_by_single_step_estimation(timeseries, nlags, dt)
    return bm.nanmedian(results) # log有可能导致nan
# endregion


# region 神经元网络模型训练
class SNNTrainer(SNNSimulator):
    '''
    注意事项:

    要训练的参数要设置成bm.TrainVar
    可以单独调用test_f_loss来建议f_loss_with_detail的正确性
    如果要共享某个TrainVar,最好在外面定义完之后直接输入到comm等的内部,并且要注意内部没有对其做额外操作
    TrainVar在外部四则运算后会转化为普通的Array,所有运算最好放到comm等的内部
    '''
    def __init__(self):
        super().__init__()
        self.current_epoch_bm = bm.Variable(bm.zeros((1, ), dtype=int))
        self.t0 = bm.Variable(bm.zeros((1, )))
    
    def set_simulation_results(self):
        self.simulation_results = defaultdict(dict)
        self.simulation_results_type = 'dict'

    def set_bm_mode(self, bm_mode=None):
        '''
        设置模式
        '''
        if bm_mode is None:
            bm_mode = bm.training_mode # 注意: 设置成training_mode之后,会产生一个batch的维度,比如说spike会变成(batch_size, time_steps, neuron_num)
        bm.set_mode(bm_mode)
        cf.print_title('Set to brainpy {} mode'.format(bm_mode))
        self.bm_mode = bm_mode

    def set_epoch(self, epoch):
        '''
        设置epoch
        '''
        self.epoch = epoch
        self.params['epoch'] = epoch

    def set_loss_tolerance(self, loss_tolerance):
        '''
        设置损失函数的容忍度(到达这个值就停止训练)
        '''
        self.loss_tolerance = loss_tolerance
        self.params['loss_tolerance'] = loss_tolerance

    def set_log_interval_epoch(self, log_interval_epoch=1):
        '''
        每log_interval_epoch记录一次
        '''
        self.log_interval_epoch = log_interval_epoch

    def set_optional_params_default(self):
        super().set_optional_params_default()
        self.set_log_interval_epoch(1)
        self.set_loss_tolerance(None)
    
    @property
    def train_vars(self):
        '''
        获取训练变量
        '''
        return self.net.train_vars().unique()

    def get_runner(self):
        pass

    @abc.abstractmethod
    def get_opt(self):
        '''
        获取优化器(子类需要实现,并且定义为self.opt)

        例如:
        lr = bp.optim.ExponentialDecayLR(lr=0.025, decay_steps=1, decay_rate=0.99975)
        self.opt = bp.optim.Adam(lr=lr, train_vars=self.train_vars)
        '''
        self.opt = None
    
    def get_f_grad(self):
        '''
        获取损失函数的梯度
        '''
        self.f_grad = bm.grad(self.f_loss_with_detail, grad_vars=self.train_vars, return_value=True, has_aux=True)

    def set_f_loss(self, batch_size, time_step, inputs=None):
        '''
        设置损失函数的信息,比如说f_loss要对比的inputs值,target值,batch_size,timestep等
        '''
        self.batch_size = batch_size
        self.params['batch_size'] = batch_size
        self.time_step = time_step
        self.params['time_step'] = time_step
        if inputs is None:
            self.inputs = bm.ones((self.batch_size, self.time_step, 1))
        else:
            self.inputs = inputs

    @property
    def warmup_monitors(self):
        '''
        将其设置为property,因为设置warmup_monitor_mode后,也许没有self.monitors,等到调用的时候再设置
        '''
        if self.warmup_monitor_mode == 'same':
            return self.monitors
        elif self.warmup_monitor_mode is None:
            return {}
        else:
            raise ValueError(f'Unknown warmup_monitor_mode: {self.warmup_monitor_mode}')

    def _process_warmup_step_list(self, warmup_step_list):
        '''
        可能输入的不是list,那么自动广播到epoch的长度
        '''
        if isinstance(warmup_step_list, int):
            warmup_step_list = [warmup_step_list] * self.epoch
        elif isinstance(warmup_step_list, list):
            if len(warmup_step_list) != self.epoch:
                raise ValueError(f'len(warmup_step_list)={len(warmup_step_list)} != epoch={self.epoch}')
        else:
            raise ValueError(f'Unknown warmup_step_list: {warmup_step_list}')
        return warmup_step_list

    def set_warmup(self, warmup_step_list, warmup_inputs_list=None, warmup_dt=None, warmup_monitor_mode=None):
        '''
        在训练之前跑一段时间

        warmup_monitor_mode: 
            None: 不监测
            'same': 监测和训练时一样的监测器(即self.monitors)
        '''
        self.warmup_step_list = self._process_warmup_step_list(warmup_step_list)
        self.params['warmup_step_list'] = self.warmup_step_list

        if warmup_inputs_list is None:
            self.warmup_inputs_list = []
            for warmup_step in self.warmup_step_list:
                self.warmup_inputs_list.append(bm.ones((self.batch_size, warmup_step, 1)))
        else:
            self.warmup_inputs_list = warmup_inputs_list

        if warmup_dt is None:
            self.warmup_dt = self.dt
        else:
            self.warmup_dt = warmup_dt
        self.params['warmup_dt'] = self.warmup_dt

        self.warmup_monitor_mode = warmup_monitor_mode
        self.params['warmup_monitor_mode'] = warmup_monitor_mode

    @abc.abstractmethod
    def f_loss_with_detail(self):
        '''
        输出为loss和其余详细信息(其余信息只占一个位置,如果有很多信息可以搞成元组一起输出)(如果不输出其他信息,用None占位)

        示例:训练网络到达指定的firing rate
        self.net.reset(self.batch_size)
        runner = bp.DSTrainer(self.net, progress_bar=False, numpy_mon_after_run=False, monitors=self.monitors)
        runner.predict(self.inputs, reset_state=False)
        output = runner.mon['E_spike']
        mean_fr_predict = bm.mean(output) * 1000 / bm.get_dt()
        return bm.square(mean_fr_predict - self.fr_target), mean_fr_predict
        '''
        pass

    @bm.cls_jit
    def f_train(self):
        grads, loss, detail = self.f_grad()
        self.opt.update(grads)
        return grads, loss, detail
    
    def initialize_model(self):
        self.get_net()
        self.get_monitors()
        self.get_opt()
        self.get_f_grad()
        cf.MetaModel.initialize_model(self)
        cf.bprt(self.train_vars, 'train_vars')

    def test_f_loss(self):
        '''
        单独调用f_loss来建议f_loss的正确性
        '''
        self.initialize_model()
        cf.print_title('test f_loss')
        print(self.f_loss_with_detail())

    def finalize_run_detail(self):
        bm.clear_buffer_memory()

    def _log_during_train_for_detail(self):
        self.simulation_results['detail'].append(self.current_detail)
        cf.better_print(f'detail: {self.current_detail}')
        self.logger.py_logger.info(f'detail: {self.current_detail}')

    def log_during_train(self):
        '''
        记录训练过程中的数据,可以在子类重写,并append到self.simulation_results对应的key中
        '''
        self.simulation_results[f'epoch_{self.current_epoch}']['epoch'] = self.current_epoch
        self.simulation_results[f'epoch_{self.current_epoch}']['loss'] = self.current_loss
        np_current_grads = {}
        for k, v in self.current_grads.items():
            np_current_grads[k] = np.array(v)
        self.simulation_results[f'epoch_{self.current_epoch}']['grads'] = np_current_grads
        cf.better_print(f'epoch: {self.current_epoch}')
        cf.better_print(f'loss: {self.current_loss}')
        self.logger.py_logger.info(f'epoch: {self.current_epoch}')
        self.logger.py_logger.info(f'loss: {self.current_loss}')
        self.logger.py_logger.info(f'grads: {self.current_grads}')
        self._log_during_train_for_detail()

        processed_wramup_mon = dict_to_np_dict(self.runner_warmup.mon)
        self.simulation_results[f'epoch_{self.current_epoch}']['warmup_mon'] = processed_wramup_mon

    def run_warmup(self):
        cf.print_title(f'Warmup with {self.warmup_step_list[self.current_epoch_bm.value[0]]} steps')
        self.runner_warmup = bp.DSTrainer(self.net, progress_bar=True, numpy_mon_after_run=False, monitors=self.warmup_monitors, dt=self.warmup_dt, t0=self.t0.value)
        self.runner_warmup.predict(self.warmup_inputs_list[self.current_epoch_bm.value[0]], reset_state=False)
        self.t0.value += self.warmup_step_list[self.current_epoch_bm.value[0]] * self.warmup_dt

    def before_f_train(self):
        '''
        每个epoch训练前的操作,可以在子类重写
        '''
        self.net.reset(self.batch_size)
        self.run_warmup()

    def after_f_train(self):
        '''
        每个epoch训练完成后的操作,可以在子类重写

        例子:
        训练SNN的权重,可以强制在每次train后调整权重的符号,利用bm.abs和-bm.abs来选择符号
        '''
        pass

    def run_detail(self):
        '''
        仍然叫做run_detail,但是实际上是训练,可以在子类重写
        '''
        for self.current_epoch in tqdm(range(self.epoch)):
            self.before_f_train()
            self.current_grads, self.current_loss, self.current_detail = self.f_train()
            self.after_f_train()
            if self.current_epoch % self.log_interval_epoch == 0:
                self.log_during_train()
                self.save_simulation_results(key_to_save=[f'epoch_{self.current_epoch}'], max_depth=2)
            if self.loss_tolerance is not None and self.current_loss < self.loss_tolerance:
                print(f'Loss tolerance reached: {self.current_loss} < {self.loss_tolerance}')
                break
            self.current_epoch_bm.value += 1


class SNNFitter(SNNTrainer):
    '''
    专门用来让SNN的某种性质达到目标值,不需要输入

    支持连续模式,即在每个epoch结束后,将当前epoch的结果传入下一个epoch
    '''
    def set_continuous_mode(self, continuous_mode=True):
        '''
        设置连续模式,如果设置为True,那么在每个epoch结束后,会将当前epoch的结果传入下一个epoch
        '''
        self.continuous_mode = continuous_mode
        self.params['continuous_mode'] = continuous_mode

    def set_optional_params_default(self):
        super().set_optional_params_default()
        self.set_continuous_mode(True)

    def set_f_loss(self, batch_size, time_step, single_loss_config_list, multi_loss_config_list=None):
        '''
        single_loss_config_list: list of dict,每个字典包含单个monitor的损失配置
            Required keys: 'monitor_key', 'target_value'
            Optional keys: 'loss_coef', 'loss_func', 'transform_func'
        multi_loss_config_list: list of dict,每个字典包含多个monitor联合的损失配置
            Required keys: 'monitor_key' (tuple), 'target_value', 'transform_func'
            Optional keys: 'loss_coef', 'loss_func'

        monitor_key: str, 监测器的key, 需要在self.monitors中定义
        target_value: float, 目标值
        loss_coef: float, 乘在每个term的loss上, 默认为1.0
        loss_func: function, 二元, 第一个参数是预测值, 第二个参数是目标值, 默认是bm_abs_loss
        transform_func: function, 将runner.mon[monitor_key]转换为目标值的函数, 当multi时, 要注意顺序和monitor_key的顺序匹配, 当single时, 默认是cf.identical_func
        '''
        super().set_f_loss(batch_size, time_step)

        if multi_loss_config_list is None:
            multi_loss_config_list = []

        # 处理单变量损失配置
        single_defaults = {
            'loss_coef': 1.0,
            'loss_func': bm_abs_loss,
            'transform_func': cf.identical_func
        }
        self.single_loss_terms = []
        for config in single_loss_config_list:
            validated = cf.update_dict(single_defaults, config)
            self.single_loss_terms.append(validated)
        self.single_loss_config_list = single_loss_config_list

        # 处理多变量联合损失配置
        multi_defaults = {
            'loss_coef': 1.0,
            'loss_func': bm_abs_loss
        }
        self.multi_loss_terms = []
        for config in multi_loss_config_list:
            validated = cf.update_dict(multi_defaults, config)
            self.multi_loss_terms.append(validated)
        self.multi_loss_config_list = multi_loss_config_list

    def f_loss_with_detail(self):
        runner = bp.DSTrainer(self.net, progress_bar=False, numpy_mon_after_run=False, monitors=self.monitors, t0=self.t0, dt=self.dt)
        runner.predict(self.inputs, reset_state=False)
        self.t0.value += self.time_step * self.dt
        loss = 0.
        single_loss_terms_detail = []
        multi_loss_terms_detail = []

        for term in self.single_loss_terms:
            monitor_key = term['monitor_key']
            target_value = term['target_value']
            loss_coef = term['loss_coef']
            loss_func = term['loss_func']
            transform_func = term['transform_func']

            output = runner.mon[monitor_key]
            transformed_output = transform_func(output)
            loss += loss_coef * loss_func(transformed_output, target_value)
            single_loss_terms_detail.append({
                'target_value': target_value,
                'loss_coef': loss_coef,
                'transformed_output': transformed_output
            })

        for term in self.multi_loss_terms:
            monitor_key = term['monitor_key']
            target_value = term['target_value']
            loss_coef = term['loss_coef']
            loss_func = term['loss_func']
            transform_func = term['transform_func']
            
            output = [runner.mon[k] for k in monitor_key]
            transformed_output = transform_func(*output)
            loss += loss_coef * loss_func(transformed_output, target_value)
            multi_loss_terms_detail.append({
                'target_value': target_value,
                'loss_coef': loss_coef,
                'transformed_output': transformed_output
            })

        return loss, (single_loss_terms_detail, multi_loss_terms_detail, runner.mon)
    
    def inherit_previous_state_when_continuous_mode(self):
        '''
        当连续模式下,将上一个epoch的状态传入下一个epoch
        '''
        if self.current_epoch > 0 and self.continuous_mode:
            assign_state(self.net, self.state)

    def before_f_train(self):
        '''
        每个epoch训练前的操作,可以在子类重写
        '''
        self.net.reset(self.batch_size)
        self.inherit_previous_state_when_continuous_mode()
        self.run_warmup()

    def log_state(self):
        self.state = extract_state(self.net)
        cf.save_dict(self.state, cf.pj(self.outcomes_dir, 'state', f'epoch_{self.current_epoch}'))

    def after_f_train(self):
        super().after_f_train()
        self.log_state()

    def _process_detail(self, detail_part, config_list, loss_type):
        '''
        将f_loss_with_detail的输出整理,将config_list中的monitor_key加入其中,并且将bp数组转换为np数组
        '''
        for i, (detail, config) in enumerate(zip(detail_part, config_list)):
            new_term = dict_to_np_dict(detail)
            new_term['monitor_key'] = config['monitor_key']
            self.simulation_results[f'epoch_{self.current_epoch}'][f'{loss_type}_term_{i}'] = new_term
            self.logger.py_logger.info(f'{loss_type}_term_{i}: {new_term}')
            cf.better_print(f'{loss_type}_term_{i}: {new_term}')

    def _log_during_train_for_detail(self):
        self._process_detail(self.current_detail[0], self.single_loss_config_list, 'single')
        self._process_detail(self.current_detail[1], self.multi_loss_config_list, 'multi')
        processed_mon = dict_to_np_dict(self.current_detail[2])
        self.simulation_results[f'epoch_{self.current_epoch}']['mon'] = processed_mon
# endregion


# region 神经元网络模型
class MultiNet(bp.DynSysGroup):
    def __init__(self, neuron, synapse, inp_neuron, inp_synapse, neuron_params, synapse_params, inp_neuron_params, inp_synapse_params, comm, inp_comm, print_info=True, clear_name_cache=True):
        """
        参数:
            neuron (dict): 包含每个组的神经元类型的字典
            synapse (dict): 包含每个连接的突触类型的字典(key必须是(s, t, name)的元组,如果不需要name,则name设置为None或者空字符串)
            inp_neuron (dict): 包含输入神经元类型的字典
            inp_synapse (dict): 包含输入突触类型的字典
            neuron_params (dict): 包含神经元初始化参数的字典
            synapse_params (dict): 包含突触初始化参数的字典
            inp_neuron_params (dict): 包含输入神经元初始化参数的字典
            inp_synapse_params (dict): 包含输入突触初始化参数的字典
            comm (dict): 包含组之间通信参数的字典
            inp_comm (dict): 包含输入组和其他组通信参数的字典
            print_info (bool): 是否打印信息
            
        建议:
            当需要很多group的时候,不一定要每个group一个neuron的key

            可以先考虑给不同group设定不同的参数来解决,这样更加高效
            tau_ref_1 = np.ones(ne_1) * 2
            tau_ref_2 = np.ones(ne_2) * 3
            tau_ref = np.concatenate([tau_ref_1, tau_ref_2])
            self.E = bp.dyn.LifRef(ne_1+ne_2, V_rest=-70., V_th=-50., V_reset=-60., tau=20., tau_ref=tau_ref,
                                V_initializer=bp.init.Normal(-55., 2.))
        
        注意:
            不要往self里面放多余的东西,只放neuron,synapse,inp_neuron,inp_synapse
            多放东西会直接报错,比如把comm放进去
        """
        super().__init__()
        if clear_name_cache:
            self.clear_name_cache()

        self.group = neuron.keys()
        self.inp_group = inp_neuron.keys()
        self.synapse_group = synapse.keys()
        self.inp_synapse_group = inp_synapse.keys()

        for g in self.group:
            setattr(self, g, neuron[g](**neuron_params[g]))

        for inp_g in self.inp_group:
            setattr(self, inp_g, inp_neuron[inp_g](**inp_neuron_params[inp_g]))
        
        for syn_type in synapse.keys():
            s, t, name = syn_type
            if print_info:
                cf.print_title(f'{s}2{t} {name} synapse')
            setattr(self, cf.concat_str([f'{s}2{t}', name]), synapse[syn_type](pre=getattr(self, s), post=getattr(self, t), comm=comm[syn_type], **synapse_params[syn_type]))

        for inp_syn_type in inp_synapse.keys():
            s, t, name = inp_syn_type
            if print_info:
                cf.print_title(f'{s}2{t} {name} synapse')
            setattr(self, cf.concat_str([f'{s}2{t}', name]), inp_synapse[inp_syn_type](pre=getattr(self, s), post=getattr(self, t), comm=inp_comm[inp_syn_type], **inp_synapse_params[inp_syn_type]))

    def clear_name_cache(self):
        '''
        2025_5_7,这样做了之后可以让Lif等对象的名字可以重复
        '''
        bp.math.clear_name_cache()

    def update(self, *args, **kwargs):
        '''
        2025_4_17,为了让其可以接收输入用于训练(在我的框架下,输入是虚假的,只是让bp来确定batch等,所以这里没有输入给原先的update)
        '''
        super().update()
    
    def reset_state(self, batch_size=1):
        '''
        2025_4_17,为了训练需要增添的
        '''
        for g in self.group:
            getattr(self, g).reset_state(batch_size)
        for inp_g in self.inp_group:
            getattr(self, inp_g).reset_state(batch_size)
        for syn_type in self.synapse_group:
            s, t, name = syn_type
            getattr(self, cf.concat_str([f'{s}2{t}', name])).reset_state(batch_size)
        for inp_syn_type in self.inp_synapse_group:
            s, t, name = inp_syn_type
            getattr(self, cf.concat_str([f'{s}2{t}', name])).reset_state(batch_size)


class SNNNetMixin:
    def get_neuron(self):
        pass

    def get_synapse(self):
        pass

    def get_inp_neuron(self):
        pass

    def get_inp_synapse(self):
        pass

    def get_neuron_params(self):
        pass

    def get_synapse_params(self):
        pass

    def get_inp_neuron_params(self):
        pass

    def get_inp_synapse_params(self):
        pass

    def get_comm(self):
        pass

    def get_inp_comm(self):
        pass

    def before_get_net(self):
        self.get_neuron()
        self.get_synapse()
        self.get_inp_neuron()
        self.get_inp_synapse()
        self.get_neuron_params()
        self.get_synapse_params()
        self.get_inp_neuron_params()
        self.get_inp_synapse_params()
        self.get_comm()
        self.get_inp_comm()

    def get_net(self):
        '''
        获取网络模型(子类可以改写,并且定义为self.net)

        注意:
        推荐将过程拆分到上面定义的函数中
        '''
        self.before_get_net()
        self.net = MultiNet(neuron=self.neuron, synapse=self.synapse, inp_neuron=self.inp_neuron, inp_synapse=self.inp_synapse, neuron_params=self.neuron_params, synapse_params=self.synapse_params, inp_neuron_params=self.inp_neuron_params, inp_synapse_params=self.inp_synapse_params, comm=self.comm, inp_comm=self.inp_comm)


class SNNNetGenerator(abc.ABC):
    @abc.abstractmethod
    def _get_neuron(self):
        pass

    @abc.abstractmethod
    def _get_synapse(self):
        pass

    @abc.abstractmethod
    def _get_inp_neuron(self):
        pass

    @abc.abstractmethod
    def _get_inp_synapse(self):
        pass

    @abc.abstractmethod
    def _get_neuron_params(self):
        pass

    @abc.abstractmethod
    def _get_synapse_params(self):
        pass

    @abc.abstractmethod
    def _get_inp_neuron_params(self):
        pass

    @abc.abstractmethod
    def _get_inp_synapse_params(self):
        pass

    def inject_conn(self, conn_dict):
        '''
        conn_dict是字典,key是(s, t, name)的元组,value是conn对象
        '''
        pass

    @abc.abstractmethod
    def _get_comm(self):
        pass

    @abc.abstractmethod
    def _get_inp_comm(self):
        pass

    def _before_get_net(self):
        self._get_neuron()
        self._get_synapse()
        self._get_inp_neuron()
        self._get_inp_synapse()
        self._get_neuron_params()
        self._get_synapse_params()
        self._get_inp_neuron_params()
        self._get_inp_synapse_params()
        self._get_comm()
        self._get_inp_comm()

    def get_net(self):
        '''
        获取网络模型(子类可以改写,并且定义为self.net)

        注意:
        推荐将过程拆分到上面定义的函数中
        '''
        self._before_get_net()
        self.net = MultiNet(neuron=self.neuron, synapse=self.synapse, inp_neuron=self.inp_neuron, inp_synapse=self.inp_synapse, neuron_params=self.neuron_params, synapse_params=self.synapse_params, inp_neuron_params=self.inp_neuron_params, inp_synapse_params=self.inp_synapse_params, comm=self.comm, inp_comm=self.inp_comm)
        return self.net
# endregion


# region 保存和加载
def bp_load_state_adapted(target, state_dict, **kwargs):
    """
    2025_5_8, adapted from brainpy, the order of missing and unexpected keys is changed since the brainpy bug

    also print things to make it easier to debug
    """
    nodes = target.nodes().subset(DynamicalSystem).not_subset(DynView).unique()
    missing_keys = []
    unexpected_keys = []
    failed_names = []
    for name, node in nodes.items():
        try:
            r = node.load_state(state_dict[name], **kwargs)
        except:
            r = None
            failed_names.append(name)
        if r is not None:
            missing, unexpected = r
            missing_keys.extend([f'{name}.{key}' for key in missing])
            unexpected_keys.extend([f'{name}.{key}' for key in unexpected])
    if bp.__version__ == '2.6.0':
        unexpected_keys, missing_keys = missing_keys, unexpected_keys
        if len(unexpected_keys) > 0 or len(missing_keys) > 0:
            # 两个都空的话也没必要打印warning
            cf.print_title('Note', char='!')
            cf.print_title('bp version is 2.6.0, so the order of missing and unexpected keys is changed since the brainpy bug', char='!')
            cf.print_sep(char='!')
    print(f'Failed names: {failed_names}')
    return StateLoadResult(missing_keys, unexpected_keys)


def extract_state(net):
    '''
    提取net的状态
    '''
    return bp.save_state(net)


def save_state_to_disk(net, filename, **kwargs):
    '''
    保存net的状态
    '''
    cf.mkdir(filename)
    bp.checkpoints.save_pytree(filename, net.state_dict(), **kwargs)


def load_state_from_disk(filename):
    '''
    读取net的状态
    '''
    return bp.checkpoints.load_pytree(filename)


def assign_state(net, state):
    '''
    将state赋值给net
    '''
    bp_load_state_adapted(net, state)


def load_and_assign_state(net, filename):
    '''
    读取net的状态并赋值
    '''
    state = load_state_from_disk(filename)
    assign_state(net, state)
# endregion