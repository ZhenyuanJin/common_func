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


# 数据处理和可视化库
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, ScalarFormatter
from matplotlib.colors import BoundaryNorm, Normalize
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable


# 网络分析库
import networkx as nx


# 自定义库
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import common_functions as cf
# endregion


# region 各种格式的network的转换
def get_graph_by_mode(mode):
    if mode == 'undirected':
        G = nx.Graph()
    elif mode == 'directed':
        G = nx.DiGraph()
    else:
        raise ValueError("mode must be 'undirected' or 'directed'")
    return G


def binary_adj_mat_to_networkx_graph(matrix, mode='directed'):
    create_using = get_graph_by_mode(mode)
    if sps.issparse(matrix):
        matrix = matrix.astype(bool).tocoo()
        G = nx.from_scipy_sparse_array(matrix, create_using=create_using, edge_attribute=None)
    else:
        matrix = matrix.astype(bool)
        G = nx.from_numpy_array(matrix, create_using=create_using)
    return G


def weighted_adj_mat_to_networkx_graph(matrix, mode='directed'):
    create_using = get_graph_by_mode(mode)
    if sps.issparse(matrix):
        G = nx.from_scipy_sparse_array(matrix, create_using=create_using)
    else:
        G = nx.from_numpy_array(matrix, create_using=create_using)
    return G
# endregion


# region 添加空间信息
def add_coordinates_to_graph(G, coordinates_dict):
    '''为图节点添加坐标信息'''
    for node_id, coords in coordinates_dict.items():
        if node_id in G:
            G.nodes[node_id]['pos'] = coords
    return G


def add_distances_from_coordinates(G):
    '''
    为图的边添加距离属性(基于节点坐标)
    G[u][v]代表的是边(u, v)的属性字典,起点为u,终点为v
    '''
    for u, v in G.edges():
        if 'pos' in G.nodes[u] and 'pos' in G.nodes[v]:
            pos_u, pos_v = G.nodes[u]['pos'], G.nodes[v]['pos']
            dist = np.linalg.norm(np.array(pos_u) - np.array(pos_v))
            G[u][v]['distance'] = dist
    return G


def add_distances_to_graph(G, distance_dict):
    '''
    为图的边添加自定义距离属性
    G[u][v]代表的是边(u, v)的属性字典,起点为u,终点为v
    '''
    for (u, v), dist in distance_dict.items():
        G[u][v]['distance'] = dist
    return G
# endregion


# region 基于距离的边的筛选和移除
def get_edges_in_distance_range(G, min_dist=None, max_dist=None):
    '''获取距离范围内的边'''
    edges_in_range = set()
    
    for u, v, data in G.edges(data=True):
        if 'distance' not in data:
            continue
            
        dist = data['distance']
        if min_dist is not None and dist < min_dist:
            continue
        if max_dist is not None and dist > max_dist:
            continue
            
        edges_in_range.add((u, v))
    
    return edges_in_range


def remove_edges_outside_distance_range(G, min_dist=None, max_dist=None):
    '''移除距离范围外的边(保留范围内的边)'''
    H = G.copy()
    edges_to_remove = set()
    
    for u, v, data in H.edges(data=True):
        dist = data['distance']
        # 如果边不在指定范围内,则标记为要移除
        if (min_dist is not None and dist < min_dist) or (max_dist is not None and dist > max_dist):
            edges_to_remove.add((u, v))
    
    H.remove_edges_from(edges_to_remove)
    return H


def remove_edges_inside_distance_range(G, min_dist=None, max_dist=None):
    '''移除距离范围内的边(保留范围外的边)'''
    H = G.copy()
    edges_to_remove = set()
    
    for u, v, data in H.edges(data=True):
        dist = data['distance']
        # 如果边在指定范围内,则标记为要移除
        in_range = True
        if min_dist is not None and dist < min_dist:
            in_range = False
        if max_dist is not None and dist > max_dist:
            in_range = False
            
        if in_range:
            edges_to_remove.add((u, v))
    
    H.remove_edges_from(edges_to_remove)
    return H


def get_weight_proportion_in_distance_range(G, min_dist=None, max_dist=None):
    '''计算距离范围内边的权重比例'''
    total_weight = sum(data.get('weight', 0) for _, _, data in G.edges(data=True))
    
    filtered_weight = 0
    for u, v, data in G.edges(data=True):
        if 'distance' not in data:
            continue
            
        dist = data['distance']
        if min_dist is not None and dist < min_dist:
            continue
        if max_dist is not None and dist > max_dist:
            continue
            
        filtered_weight += data.get('weight', 0)
    
    return filtered_weight / total_weight
# endregion


# region 改变连接
def add_edges_to_graph(G, edges):
    '''
    edges: list of (u, v) tuples, u代表起点, v代表终点
    例如: [(0, 1), (1, 2)]
    '''
    if isinstance(edges, tuple):
        edges = [edges]
    G.add_edges_from(edges)
    return G
# endregion


# region 常见图的生成
def generate_Erdos_Renyi_graph(n, p, seed=None, mode='undirected'):
    '''
    n: 节点数
    p: 边的生成概率
    seed: 随机种子
    mode: 'undirected'
    '''
    if mode != 'undirected':
        raise ValueError("Erdos-Renyi graph only supports 'undirected' mode now.")
    try:
        create_using = get_graph_by_mode(mode)
        G = nx.erdos_renyi_graph(n=n, p=p, seed=seed, directed=(mode=='directed'), create_using=create_using)
        return G
    except:
        G = nx.erdos_renyi_graph(n=n, p=p, seed=seed, directed=(mode=='directed'))
        return G


def generate_lattice_graph(n, k, mode='undirected'):
    '''
    n: 节点数
    k: 每个节点的最近邻连接数(单侧),两侧为2k
    mode: 'undirected'

    一个环形网络,每个节点连接其前后各k个邻居
    '''
    if mode != 'undirected':
        raise ValueError("Lattice graph only supports 'undirected' mode now.")
    G = generate_newman_watts_strogatz_graph(n=n, k=k, p=0.0, seed=None, mode=mode)
    return G


def generate_newman_watts_strogatz_graph(n, k, p, seed=None, mode='undirected'):
    '''
    n: 节点数
    k: 每个节点的最近邻连接数(单侧),两侧为2k
    p: 额外边的生成概率
    seed: 随机种子
    mode: 'undirected'

    一个小世界网络,基于环形网络添加额外边,添加边的数量为n*k*p

    注意:
    与networkx中的newman_watts_strogatz_graph函数不同,这里的k表示单侧连接数,而networkx中表示的是总连接数(两侧)
    例如: k=2表示每个节点连接其前后各2个邻居,总共4个邻居
    '''
    if mode != 'undirected':
        raise ValueError("Newman-Watts-Strogatz graph only supports 'undirected' mode now.")
    try:
        create_using = get_graph_by_mode(mode)
        G = nx.newman_watts_strogatz_graph(n=n, k=2*k, p=p, seed=seed, create_using=create_using)
        return G
    except:
        G = nx.newman_watts_strogatz_graph(n=n, k=2*k, p=p, seed=seed)
        return G
# endregion


# region 图论指标计算
def get_connection_num(G):
    '''
    计算图的连接数(边的数量)
    '''
    return G.number_of_edges()


def get_connection_weight_sum(G):
    '''
    计算图的连接权重和
    '''
    weight_sum = 0
    for u, v, data in G.edges(data=True):
        weight_sum += data.get('weight', 1)  # 如果没有权重属性,则默认为1
    return weight_sum


def get_connection_num_density(G):
    '''
    计算图的连接数密度
    '''
    n = G.number_of_nodes()
    m = G.number_of_edges()
    if G.is_directed():
        max_edges = n * (n - 1)
    else:
        max_edges = n * (n - 1) / 2
    return m / max_edges


def get_average_shortest_path_length(G, **kwargs):
    if not nx.is_connected(G):
        return float('inf')
    return nx.average_shortest_path_length(G, **kwargs)


def get_average_clustering_coefficient(G, **kwargs):
    return nx.average_clustering(G, **kwargs)


def get_degree_dict(G):
    degree_dict = dict(G.degree())
    return degree_dict


def get_small_world_index(G, seed=0):
    ASPL_sw = get_average_shortest_path_length(G)
    C_sw = get_average_clustering_coefficient(G)
    
    n = G.number_of_nodes()
    p = (2 * G.number_of_edges()) / (n * (n - 1))
    G_rand = generate_Erdos_Renyi_graph(n=n, p=p, seed=seed, mode='undirected')
    ASPL_rand = get_average_shortest_path_length(G_rand)
    C_rand = get_average_clustering_coefficient(G_rand)
    
    if np.allclose(C_rand, 0):
        C_ratio = np.nan
        print("Warning: C_rand is zero, cannot compute C_ratio.")
    else:
        C_ratio = C_sw / C_rand
    if np.allclose(ASPL_rand, 0):
        ASPL_ratio = np.nan
        print("Warning: ASPL_rand is zero, cannot compute ASPL_ratio.")
    else:
        ASPL_ratio = ASPL_sw / ASPL_rand
    if np.allclose(ASPL_ratio, 0):
        sigma = np.nan
        print("Warning: ASPL_ratio is zero, cannot compute small-world index sigma.")
    else:
        sigma = C_ratio / ASPL_ratio
    results = {}
    results['clustering_coefficient'] = C_sw
    results['average_shortest_path_length'] = ASPL_sw
    results['clustering_coefficient_random'] = C_rand
    results['average_shortest_path_length_random'] = ASPL_rand
    results['clustering_coefficient_ratio'] = C_ratio
    results['average_shortest_path_length_ratio'] = ASPL_ratio
    results['small_world_index'] = sigma
    return results
# endregion


# region 测试本py中的函数
class TestNetworkFunctions:
    def test_binary_adj_mat_to_networkx_graph(self):
        '''测试二进制邻接矩阵转换'''
        adj_mat = np.array([[0, 1, 0],
                            [1, 0, 1],
                            [0, 1, 0]])
        G = binary_adj_mat_to_networkx_graph(adj_mat, mode='undirected')
        assert G.number_of_nodes() == 3
        assert G.number_of_edges() == 2
        G = binary_adj_mat_to_networkx_graph(adj_mat, mode='directed')
        assert G.number_of_nodes() == 3
        assert G.number_of_edges() == 4 # 有向图,两倍
        print("binary_adj_mat_to_networkx_graph passed.")

    def test_weighted_adj_mat_to_networkx_graph(self):
        '''测试加权邻接矩阵转换'''
        adj_mat = np.array([[0, 2, 0],
                            [2, 0, 3],
                            [0, 3, 0]])
        G = weighted_adj_mat_to_networkx_graph(adj_mat, mode='undirected')
        assert G.number_of_nodes() == 3
        assert G.number_of_edges() == 2
        # 检查权重
        assert G[0][1]['weight'] == 2
        assert G[1][2]['weight'] == 3

        G = weighted_adj_mat_to_networkx_graph(adj_mat, mode='directed')
        assert G.number_of_nodes() == 3
        assert G.number_of_edges() == 4 # 有向图,两倍
        # 检查权重
        assert G[0][1]['weight'] == 2
        assert G[1][0]['weight'] == 2
        assert G[1][2]['weight'] == 3
        assert G[2][1]['weight'] == 3
        print("weighted_adj_mat_to_networkx_graph passed.")

    def test_sparse_matrix_conversion(self):
        '''测试稀疏矩阵转换'''
        adj_mat = sps.csr_matrix(np.array([[0, 1, 0],
                                         [1, 0, 1],
                                         [0, 1, 0]]))
        G = binary_adj_mat_to_networkx_graph(adj_mat, mode='undirected')
        assert G.number_of_nodes() == 3
        assert G.number_of_edges() == 2

        G = binary_adj_mat_to_networkx_graph(adj_mat, mode='directed')
        assert G.number_of_nodes() == 3
        assert G.number_of_edges() == 4 # 有向图,两倍
        print("sparse_matrix_conversion passed.")

    def test_directed_graph_conversion(self):
        '''测试有向图转换'''
        adj_mat = np.array([[0, 1, 0],
                            [0, 0, 1],
                            [0, 0, 0]])
        G = binary_adj_mat_to_networkx_graph(adj_mat, mode='directed')
        assert G.number_of_nodes() == 3
        assert G.number_of_edges() == 2
        assert G.has_edge(0, 1)
        assert G.has_edge(1, 2)
        assert not G.has_edge(1, 0)  # 有向图,反向边不存在
        print("directed_graph_conversion passed.")

    def test_add_edges_to_graph(self):
        '''测试添加边功能'''
        G = nx.Graph()
        G.add_nodes_from([0, 1, 2])
        edges = [(0, 1), (1, 2)]
        G = add_edges_to_graph(G, edges)
        assert G.number_of_edges() == 2
        print("add_edges_to_graph passed.")

    def test_generate_Erdos_Renyi_graph(self):
        '''测试Erdos-Renyi图生成'''
        G = generate_Erdos_Renyi_graph(n=10, p=0.3, seed=42, mode='undirected')
        assert G.number_of_nodes() == 10
        assert not G.is_directed()
        print("generate_Erdos_Renyi_graph passed.")

    def test_generate_lattice_graph(self):
        '''测试格子图生成'''
        G = generate_lattice_graph(n=10, k=2, mode='undirected')
        assert G.number_of_nodes() == 10
        degrees = [deg for node, deg in G.degree()]
        assert np.allclose(np.sum(degrees) / len(degrees), 4), np.sum(degrees) / len(degrees)  # 每个节点度数应为4
        print("generate_lattice_graph passed.")

    def test_generate_newman_watts_strogatz_graph(self):
        '''测试Newman-Watts-Strogatz图生成'''
        G = generate_newman_watts_strogatz_graph(n=20, k=2, p=0.1, seed=42, mode='undirected')
        assert G.number_of_nodes() == 20
        print('Number of edges:', G.number_of_edges(), 'Expected approx:', 20 * 2 + 20 * 2 * 0.1)
        print("generate_newman_watts_strogatz_graph passed.")

    def test_get_connection_num(self):
        """测试连接数计算"""
        G = nx.Graph()
        G.add_edges_from([(1, 2), (2, 3), (3, 1)])
        
        result = get_connection_num(G)
        expected = 3
        assert result == expected
    
    def test_get_connection_weight_sum(self):
        """测试连接权重和计算"""
        G = nx.Graph()
        G.add_weighted_edges_from([(1, 2, 2.5), (2, 3, 1.0), (3, 1, 0.5)])
        
        result = get_connection_weight_sum(G)
        expected = 2.5 + 1.0 + 0.5
        assert np.allclose(result, expected)
    
    def test_get_connection_num_density(self):
        """测试连接数密度计算"""
        G = nx.Graph()
        G.add_edges_from([(1, 2), (2, 3), (3, 1)])  # 3个节点的三角形
        
        result = get_connection_num_density(G)
        n = 3
        max_edges = n * (n - 1) / 2  # 3
        expected = 3 / max_edges  # 1.0
        assert np.allclose(result, expected)

    def test_get_average_shortest_path_length(self):
        '''测试平均最短路径长度计算'''
        # 完全图,平均最短路径长度应为1
        G = nx.complete_graph(5)
        aspl = get_average_shortest_path_length(G)
        assert aspl == 1.0
        
        # 不连通图
        G_disconnected = nx.Graph()
        G_disconnected.add_edges_from([(0, 1), (2, 3)])
        aspl_disconnected = get_average_shortest_path_length(G_disconnected)
        assert np.isinf(aspl_disconnected)
        print("get_average_shortest_path_length passed.")

    def test_get_average_clustering_coefficient(self):
        '''测试平均聚类系数计算'''
        # 完全图的聚类系数应为1
        G = nx.complete_graph(5)
        acc = get_average_clustering_coefficient(G)
        assert acc == 1.0
        print("get_average_clustering_coefficient passed.")

    def test_get_degree_dict(self):
        '''测试度字典获取'''
        G = nx.Graph()
        G.add_edges_from([(0, 1), (0, 2), (1, 2), (2, 3)])
        degree_dict = get_degree_dict(G)
        expected_degrees = {0: 2, 1: 2, 2: 3, 3: 1}
        assert degree_dict == expected_degrees
        print("get_degree_dict passed.")

    def test_get_small_world_index(self):
        '''测试小世界指数计算'''
        # 使用一个已知的小世界网络
        G = generate_newman_watts_strogatz_graph(n=30, k=4, p=0.1, seed=42)
        results = get_small_world_index(G, seed=42)
        
        # 检查返回的字典包含所有必要的键
        expected_keys = ['clustering_coefficient', 'average_shortest_path_length',
                        'clustering_coefficient_random', 'average_shortest_path_length_random',
                        'clustering_coefficient_ratio', 'average_shortest_path_length_ratio',
                        'small_world_index']
        for key in expected_keys:
            assert key in results
        
        # 小世界网络的sigma应该大于1
        if not np.isnan(results['small_world_index']):
            assert results['small_world_index'] > 1
        print("get_small_world_index passed.")

    def run_all(self, exclude=None):
        '''
        运行所有公共方法
        exclude: 要排除的方法名列表
        '''
        if exclude is None:
            exclude = ['run_all']
        
        for method_name in dir(self):
            # 排除特殊方法和私有方法
            if (method_name.startswith('_') or 
                method_name in exclude):
                continue
            
            method = getattr(self, method_name)
            if callable(method):
                print(f"执行 {method_name}:")
                result = method()
                if result is not None:
                    print(f"  返回: {result}")
# endregion