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
import igraph as ig

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


def get_sparse_adj_from_networkx_graph(G, weight='weight'):
    """
    获取图的稀疏邻接矩阵
    
    Parameters
    ----------
    G : networkx.Graph
        输入图
    weight : str or None, default='weight'
        边权重属性名,None表示无权图
    
    Returns
    -------
    scipy.sparse.spmatrix
        稀疏邻接矩阵
    """
    try:
        # 尝试使用推荐的to_scipy_sparse_array
        return nx.to_scipy_sparse_array(G, weight=weight, format='csr')
    except AttributeError:
        # 兼容旧版本的to_scipy_sparse_matrix
        return nx.to_scipy_sparse_matrix(G, weight=weight, format='csr')


class TestGraphConversion(cf.RunAllMixin):
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
# endregion


# region 各种格式的igraph的转换
def get_graph_by_mode_igraph(mode):
    if mode == 'directed':
        g = ig.Graph(directed=True)
    else:
        g = ig.Graph(directed=False)
    return g


def binary_adj_mat_to_igraph_graph(matrix, mode='directed'):
    g = get_graph_by_mode_igraph(mode)
    n = matrix.shape[0]
    is_sparse = sps.issparse(matrix)
    g.add_vertices(n)
    
    if is_sparse:
        coo_matrix = matrix.tocoo()
        edges = list(zip(coo_matrix.row, coo_matrix.col))
    else:
        rows, cols = np.where(matrix)
        edges = list(zip(rows, cols))
    
    if edges:
        g.add_edges(edges)
    
    return g


def weighted_adj_mat_to_igraph_graph(matrix, mode='directed'):
    g = get_graph_by_mode_igraph(mode)
    n = matrix.shape[0]
    is_sparse = sps.issparse(matrix)
    g.add_vertices(n)
    
    if is_sparse:
        coo_matrix = matrix.tocoo()
        rows, cols, weights = coo_matrix.row, coo_matrix.col, coo_matrix.data
    else:
        rows, cols = np.where(matrix != 0)
        weights = matrix[rows, cols]
    
    if len(rows) > 0:
        edges = list(zip(rows, cols))
        g.add_edges(edges)
        g.es["weight"] = list(weights)
    
    return g


def get_sparse_adj_from_igraph_graph(g, weight='weight'):
    """
    从igraph图中获取稀疏邻接矩阵
    
    Parameters
    ----------
    G : igraph.Graph
        输入图
    weight : str or None, default='weight'
        边权重属性名,None表示无权图
    
    Returns
    -------
    scipy.sparse.spmatrix
        稀疏邻接矩阵
    """
    n = G.vcount()
    edges = G.get_edgelist()
    rows, cols, data = [], [], []
    
    for i, (src, dst) in enumerate(edges):
        if weight is None or weight not in G.es.attribute_names():
            w = 1.0
        else:
            w = G.es[i][weight]
        
        rows.append(src)
        cols.append(dst)
        data.append(w)
        
        if not G.is_directed() and src != dst:
            rows.append(dst)
            cols.append(src)
            data.append(w)
    
    return sp.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float64)
# endregion


# region networkx和igraph的转换
def convert_nx_to_ig(nx_graph):
    """Convert networkx graph to igraph graph"""
    if not nx_graph:
        return ig.Graph(directed=False)
    
    nodes = list(nx_graph.nodes())
    edges = list(nx_graph.edges())
    
    # Create igraph with correct directionality
    ig_graph = ig.Graph(directed=nx_graph.is_directed())
    
    # Add vertices
    ig_graph.add_vertices(len(nodes))
    
    # Set vertex names
    for i, node in enumerate(nodes):
        ig_graph.vs[i]["name"] = str(node)
    
    # Add edges
    if edges:
        edge_list = []
        for u, v in edges:
            u_idx = nodes.index(u) if u in nodes else -1
            v_idx = nodes.index(v) if v in nodes else -1
            if u_idx != -1 and v_idx != -1:
                edge_list.append((u_idx, v_idx))
        
        if edge_list:
            ig_graph.add_edges(edge_list)
    
    return ig_graph


def convert_ig_to_nx(ig_graph):
    """Convert igraph graph to networkx graph"""
    if ig_graph is None or ig_graph.vcount() == 0:
        is_directed = getattr(ig_graph, "is_directed", lambda: False)()
        return nx.DiGraph() if is_directed else nx.Graph()
    
    # Determine graph type
    is_directed = ig_graph.is_directed()
    
    # Create appropriate networkx graph
    if is_directed:
        nx_graph = nx.DiGraph()
    else:
        nx_graph = nx.Graph()
    
    # Get or create node names
    node_names = []
    for v in ig_graph.vs:
        if "name" in v.attribute_names():
            node_name = v["name"]
        else:
            node_name = str(v.index)
        node_names.append(node_name)
    
    # Add nodes
    nx_graph.add_nodes_from(node_names)
    
    # Add edges
    edges = ig_graph.get_edgelist()
    for src_idx, tgt_idx in edges:
        if 0 <= src_idx < len(node_names) and 0 <= tgt_idx < len(node_names):
            nx_graph.add_edge(node_names[src_idx], node_names[tgt_idx])
    
    return nx_graph
# endregion


# region 基于权重的边的筛选和移除
def get_edges_in_weight_range(G, min_weight=None, max_weight=None):
    '''
    获取权重范围内的边
    min_weight 与 max_weight 为 None 时表示不设下限或上限
    没有权重属性的边默认权重为1
    '''
    edges_in_range = set()
    
    for u, v, data in G.edges(data=True):
        # 没有权重属性时默认权重为1
        weight = data.get('weight', 1)
        
        if min_weight is not None and weight < min_weight:
            continue
        if max_weight is not None and weight > max_weight:
            continue
            
        edges_in_range.add((u, v))
    
    return edges_in_range


def remove_edges_outside_weight_range(G, min_weight=None, max_weight=None):
    '''
    移除权重范围外的边(保留范围内的边)
    min_weight 与 max_weight 为 None 时表示不设下限或上限
    没有权重属性的边默认权重为1
    '''
    H = G.copy()
    edges_to_remove = set()
    
    for u, v, data in H.edges(data=True):
        # 没有权重属性时默认权重为1
        weight = data.get('weight', 1)
        
        # 如果边不在指定范围内,则标记为要移除
        if (min_weight is not None and weight < min_weight) or (max_weight is not None and weight > max_weight):
            edges_to_remove.add((u, v))
    
    H.remove_edges_from(edges_to_remove)
    return H


def remove_edges_inside_weight_range(G, min_weight=None, max_weight=None):
    '''
    移除权重范围内的边(保留范围外的边)
    min_weight 与 max_weight 为 None 时表示不设下限或上限
    没有权重属性的边默认权重为1
    '''
    H = G.copy()
    edges_to_remove = set()
    
    for u, v, data in H.edges(data=True):
        # 没有权重属性时默认权重为1
        weight = data.get('weight', 1)
        
        # 如果边在指定范围内,则标记为要移除
        in_range = True
        if min_weight is not None and weight < min_weight:
            in_range = False
        if max_weight is not None and weight > max_weight:
            in_range = False
            
        if in_range:
            edges_to_remove.add((u, v))
    
    H.remove_edges_from(edges_to_remove)
    return H


def get_weight_proportion_in_weight_range(G, min_weight=None, max_weight=None):
    '''
    计算权重范围内边的权重比例
    min_weight 与 max_weight 为 None 时表示不设下限或上限
    没有权重属性的边默认权重为1
    '''
    total_weight = get_connection_weight_sum(G)
    
    filtered_weight = 0
    for u, v, data in G.edges(data=True):
        # 没有权重属性时默认权重为1
        weight = data.get('weight', 1)
        
        if min_weight is not None and weight < min_weight:
            continue
        if max_weight is not None and weight > max_weight:
            continue
            
        filtered_weight += weight
    
    return filtered_weight / total_weight


class TestWeightRangeFunctions(cf.RunAllMixin):
    def test_get_edges_in_weight_range_basic(self):
        G = nx.Graph()
        G.add_edge(1, 2, weight=5)
        G.add_edge(2, 3, weight=10)
        G.add_edge(3, 4, weight=15)
        
        result = get_edges_in_weight_range(G, min_weight=8, max_weight=12)
        expected = {(2, 3)}
        
        assert result == expected
    
    def test_get_edges_in_weight_range_no_bounds(self):
        G = nx.Graph()
        G.add_edge(1, 2, weight=5)
        G.add_edge(2, 3, weight=10)
        
        result = get_edges_in_weight_range(G)
        expected = {(1, 2), (2, 3)}
        
        assert result == expected
    
    def test_get_edges_in_weight_range_no_weight_attr(self):
        G = nx.Graph()
        G.add_edge(1, 2)  # 没有权重属性,默认权重为1
        G.add_edge(2, 3, weight=10)
        
        result = get_edges_in_weight_range(G, min_weight=5, max_weight=15)
        expected = {(2, 3)}  # 只有第二条边在范围内
        
        assert result == expected
    
    def test_get_edges_in_weight_range_default_weight(self):
        G = nx.Graph()
        G.add_edge(1, 2)  # 完全没有属性,默认权重为1
        G.add_edge(2, 3, weight=2)
        
        result = get_edges_in_weight_range(G, min_weight=0, max_weight=1.5)
        expected = {(1, 2)}  # 第一条边权重默认为1,在范围内
        
        assert result == expected
    
    def test_remove_edges_outside_weight_range(self):
        G = nx.Graph()
        G.add_edge(1, 2, weight=5)
        G.add_edge(2, 3, weight=10)
        G.add_edge(3, 4, weight=15)
        
        result_graph = remove_edges_outside_weight_range(G, min_weight=8, max_weight=12)
        
        assert result_graph.has_edge(1, 2) == False
        assert result_graph.has_edge(2, 3) == True
        assert result_graph.has_edge(3, 4) == False
    
    def test_remove_edges_outside_weight_range_original_unchanged(self):
        G = nx.Graph()
        G.add_edge(1, 2, weight=5)
        
        result_graph = remove_edges_outside_weight_range(G, min_weight=10, max_weight=20)
        
        assert G.has_edge(1, 2) == True
        assert result_graph.has_edge(1, 2) == False
    
    def test_remove_edges_inside_weight_range(self):
        G = nx.Graph()
        G.add_edge(1, 2, weight=5)
        G.add_edge(2, 3, weight=10)
        G.add_edge(3, 4, weight=15)
        
        result_graph = remove_edges_inside_weight_range(G, min_weight=8, max_weight=12)
        
        assert result_graph.has_edge(1, 2) == True
        assert result_graph.has_edge(2, 3) == False
        assert result_graph.has_edge(3, 4) == True
    
    def test_remove_edges_inside_weight_range_no_bounds(self):
        G = nx.Graph()
        G.add_edge(1, 2, weight=5)
        G.add_edge(2, 3, weight=10)
        
        result_graph = remove_edges_inside_weight_range(G)
        
        assert result_graph.number_of_edges() == 0
    
    def test_get_weight_proportion_in_weight_range(self):
        G = nx.Graph()
        G.add_edge(1, 2, weight=5)
        G.add_edge(2, 3, weight=10)
        G.add_edge(3, 4, weight=15)
        
        proportion = get_weight_proportion_in_weight_range(G, min_weight=8, max_weight=12)
        
        assert proportion == 10/30
    
    def test_get_weight_proportion_in_weight_range_no_matching_edges(self):
        G = nx.Graph()
        G.add_edge(1, 2, weight=5)
        G.add_edge(2, 3, weight=10)
        
        proportion = get_weight_proportion_in_weight_range(G, min_weight=20, max_weight=30)
        
        assert proportion == 0
    
    def test_get_weight_proportion_in_weight_range_no_distance_attr(self):
        G = nx.Graph()
        G.add_edge(1, 2, weight=5)
        G.add_edge(2, 3, weight=10)
        
        proportion = get_weight_proportion_in_weight_range(G, min_weight=3, max_weight=8)
        
        assert proportion == 5/15
    
    def test_get_weight_proportion_in_weight_range_default_weight_and_distance(self):
        G = nx.Graph()
        G.add_edge(1, 2)
        G.add_edge(2, 3, weight=2)
        
        proportion = get_weight_proportion_in_weight_range(G, min_weight=0.5, max_weight=1.5)
        
        assert proportion == 1/3
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


class TestSpatialGraph(cf.RunAllMixin):
    def test_add_coordinates_to_graph(self):
        '''测试为图添加坐标信息'''
        G = nx.Graph()
        G.add_nodes_from([0, 1, 2])
        coordinates_dict = {
            0: (0, 0),
            1: (1, 0),
            2: (0, 1)
        }
        G = add_coordinates_to_graph(G, coordinates_dict)
        assert G.nodes[0]['pos'] == (0, 0)
        assert G.nodes[1]['pos'] == (1, 0)
        assert G.nodes[2]['pos'] == (0, 1)

    def test_add_distances_from_coordinates(self):
        '''测试为图添加距离属性'''
        G = nx.Graph()
        G.add_nodes_from([0, 1])
        G.add_edge(0, 1)
        coordinates_dict = {
            0: (0, 0),
            1: (3, 4)
        }
        G = add_coordinates_to_graph(G, coordinates_dict)
        G = add_distances_from_coordinates(G)
        assert np.isclose(G[0][1]['distance'], 5.0)  # 距离应为5
# endregion


# region 基于距离的边的筛选和移除
def get_edges_in_distance_range(G, min_dist=None, max_dist=None, enable_no_distance_continue=True):
    '''
    获取距离范围内的边
    min_dist 与 max_dist 为 None 时表示不设下限或上限
    '''
    edges_in_range = set()
    
    for u, v, data in G.edges(data=True):
        if 'distance' not in data:
            if enable_no_distance_continue:
                continue
            else:
                raise ValueError(f"Edge ({u}, {v}) does not have 'distance' attribute.")
        dist = data['distance']
        if min_dist is not None and dist < min_dist:
            continue
        if max_dist is not None and dist > max_dist:
            continue
            
        edges_in_range.add((u, v))
    
    return edges_in_range


def remove_edges_outside_distance_range(G, min_dist=None, max_dist=None, enable_no_distance_continue=True):
    '''
    移除距离范围外的边(保留范围内的边)
    min_dist 与 max_dist 为 None 时表示不设下限或上限
    '''
    H = G.copy()
    edges_to_remove = set()
    
    for u, v, data in H.edges(data=True):
        if 'distance' not in data:
            if enable_no_distance_continue:
                continue
            else:
                raise ValueError(f"Edge ({u}, {v}) does not have 'distance' attribute.")
        dist = data['distance']
        # 如果边不在指定范围内,则标记为要移除
        if (min_dist is not None and dist < min_dist) or (max_dist is not None and dist > max_dist):
            edges_to_remove.add((u, v))
    
    H.remove_edges_from(edges_to_remove)
    return H


def remove_edges_inside_distance_range(G, min_dist=None, max_dist=None, enable_no_distance_continue=True):
    '''
    移除距离范围内的边(保留范围外的边)
    min_dist 与 max_dist 为 None 时表示不设下限或上限
    '''
    H = G.copy()
    edges_to_remove = set()
    
    for u, v, data in H.edges(data=True):
        if 'distance' not in data:
            if enable_no_distance_continue:
                continue
            else:
                raise ValueError(f"Edge ({u}, {v}) does not have 'distance' attribute.")
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


def get_weight_proportion_in_distance_range(G, min_dist=None, max_dist=None, enable_no_distance_continue=True):
    '''
    计算距离范围内边的权重比例
    min_dist 与 max_dist 为 None 时表示不设下限或上限
    '''
    total_weight = get_connection_weight_sum(G)
    
    filtered_weight = 0
    for u, v, data in G.edges(data=True):
        if 'distance' not in data:
            if enable_no_distance_continue:
                continue
            else:
                raise ValueError(f"Edge ({u}, {v}) does not have 'distance' attribute.")
        dist = data['distance']
        if min_dist is not None and dist < min_dist:
            continue
        if max_dist is not None and dist > max_dist:
            continue
            
        filtered_weight += data.get('weight', 1) # 如果没有权重属性,则默认为1
    
    return filtered_weight / total_weight


class TestDistanceRangeFunctions(cf.RunAllMixin):
    def test_get_edges_in_distance_range_basic(self):
        G = nx.Graph()
        G.add_edge(1, 2, distance=5, weight=1)
        G.add_edge(2, 3, distance=10, weight=2)
        G.add_edge(3, 4, distance=15, weight=3)
        
        result = get_edges_in_distance_range(G, min_dist=8, max_dist=12)
        expected = {(2, 3)}
        
        assert result == expected
    
    def test_get_edges_in_distance_range_no_bounds(self):
        G = nx.Graph()
        G.add_edge(1, 2, distance=5, weight=1)
        G.add_edge(2, 3, distance=10, weight=2)
        
        result = get_edges_in_distance_range(G)
        expected = {(1, 2), (2, 3)}
        
        assert result == expected
    
    def test_get_edges_in_distance_range_no_distance_attr(self):
        G = nx.Graph()
        G.add_edge(1, 2, weight=1)
        G.add_edge(2, 3, distance=10, weight=2)
        
        result = get_edges_in_distance_range(G, min_dist=5, max_dist=15)
        expected = {(2, 3)}
        
        assert result == expected
    
    def test_remove_edges_outside_distance_range(self):
        G = nx.Graph()
        G.add_edge(1, 2, distance=5, weight=1)
        G.add_edge(2, 3, distance=10, weight=2)
        G.add_edge(3, 4, distance=15, weight=3)
        
        result_graph = remove_edges_outside_distance_range(G, min_dist=8, max_dist=12)
        
        assert result_graph.has_edge(1, 2) == False
        assert result_graph.has_edge(2, 3) == True
        assert result_graph.has_edge(3, 4) == False
        assert result_graph[2][3]['weight'] == 2
    
    def test_remove_edges_outside_distance_range_original_unchanged(self):
        G = nx.Graph()
        G.add_edge(1, 2, distance=5, weight=1)
        
        result_graph = remove_edges_outside_distance_range(G, min_dist=10, max_dist=20)
        
        assert G.has_edge(1, 2) == True
        assert result_graph.has_edge(1, 2) == False
    
    def test_remove_edges_inside_distance_range(self):
        G = nx.Graph()
        G.add_edge(1, 2, distance=5, weight=1)
        G.add_edge(2, 3, distance=10, weight=2)
        G.add_edge(3, 4, distance=15, weight=3)
        
        result_graph = remove_edges_inside_distance_range(G, min_dist=8, max_dist=12)
        
        assert result_graph.has_edge(1, 2) == True
        assert result_graph.has_edge(2, 3) == False
        assert result_graph.has_edge(3, 4) == True
        assert result_graph[1][2]['weight'] == 1
    
    def test_remove_edges_inside_distance_range_no_bounds(self):
        G = nx.Graph()
        G.add_edge(1, 2, distance=5, weight=1)
        G.add_edge(2, 3, distance=10, weight=2)
        
        result_graph = remove_edges_inside_distance_range(G)
        
        assert result_graph.number_of_edges() == 0
    
    def test_get_weight_proportion_in_distance_range(self):
        G = nx.Graph()
        G.add_edge(1, 2, distance=5, weight=1)
        G.add_edge(2, 3, distance=10, weight=2)
        G.add_edge(3, 4, distance=15, weight=3)
        
        proportion = get_weight_proportion_in_distance_range(G, min_dist=8, max_dist=12)
        
        assert proportion == 2/6
    
    def test_get_weight_proportion_in_distance_range_no_matching_edges(self):
        G = nx.Graph()
        G.add_edge(1, 2, distance=5, weight=1)
        G.add_edge(2, 3, distance=10, weight=2)
        
        proportion = get_weight_proportion_in_distance_range(G, min_dist=20, max_dist=30)
        
        assert proportion == 0
    
    def test_get_weight_proportion_in_distance_range_no_weight_attr(self):
        G = nx.Graph()
        G.add_edge(1, 2, distance=5)
        G.add_edge(2, 3, distance=10)
        
        proportion = get_weight_proportion_in_distance_range(G, min_dist=3, max_dist=8)
        
        assert proportion == 1/2
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


class TestAdjustConnections(cf.RunAllMixin):
    def test_add_edges_to_graph(self):
        '''测试添加边功能'''
        G = nx.Graph()
        G.add_nodes_from([0, 1, 2])
        edges = [(0, 1), (1, 2)]
        G = add_edges_to_graph(G, edges)
        assert G.number_of_edges() == 2
        print("add_edges_to_graph passed.")
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


def generate_Erdos_Renyi_graph_igraph(n, p, seed=None, mode='undirected'):
    G = generate_Erdos_Renyi_graph(n=n, p=p, seed=seed, mode=mode)
    return convert_nx_to_ig(G)


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


def generate_lattice_graph_igraph(n, k, mode='undirected'):
    G = generate_lattice_graph(n=n, k=k, mode=mode)
    return convert_nx_to_ig(G)


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


def generate_newman_watts_strogatz_graph_igraph(n, k, p, seed=None, mode='undirected'):
    G = generate_newman_watts_strogatz_graph(n=n, k=k, p=p, seed=seed, mode=mode)
    return convert_nx_to_ig(G)


class TestCommonGraphs(cf.RunAllMixin):
    def test_generate_Erdos_Renyi_graph(self):
        '''测试Erdos-Renyi图生成'''
        G = generate_Erdos_Renyi_graph(n=10, p=0.3, seed=42, mode='undirected')
        assert G.number_of_nodes() == 10
        assert not G.is_directed()

    def test_generate_lattice_graph(self):
        '''测试格子图生成'''
        G = generate_lattice_graph(n=10, k=2, mode='undirected')
        assert G.number_of_nodes() == 10
        degrees = [deg for node, deg in G.degree()]
        assert np.allclose(np.sum(degrees) / len(degrees), 4), np.sum(degrees) / len(degrees)  # 每个节点度数应为4

    def test_generate_newman_watts_strogatz_graph(self):
        '''测试Newman-Watts-Strogatz图生成'''
        n = 20
        k = 2
        p = 0.1
        G = generate_newman_watts_strogatz_graph(n=20, k=2, p=0.1, seed=42, mode='undirected')
        assert G.number_of_nodes() == 20
        print('Number of edges:', G.number_of_edges(), 'Expected approx:', n * k + n * k * p)
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


def get_average_shortest_path_length_igraph(G, **kwargs):
    if not G.is_connected():
        return float('inf')
    return G.average_path_length()


def get_average_clustering_coefficient(G, **kwargs):
    return nx.average_clustering(G, **kwargs)


def get_average_clustering_coefficient_igraph(G, mode=None, **kwargs):
    if mode is None:
        mode = 'undirected' if not G.is_directed() else 'directed'
    return G.transitivity_avglocal_undirected() if mode == 'undirected' else G.transitivity_avglocal_directed()


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


def get_small_world_index_igraph(G, seed=0):
    ASPL_sw = get_average_shortest_path_length_igraph(G)
    C_sw = get_average_clustering_coefficient_igraph(G)
    
    n = G.vcount()
    m = G.ecount()
    p = (2 * m) / (n * (n - 1))
    
    G_rand = generate_Erdos_Renyi_graph_igraph(n=n, p=p, seed=seed, mode='undirected')
    ASPL_rand = get_average_shortest_path_length_igraph(G_rand)
    C_rand = get_average_clustering_coefficient_igraph(G_rand)
    
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
    
    results = {
        'clustering_coefficient': C_sw,
        'average_shortest_path_length': ASPL_sw,
        'clustering_coefficient_random': C_rand,
        'average_shortest_path_length_random': ASPL_rand,
        'clustering_coefficient_ratio': C_ratio,
        'average_shortest_path_length_ratio': ASPL_ratio,
        'small_world_index': sigma
    }
    return results


def get_adjacency_matrix_power(adj_mat, power):
    '''
    计算邻接矩阵的幂
    adj_mat: 邻接矩阵,可以是numpy数组或scipy稀疏矩阵
    power: 幂次
    mask的作用: 保证一定为0的位置在结果中仍为0
    '''
    mask = (adj_mat != 0).astype(int)
    if sps.issparse(adj_mat):
        result = adj_mat ** power
        mask = mask ** power
        mask = mask.astype(bool)
        result = result.multiply(mask)
    else:
        result = np.linalg.matrix_power(adj_mat, power)
        mask = np.linalg.matrix_power(mask, power)
        result = result * (mask > 0)
    return result


class TestNetworkIndex(cf.RunAllMixin):
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

    def test_get_average_clustering_coefficient(self):
        '''测试平均聚类系数计算'''
        # 完全图的聚类系数应为1
        G = nx.complete_graph(5)
        acc = get_average_clustering_coefficient(G)
        assert acc == 1.0

    def test_get_degree_dict(self):
        '''测试度字典获取'''
        G = nx.Graph()
        G.add_edges_from([(0, 1), (0, 2), (1, 2), (2, 3)])
        degree_dict = get_degree_dict(G)
        expected_degrees = {0: 2, 1: 2, 2: 3, 3: 1}
        assert degree_dict == expected_degrees

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
# endregion


# region 测试本py中的函数
class TestNetworkFunctions(cf.RunAllMixin):
    def test_graph_conversions(self):
        t = TestGraphConversion()
        t.run_all()
    
    def test_weight_range_functions(self):
        t = TestWeightRangeFunctions()
        t.run_all()

    def test_spatial_graph(self):
        t = TestSpatialGraph()
        t.run_all()
    
    def test_distance_range_functions(self):
        t = TestDistanceRangeFunctions()
        t.run_all()
    
    def test_adjust_connections(self):
        t = TestAdjustConnections()
        t.run_all()
    
    def test_common_graphs(self):
        t = TestCommonGraphs()
        t.run_all()
    
    def test_network_index(self):
        t = TestNetworkIndex()
        t.run_all()
# endregion