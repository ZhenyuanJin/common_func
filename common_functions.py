# region 导入模块
# 标准库导入
import subprocess
import os
import re
import sys
import json
import pickle
import joblib
import random
import shutil
import time
import datetime
import warnings
from math import ceil
import multiprocessing
from multiprocessing import Process
from pathlib import Path
import inspect
import types
from functools import wraps, partial, lru_cache
import importlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import StringIO, BytesIO
import copy
import hashlib
import json
import logging
from cProfile import Profile
from line_profiler import LineProfiler
from pstats import Stats
import tracemalloc
import itertools
import contextlib
import abc
import glob
import pdb
import types
from collections import defaultdict, Counter
import yaml
import lmdb
from pympler import asizeof
import traceback
import tqdm


# 数学和科学计算库
import numpy as np
import math
import scipy
import scipy.stats as st
from scipy.stats import gaussian_kde, zscore
from scipy.integrate import quad
from scipy.sparse import coo_matrix, csr_matrix
import scipy.sparse as sps
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA, NMF
import statsmodels
from statsmodels.tsa.stattools import acf, acovf, ccf
from statsmodels.distributions.empirical_distribution import ECDF
from statsmodels.nonparametric.smoothers_lowess import lowess


# 数据处理和可视化库
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure, SubFigure
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import FuncFormatter, ScalarFormatter, LogFormatter, LogFormatterExponent, LogFormatterMathtext, LogFormatterSciNotation, PercentFormatter
import matplotlib.ticker as ticker
from matplotlib.colors import BoundaryNorm, Normalize, ListedColormap, SymLogNorm, LogNorm, TwoSlopeNorm
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.collections as mcoll
import matplotlib.font_manager as fm
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import cv2
from reportlab.pdfgen import canvas
from pdf2image import convert_from_path
import imageio
# endregion


# region cf目录
CF_DIR = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件所在目录
# endregion


# region 定义颜色,为了防止一些奇怪的错误,不使用0而使用0.01
BLUE = (0.01, 0.45, 0.70)
RED = (1.00, 0.27, 0.01)
GREEN = (0.01, 0.65, 0.50)
YELLOW = (0.99, 0.70, 0.01)
BLACK = (0.01, 0.01, 0.01)
WHITE = (1.00, 1.00, 1.00)
GRAY = (0.50, 0.50, 0.50)
GREY = (0.50, 0.50, 0.50)
ORANGE = (1.00, 0.50, 0.01)
PURPLE = (0.50, 0.01, 0.50)
BROWN = (0.50, 0.16, 0.16)
CRIMSON = (220 / 255, 20 / 255, 60 / 255)
CYAN = (0.01, 0.75, 0.75)
MEGNETA = (1.00, 0.01, 1.00)
TRANSPARENT = (0.00, 0.00, 0.00, 0.00)

# region mygo颜色
RANA = (235 / 255, 235 / 255, 235 / 255)
ANON = (255 / 255, 178 / 255, 212 / 255)
UIKA = (251 / 255, 230 / 255, 153 / 255)
MUTSUMI = (229 / 255, 241 / 255, 231 / 255)
SOYO = (203 / 255, 172 / 255, 158 / 255)
SAKIKO = (198 / 255, 211 / 255, 225 / 255)
TOMORI = (158 / 255, 148 / 255, 166 / 255)
TAKI = (105 / 255, 92 / 255, 97 / 255)
NYAMU = (135 / 255, 124 / 255, 157 / 255)
MYGO_LIST = [RANA, ANON, UIKA, MUTSUMI, SOYO, SAKIKO, TOMORI, TAKI, NYAMU]
MYGO_DICT = {'RANA': RANA, 'ANON': ANON, 'UIKA': UIKA, 'MUTSUMI': MUTSUMI, 'SOYO': SOYO, 'SAKIKO': SAKIKO, 'TOMORI': TOMORI, 'TAKI': TAKI, 'NYAMU': NYAMU}
# endregion

# region 孤独摇滚颜色
BOCCHI = (239 / 255, 176 / 255, 193 / 255)
NIJIKA = (238 / 255, 218 / 255, 137 / 255)
KITA = (200 / 255, 102 / 255, 91 / 255)
RYO = (93 / 255, 116 / 255, 165 / 255)
BOCCHI_THE_ROCK_LIST = [BOCCHI, NIJIKA, KITA, RYO]
BOCCHI_THE_ROCK_DICT = {'BOCCHI': BOCCHI, 'NIJIKA': NIJIKA, 'KITA': KITA, 'RYO': RYO}
# endregion

# region GOOGLE颜色
GOOGLE_BLUE = (66/255, 133/255, 244/255)
GOOGLE_RED = (234/255, 67/255, 53/255)
GOOGLE_YELLOW = (251/255, 188/255, 5/255)
GOOGLE_GREEN = (52/255, 168/255, 83/255)
GOOGLE_LIST = [GOOGLE_BLUE, GOOGLE_RED, GOOGLE_YELLOW, GOOGLE_GREEN]
GOOGLE_DICT = {'GOOGLE_BLUE': GOOGLE_BLUE, 'GOOGLE_RED': GOOGLE_RED, 'GOOGLE_YELLOW': GOOGLE_YELLOW, 'GOOGLE_GREEN': GOOGLE_GREEN}
# endregion

# region F1颜色
FERRARI = (211 / 255, 45 / 255, 53 / 255)
MCLAREN = (239 / 255, 136 / 255, 51 / 255)
ALPINE = (239 / 255, 142 / 255, 187 / 255)
REDBULL = (72 / 255, 114 / 255, 192 / 255)
ALPHATAURI = (113 / 255, 146 / 255, 246 / 255)
WILLIAMS = (124 / 255, 194 / 255, 250 / 255)
HAAS = (179 / 255, 183 / 255, 185 / 255)
MERCEDES = (118 / 255, 243 / 255, 214 / 255)
SAUBER = (130 / 255, 225 / 255, 108 / 255)
ASTONMARTIN = (75 / 255, 151 / 255, 116 / 255)
F1_LIST = [FERRARI, MCLAREN, ALPINE, REDBULL, ALPHATAURI, WILLIAMS, HAAS, MERCEDES, SAUBER, ASTONMARTIN]
F1_DICT = {'FERRARI': FERRARI, 'MCLAREN': MCLAREN, 'ALPINE': ALPINE, 'REDBULL': REDBULL, 'ALPHATAURI': ALPHATAURI, 'WILLIAMS': WILLIAMS, 'HAAS': HAAS, 'MERCEDES': MERCEDES, 'SAUBER': SAUBER, 'ASTONMARTIN': ASTONMARTIN}
# endregion
# endregion


# region 用于plt设置
MARGIN = {'left': 0.2, 'right': 0.85, 'bottom': 0.2, 'top': 0.85}     # 默认图形边距
SPACIOUS_MARGIN = {'left': 0.25, 'right': 0.8, 'bottom': 0.25, 'top': 0.8}     # 边框较宽的图形边距
COMPACT_MARGIN = {'left': 0.1, 'right': 0.95, 'bottom': 0.1, 'top': 0.95}     # 边框较窄的图形边距
CBAR_MARGIN = {'left': 0.2, 'right': 0.8, 'bottom': 0.2, 'top': 0.85}     # 当右侧添加了colorbar时, 推荐的图形边距
MARGIN_3D = {'left': 0.15, 'right': 0.85, 'bottom': 0.1, 'top': 0.85}     # 3d图形的默认边距
CBAR_MARGIN_3D = {'left': 0.15, 'right': 0.7, 'bottom': 0.1, 'top': 0.85}     # 3d图形带有cbar的默认边距
ADJUST_PARAMS_CUSTOM = {'left': 0.2, 'right': 0.85, 'bottom': 0.2, 'top': 0.85, 'wspace': 0.35, 'hspace': 0.35}     # 自定义的调整参数
ADJUST_PARAMS_CUSTOM_3D = {'left': 0.15, 'right': 0.85, 'bottom': 0.1, 'top': 0.85, 'wspace': 0.3, 'hspace': 0.35}     # 默认的调整参数
FONT_SIZE = 15     # 默认字体大小
TITLE_SIZE = FONT_SIZE*2     # 默认标题字体大小
SUP_TITLE_SIZE = FONT_SIZE*3     # 默认总标题字体大小
LABEL_SIZE = FONT_SIZE*2     # 默认标签字体大小
TICK_SIZE = FONT_SIZE*4/3     # 默认刻度标签字体大小
LEGEND_SIZE = FONT_SIZE*4/3     # 默认图例字体大小
LINE_WIDTH = 3     # 默认线宽
MARKER_SIZE = LINE_WIDTH*2     # 默认标记大小,注意散点图的s是面积,markersize是直径
AX_WIDTH = 5   # 默认图形宽度(单个图形)
AX_HEIGHT = 5   # 默认图形高度(单个图形)
FIG_SIZE = (AX_WIDTH / ( MARGIN['right'] - MARGIN['left'] ), AX_HEIGHT / ( MARGIN['top'] - MARGIN['bottom'] ))     # 默认图形大小
FIG_DPI = 100     # 默认图形分辨率
SAVEFIG_DPI = 300     # 默认保存图形分辨率
SAVEFIG_VECTOR_FORMAT = 'pdf'   # 默认保存图形的矢量图格式
SAVEFIG_RASTER_FORMAT = 'png'   # 默认保存图形的位图格式
SAVEFIG_FORMAT = SAVEFIG_VECTOR_FORMAT     # 默认保存图形的格式
TOP_SPINE = False     # 默认隐藏上方的轴脊柱
RIGHT_SPINE = False     # 默认隐藏右方的轴脊柱
LEFT_SPINE = True     # 默认显示左方的轴脊柱
BOTTOM_SPINE = True     # 默认显示下方的轴脊柱
LEGEND_LOC = 'upper right'      # 图例位置
PDF_FONTTYPE = 42     # 使pdf文件中的文字可编辑
PS_FONTTYPE = 42     # 适用于PostScript导出
PDF_USE14COREFONTS = False     # 禁用 14 种核心字体(强制嵌入自定义字体)
PAD_INCHES = 0.2     # 默认图形边距
USE_MATHTEXT = True     # 使用数学文本
USE_OFFSET = False     # 不使用偏移
AXES_LINEWIDTH = LINE_WIDTH     # 设置坐标轴线宽
TICK_MAJOR_WIDTH = 2    # x轴主刻度线的宽度
TICK_MINOR_WIDTH = 1    # x轴次刻度线的宽度
TICK_MAJOR_SIZE = 8     # x轴主刻度线的长度
TICK_MINOR_SIZE = 4     # x轴次刻度线的长度
TICK_DIRECTION = 'out'     # 刻度线的方向
RM_REPEAT_TICK_LABEL_WHEN_SHARE = True     # 当共享坐标轴时,去除重复的刻度标签
AX_FACECOLOR = (1., 1., 1., 0.5)     # 背景具有一定的透明度,防止遮挡


def set_font_size(font_size):
    '''
    目前并未生效,函数的默认值绑定导致了这个问题,需要设置为none在内部赋予
    '''
    global FONT_SIZE, TITLE_SIZE, LABEL_SIZE, TICK_SIZE, LEGEND_SIZE
    
    FONT_SIZE = font_size
    TITLE_SIZE = FONT_SIZE*2
    SUP_TITLE_SIZE = FONT_SIZE*3
    LABEL_SIZE = FONT_SIZE*2
    TICK_SIZE = FONT_SIZE*4/3
    LEGEND_SIZE = FONT_SIZE*4/3
    
    # 更新Matplotlib全局设置
    plt.rcParams.update({
        'font.size': FONT_SIZE,
        'axes.titlesize': TITLE_SIZE,
        'axes.labelsize': LABEL_SIZE,
        'xtick.labelsize': TICK_SIZE,
        'ytick.labelsize': TICK_SIZE,
        'legend.fontsize': LEGEND_SIZE,
        'figure.titlesize': SUP_TITLE_SIZE
    })
# endregion


# region plt全局参数设置
plt.rcParams['figure.figsize'] = FIG_SIZE   # 图形的大小
plt.rcParams['font.size'] = FONT_SIZE       # 字体大小
plt.rcParams['axes.labelsize'] = LABEL_SIZE       # 坐标轴标签的字体大小
plt.rcParams['axes.titlesize'] = TITLE_SIZE       # 标题的字体大小
plt.rcParams['xtick.labelsize'] = TICK_SIZE       # x轴刻度标签的字体大小
plt.rcParams['ytick.labelsize'] = TICK_SIZE       # y轴刻度标签的字体大小
plt.rcParams['legend.fontsize'] = LEGEND_SIZE       # 图例的字体大小
plt.rcParams['axes.spines.top'] = TOP_SPINE    # 设定上方的轴脊柱
plt.rcParams['axes.spines.right'] = RIGHT_SPINE    # 设定右方的轴脊柱
plt.rcParams['axes.spines.left'] = LEFT_SPINE    # 设定左方的轴脊柱
plt.rcParams['axes.spines.bottom'] = BOTTOM_SPINE    # 设定下方的轴脊柱
plt.rcParams['axes.linewidth'] = AXES_LINEWIDTH    # 设置坐标轴线宽
plt.rcParams['lines.linewidth'] = LINE_WIDTH    # 线宽
plt.rcParams['lines.markersize'] = MARKER_SIZE    # 标记大小
plt.rcParams['savefig.format'] = SAVEFIG_FORMAT    # 保存图形的格式
plt.rcParams['figure.dpi'] = FIG_DPI      # 图形的分辨率
plt.rcParams['savefig.dpi'] = SAVEFIG_DPI      # 保存图像的分辨率
plt.rcParams['legend.loc'] = LEGEND_LOC      # 图例位置
plt.rcParams['pdf.fonttype'] = PDF_FONTTYPE     # 使pdf文件中的文字可编辑
plt.rcParams['ps.fonttype'] = PS_FONTTYPE     # 适用于PostScript导出
plt.rcParams['pdf.use14corefonts'] = PDF_USE14COREFONTS     # 禁用14种核心字体(强制嵌入自定义字体)
plt.rcParams['savefig.pad_inches'] = PAD_INCHES     # 图形边距
plt.rcParams['axes.formatter.use_mathtext'] = USE_MATHTEXT     # 使用数学文本
plt.rcParams['axes.formatter.useoffset'] = USE_OFFSET     # 使用偏移
plt.rcParams['axes.linewidth'] = LINE_WIDTH     # 设置坐标轴线宽
plt.rcParams['xtick.major.width'] = TICK_MAJOR_WIDTH   # x轴主刻度线的宽度
plt.rcParams['xtick.minor.width'] = TICK_MINOR_WIDTH   # x轴次刻度线的宽度
plt.rcParams['ytick.major.width'] = TICK_MAJOR_WIDTH   # y轴主刻度线的宽度
plt.rcParams['ytick.minor.width'] = TICK_MINOR_WIDTH   # y轴次刻度线的宽度
plt.rcParams['xtick.major.size'] = TICK_MAJOR_SIZE  # x轴主刻度线的长度
plt.rcParams['xtick.minor.size'] = TICK_MINOR_SIZE   # x轴次刻度线的长度
plt.rcParams['ytick.major.size'] = TICK_MAJOR_SIZE  # y轴主刻度线的长度
plt.rcParams['ytick.minor.size'] = TICK_MINOR_SIZE   # y轴次刻度线的长度
plt.rcParams['xtick.direction'] = TICK_DIRECTION   # x轴刻度线的方向
plt.rcParams['ytick.direction'] = TICK_DIRECTION   # y轴刻度线的方向
plt.rcParams['axes.facecolor'] = AX_FACECOLOR  # 背景颜色
plt.rcParams['figure.subplot.wspace'] = ADJUST_PARAMS_CUSTOM['wspace']  # 调整子图之间的间距
plt.rcParams['figure.subplot.hspace'] = ADJUST_PARAMS_CUSTOM['hspace']  # 调整子图之间的间距
fm.fontManager.addfont(os.path.join(CF_DIR, 'font/arial.ttf'))  # Specify the path to your local Arial TTF file from https://github.com/gasharper/linux-fonts/blob/master/arial.ttf
plt.rcParams['font.family'] = 'Arial'  # Set Arial as the default font
# endregion


# region 全局参数
# region 随机种子
SEED = 421
# endregion

# region 数值处理参数
ROUND_DIGITS = 3
ROUND_FORMAT = 'general'
INF_POLICY = 'to_nan'
NAN_POLICY = 'drop'
# endregion

# region 文本处理参数
TEXT_PROCESS = {'capitalize': True, 'replace_underscore': ' ', 'ignore_equation_underscore': True}
REPLACE_DOT = 'd'
FILENAME_PROCESS = {'replace_blank': '_', 'replace_dot': REPLACE_DOT} # replace_blank: 替换空格, replace_dot: 替换点
# endregion

# region 图形样式参数
BAR_WIDTH = 0.8
BIN_NUM = 20
XTICK_ROTATION = 90
YTICK_ROTATION = 0
STAR = '*'
STAR_SIZE = LINE_WIDTH*5
AUXILIARY_LINE_STYLE = '--'
PLT_CAP_SIZE = 5
SNS_CAP_SIZE = 0.1
ARROW_STYLE = '-|>'
ARROW_HEAD_WIDTH = 0.3
ARROW_HEAD_LENGTH = 0.5
ARROW_PROPS = {'fc': RED, 'ec': RED, 'linewidth': LINE_WIDTH,
                      'arrowstyle': ARROW_STYLE, 'head_length': ARROW_HEAD_LENGTH, 'head_width': ARROW_HEAD_WIDTH}
FAINT_ALPHA = 0.5
TEXT_X = 0.05
TEXT_Y = 0.95
ELEV = 30
AZIM = 30
CBAR_LABEL_SIZE = TICK_SIZE
CBAR_TICK_SIZE = TICK_SIZE
MASK_COLOR = GRAY
LABEL_PAD = LABEL_SIZE/3     # label间距(间距的单位是字体大小)
TITLE_PAD = TITLE_SIZE/3     # 标题间距(间距的单位是字体大小)
TEXT_VA = 'top'
TEXT_HA = 'left'
# endregion

# region 图形布局参数
BBOX_INCHES = None
SIDE_PAD = 0.03
CBAR_POSITION = {'position': 'right', 'size': 0.05, 'pad': SIDE_PAD}
CBAR_POSITION_3D = {'position': 'right', 'size': 0.05, 'pad': SIDE_PAD * 6}
TICK_PROPORTION = 0.9    # 刻度标签的拥挤程度,1代表完全贴住,见suitable_tick_size与adjust_ax_tick
# endregion

# region 保存fig参数
SAVEFIG_PKL = False
# endregion

# region 颜色映射参数(注意,颜色映射在使用整数调用时会出现和使用浮点数调用时不同的效果,比如输入1会映射到第一个颜色,但输入1.0会映射到cmap的最顶端)
CMAP = plt.cm.viridis
HEATMAP_CMAP = plt.cm.jet
DENSITY_CMAP = mcolors.LinearSegmentedColormap.from_list("density_cmap", [RANA, BLUE])
CONTRAST_CMAP = mcolors.LinearSegmentedColormap.from_list("contrast_cmap", [REDBULL, WHITE, FERRARI])
CONTRAST_WITH_MID_CMAP = mcolors.LinearSegmentedColormap.from_list("contrast_with_mid_cmap", [REDBULL, (0.5, 0.6, 0.4), FERRARI])
CONTRAST_GRAY_CMAP = mcolors.LinearSegmentedColormap.from_list("contrast_gray_cmap", [REDBULL, RANA, FERRARI])
RGB_CMAP = mcolors.LinearSegmentedColormap.from_list("rgb_cmap", ['#FF0000', '#00FF00', '#0000FF'])
MARS_CMAP = mcolors.LinearSegmentedColormap.from_list("mars_cmap", [MERCEDES, FERRARI, REDBULL])
OCEAN_CMAP = mcolors.LinearSegmentedColormap.from_list("ocean_cmap", ['#FF7F46', '#FFDC87', '#B3E5EF', '#59CCE3', '#49A8D0', '#0C5582'])
MARCH_7TH_CMAP = mcolors.LinearSegmentedColormap.from_list("march_7th_cmap", ['#59CCE3', '#D587D4'])
MARCH_7TH_WHITE_CMAP = mcolors.LinearSegmentedColormap.from_list("march_7th_white_cmap", ['#59CCE3', '#FFFFFF', '#D587D4'])
SAKURA_CMAP = mcolors.LinearSegmentedColormap.from_list("sakura_cmap", [ANON, REDBULL])
FOREST_CMAP = mcolors.LinearSegmentedColormap.from_list("forest_cmap", [BROWN, '#087532', '#8BC34A'])
DESERT_CMAP = mcolors.LinearSegmentedColormap.from_list("desert_cmap", ['#A0522D', '#F0E68C'])
LAVENDER_CMAP = mcolors.LinearSegmentedColormap.from_list("lavender_cmap", ['#B8ACD5', '#9400dd'])
SUNSET_CMAP = mcolors.LinearSegmentedColormap.from_list("sunset_cmap", ['#FF4500', '#FFD700'])
CLOUD_CMAP = mcolors.LinearSegmentedColormap.from_list("cloud_cmap", ['#F0F8FF', '#4682B4'])
PINEAPPLE_CMAP = mcolors.LinearSegmentedColormap.from_list("pineapple_cmap", ['#FFEE58', '#D4E157', '#7DB93D'])
CMAP_DICT = {'viridis': plt.cm.viridis, 'jet': plt.cm.jet, 'density': DENSITY_CMAP, 'contrast': CONTRAST_CMAP, 'contrast_with_mid': CONTRAST_WITH_MID_CMAP, 'contrast_gray': CONTRAST_GRAY_CMAP, 'rgb': RGB_CMAP, 'mars': MARS_CMAP, 'ocean': OCEAN_CMAP, 'march_7th': MARCH_7TH_CMAP, 'sakura': SAKURA_CMAP, 'forest': FOREST_CMAP, 'desert': DESERT_CMAP, 'lavender': LAVENDER_CMAP, 'sunset': SUNSET_CMAP, 'cloud': CLOUD_CMAP, 'pineapple': PINEAPPLE_CMAP}
CLABEL_KWARGS = {'inline': True, 'fontsize': FONT_SIZE, 'fmt': f'%.{ROUND_DIGITS}g'}
# endregion

# region 视频参数
SAVEVIDEO_FORMAT = 'mp4'
FRAME_RATE = 5
# endregion

# region 添加图片tag相关参数
TAG_CASE = 'lower'     # 默认tag大小写
TAG_PARENTHESES = False     # 默认tag是否加括号
FIG_TAG_POS = (0.05, 0.95)     # 默认图形tag位置
AX_TAG_POS = (-0.2, 1.2)     # 默认坐标轴tag位置
TAG_SIZE = SUP_TITLE_SIZE     # 默认tag字体大小
TAG_VA = TEXT_VA     # 默认tag垂直对齐方式
TAG_HA = TEXT_HA     # 默认tag水平对齐方式
# endregion

# region print相关参数
PRINT_WIDTH = 80     # 默认打印宽度
PRINT_CHAR = '-'     # 默认打印字符
# endregion

# region multiprocess
PROCESS_NUM = min(40, int(multiprocessing.cpu_count()/3))    # 默认multiprocess的数量
# endregion
# endregion


# region 设定gpu
def set_gpu(id):
    '''
    例子: set_gpu('0'), set_gpu("1,2,3")
    '''
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = id  # specify which GPU(s) to be used
    print_title(f"Set to use GPU: {id}", char='#')


def find_least_used_gpu():
    # 执行nvidia-smi命令获取GPU状态
    smi_output = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits']).decode()

    # 初始化最小内存使用量和对应的GPU索引
    min_memory_used = float('inf')
    least_used_gpu_index = None

    # 解析输出，找到内存使用最少的GPU
    for line in smi_output.strip().split('\n'):
        gpu_index, memory_used = line.split(', ')
        memory_used = int(memory_used)
        if memory_used < min_memory_used:
            min_memory_used = memory_used
            least_used_gpu_index = gpu_index

    return least_used_gpu_index


def set_least_used_gpu():
    least_used_gpu = find_least_used_gpu()
    if least_used_gpu is not None:
        # 设置环境变量以使用内存使用最少的GPU
        set_gpu(least_used_gpu)
        print(f"Set to use the least used GPU: {least_used_gpu}")
    else:
        print("No GPU found.")
# endregion


# region 获取cpu核心数
def get_core_num():
    # 返回CPU核心数量
    return multiprocessing.cpu_count()
# endregion


# region 获取内存大小
def get_object_memory(obj, unit='gb', print_info=True):
    memory_size = sys.getsizeof(obj)
    normalize_size = get_size_in_bytes(1, unit)
    memory_size = memory_size / normalize_size

    if print_info:
        print(f"Estimated memory usage for object of type {type(obj).__name__}: {memory_size:.2f} {unit}")
    return memory_size


def estimate_array_memory(shape, data_type='float64', unit='gb', print_info=True):
    '''
        可以利用arr.dtype来获取数组的data type
    '''
    bytes_per_element = np.dtype(data_type).itemsize
    total_elements = np.prod(shape)
    total_bytes = total_elements * bytes_per_element

    if unit == 'bytes':
        memory_estimate = total_bytes
    elif unit == 'kb':
        memory_estimate = total_bytes / 1024
    elif unit == 'mb':
        memory_estimate = total_bytes / (1024 ** 2)
    elif unit == 'gb':
        memory_estimate = total_bytes / (1024 ** 3)
    else:
        raise ValueError("Invalid unit. Please choose from 'bytes', 'kb', 'mb', or 'gb'.")
    if print_info:
        print(f"Estimated memory usage for array with shape {shape} and data type {data_type}: {memory_estimate:.2f} {unit}")
    return memory_estimate


def get_dict_memory(d, unit='gb', print_info=True):
    """
    计算字典中每个键的 Value 内存占用,忽略 Key 的内存
    
    :param d: 要分析的字典
    :return: 字典,键为原字典的键,值为 Value 的内存占用
    """
    normalize_size = get_size_in_bytes(1, unit)
    mem_info = {}
    for key, value in d.items():
        value_size = asizeof.asizeof(value)
        mem_info[key] = value_size / normalize_size
    
    if print_info:
        total_size = sum(mem_info.values())
        print(f"Estimated memory usage for dictionary with {len(d)} items: {total_size:.2f} {unit}")
    return mem_info


def get_top_n_memory_item(d, top_n=5, unit='gb', print_info=True):
    """
    找出内存占用最大的前 top_n 个键值对(仅基于 Value 的内存)
    
    :param d: 要分析的字典
    :param top_n: 返回前多少个最大元素
    :return: 列表,包含前 top_n 个键值对的内存信息
    """
    mem_info = get_dict_memory(d, unit, print_info)
    items = list(mem_info.items())
    
    # 按内存大小降序排序
    sorted_items = sorted(items, key=lambda x: -x[1])
    top_items = sorted_items[:top_n]
    
    # 构造返回结果
    result = []
    for key, size_value in top_items:
        result.append({
            'key': key,
            'mem': size_value
        })
        if print_info:
            print(f"Key: {key}, Memory Size: {size_value:.2f} {unit}")
    return result
# endregion


# region 内存管理
class MemoryTracker:
    def __init__(self, stack_depth=10):
        """
        初始化 MemoryDebugger 类
        :param stack_depth: 堆栈追踪深度，默认为 10
        """
        self.stack_depth = stack_depth
        self.snapshot_before = None
        self.snapshot_after = None

    def start(self):
        """
        开始内存追踪
        """
        tracemalloc.start(self.stack_depth)
        self.snapshot_before = tracemalloc.take_snapshot()

    def stop(self):
        """
        停止内存追踪并记录第二个快照
        """
        self.snapshot_after = tracemalloc.take_snapshot()

    def compare_by_lineno(self, top_n=10):
        """
        按行号统计内存差异
        :param top_n: 显示内存占用最多的前 N 个结果
        :return: 打印统计信息
        """
        if not self.snapshot_before or not self.snapshot_after:
            print("Error: Please ensure both snapshots are taken (call start() and stop()).")
            return

        stats = self.snapshot_after.compare_to(self.snapshot_before, 'lineno')
        print(f"\nTop {top_n} memory usage by line number:\n")
        for stat in stats[:top_n]:
            print(stat)

    def compare_by_traceback(self, top_n=5):
        """
        按堆栈追踪统计内存差异
        :param top_n: 显示内存占用最多的前 N 个结果
        :return: 打印堆栈信息
        """
        if not self.snapshot_before or not self.snapshot_after:
            print("Error: Please ensure both snapshots are taken (call start() and stop()).")
            return

        stats = self.snapshot_after.compare_to(self.snapshot_before, 'traceback')
        print(f"\nTop {top_n} memory usage by traceback:\n")
        for stat in stats[:top_n]:
            print("\n".join(stat.traceback.format()))
            print(f"Memory Usage: {stat.size / 1024:.2f} KiB, Count: {stat.count}")

    def reset(self):
        """
        重置快照，重新开始内存追踪
        """
        tracemalloc.stop()
        self.snapshot_before = None
        self.snapshot_after = None
        tracemalloc.start(self.stack_depth)
        print(f"{self.__class__.__name__} has been reset.")


class WithMemoryTracker:
    def __init__(self, mode='lineno', top_n=5, stack_depth=10):
        """
        :param mode: 对比模式 - 'lineno'按行号 / 'traceback'按堆栈
        :param top_n: 结果显示条数
        :param stack_depth: 堆栈捕获深度
        """
        self.mode = mode
        self.top_n = top_n
        self.tracker = MemoryTracker(stack_depth)
    
    def __enter__(self):
        """进入上下文时启动追踪"""
        self.tracker.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文时停止追踪并打印结果"""
        self.tracker.stop()
        if self.mode == 'lineno':
            print(self.tracker.compare_by_lineno(self.top_n))
        else:
            print(self.tracker.compare_by_traceback(self.top_n))
        return False  # 不抑制异常
# endregion


# region 获取储存大小
def get_storage_size():
    '''
    du -h --max-depth=1 /path/to/directory (查看目录下所有文件夹的大小)
    df -h (查看磁盘空间)
    du -sh /path/to/directory (查看目录大小)
    '''
    pass
# endregion


# region 随机种子
def set_seed(seed=SEED):
    '''设置随机种子'''
    random.seed(seed)
    np.random.seed(seed)


def get_local_rng(seed=SEED):
    '''
    返回局部随机数生成器(rng:Random Number Generator)
    '''
    return np.random.RandomState(seed)


def get_time_seed():
    '''
    返回时间种子
    '''
    # 获取时间戳和微秒数
    timestamp = int(time.time())
    microsecond = datetime.datetime.now().microsecond

    # 结合时间戳和微秒数作为种子,为了防止超出随机数生成器的范围,取模
    return (timestamp * 1000000 + microsecond) % (2**31-1)


def get_time_seed_or_retain(seed=None):
    '''
    如果seed为None,则返回时间种子,否则返回seed
    '''
    if seed is None:
        return get_time_seed()
    else:
        return seed


def get_local_rng_by_time():
    '''
    返回局部随机数生成器(rng:Random Number Generator)
    '''
    return np.random.RandomState(get_time_seed())
# endregion


# region 随机数
def rand_unit_vec(num, dim, rng=None):
    '''
    生成随机单位向量，使其在单位球上均匀分布。(注意,三维的时候,单位球上均匀不代表azim和elev均匀)

    参数:
    num -- 要生成的随机单位向量的数量
    dim -- 向量的维度
    rng -- 随机数生成器，默认为None，此时会使用numpy的默认随机数生成器

    返回:
    一个numpy数组，包含生成的随机单位向量
    '''
    if rng is None:
        rng = np.random.default_rng()
    gaussian = rng.normal(size=(num, dim))
    return gaussian / np.linalg.norm(gaussian, axis=1, keepdims=True)
# endregion


# region 时间相关函数
def get_time(char='_', unit='us'):
    """
    获取当前时间，支持秒/毫秒/微秒级精度
    
    :param char: 时间各部分分隔符，默认为下划线
    :param unit: 时间单位（'s'-秒, 'ms'-毫秒, 'us'-微秒），默认为'us'
    :return: 格式化后的时间字符串
    """
    unit = unit.lower()
    valid_units = {'s', 'ms', 'us'}
    if unit not in valid_units:
        raise ValueError(f"Invalid unit: {unit}. Must be one of {valid_units}")
    
    # 单位到精度的映射
    unit_config = {
        's':  ('', 0),
        'ms': (f'{char}' + '%%03d', 3),  # 添加毫秒部分格式
        'us': (f'{char}' + '%%06d', 6)   # 添加微秒部分格式
    }
    
    now = datetime.datetime.now()
    
    # 基础时间格式
    base_format = f'%Y{char}%m{char}%d{char}%H{char}%M{char}%S'
    
    # 拼接完整格式
    suffix_format, precision = unit_config[unit]
    full_format = base_format + suffix_format
    
    # 处理不同精度需求
    if precision == 0:
        return now.strftime(full_format)
    else:
        truncated = now.microsecond // (10 ** (6 - precision))
        return now.strftime(full_format) % truncated


def get_start_time():
    '''获取开始时间'''
    return time.perf_counter()


def get_end_time():
    '''获取结束时间'''
    return time.perf_counter()


def _readable_time(seconds):
    if seconds < 60:
        return f"{seconds:.9f} s"
    elif seconds < 3600:
        return f"{seconds / 60:.2f} min"
    elif seconds < 86400:
        return f"{seconds / 3600:.2f} h"
    else:
        return f"{seconds / 86400:.2f} d"


def get_interval_time(start, title='', print_info=True, **kwargs):
    '''获取时间间隔'''
    interval = time.perf_counter() - start
    if print_info:
        if 'digits' in kwargs:
            print('do not use digits, this will be deprecated soon')
        if 'format_type' in kwargs:
            print('do not use format_type, this will be deprecated soon')
        # print_title(f'{title} takes {round_float(interval, digits, format_type)} s')
        print(f'{title} takes {_readable_time(interval)}')
    return interval


def func_timer(func):
    '''
    计时器修饰器
    使用例子:
    @func_timer
    def some_function():
        # some code here
    '''
    @wraps(func)  # 使用wraps装饰内部函数
    def wrapper(*args, **kwargs):
        start = get_start_time()
        result = func(*args, **kwargs)
        interval = get_interval_time(start, title=func.__name__, print_info=True)
        return result
    return wrapper


class Timer:
    '''
    计时器类
    '''
    def __init__(self, title='', print_info=True):
        self.title = title
        self.print_info = print_info
        self.start_time = None

    def start(self):
        self.start_time = get_start_time()
        if self.print_info:
            print_title(f'{self.title} starts')

    def end(self):
        if self.start_time is None:
            print('Please start the timer first')
            return None
        self.interval = get_interval_time(self.start_time, title=self.title, print_info=self.print_info)
        return self.interval

    def stop(self):
        '''停止计时,为了方便使用,end,stop都可以用'''
        self.interval = self.end()
        return self.interval


class WithTimer(Timer):
    '''
    利用with语句计时

    例子:
    with WithTimer('test') as with_timer:
        print('hello world')
    print(with_timer.interval)
    '''
    def __init__(self, title='', print_info=True):
        super().__init__(title, print_info)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end()


def sleep(seconds):
    '''睡眠'''
    time.sleep(seconds)
# endregion


# region 变量类型
def is_mutable(obj_type):
    '''判断变量是否可变'''
    if obj_type in [list, dict, set, np.ndarray, pd.DataFrame, pd.Series]:
        print(f'{obj_type} is mutable')
    elif obj_type in [int, float, str, tuple, bool, complex, frozenset]:
        # t = (1, 2, [3, 4])
        # # t[0] = 2  # 这会抛出TypeError，因为你不能改变元组中元素的值
        # t[2].append(5)  # 这是合法的，因为你改变的是元组中列表的内容，而不是元组本身
        print(f'{obj_type} is immutable')
    else:
        print(f'{obj_type} is unknown')
# endregion


# region gpt
def get_prompt(prompt_type):
    '''
    获取常用prompt
    '''
    if prompt_type == 'change_code':
        return '帮我修改这个函数使得注释完整(注释使用中文,但注释的标点是英文标点)，功能完整且正确，函数名和变量名规范协调简洁(但是并不是修改越多就越好,需要尊重我的命名方式但是在必要的时刻作出适当的改动)，并给出运行测试例子甚至可视化的代码'
    if prompt_type == 'write_code':
        return '帮我写一个函数使得注释完整(注释使用中文,但注释的标点是英文标点)，功能完整且正确，函数名和变量名规范协调简洁，并给出运行测试例子甚至可视化的代码'
    if prompt_type == 'write_annotation':
        return 'see rule'
    if prompt_type == 'paper':
        return '帮我修改这段文字使得语法通顺，逻辑清晰，表达准确，且符合学术规范'
    if prompt_type == 'translate_english':
        return '帮我翻译这段文字成英文使得语法通顺，逻辑清晰，表达准确'
    if prompt_type == 'translate_chinese':
        return '帮我翻译这段文字成中文使得语法通顺，逻辑清晰，表达准确'
# endregion


# region LATEX
def get_latex(latex_type):
    '''
    获取常用latex代码
    '''
    if latex_type == 'fig':
        return r'\begin{figure}[H]' + '\n' + r'\centering' + '\n' + r'\includegraphics[width=0.8\textwidth]{path/to/figure}' + '\n' + r'\caption{caption}' + '\n' + r'\label{fig:label}' + '\n' + r'\end{figure}'
# endregion


# region print
def flex_print_sep(print_func=print, n=PRINT_WIDTH, char=PRINT_CHAR):
    '''灵活打印分隔符'''
    print_func(char * n)


def flex_print_title(print_func=print, title=None, n=PRINT_WIDTH, char=PRINT_CHAR):
    '''灵活打印标题'''
    if title is not None:
        title = f'{title}'
        length_start = (n - len(title)) // 2
        length_end = n - len(title) - length_start
        print_func(length_start * char + title + length_end * char)
    else:
        print_func(char * n)


def flex_print_with_name(print_func=print, variable=None, name=None):
    '''灵活打印变量名和变量值'''
    print_func(f'{name}: {variable}')


def flex_print_type(print_func=print, variable=None, name=None, n=PRINT_WIDTH, char=PRINT_CHAR, lite=True, print_title=True):
    '''灵活打印变量类型和值'''
    if print_title:
        flex_print_title(print_func, title=name, n=n, char=char)
    if name is None:
        if lite:
            print_func(f'type: {type(variable)}')
        else:
            print_func(f'type: {type(variable)}, value: {variable}')
    else:
        if lite:
            print_func(f'{name} type: {type(variable)}')
        else:
            print_func(f'{name} type: {type(variable)}, value: {variable}')


def flex_print_df(print_func=print, df=None, name=None, n=PRINT_WIDTH, char=PRINT_CHAR, lite=True, print_title=True):
    '''灵活打印DataFrame信息'''
    if print_title:
        flex_print_title(print_func, title=name, n=n, char=char)
    elif name is not None:
        # print_func(f'# {name}:')
        print_func(f'{name}:')
    print_func(f'shape: {df.shape}')
    print_func(f'columns: {list(df.columns)}')
    print_func(f'index: {df.index}')
    if lite:
        print_func(f'data head: {df.head()}')
    else:
        print_func(f'data:\n{df}')


def flex_print_array(print_func=print, arr=None, name=None, n=PRINT_WIDTH, char=PRINT_CHAR, lite=True, print_title=True):
    '''灵活打印数组或列表或稀疏矩阵信息'''
    if print_title:
        flex_print_title(print_func, title=name, n=n, char=char)
    elif name is not None:
        # print_func(f'# {name}:')
        print_func(f'{name}:')
    if isinstance(arr, np.ndarray):
        print_func(f'Array shape: {arr.shape}')
    elif isinstance(arr, list):
        print_func(f'List shape: {list_shape(arr)}')
    elif isinstance(arr, tuple):
        print_func(f'Tuple shape: {tuple_shape(arr)}')
    elif isinstance(arr, sps.spmatrix):
        print_func(f'Sparse matrix shape: {arr.shape}')
    if lite:
        print_func(f'value head: {arr[:5]}')
    else:
        print_func(f'value: {arr}')


def flex_print_dict(print_func=print, dic=None, name=None, n=PRINT_WIDTH, char=PRINT_CHAR, lite=True, print_title=True, level=''):
    '''灵活打印字典信息'''
    # 这里如果print_title为True,则会打印标题(不管name是否为None),如果不打印标题,则name不为None时打印name
    if print_title:
        flex_print_title(print_func, title=name, n=n, char=char)
    elif name is not None:
        if level == '':
            print_func(f"{name}:")
        else:
            print_func(f"{level}:")
    for k, v in dic.items():
        if isinstance(v, dict):
            if level == '':
                flex_print_dict(print_func, dic=v, name=k, lite=lite, print_title=False, level=f"{k}")
            else:
                flex_print_dict(print_func, dic=v, name=k, lite=lite, print_title=False, level=f"{level}['{k}']")
        else:
            if level == '':
                flex_better_print(print_func, variable=v, name=f'{k}', lite=lite, print_title=False)
            else:
                flex_better_print(print_func, variable=v, name=f"{level}['{k}']", lite=lite, print_title=False)


def flex_print_set(print_func=print, st=None, name=None, n=PRINT_WIDTH, char=PRINT_CHAR, lite=True, print_title=True):
    '''灵活打印集合信息'''
    if print_title:
        flex_print_title(print_func, title=name, n=n, char=char)
    elif name is not None:
        # print_func(f'# {name}:')
        print_func(f'{name}:')
    print_func(f'value num: {len(st)}')
    if lite:
        print_func(f'value head: {list(st)[:5]}')
    else:
        print_func(f'value: {st}')


def flex_better_print(print_func=print, variable=None, name=None, n=PRINT_WIDTH, char=PRINT_CHAR, lite=True, print_title=True):
    '''更好的打印'''
    if isinstance(variable, pd.DataFrame):
        flex_print_df(print_func, df=variable, name=name, n=n, char=char, lite=lite, print_title=print_title)
    elif isinstance(variable, (np.ndarray, list, tuple, sps.spmatrix)):
        flex_print_array(print_func, arr=variable, name=name, n=n, char=char, lite=lite, print_title=print_title)
    elif isinstance(variable, dict):
        flex_print_dict(print_func, dic=variable, name=name, n=n, char=char, lite=lite, print_title=print_title)
    elif isinstance(variable, set):
        flex_print_set(print_func, st=variable, name=name, n=n, char=char, lite=lite, print_title=print_title)
    elif isinstance(variable, (int, float, str, np.float128, np.float64, np.float32, np.float16, np.int64, np.int32, np.int16, np.int8)):
        # 对于这几个类型,不论lite是否为True,都会打印出变量值
        flex_print_type(print_func, variable=variable, name=name, n=n, char=char, lite=False, print_title=print_title)
    else:
        flex_print_type(print_func, variable=variable, name=name, n=n, char=char, lite=lite, print_title=print_title)


def print_sep(n=PRINT_WIDTH, char=PRINT_CHAR):
    '''打印分隔符'''
    flex_print_sep(print_func=print, n=n, char=char)


def print_title(title, n=PRINT_WIDTH, char=PRINT_CHAR):
    '''打印标题'''
    flex_print_title(print_func=print, title=title, n=n, char=char)


def print_with_name(variable, name):
    '''打印变量名和变量值'''
    flex_print_with_name(print_func=print, variable=variable, name=name)


def print_type(variable, name=None, n=PRINT_WIDTH, char=PRINT_CHAR, lite=True, print_title=True):
    '''打印变量类型'''
    flex_print_type(print_func=print, variable=variable, name=name, n=n, char=char, lite=lite, print_title=print_title)


def print_df(df, name=None, n=PRINT_WIDTH, char=PRINT_CHAR, lite=True, print_title=True):
    '''打印DataFrame'''
    flex_print_df(print_func=print, df=df, name=name, n=n, char=char, lite=lite, print_title=print_title)


def print_array(arr, name=None, n=PRINT_WIDTH, char=PRINT_CHAR, lite=True, print_title=True):
    '''打印数组'''
    flex_print_array(print_func=print, arr=arr, name=name, n=n, char=char, lite=lite, print_title=print_title)


def print_dict(dic, name=None, n=PRINT_WIDTH, char=PRINT_CHAR, lite=True, print_title=True):
    '''打印字典'''
    flex_print_dict(print_func=print, dic=dic, name=name, n=n, char=char, lite=lite, print_title=print_title)


def better_print(variable, name=None, n=PRINT_WIDTH, char=PRINT_CHAR, lite=True, print_title=True):
    '''更好的打印'''
    flex_better_print(print_func=print, variable=variable, name=name, n=n, char=char, lite=lite, print_title=print_title)


def bprt(variable, name=None, n=PRINT_WIDTH, char=PRINT_CHAR, lite=True, print_title=True):
    '''更好的打印,bp为better_print的缩写(防止和brainpy冲突所以叫做bprt)'''
    better_print(variable, name=name, n=n, char=char, lite=lite, print_title=print_title)
# endregion


# region log
class Capturing(list):
    '''
    利用此类,和with语句,可以捕捉print输出,并保存到log中(见Logger类的capture方法)
    '''
    def __init__(self, loggers=None, print_to_console=True):
        super().__init__()
        self.loggers = loggers if isinstance(loggers, list) else [loggers]  # Store multiple loggers in a list
        self.print_to_console = print_to_console  # Control real-time console printing
        self._buffer = ''  # Temporary storage to reduce unnecessary newlines

    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self
        return self

    def __exit__(self, *args):
        sys.stdout = self._stdout

    def write(self, message):
        # Accumulate content in the buffer
        self._buffer += message
        # Process buffer when a newline character is encountered
        if '\n' in self._buffer:
            self._log_message(self._buffer)
            self._buffer = ''  # Clear the buffer

    def flush(self):
        # Ensure any remaining content is written out when flush() is called
        if self._buffer:
            self._log_message(self._buffer)
            self._buffer = ''

    def _log_message(self, message):
        message = message.strip()  # Strip extra whitespace and newlines
        if message:  # Ensure it's not an empty message
            self.append(message)
            if self.print_to_console:
                self._stdout.write(message + '\n')
                self._stdout.flush()
            # Log message to each logger if loggers are provided
            if self.loggers:
                for logger in self.loggers:
                    logger.add(message, end='\n')


class Logger:
    '''
    需要利用logger.prt等来同时打印和记录log
    add方法只记录log不打印
    
    使用方法:

    - 使用logger的prt
    logger = Logger()
    logger.prt('message')
    
    - 利用python logging库实现,实时保存到文件
    logger.get_py_logger(filename='log.txt', name='logger')
    logger.py_logger即为python logging库的logger,可以按照python logging库的方式使用

    - 捕捉其他函数的print
    new_func = logger.capture(func)
    new_func(*args, **kwargs)

    - 捕捉其他class的print
    new_class = logger.capture(class)
    nc = new_class(*args, **kwargs)
    nc.some_method()

    - 捕捉某个代码块的print
    with Capturing(loggers=logger):
        some_code

    - 使用多个logger同时捕捉某个代码块的print
    with Capturing(loggers=[logger1, logger2]):
        some_code
    
    - 不使用with语句,直接捕捉print(记得手动调用end)
    cm = CaptureManager(loggers=[logger1, logger2]) 不需要调用enter,init中已经调用了
    some_code
    cm.end()

    - CaptureManager也可以使用with语句
    
    - 只添加log不打印
    logger.add('message')
    '''
    def __init__(self, n=PRINT_WIDTH, char=PRINT_CHAR, prt_mode=True, lite=True, print_title=True):
        '''
        prt_mode -- 是否打印
        '''
        # 自定义的log和参数
        self.log = []
        self.n = n
        self.char = char
        self.prt_mode = prt_mode
        self.lite = lite
        self.print_title = print_title
        # python logging库的logger(先设置为None)
        self.py_logger = None

    def get_py_logger(self, basedir, filename=None, name='', level=logging.DEBUG, format=None, datefmt='%Y-%m-%d %H:%M:%S', filemode='w', print_to_console=False, create_dir=True):
        '''
        注意: 特别不建议打开print_to_console,因为使用普通的抓取时,由于记录到logger中,这里会重复打印一遍
        '''
        self.basedir = basedir
        if create_dir:
            # 创建文件夹
            mkdir(self.basedir)

        # 获取filename
        if filename is None:
            filename = 'full.log'
        filename = safe_path_join(self.basedir, filename)

        # 设置format
        if format is None:
            if name == '':
                format = '%(asctime)s - %(levelname)s - %(message)s'
            else:
                format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

        # 获取python logging库的logger,注意,获得py_logger后,本类的所有方法才会记录到py_logger中
        self.py_logger = logging.getLogger(name)
        self.py_logger.propagate = False

        # 配置日志格式
        formatter = logging.Formatter(format, datefmt=datefmt)

        # 移除之前的所有处理器
        self.py_logger.handlers.clear()

        # 配置文件处理器
        if filename:
            file_handler = logging.FileHandler(filename, mode=filemode, delay=True)
            file_handler.setFormatter(formatter)
            self.py_logger.addHandler(file_handler)

        # 配置控制台处理器(如果设置了会同时打印到控制台)
        if print_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.py_logger.addHandler(console_handler)

        # 设置日志级别
        self.py_logger.setLevel(level)

    def prt(self, *message, end='\n'):
        if self.prt_mode:
            print(*message)
        for i, m in enumerate(message):
            self.log.append(m)
            if i < len(message) - 1:
                s = ' '
            else:
                s = end
            if s is not None:
                self.log.append(s)
        if self.py_logger is not None:
            # 如果message只是一个'\n',则跳过
            if len(message) == 1 and message[0] == '\n':
                pass
            else:
                self.py_logger.info(' '.join(message))

    def prt_sep(self):
        flex_print_sep(print_func=self.prt, n=self.n, char=self.char)

    def prt_title(self, title):
        flex_print_title(print_func=self.prt, title=title, n=self.n, char=self.char)

    def prt_with_name(self, variable, name):
        flex_print_with_name(print_func=self.prt, variable=variable, name=name)

    def prt_type(self, variable, name=None):
        flex_print_type(print_func=self.prt, variable=variable, name=name, n=self.n, char=self.char, lite=self.lite, print_title=self.print_title)

    def prt_df(self, df, name=None):
        flex_print_df(print_func=self.prt, df=df, name=name, n=self.n, char=self.char, lite=self.lite, print_title=self.print_title)

    def prt_array(self, arr, name=None):
        flex_print_array(print_func=self.prt, arr=arr, name=name, n=self.n, char=self.char, lite=self.lite, print_title=self.print_title)

    def prt_dict(self, dic, name=None):
        flex_print_dict(print_func=self.prt, dic=dic, name=name, n=self.n, char=self.char, lite=self.lite, print_title=self.print_title)

    def better_prt(self, variable, name=None):
        flex_better_print(print_func=self.prt, variable=variable, name=name, n=self.n, char=self.char, lite=self.lite, print_title=self.print_title)

    def func_prter(self, func, multi_result=False):
        return flex_func_printer(func, print_func=self.prt, lite=self.lite, multi_result=multi_result)
    
    def func_tester(self, func, multi_result=False):
        return flex_func_tester(func, print_func=self.prt, lite=self.lite, multi_result=multi_result)

    def add(self, *message, end='\n'):
        for i, m in enumerate(message):
            self.log.append(m)
            if i < len(message) - 1:
                s = ' '
            else:
                s = end
            if s is not None:
                self.log.append(s)
        if self.py_logger is not None:
            # 如果message只是一个'\n',则跳过
            if len(message) == 1 and message[0] == '\n':
                pass
            else:
                self.py_logger.info(' '.join(message))

    def add_sep(self):
        flex_print_sep(print_func=self.add, n=self.n, char=self.char)

    def add_title(self, title):
        flex_print_title(print_func=self.add, title=title, n=self.n, char=self.char)

    def add_with_name(self, variable, name):
        flex_print_with_name(print_func=self.add, variable=variable, name=name)

    def add_type(self, variable, name=None):
        flex_print_type(print_func=self.add, variable=variable, name=name, n=self.n, char=self.char, lite=self.lite, print_title=self.print_title)

    def add_df(self, df, name=None):
        flex_print_df(print_func=self.add, df=df, name=name, n=self.n, char=self.char, lite=self.lite, print_title=self.print_title)

    def add_array(self, arr, name=None):
        flex_print_array(print_func=self.add, arr=arr, name=name, n=self.n, char=self.char, lite=self.lite, print_title=self.print_title)

    def add_dict(self, dic, name=None):
        flex_print_dict(print_func=self.add, dic=dic, name=name, n=self.n, char=self.char, lite=self.lite, print_title=self.print_title)

    def better_add(self, variable, name=None):
        flex_better_print(print_func=self.add, variable=variable, name=name, n=self.n, char=self.char, lite=self.lite, print_title=self.print_title)

    def add_func_prter(self, func, multi_result=False):
        '''将func_prter仅添加到log中'''
        return flex_func_printer(func, print_func=self.add, lite=self.lite, multi_result=multi_result)

    def add_func_tester(self, func, multi_result=False):
        '''将func_tester仅添加到log中'''
        return flex_func_tester(func, print_func=self.add, lite=self.lite, multi_result=multi_result)

    def capture(self, obj, print_to_console=True):
        '''
        收集obj中的print到log中
        如果obj是函数,则捕获其print输出
        如果obj是类,则捕获类中所有方法的print输出
        
        参数:
        - obj: 要装饰的函数或类
        - print_to_console: 是否打印到控制台,默认为True
        
        返回:
        - 装饰后的函数或类
        '''
        if isinstance(obj, type):  # 检查是否为类
            # 创建一个新的类，继承自原始类
            class_name = obj.__name__
            decorated_class = type(class_name, (obj,), {})
            
            # 遍历类中的所有方法
            for attr_name, attr_value in obj.__dict__.items():
                if callable(attr_value):  # 只处理可调用的方法
                    # 将捕获装饰器应用到每个方法
                    setattr(decorated_class, attr_name, 
                        self._capture_method(attr_value, print_to_console))
            return decorated_class
        elif callable(obj):  # 如果是函数或方法
            return self._capture_method(obj, print_to_console)
        else:
            raise TypeError("capture只支持函数或类")

    def _capture_method(self, method, print_to_console):
        '''
        为函数或方法应用捕获装饰器
        '''
        @wraps(method)
        def wrapper(*args, **kwargs):
            with Capturing(loggers=self, print_to_console=print_to_console):
                return method(*args, **kwargs)
        return wrapper

    def save(self, basedir=None, filename=None, mode='w'):
        if self.py_logger is not None:
            basedir = self.basedir
        mkdir(basedir)
        if filename is None:
            filename = 'brief.log'
        elif not filename.endswith('.log'):
            filename += '.log'
        with open(os.path.join(basedir, filename), mode=mode) as f:
            for line in self.log:
                f.write(line)


class CaptureManager:
    '''
    输入loggers,开始后,会将print输出保存到log中(initialize后,立即开始捕获;使用结束后,注意调用end;或者使用with语句打开)
    '''
    def __init__(self, loggers, print_to_console=True):
        self.capture_print = Capturing(loggers=loggers, print_to_console=print_to_console)
        self.capture_print.__enter__()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.capture_print.__exit__()
    
    def end(self):
        self.capture_print.__exit__()
# endregion


# region decorator
def select_output(index):
    """
    装饰器工厂，允许指定要提取的返回值位置
    
    输入: index, 指定被装饰函数需要的返回值的索引
    输出: 装饰器

    例子:
    直接装饰
    @select_output(0)
    def f(a, b):
        return a, b

    复合使用
    select_output(index=1)(f)(a, b)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            # 如果是序列类型（如元组、列表）则按索引提取
            if hasattr(result, "__getitem__"):
                return result[index]
            else:
                raise ValueError("返回值不是可索引的序列类型")
        return wrapper
    return decorator
# endregion


# region class decorator
def method_decorator(decorator):
    '''
    将装饰器应用于类的所有方法

    使用方法:
    已有一个decorator,可以使用如下方式将其应用到类的所有方法
    @class_decorator(decorator)
    '''
    def decorate(cls):
        for attr_name, attr_value in cls.__dict__.items():
            if callable(attr_value):  # 只处理可调用的方法
                # 将装饰器应用到每个方法
                setattr(cls, attr_name, decorator(attr_value))
        return cls
    return decorate
# endregion


# region class method recorder
class MethodRecorder:
    '''
    使用例子:
    class MyClass:
        def __init__(self):
            self.recorder = MethodRecorder(self)

        def my_method(self):
            print("This is my method")
    my_instance = MyClass()
    my_instance.my_method()  # 调用方法
    print(my_instance.recorder.was_called('my_method'))  # 检查方法是否被调用过
    '''
    def __init__(self, target_instance, methods_to_record=None):
        """
        :param target_instance: 要记录的目标实例
        :param methods_to_record: 要记录的方法名称列表,None 表示记录所有公共方法
        """
        self._call_records = {}
        self._methods_to_record = methods_to_record
        self._decorate_instance_methods(target_instance)
    
    def _decorate_instance_methods(self, instance):
        cls = instance.__class__
        for name, attr in inspect.getmembers(cls, inspect.isfunction):
            # 跳过特殊方法和私有方法
            if name.startswith("__") and name.endswith("__"):
                continue
            
            # 如果指定了要记录的方法列表,且当前方法不在列表中,则跳过
            if self._methods_to_record and name not in self._methods_to_record:
                continue
            
            # 创建方法装饰器
            decorated = self._create_recorder_decorator(attr)
            # 绑定装饰后的方法到实例
            setattr(instance, name, types.MethodType(decorated, instance))
    
    def _create_recorder_decorator(self, method):
        @wraps(method)
        def recorder_wrapper(instance, *args, **kwargs):
            # 调用实际方法前记录
            self.record(method.__name__)
            return method(instance, *args, **kwargs)
        return recorder_wrapper
    
    def record(self, method_name):
        self._call_records[method_name] = True
    
    def was_called(self, method_name):
        return self._call_records.get(method_name, False)
    
    def reset(self, method_name=None):
        if method_name:
            if method_name in self._call_records:
                del self._call_records[method_name]
        else:
            self._call_records.clear()

    def __repr__(self):
        called = [m for m in self._call_records.keys() if self._call_records[m]]
        return f"<FlexibleMethodRecorder: Called={called}>"


class CallOnceExecutor(MethodRecorder):
    def __init__(self, target_instance, methods_to_protect=None, action="skip"):
        """
        :param target_instance: 要保护的目标实例
        :param methods_to_protect: 需要保护的方法名称列表
        :param action: 重复调用时的行为 - "skip"(跳过) 或 "block"(抛出异常)
        """
        super().__init__(target_instance, methods_to_protect)
        self.action = action  # 重复调用时的行为
    
    def _create_recorder_decorator(self, method):
        @wraps(method)
        def executor_wrapper(instance, *args, **kwargs):
            method_name = method.__name__
            
            # 检查方法是否已被调用过
            if self.was_called(method_name):
                # 根据设置的行为处理重复调用
                if self.action == "skip":
                    print(f"Warning: 跳过重复执行的 '{method_name}' 方法")
                    return None  # 直接返回 None
                elif self.action == "block":
                    raise RuntimeError(f"方法 '{method_name}' 已被执行过,禁止二次执行")
                else:
                    # 默认行为 - 只是记录,不阻止执行
                    print(f"Notice: 方法 '{method_name}' 重复执行")
            
            # 记录方法调用
            self.record(method_name)
            
            # 执行原始方法
            try:
                return method(instance, *args, **kwargs)
            except Exception as e:
                # 如果执行失败,重置记录以便可以重试
                self.reset(method_name)
                raise e
        
        return executor_wrapper
    
    def can_execute(self, method_name):
        """检查方法是否可以执行（未被调用过）"""
        return not self.was_called(method_name)
    
    def get_action(self):
        """获取当前设置的行为模式"""
        return self.action
    
    def set_action(self, new_action):
        """设置新的行为模式"""
        valid_actions = ["skip", "block", "record_only"]
        if new_action in valid_actions:
            self.action = new_action
        else:
            raise ValueError(f"无效的行为: {new_action}. 可选值: {valid_actions}")
    
    def __repr__(self):
        action_desc = {
            "skip": "跳过重复执行",
            "block": "阻止重复执行",
            "record_only": "仅记录"
        }
        return f"<CallOnceExecutor: 模式={action_desc.get(self.action, '未知')}, {super().__repr__()}>"
# endregion


# region 测试相关函数
def flex_func_printer(func, print_func=print, lite=True, multi_result=False):
    '''
    打印出函数的所有输入,输出以及变量名

    参数:
    func -- 要测试的函数
    print_func -- 打印函数,默认为print
    lite -- 打印是否简化,默认为True
    multi_result -- 函数是否有多个返回值,默认为False

    返回:
    一个函数,用于测试func

    使用例子:
    @flex_func_printer
    def some_function():
        # some code here
    
    flex_func_printer(some_function)(*args, **kwargs)
    '''
    @wraps(func)
    def wrapper(*args, **kwargs):
        flex_print_title(print_func, f'Function: {func.__name__}')
        
        # 打印 args 及其位置索引
        flex_print_title(print_func, 'Arguments')
        sig = inspect.signature(func)
        params = list(sig.parameters.values())
        for i, arg in enumerate(args):
            flex_better_print(print_func, arg, name=params[i].name, lite=lite, print_title=False)
        
        # 打印 kwargs 及其变量名
        flex_print_title(print_func, 'Keyword Arguments')
        flex_print_dict(print_func, kwargs, name='Keyword Arguments', lite=lite, print_title=False)
        
        result = func(*args, **kwargs)
        
        # 打印结果
        flex_print_title(print_func, 'Result')
        if multi_result:
            for i, r in enumerate(result):
                flex_better_print(print_func, r, name=f'result_{i}', lite=lite, print_title=False)
        else:
            flex_better_print(print_func, result, name='result', lite=lite, print_title=False)
        return result
    return wrapper


def flex_func_tester(func, print_func=print, lite=True, multi_result=False):
    '''
    计时,打印出函数的所有输入,输出以及变量名

    参数:
    func -- 要测试的函数
    print_func -- 打印函数,默认为print
    lite -- 打印是否简化,默认为True
    multi_result -- 函数是否有多个返回值,默认为False

    返回:
    一个函数,用于测试func

    使用例子:
    @flex_func_tester
    def some_function():
        # some code here
    
    flex_func_tester(some_function)(*args, **kwargs)
    '''
    return flex_func_printer(func_timer(func), print_func, lite, multi_result)


def message_decorator(message, print_func=print):
    """
    一个可以自定义输出消息的装饰器函数

    该装饰器可以用于装饰函数或类,在函数调用或类实例化时,
    会打印一个自定义的消息

    参数:
    message (str): 要输出的消息,可以使用 {0} 占位符来表示被装饰的对象名称

    返回:
    function: 返回一个装饰器函数

    使用示例:
    @message_decorator("The {0} needs improvement.")
    def my_function():
        pass

    @message_decorator("The {0} needs to be tested.")
    class MyClass:
        pass

    # 输出:
    # The my_function needs improvement.
    # The MyClass needs to be tested.
    """
    def decorator(obj):
        if inspect.isfunction(obj):
            @wraps(obj)
            def wrapper(*args, **kwargs):
                print_func(message.format(obj.__name__))
                return obj(*args, **kwargs)
            return wrapper
        elif inspect.isclass(obj):
            for name, method in obj.__dict__.items():
                if callable(method):
                    if name == '__init__':
                        message_formatted_by_cls_name = message.format(obj.__name__)
                        setattr(obj, name, message_decorator(message=message_formatted_by_cls_name, print_func=print_func)(method))
            return obj
        else:
            raise TypeError("This decorator supports only classes and functions.")
    return decorator


def to_be_improved(obj):
    return message_decorator("The '{0}' needs improvement.")(obj)


def to_be_test(obj):
    return message_decorator("The '{0}' needs to be tested.")(obj)


def deprecated(obj):
    return message_decorator("The '{0}' is deprecated.")(obj)


def not_recommend(obj):
    return message_decorator("The '{0}' is not recommended.")(obj)


def direct_use(obj):
    return message_decorator("Directly use the code in '{0}'.")(obj)


def func_printer(func, lite=True, multi_result=False):
    '''
    打印出函数的所有输入,输出以及变量名

    参数:
    func -- 要测试的函数

    返回:
    一个函数,用于测试func

    使用例子:
    @func_printer
    def some_function():
        # some code here
    
    func_printer(some_function)(*args, **kwargs)
    '''
    return flex_func_printer(func, print, lite, multi_result)


def func_tester(func, lite=True, multi_result=False):
    '''
    计时,打印出函数的所有输入,输出以及变量名

    参数:
    func -- 要测试的函数

    返回:
    一个函数,用于测试func

    使用例子:
    @func_tester
    def some_function():
        # some code here
    
    func_tester(some_function)(*args, **kwargs)
    '''
    return flex_func_tester(func, print, lite, multi_result)


def func_tracker(func, print_func=print_title):
    '''
    当函数被调用时,打印函数名
    '''
    return message_decorator("Running: {0}", print_func=print_func)(func)


def init_tracker(func, print_func=print_title, cls_name=None):
    '''
    实例化时打印类名
    '''
    return message_decorator(f"Instance of {cls_name} created", print_func=print_func)(func)


def cls_tracker(cls):
    '''
    遍历类的所有方法,给每个方法添加装饰器,实例化时打印类名,当方法被调用时,打印方法名
    '''
    for name, method in cls.__dict__.items():
        if callable(method):
            if name == '__init__':
                setattr(cls, name, init_tracker(method, cls_name=cls.__name__))
            else:
                setattr(cls, name, func_tracker(method))
    return cls


def tracker(obj):
    '''
    根据对象的类型,选择合适的tracker
    '''
    if inspect.isfunction(obj):
        return func_tracker(obj)
    elif inspect.isclass(obj):
        return cls_tracker(obj)


def tracker_with_time(obj):
    '''
    tracker的增强版,在打印函数名的同时,还会打印时间
    '''
    if inspect.isfunction(obj):
        return func_timer(func_tracker(obj))
    elif inspect.isclass(obj):
        return method_decorator(func_timer)(cls_tracker(obj))


def break_point(message='Break Point'):
    '''
    停止运行，打印消息

    注意:
    python有内置的breakpoint,此函数与内置的函数不同
    '''
    assert False, message


def print_current_line():
    frame = inspect.currentframe()
    print(f"Current line number: {frame.f_lineno}")


def func_profiler(func):
    """
    Decorator to profile the execution of a function.
    
    Parameters:
    - func: The function to profile.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        with Profile() as profiler:
            result = profiler.runcall(func, *args, **kwargs)
        
        # 创建Stats对象来查看分析结果
        stats = Stats(profiler)
        
        # 清理文件路径信息，使输出更简洁
        stats.strip_dirs()
        
        # 按累积时间排序并打印统计信息
        stats.sort_stats('cumulative')
        stats.print_stats()
        
        # 打印哪些函数调用了目标函数
        print("\nCallers for each function:")
        stats.print_callers()

        # 返回函数结果
        return result
    
    return wrapper


def func_line_profiler(func):
    '''
    记录函数的每一行的执行时间
    '''
    # 创建 LineProfiler 实例，并包装目标函数
    profiler = LineProfiler()
    profiled_func = profiler(func)  # 将函数注册到分析器
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            # 执行被分析的函数
            result = profiled_func(*args, **kwargs)
        finally:
            # 输出逐行分析结果（单位：微秒）
            profiler.print_stats(output_unit=1e-6)
        return result
    return wrapper


def func_memory_tracer(func):
    '''
    记录函数的内存使用情况
    '''
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 检查当前是否已启用跟踪
        original_tracing = tracemalloc.is_tracing()
        if not original_tracing:
            tracemalloc.start()  # 若未启用，则启动跟踪
        
        # 拍摄执行前的内存快照
        snapshot_before = tracemalloc.take_snapshot()
        
        # 执行被装饰的函数
        result = func(*args, **kwargs)
        
        # 拍摄执行后的内存快照
        snapshot_after = tracemalloc.take_snapshot()
        
        # 对比快照，按内存占用排序
        top_stats = snapshot_after.compare_to(snapshot_before, 'lineno')
        
        # 打印内存分析结果
        print(f"\nMemory usage analysis for {func.__name__}:")
        for stat in top_stats[:5]:  # 显示前5个内存分配点
            print(f"  {stat}")
        
        # 如果装饰器自己启用了跟踪，则停止
        if not original_tracing:
            tracemalloc.stop()
        
        return result
    return wrapper


def get_detailed_error():
    exc_type, exc_value, exc_trace = sys.exc_info()
    if not exc_type:
        return
    
    # 获取错误基本信息
    error_name = exc_type.__name__
    error_msg = str(exc_value)
    
    # 获取最后一条堆栈记录
    stack_summary = traceback.extract_tb(exc_trace)
    last_frame = stack_summary[-1] if stack_summary else None
    
    # 构造详细日志
    error_info = f"""
    ======== 错误详情 ========
    错误类型: {error_name}
    错误信息: {error_msg}
    发生文件: {last_frame.filename if last_frame else 'unknown'}
    错误行号: {last_frame.lineno if last_frame else 0}
    问题代码: {last_frame.line if last_frame else 'unknown'}
    ======== 堆栈跟踪 ========
    {traceback.format_exc()}
    """
    return error_info


class FlexibleTry:
    '''
    示例:
    with FlexibleTry(enable_try=True):
        print(1/0)  # 会捕获异常并打印详细错误信息
    
    with FlexibleTry(enable_try=False):
        print(1/0)  # 会抛出异常
    
    with FlexibleTry(enable_try=True) as ft:
        print(1/0)  # 会捕获异常并打印详细错误信息
    print(ft.success)  # True or False, 是否成功
    print(ft.exception)  # 如果发生异常, 则为异常对象, 否则为None
    '''
    def __init__(self, enable_try=True, print_info=True):
        self.enable_try = enable_try
        self.print_info = print_info
        
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 没有发生异常
        if exc_type is None:
            self.success = True
            self.exception = None
            return False
        
        # 发生异常时的处理
        self.success = False
        self.exception = exc_val
        
        if self.enable_try and exc_type:
            # 当启用异常捕获且发生异常时，获取详细错误信息
            error_info = get_detailed_error()
            if self.print_info:
                print(error_info)  # 打印详细错误信息
            return True  # 吞掉异常
        
        return False  # 抛出异常


def flexible_try(enable_try=True, print_info=True):
    """
    异常处理装饰器,可同时用于函数和类方法
    
    参数:
        enable_try: 是否启用异常捕获 (默认True)
        
    示例:
        @flexible_try(enable_try=True)
        def func():
            ...
            
        class MyClass:
            @flexible_try(enable_try=False)
            def method(self):
                ...
                
            @classmethod
            @flexible_try(enable_try=True)
            def class_method(cls):
                ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if enable_try:
                try:
                    return func(*args, **kwargs)
                except Exception:
                    # 捕获并打印详细错误信息
                    error_info = get_detailed_error()
                    if print_info:
                        print(f"{func.__name__}:\n{error_info}")
                    return None  # 可以选择返回默认值或None
            else:
                # 直接执行不捕获异常
                return func(*args, **kwargs)
        return wrapper
    return decorator
# endregion


# region 文件操作相关函数
@message_decorator("it will always print the common_functions.py. If you want to use it in another file, you can directly copy the code in the file, or in a better way, just use the current_file function")
def current_file_py():
    '''获取当前文件名(不带路径,带后缀)'''
    return os.path.basename(__file__)


def caller_filename():
    '''
    打印调用此函数的文件名(带整个路径)

    如果是在Jupyter Notebook中运行,返回的是一个临时文件名,请使用current_file_ipynb函数获取当前文件名
    如果是在.py文件中运行,返回的是调用者的文件名(这也是为什么current_file需要显式复制这段代码,而不是调用这个函数,因为在common_functions.py中调用这个函数会返回common_functions.py)
    '''
    # 获取调用栈
    stack = inspect.stack()
    # 获取调用者的帧信息
    caller_frame = stack[1]
    # 获取调用者文件名
    caller_filename = caller_frame.filename
    # 打印文件名
    return caller_filename


def current_file_ipynb():
    '''获取当前文件名(带整个路径)'''
    from IPython import get_ipython
    ip = get_ipython()
    path = None
    if '__vsc_ipynb_file__' in ip.user_ns:
        path = ip.user_ns['__vsc_ipynb_file__']
    return path


def current_file():
    '''获取当前文件名(带整个路径)'''
    try:
        return current_file_ipynb()
    except:
        # 获取调用栈
        stack = inspect.stack()
        # 获取调用者的帧信息
        caller_frame = stack[1]
        # 获取调用者文件名
        caller_filename = caller_frame.filename
        # 打印文件名
        return caller_filename


def current_filename():
    '''获取当前文件名(不带后缀,不带路径)'''
    try:
        caller_filename = current_file_ipynb()
    except:
        # 获取调用栈
        stack = inspect.stack()
        # 获取调用者的帧信息
        caller_frame = stack[1]
        # 获取调用者文件名
        caller_filename = caller_frame.filename
        # 打印文件名
    return os.path.splitext(os.path.basename(caller_filename))[0]


def current_dir():
    '''获取当前路径(并不是common_functions.py所在的路径,而是调用common_functions.py的文件所在的路径)'''
    return os.getcwd()


def get_parent_dir(basedir):
    return Path(basedir).parent


def get_script_name(extension=True):
    '''
    获取当前脚本的文件名。如果在 Jupyter Notebook 中运行，则返回 None 并显示警告。

    参数:
    extension (bool): 是否包括文件扩展名，默认为 True。

    返回:
    str or None: 当前脚本的文件名或 None。
    '''
    try:
        # 尝试检测是否在 Jupyter Notebook 环境中运行
        from IPython import get_ipython
        ipython = get_ipython()
        if ipython and 'IPKernelApp' in ipython.config:
            # 在 Jupyter Notebook 中运行
            print("Warning: Running in a Jupyter Notebook environment. Unable to determine script name.")
            return None
    except ImportError:
        pass  # 不在 Jupyter 环境中，忽略

    # 对于非 Jupyter 环境，按常规方式获取脚本名称
    caller_frame = inspect.stack()[1]
    caller_script_path = caller_frame.filename
    caller_script_name = os.path.basename(caller_script_path)
    
    if not extension:
        caller_script_name = os.path.splitext(caller_script_name)[0]
    
    return caller_script_name


def safe_path_join(*paths):
    '''
    os.path.join的再封装,保证不会因为中间的多余/或\导致路径不是预想的路径
    并且如果输入了空字符串或者None,则会自动忽略
    '''
    # 清理每个路径，去除前导和尾随的斜杠（只保留中间路径的斜杠）
    cleaned_paths = []
    for i, path in enumerate(paths):
        if path is None or path == "":
            continue
        else:
            if i == 0:
                # 第一个路径可以保留前导的斜杠（如果是绝对路径），但去掉尾部的斜杠
                cleaned_paths.append(path.rstrip("/\\"))
            else:
                # 后续路径要去掉前导和尾随斜杠
                cleaned_paths.append(path.strip("/\\"))
    
    # 使用 os.path.join 拼接清理后的路径
    return os.path.join(*cleaned_paths)


def pj(*args):
    '''
    极度缩写的safe_path_join
    '''
    return safe_path_join(*args)


def split_path(path, mode="full"):
    """
    将路径分离成每一层的文件夹/文件名,根据模式选择分割方式
    
    参数:
        path (str): 需要分离的文件路径
        mode (str): 分割模式,"full" 表示完全分开,"cumulative" 表示逐层累积,默认 "full"
    
    返回:
        list: 路径的分层列表,根据模式选择分割方式
    
    使用例子:
        split_path("/home/user/documents/file.txt", mode="full")  
        # 返回 ['/', 'home', 'user', 'documents', 'file.txt']

        split_path("/home/user/documents/file.txt", mode="cumulative")  
        # 返回 ['/', '/home', '/home/user', '/home/user/documents', '/home/user/documents/file.txt']
    """
    parts = []
    cumulative_parts = []

    while True:
        path, tail = os.path.split(path)
        if tail:
            parts.insert(0, tail)
        else:
            if path:
                parts.insert(0, path)
            break
    
    if mode == "cumulative":
        for i in range(1, len(parts) + 1):
            cumulative_parts.append(os.path.join(*parts[:i]))
        return cumulative_parts
    elif mode == "full":
        return parts
    else:
        raise ValueError("Invalid mode. Choose 'full' or 'cumulative'.")


def mkdir(fn):
    '''创建文件夹及中间文件夹'''
    os.makedirs(fn, exist_ok=True)


def rmdir(fn):
    '''删除文件夹及其所有内容'''
    if os.path.isdir(fn):
        shutil.rmtree(fn, ignore_errors=True)


def cpdir(src, dst, dirs_exist_ok=True, overwrite=False):
    '''
    复制一个文件夹下面的所有内容(不是大文件夹本身)到另一个文件夹里面

    参数:
    - src: 源文件夹
    - dst: 目标文件夹
    - dirs_exist_ok: 是否允许目标文件夹存在,默认为True,允许目标文件夹存在
    - overwrite: 是否允许覆盖目标文件夹,默认为False,不允许覆盖目标文件夹

    注意:
    如果你想要把整个文件夹复制到另一个文件夹,方案一是把dst内创建一个同名文件夹,然后调用此函数;方案二是使用cp_as_subdir函数
    '''
    if not os.path.isdir(src):
        raise ValueError("源路径不是一个目录")

    if not os.path.exists(dst):
        mkdir(dst)
    elif not dirs_exist_ok:
        raise FileExistsError("目标目录已存在")

    for root, dirs, files in os.walk(src):
        # 构造当前遍历的目录在目标路径中的对应路径
        rel_path = os.path.relpath(root, src)
        dst_path = os.path.join(dst, rel_path)

        if not os.path.exists(dst_path):
            mkdir(dst_path)

        for file in files:
            src_file_path = os.path.join(root, file)
            dst_file_path = os.path.join(dst_path, file)

            # 如果不允许覆盖，且目标文件已存在，则跳过
            if not overwrite and os.path.exists(dst_file_path):
                continue

            shutil.copy2(src_file_path, dst_file_path)


def cp_as_subdir(src, dst, dirs_exist_ok=True, overwrite=False):
    '''
    复制一个文件夹到另一个文件夹里面
    
    参数:
    - src: 源文件夹
    - dst: 目标文件夹
    - dirs_exist_ok: 是否允许目标文件夹存在,默认为True,允许目标文件夹存在
    - overwrite: 是否允许覆盖目标文件夹,默认为False,不允许覆盖目标文件夹
    '''
    # 获取src的最后一个文件夹名
    src_folder_name = os.path.basename(src)

    # 构建最终目标路径
    final_dst = os.path.join(dst, src_folder_name)

    # 调用cpdir函数
    cpdir(src, final_dst, dirs_exist_ok=dirs_exist_ok, overwrite=overwrite)


def cp_file(src, dst, overwrite=False):
    """
    复制文件从 src 到 dst

    :param src: 源文件路径
    :param dst: 目标文件路径或目标文件夹路径
    :param overwrite: 是否覆盖目标文件，默认为 True
    """
    # 如果 dst 是文件夹，将其转化为完整路径
    if os.path.isdir(dst):
        dst = os.path.join(dst, os.path.basename(src))
    
    # 检查是否需要跳过已存在的目标文件
    if not overwrite and os.path.exists(dst):
        print(f"目标文件已存在，跳过复制: {dst}")
        return

    shutil.copy(src, dst)


def mvdir(src, dst, overwrite=False):
    '''
    移动一个目录的最后一个层级到另一个目录里
    例子:
    mvdir('folder/subfolder', 'newfolder/')  # 将subfolder移动到newfolder下

    参数:
    - src: 源目录
    - dst: 目标目录
    - overwrite: 是否允许覆盖目标目录,默认为False,不允许覆盖目标目录
    '''
    # 如果src最后一个字符是'/',则去掉
    if src.endswith('/'):
        local_src = src[:-1]
    else:
        local_src = src

    # 使用os.path.basename直接获取目录名,无需手动查找最后一个'/'的位置
    dir_name = os.path.basename(local_src)

    # 构建最终目标路径,假设要保持原始目录名称
    final_dst = os.path.join(dst, dir_name)

    if overwrite:
        shutil.move(local_src, final_dst)
    else:
        # 检查目标路径是否存在
        if os.path.exists(final_dst):
            # 如果目标路径已存在,打印提示信息并终止操作
            print(
                f"The target folder '{final_dst}' already exists. Operation aborted.")
        else:
            # 如果目标路径不存在,执行移动操作
            shutil.move(local_src, final_dst)


def mv_file(source_file_path, destination_folder_path):
    """
    Moves a file from the source path to the destination folder.

    :param source_file_path: The full path of the file to be moved.
    :param destination_folder_path: The folder where the file should be moved.
    :return: The new path of the moved file.
    """
    try:
        # Ensure the destination folder exists
        mkdir(destination_folder_path)

        # Get the base name of the file
        file_name = os.path.basename(source_file_path)

        # Construct the full destination file path
        destination_file_path = os.path.join(destination_folder_path, file_name)

        # Move the file
        shutil.move(source_file_path, destination_file_path)

        # print(f"File moved to: {destination_file_path}")
        return destination_file_path
    except Exception as e:
        print(f"An error occurred while moving the file: {e}")
        return None


def delete_file(file_path):
    try:
        # 尝试删除文件
        os.remove(file_path)
        print(f"文件 '{file_path}' 已成功删除。")
    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 不存在。")
    except PermissionError:
        print(f"错误: 无权限删除文件 '{file_path}'。")
    except Exception as e:
        print(f"删除文件时发生意外错误: {e}")


def mvdir_cp_like(src, dst, dirs_exist_ok=True, overwrite=False, rm_src=True):
    '''
    将一个文件夹下面的所有内容(不包括大文件夹本身)移动到另一个文件夹里面(与cpdir类似,但是是移动而不是复制)

    参数:
    - src: 源文件夹
    - dst: 目标文件夹
    - dirs_exist_ok: 是否允许目标文件夹存在,默认为True,允许目标文件夹存在
    - overwrite: 是否允许覆盖目标文件夹中的文件,默认为False,不允许覆盖目标文件夹中的文件
    '''
    if not os.path.isdir(src):
        raise ValueError("源路径不是一个目录")

    if not os.path.exists(dst):
        os.mkdir(dst)
    elif not dirs_exist_ok:
        raise FileExistsError("目标目录已存在")

    for root, dirs, files in os.walk(src, topdown=False):
        # 构造当前遍历的目录在目标路径中的对应路径
        rel_path = os.path.relpath(root, src)
        dst_path = os.path.join(dst, rel_path)

        if not os.path.exists(dst_path):
            os.mkdir(dst_path)

        for file in files:
            src_file_path = os.path.join(root, file)
            dst_file_path = os.path.join(dst_path, file)

            # 如果不允许覆盖，并且目标文件已存在，则跳过
            if not overwrite and os.path.exists(dst_file_path):
                continue

            shutil.move(src_file_path, dst_file_path)

    # 删除源目录中已经为空的文件夹
    if rm_src:
        for root, dirs, files in os.walk(src, topdown=False):
            if not files and not dirs:
                os.rmdir(root)


def insert_dir(fn, insert_str):
    '''将文件夹下所有文件移动到insert_str文件夹下,insert_str文件夹放在fn文件夹下'''
    # 获取fn文件夹下的所有文件
    files = os.listdir(fn)
    # 创建insert_str文件夹
    mkdir(os.path.join(fn, insert_str))
    # 移动所有文件到insert_str文件夹下
    for file in files:
        shutil.move(os.path.join(fn, file), os.path.join(fn, insert_str, file))


def check_all_file_exist(*paths):
    '''
    检查所有文件路径是否存在
    参数: *paths (str): 可变数量的文件路径
    返回: bool: 仅当所有路径对应的文件/目录均存在时返回True,否则返回False
    '''
    return all(os.path.exists(path) for path in paths)


def find_files_with_any_extension(path):
    """
    查找指定路径的所有匹配文件（包括原始路径和任意后缀版本）
    
    参数:
        path (str): 文件路径
        
    返回:
        list: 包含所有匹配文件的完整路径列表（可能为空）
    """
    result = []
    
    # 如果原始路径存在（文件或目录），加入结果
    if os.path.exists(path):
        result.append(path)
    
    # 分离目录和文件名
    dir_path = os.path.dirname(path) or '.'  # 处理无目录的情况
    file_name = os.path.basename(path)
    
    # 构建匹配模式（文件名.*）
    pattern = os.path.join(dir_path, f"{file_name}.*")
    
    # 查找所有符合条件的普通文件（排除目录）
    for p in glob.glob(pattern):
        # 排除以点结尾的无效文件（如 "file."）
        if os.path.isfile(p) and not p.endswith('.'):
            # 避免重复添加原始路径（如果已存在）
            if p != path:
                result.append(p)
    
    return result


def check_all_file_exist_with_any_extension(*paths):
    """
    检查多个文件路径或该路径加任意后缀的文件是否存在
    
    参数:
        *paths (str): 可变数量的文件路径
        
    返回:
        bool: 所有路径都至少有一个匹配文件时返回True
    """
    for path in paths:
        # 如果找不到任何匹配文件，立即返回False
        if not find_files_with_any_extension(path):
            return False
    return True


def get_subfile(basedir, full=True):
    '''找到文件夹下的所有文件(非文件夹),full为True则返回全路径,否则只返回文件名'''
    if full:
        return [os.path.join(basedir, f) for f in os.listdir(basedir) if os.path.isfile(os.path.join(basedir, f))]
    else:
        return [f for f in os.listdir(basedir) if os.path.isfile(os.path.join(basedir, f))]


def get_subdir(basedir, full=True):
    '''找到文件夹下的一级子文件夹,full为True则返回全路径,否则只返回文件夹名'''
    if full:
        return [os.path.join(basedir, d) for d in os.listdir(basedir) if os.path.isdir(os.path.join(basedir, d))]
    else:
        return [d for d in os.listdir(basedir) if os.path.isdir(os.path.join(basedir, d))]


def get_first_subdir(full_dir, part_dir):
    '''
    例子:
    full_dir = '/home/user/documents/projects/some_project'
    part_dir = '/home/user/documents'
    get_first_subdir(full_dir, part_dir)  # 返回 'projects'
    '''
    full_abs = os.path.abspath(full_dir)
    part_abs = os.path.abspath(part_dir)
    try:
        common_path = os.path.commonpath([full_abs, part_abs])
    except ValueError:
        return None
    if common_path != part_abs:
        return None
    relative = os.path.relpath(full_abs, part_abs)
    parts = relative.split(os.sep)
    if not parts or parts[0] == '.':
        return None
    return parts[0]


@not_recommend
def get_common_subdir(basedir, subdir_names=None):
    '''
    生成子目录的路径列表。

    Args:
        basedir (str): 目录的路径
        subdir_names (list, optional): 子目录名称列表。默认为 ['outcomes', 'models', 'figs', 'logs', 'params']

    Returns:
        list: 子目录的完整路径列表。
    '''
    if subdir_names is None:
        subdir_names = ['outcomes', 'models', 'figs', 'logs', 'params']
    return [os.path.join(basedir, subdir_name) for subdir_name in subdir_names]


@not_recommend
def get_common_subdir_dict(basedir, subdir_names=None):
    '''
    生成子目录的路径字典
    
    Args:
        basedir (str): 目录的路径
        subdir_names (list, optional): 子目录名称列表。默认为 ['outcomes', 'models', 'figs', 'logs', 'params']

    Returns:
        dict: 子目录的完整路径字典。
    '''
    if subdir_names is None:
        subdir_names = ['outcomes', 'models', 'figs', 'logs', 'params']
    return {subdir_name: os.path.join(basedir, subdir_name) for subdir_name in subdir_names}


def find_fig(filename, order=None):
    '''查找文件夹下的图片文件'''
    if order is None:
        order = ['.png', '.pdf', '.eps']
    for ext in order:
        # Check both the original filename and the filename with each extension.
        if os.path.exists(filename) and filename.endswith(ext):
            return filename, True
        elif os.path.exists(filename+ext):
            return filename+ext, True
        elif os.path.exists(filename[-4:]+ext):
            return filename[-4:]+ext, True
    return '', False


def compare_line_count(file1, file2):
    """
    比较两个文件的行数是否相同

    可以用于粗略检查两个文件是否相同,但有可能错判恰好有相同的行数但不相同的文件
    """
    with open(file1, 'r', encoding='utf-8') as f1:
        lines1 = sum(1 for _ in f1)
    
    with open(file2, 'r', encoding='utf-8') as f2:
        lines2 = sum(1 for _ in f2)
    
    return lines1 == lines2
# endregion


# region 循环相关函数
def get_reusable_zip(*iterables):
    '''
    用于将多个可迭代对象压缩成一个列表的函数,并且当输入的可迭代对象长度不一致时,提示
    '''
    # Convert all iterables to lists to check lengths
    iterables = [list(it) for it in iterables]
    
    # Check if all iterables have the same length
    lengths = [len(it) for it in iterables]
    if len(set(lengths)) > 1:
        print("Warning: Iterables have different lengths. Zip will truncate to the shortest length.")
    
    # Return a list of tuples
    return list(zip(*iterables))
# endregion


# region enum枚举类型
def enum_tutorial():
    '''
    使用enum来创建枚举类型

    示例:
    输入函数中作为一个选项,而不是直接输入字符串,这样可以规范输入防止拼写问题,并且方便扩展
    一般来说,布尔值也不建议直接输入到函数中,而是使用枚举类型
    class some_mode(Enum):
        mode1 = 1
        mode2 = 2

    def some_func(mode):
        if mode == some_mode.mode1:
            do something
        elif mode == some_mode.mode2:
            do something
    '''
    pass
# endregion


# region 保存和加载,删除相关函数
def save_code_copy(destination_folder, code_file):
    '''
    将code保存为一个带时间戳的副本(code_file需要带后缀)

    示例:
    # 保存当前脚本
    cf.save_code_copy(basedir_code, cf.current_file())
    # 保存指定脚本
    cf.save_code_copy(basedir_code, r'../util/common_functions.py')
    '''
    os.makedirs(destination_folder, exist_ok=True)

    abs_code_path = os.path.abspath(code_file)
    
    code_name = os.path.basename(abs_code_path)
    file_prefix, file_extension = os.path.splitext(code_name)
    
    # 检查目标文件夹中是否存在行数相同的同名文件(前缀匹配)
    for existing_file in os.listdir(destination_folder):
        # 只检查前缀相同的文件
        if existing_file.startswith(file_prefix + '_') and existing_file.endswith(file_extension):
            existing_file_path = os.path.join(destination_folder, existing_file)
            # 如果找到行数相同的文件，直接返回不保存
            if compare_line_count(abs_code_path, existing_file_path):
                return
    
    # 生成带时间戳的文件名
    new_filename = f"{file_prefix}_{get_time()}{file_extension}"
    
    # 生成目标路径
    destination_path = os.path.join(destination_folder, new_filename)
    
    # 复制文件
    shutil.copy(abs_code_path, destination_path)


def save_dict_yaml(data, file_path):
    """
    将字典保存为 YAML 文件

    优势在于: 可以直接打开预览,修改
    劣势在于: numpy array等不适合保存
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def load_dict_yaml(file_path):
    """
    从 YAML 文件读取字典
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_dict_txt(filename, dict_data, key_to_save=None):
    """
    将字典数据保存为结构化文本文件
    
    参数：
    filename: 文件名(无需扩展名)
    dict_data: 要保存的字典数据
    key_to_save: 需要保存的键列表(默认保存全部键)
    """
    # 确定要保存的数据子集
    data_to_save = dict_data if key_to_save is None else {k: dict_data[k] for k in key_to_save}
    
    # 处理后缀
    if not filename.endswith('.txt'):
        filename += '.txt'

    # 创建文件夹
    mkdir(os.path.dirname(filename))

    # 保存数据到文本文件
    with open(filename, 'w') as txt_file:
        def write_dict(d, indent):
            """递归写入字典的辅助函数"""
            for key, value in d.items():
                if isinstance(value, dict):
                    txt_file.write(' ' * indent + f'{key}:\n')
                    write_dict(value, indent + 4)  # 增加缩进
                else:
                    txt_file.write(' ' * indent + f'{key}: {value}\n')
        
        write_dict(data_to_save, 0)  # 从0缩进开始写入


def save_dict(dict_data, filename, format_list=None, key_to_save=None):
    '''保存字典到txt和pkl文件'''
    if format_list is None:
        format_list = ['txt', 'pkl']
    if key_to_save is None:
        key_to_save = list(dict_data.keys())

    # 创建文件夹
    mkdir(os.path.dirname(filename))

    # 假如filename有后缀,则添加到format_list中
    if filename.endswith('.txt'):
        format_list.append(filename.split('.')[-1])
        filename = os.path.splitext(filename)[0]
    if filename.endswith('.pkl') or filename.endswith('.pickle') or filename.endswith('.joblib'):
        format_list.append('.pkl')
        filename = os.path.splitext(filename)[0]

    # 保存到txt
    if 'txt' in format_list:
        save_dict_txt(filename, dict_data, key_to_save=key_to_save)

    # 保存到pkl
    if 'pkl' in format_list:
        save_pkl({k: dict_data[k] for k in key_to_save}, filename)


def save_dict_merge_to_saved(dict_data, filename, format_list=None, key_to_save=None, overwrite_separate=True):
    '''
    将dict_data保存到filename中,如果filename已经存在,则将dict_data与已存在的字典合并
    '''
    if key_to_save is None:
        key_to_save = list(dict_data.keys())

    try:
        # 尝试加载已存在的字典
        previous_dict = load_dict(filename)
    except:
        previous_dict = {}

    if overwrite_separate is False:
        for k in key_to_save:
            if k in previous_dict:
                raise ValueError(f"Key '{k}' already exists in the saved dictionary. Use overwrite_separate=True to overwrite.")
            
    # 合并字典
    dict_data_to_save = {k: dict_data[k] for k in key_to_save}
    merged_dict = {**previous_dict, **dict_data_to_save}
    save_dict(merged_dict, filename, format_list=format_list, key_to_save=key_to_save)


def get_load_function(format):
    '''
    根据格式获取合适的读取函数
    '''
    d = {'npy': load_array, 'npz': load_sps_array, 'pkl': load_pkl, 'joblib': load_pkl, 'pickle': load_pkl, '': load_dict_separate}
    return d[format]


def get_save_function_for_object(obj):
    '''
    根据数据类型获取合适的保存格式
    '''
    if isinstance(obj, np.ndarray):
        return save_array
    elif isinstance(obj, sps.spmatrix):
        return save_sps_array
    else:
        return save_pkl


def save_key_value_pair(key, value, save_func_dict, save_kwargs_dict, save_dir):
    '''
    用于save_dict_separate函数,保存键值对到文件
    '''
    str_key = hash_or_str(key)  # 将键转换为字符串或哈希值
    save_func = save_func_dict[key]  # 获取保存函数
    save_kwargs = save_kwargs_dict[key]  # 获取保存函数的参数
    save_func(value, os.path.join(save_dir, str_key), **save_kwargs)  # 保存值到文件
    return str_key, key


def save_dict_separate(dict_data, save_dir, save_func_dict=None, save_kwargs_dict=None, overwrite=False, save_as_txt=True, process_num=1, key_to_save=None, max_depth=0, current_depth=0, remove_after_save=False):
    """
    保存包含非字符串键的字典,将每个值保存为单独的文件,并保存键的映射关系

    支持嵌套使用(即字典的值为字典时,也会separate保存),但是不支持无限递归

    参数：
    dict_data (dict): 要保存的字典,字典的键可以是任何可哈希的类型
    save_dir (str): 保存文件的目标目录
    save_func_dict (dict): 指定每个键的保存函数,None则自动选择
    save_kwargs_dict (dict): 指定每个键的保存函数的参数,None则不输入参数
    overwrite (bool): 是否覆盖整个已存在的目标目录,默认为 False
    save_as_txt (bool): 是否保存为txt文件,默认为 True(txt保存的是整个字典,方便预览)
    process_num (int): 并行处理的进程数,默认为 1
    key_to_save (list): 指定要保存的键的列表,默认为 None,保存所有键
    max_depth (int, optional): 最大递归深度,默认为 0,表示不嵌套;不允许无限递归
    current_depth (int): 内部使用的当前递归深度,用户无需设置
    remove_after_save (bool): 是否删除已经保存的键值对
    """
    # 假设dict_data为空字典,则不保存
    if len(dict_data) == 0:
        print_title("Warning: The dictionary is empty. No files will be saved.", char='*')
        return
    
    if key_to_save is None:
        key_to_save = list(dict_data.keys())

    save_func_dict = fill_missing_key(save_func_dict, key_to_save, None)
    save_kwargs_dict = fill_missing_key(save_kwargs_dict, key_to_save, {})

    for k in key_to_save:
        if save_func_dict.get(k, None) is not None:
            continue  # 用户已指定保存函数，跳过
        value = dict_data[k]
        if isinstance(value, dict) and current_depth < max_depth:
            save_func_dict[k] = partial(save_dict_separate, save_as_txt=save_as_txt, max_depth=max_depth, current_depth=current_depth + 1)
        else:
            save_func_dict[k] = get_save_function_for_object(value)
    
    if overwrite and os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    metadata = {}

    dict_data_to_save = {k: dict_data[k] for k in key_to_save}
    results = multi_process_items_for(process_num=process_num, func=save_key_value_pair, for_dict=dict_data_to_save, kwargs={'save_func_dict': save_func_dict, 'save_kwargs_dict': save_kwargs_dict, 'save_dir': save_dir}, func_name=f'save dict to {save_dir}', use_tqdm=False)
    for r in results:
        metadata[r[0]] = r[1]

    # 保存键映射
    save_dict(metadata, os.path.join(save_dir, 'metadata'))

    # 保存原始字典的预览
    if save_as_txt:
        for k, v in metadata.items():
            if k in key_to_save:
                save_txt(dict_data[k], os.path.join(save_dir, 'preview', v))
    
    # 如果remove_after_save为True,则删除已经保存的键值对
    if remove_after_save:
        for k in key_to_save:
            del dict_data[k]


def save_dict_separate_merge_to_saved(dict_data, save_dir, save_func_dict=None, save_kwargs_dict=None, overwrite_separate=True, save_as_txt=True, process_num=1, key_to_save=None, max_depth=0, current_depth=0, remove_after_save=False):
    '''
    save_dict_separate,但是将保存的文件合并到已经存在的separate saved dict中,更新metadata

    注意:
    不允许save_dict_separate中的overwrite(会导致原先保存的dict被覆盖)
    可设置overwrite_separate,仅覆盖相同的键
    当overwrite_separate为False时,不允许原先dict和新dict中要key_to_save的键相同
    '''
    def _save():
        # 保存
        save_dict_separate(dict_data, save_dir, save_func_dict=save_func_dict, save_kwargs_dict=save_kwargs_dict, overwrite=False, save_as_txt=save_as_txt, process_num=process_num, key_to_save=key_to_save, max_depth=max_depth, current_depth=current_depth, remove_after_save=remove_after_save)
    
    # 如果metadata存在,则读取后更新metadata
    if check_all_file_exist_with_any_extension(os.path.join(save_dir, 'metadata')):
        # 读取已存在的metadata
        exist_metadata = load_pkl(os.path.join(save_dir, 'metadata'))

        # 检查是否存在相同的键
        if overwrite_separate is False:
            for k in key_to_save:
                if k in exist_metadata.values():
                    raise ValueError(f"Key '{k}' in key_to_save already exists in the existing metadata. This is not allowed when overwrite_separate is False.")
    
        # 保存(注意,这个和else是重复的,但是不能直接写到if else外面,否则只会保存新的metadata,旧的metadata会被覆盖)
        _save()

        # 更新metadata
        new_metadata = load_pkl(os.path.join(save_dir, 'metadata')) # 读取新保存的metadata
        metadata = update_dict(exist_metadata, new_metadata) # 合并metadata
        
        # 保存更新后的metadata
        save_dict(metadata, os.path.join(save_dir, 'metadata'))
    else:
        # 保存
        _save()


def delete_saved_dict_separate(save_dir, key_to_delete=None):
    if key_to_delete is None:
        rmdir(save_dir)
    else:
        metadata = load_pkl(os.path.join(save_dir, 'metadata'))
        for k in key_to_delete:
            if k in metadata.values(): # 注意,metadata的key是文件名,value是原始字符串
                metadata_key = get_key_by_value(metadata, k)
                # 找到对应的文件
                file_list = find_files_with_any_extension(os.path.join(save_dir, metadata_key))
                # 只有仅找到一个文件时才删除
                if len(file_list) == 1:
                    delete_file(file_list[0])  # 删除文件
                    del metadata[metadata_key]  # 删除metadata中的键
                elif len(file_list) > 1:
                    print(f"Warning: Multiple files found for key '{k}'. Not deleting any files to avoid confusion.")
            else:
                print(f"{k} not found in {os.path.join(save_dir, 'metadata')}")
        save_dict(metadata, os.path.join(save_dir, 'metadata'))  # 保存更新后的metadata


def check_saved_dict_completeness(save_dir):
    '''
    检查保存的字典是否完整,即metadata文件和所有子文件都存在
    '''
    completeness = True

    # 检查metadata文件是否存在
    if not check_all_file_exist_with_any_extension(os.path.join(save_dir, 'metadata')):
        completeness = False
        print(f"Metadata file not found in {save_dir}")
        return completeness

    # 检查子文件是否存在
    metadata = load_pkl(os.path.join(save_dir, 'metadata'))
    for k, v in metadata.items():
        if not check_all_file_exist_with_any_extension(os.path.join(save_dir, v)):
            completeness = False
            print(f"File {k} not found in {save_dir}")
            return completeness
    return completeness


def part_load_dict_separate(load_dir, subfile, metadata, key_to_load):
    # 获取文件名和后缀
    filename, ext = os.path.splitext(subfile)
    # 跳过键映射文件
    if filename == 'metadata':
        return None, None
    if filename == 'preview':
        return None, None
    if filename in metadata.keys():
        if metadata[filename] in key_to_load:
            # 加载文件
            load_func = get_load_function(ext[1:])
            return metadata[filename], load_func(os.path.join(load_dir, subfile))
    return None, None


def _compatible_data_container(obj):
    '''
    定义一个修饰器,使读取函数兼容DataContainer
    '''
    @wraps(obj)
    def wrapper(*args, **kwargs):
        r = obj(*args, **kwargs)
        if isinstance(r, DataContainer):
            # 如果读取的对象是DataContainer,则不需要转换(当直接读pickle文件时会保持DataContainer)
            pass
        elif isinstance(r, dict):
            # 如果读取的对象是字典,则需要转换为DataContainer(当分开读取的时候会先读成字典)
            if '_config' in r:
                r = _dict_to_data_container(r)
        return r
    return wrapper

@_compatible_data_container
def load_dict_separate(load_dir, key_to_load=None, filter_str=None, filter_mode='include', filter_logic='or', filter_func=None, process_num=1, ensure_config=True):
    """
    从指定目录中加载字典，并还原原始的键和值类型。

    参数:
    load_dir (str): 包含文件及键映射的目录。
    key_to_load (list): 指定要加载的键的列表,默认为 None,加载所有键
    filter_str (str): 指定要加载的键的字符串,默认为 None,不过滤(如果希望过滤多个str,可以输入list或者tuple)
    filter_mode (str): 指定过滤模式,默认为 'include',可选 'include' 或 'exclude';'include'表示只加载包含指定字符串的键,'exclude'表示不加载包含指定字符串的键
    filter_logic (str): 指定过滤逻辑,默认为 'or',可选 'or' 或 'and';'or'表示只要有一个字符串匹配即可,'and'表示所有字符串都要匹配
    filter_func (function): 指定过滤函数,默认为 None,不过滤(过滤函数输出True则保留,False则删除)(当需要的逻辑无法通过字符串实现时,可以使用函数)
    process_num (int): 并行处理的进程数,默认为 1
    ensure_config (bool): 是否确保'_config'在key_to_load中,默认为 True,如果'_config'不在key_to_load中,则添加'_config'到key_to_load中
    
    返回:
    dict: 恢复的原始字典

    注意:
    filter_str 和 filter_func 可以同时使用,但进行的是一个串行的过滤操作,即先根据字符串过滤,再根据函数过滤
    """
    # 加载键的映射关系
    metadata = load_pkl(os.path.join(load_dir, 'metadata'))
    if key_to_load is None:
        # 注意metadata的key是哈希值或者字符串,value才是原始键
        key_to_load = list(metadata.values())

    # 根据过滤条件过滤键
    if filter_str is not None:
        # 确保 filter_str 是可迭代的
        if isinstance(filter_str, str):
            filter_str = [filter_str]
        
        if filter_mode == 'include':
            if filter_logic == 'or':
                key_to_load = [k for k in key_to_load if any(s in k for s in filter_str)]
            elif filter_logic == 'and':
                key_to_load = [k for k in key_to_load if all(s in k for s in filter_str)]
        elif filter_mode == 'exclude':
            if filter_logic == 'or':
                key_to_load = [k for k in key_to_load if not any(s in k for s in filter_str)]
            elif filter_logic == 'and':
                key_to_load = [k for k in key_to_load if not all(s in k for s in filter_str)]

    # 根据函数过滤键
    if filter_func is not None:
        key_to_load = [k for k in key_to_load if filter_func(k)]
    
    if ensure_config:
        # 确保'_config'在key_to_load中
        if '_config' in metadata.keys() and '_config' not in key_to_load:
            key_to_load.append('_config')

    loaded_data = {}

    r = multi_process_list_for(process_num=process_num, func=part_load_dict_separate, for_list=os.listdir(load_dir), kwargs={'load_dir':load_dir, 'metadata': metadata, 'key_to_load': key_to_load}, func_name=f'load dict from {load_dir}', for_idx_name='subfile', use_tqdm=False)
    # 遍历r并提取非None值
    while r:
        k, v = r.pop(0)  # 每次从列表前端弹出一个元素
        if k is not None:  # 检查键是否非None
            loaded_data[k] = v  # 存储到字典中

    # 检查loaded_data是否读取了所有key_to_load
    if set(loaded_data.keys()) != set(key_to_load):
        missing_keys = set(key_to_load) - set(loaded_data.keys())
        raise ValueError(f"Not all keys were loaded. Missing keys: {missing_keys}. Please check the files in {load_dir}.")
    return loaded_data


def load_dict_separate_merge_to_exist(exist_dict, load_dir, key_to_load=None, filter_str=None, filter_mode='include', filter_logic='or', filter_func=None, process_num=1, ensure_config=True):
    '''
    从指定目录中加载字典,并且与已存在的字典合并

    注意,exist_dict必须和loaded_dict拥有相同的结构,比如同时为dict或者同时为DataContainer

    在OrderedDataContainer中,'_config'里面有'param_map'
    由于这种嵌套性,_config不能直接像其他的key,value那样直接使用update

    例子:
    loaded_data = {'_config': {'param_map': {'a': '1'}}, ...}
    exist_dict['_config']['param_map'] = {'b': '2'}

    直接update
    exist_dict.update(config) # exist_dict['_config']['param_map'] = {'a': '1'}

    使用此函数, exist_dict['_config']['param_map'] = {'a': '1', 'b': '2'}
    '''
    loaded_data = load_dict_separate(load_dir, key_to_load=key_to_load, filter_str=filter_str, filter_mode=filter_mode, filter_logic=filter_logic, filter_func=filter_func, process_num=process_num, ensure_config=ensure_config)
    if '_config' in loaded_data.keys():
        config = loaded_data.pop('_config')
        exist_dict['_config']['param_map'].update(config['param_map'])
    exist_dict.update(loaded_data)
    return exist_dict


@_compatible_data_container
def load_dict_auto(filename, key_to_load=None, filter_str=None, filter_mode='include', filter_logic='or', filter_func=None, process_num=1, ensure_config=True):
    '''
    自动选择普通模式或者separate模式加载字典
    '''
    try:
        return load_dict_separate(filename, key_to_load=key_to_load, filter_str=filter_str, filter_mode=filter_mode, filter_logic=filter_logic, filter_func=filter_func, process_num=process_num, ensure_config=ensure_config)
    except:
        return load_pkl(os.path.join(filename))

@_compatible_data_container
def load_dict(*args, **kwargs):
    '''
    load_dict_auto的缩写
    '''
    return load_dict_auto(*args, **kwargs)


def pop_dict_get_dir(dict_data, value_dir_key, both_dir_key, basedir):
    '''
    弹出一部分参数,并且返回路径名
    '''
    local_dict_data = dict_data.copy()
    if value_dir_key is None:
        value_dir_key = []
    if both_dir_key is None:
        both_dir_key = []
    for key in value_dir_key:
        basedir = os.path.join(basedir, str(dict_data[key]))
        local_dict_data.pop(key)
    for key in both_dir_key:
        basedir = os.path.join(basedir, concat_str([key, str(dict_data[key])]))
        local_dict_data.pop(key)
    return local_dict_data, basedir


def save_dir_dict(dict_data, basedir, dict_name, value_dir_key=None, both_dir_key=None, format_list=None, pop=True):
    '''
    把一部分参数保存到路径名里,另一部分参数保存到文件里

    注意:
    pop为True,则保存在路径名里的参数会被弹出,不会保存到文件里;pop为False,则保存原先的字典
    '''
    local_dict_data, dictdir = pop_dict_get_dir(dict_data, value_dir_key, both_dir_key, basedir)
    if pop:
        save_dict(local_dict_data, os.path.join(dictdir, dict_name), format_list)
    else:
        save_dict(dict_data, os.path.join(dictdir, dict_name), format_list)
    return dictdir


def save_timed_dir_dict(dict_data, basedir, dict_name, value_dir_key=None, both_dir_key=None, after_timedir='', current_time=None, format_list=None, pop=True):
    '''
    把一部分参数保存到路径名里,另一部分参数保存到文件里,并且在路径名里加入时间

    注意:
    pop为True,则保存在路径名里的参数会被弹出,不会保存到文件里;pop为False,则保存原先的字典
    '''
    if current_time is None:
        current_time = get_time()
    local_dict_data, basedir = pop_dict_get_dir(dict_data, value_dir_key, both_dir_key, basedir)
    timedir = os.path.join(basedir, current_time)
    dictdir = os.path.join(timedir, after_timedir)
    if pop:
        save_dict(local_dict_data, os.path.join(dictdir, dict_name), format_list)
    else:
        save_dict(dict_data, os.path.join(dictdir, dict_name), format_list)
    return timedir, dictdir


def save_pkl(obj, filename, format='joblib', compress=0, protocol=None, add_ext='auto'):
    '''
    保存对象到pkl文件(在本代码中pkl被认为是一种通用的保存格式,可以用joblib方法或者pickle方法保存,默认使用joblib方法[这可能会让人觉得有疑问,但是各类函数中的pkl都会优先使用joblib方法])
    虽然保存时优先使用joblib,但是加载时会兼容pkl,pickle,joblib三种格式,以对旧版本或者外源性的pkl文件进行兼容
    '''
    if format == 'joblib':
        save_joblib(obj, filename, compress=compress, protocol=protocol, add_ext=add_ext)
    elif format == 'pickle':
        save_pickle(obj, filename, add_ext=add_ext)


def load_pkl(filename):
    '''
    从pkl文件加载对象

    会采用各种方式,包括添加合适的后缀,直接读取;joblib和pickle都会尝试加载
    '''
    # 如果有后缀,且后缀已经在'.joblib','.pickle','.pkl'中,则直接加载
    if filename.endswith('.joblib'):
        return load_joblib(filename, add_ext=False)
    elif filename.endswith('.pickle'):
        return load_pickle(filename, add_ext=False)
    elif filename.endswith('.pkl'): # 兼容旧版本或者外源性的pkl文件
        return load_pickle(filename, add_ext=False)
    else: # 如果文件名没有后缀,或者后缀不在'.joblib','.pickle','.pkl'中,则尝试添加后缀并加载或者直接加载
        # 尝试添加后缀并加载
        for ext in ['.joblib', '.pickle', '.pkl']:
            if os.path.exists(filename + ext):
                return load_pkl(filename + ext)
        # 尝试直接加载(因为有可能输入的时候就有后缀)
        try:
            return load_joblib(filename, add_ext=False)
        except:
            try:
                return load_pickle(filename, add_ext=False)
            except:
                raise FileNotFoundError(f"File {filename} not found")


def save_pickle(obj, filename, add_ext='auto'):
    '''
    使用pickle保存对象
    
    参数:
    - add_ext:是否自动添加后缀,默认为 'auto',会自动添加后缀;如果不需要自动添加后缀,可以将此参数设置为 False
    '''
    # 创建文件夹
    mkdir(os.path.dirname(filename))

    # 保存到pickle
    if add_ext == 'auto':
        if not filename.endswith('.pickle'):
            filename += '.pickle'
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filename, add_ext='auto'):
    '''使用pickle加载对象,会自动添加后缀;由于后缀的多样性,当不需要自动添加后缀时,请将add_ext设置为False'''
    if add_ext == 'auto':
        ext_list = ['.pickle', '.pkl']
        detected_ext = False
        # 检查输入文件是否有指定后缀
        for ext in ext_list:
            if filename.endswith(ext):
                detected_ext = True
                break
        # 如果没有指定后缀,则尝试添加后缀并加载
        if not detected_ext:
            for ext in ext_list:
                if os.path.exists(filename + ext):
                    return load_pickle(filename + ext, add_ext=False)
            # 如果没有找到对应的文件,则抛出异常
            raise FileNotFoundError(f"File {filename} not found")
    with open(filename, 'rb') as f:
        return pickle.load(f)


def save_joblib(obj, filename, compress=0, protocol=None, add_ext='auto'):
    """
    使用joblib保存对象

    参数:
    - obj: 需要保存的对象
    - filename: 保存的文件名（如果没有以 .joblib 结尾，会自动添加）
    - compress: 压缩级别，默认为 0(范围为 0 到 9),0 表示不压缩,9 表示最大压缩
    - protocol: 序列化协议版本（默认使用 joblib 的默认协议）。可以传递一个整数(如 3、4 或 5)
      表示 Python 的 Pickle 序列化协议版本
    - add_ext:是否自动添加后缀,默认为 'auto',会自动添加后缀;如果不需要自动添加后缀,可以将此参数设置为 False
    """
    # 创建文件夹
    mkdir(os.path.dirname(filename))

    if add_ext == 'auto':
        # 如果文件名没有以 .joblib 结尾，自动添加后缀
        if not filename.endswith('.joblib'):
            filename += '.joblib'
    
    # 保存对象
    joblib.dump(obj, filename, compress=compress, protocol=protocol)


def load_joblib(filename, add_ext='auto'):
    """
    使用joblib加载对象,会自动添加后缀;由于后缀的多样性,当不需要自动添加后缀时,请将add_ext设置为False
    """
    # 检查文件是否以 .joblib 结尾
    if add_ext == 'auto':
        if not filename.endswith('.joblib'):
            filename += '.joblib'

    # 检查文件是否存在
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} not found")
    
    # 加载对象
    obj = joblib.load(filename)
    return obj


def load_multi_pkl(basedir, params_name_list, pkl_name, ext='joblib', sep='_'):
    """
    在basedir下寻找pkl文件,文件路径结构如下:
    {basedir}/{params_name[0]}{sep}{params_name[0]_value}/{params_name[1]}{sep}{params_name[1]_value}/.../{after_subdir}/{pkl_name}.{ext}

    例如
    basedir = "/path/to/data"
    params_name = ["model", "lr"]
    pkl_name = "result"
    ext = "joblib"
    after_subdir = "metrics"

    则函数会在以下路径寻找pkl文件
    /path/to/data/model_{model_value}/lr_{lr_value}/metrics/result.joblib

    Args:
        basedir (str): 根目录
        params_name_list (list): 参数名列表
        pkl_name (str): pkl文件名(不含扩展名)
        ext (str, optional): pkl文件扩展名. Defaults to 'joblib'.
        sep (str, optional): 参数名和参数值之间的分隔符. Defaults to '_'.

    Returns:
        dict: key为参数组合的元组表示(包含param名和value),value为对应的pkl文件内容
    """
    all_pkl = {}

    for root, dirs, files in os.walk(basedir):
        # 查找是否有符合条件的pkl文件
        if f'{pkl_name}.{ext}' in files:
            # 获取当前路径中的各级参数值
            relative_path = os.path.relpath(root, basedir)
            subdir_parts = relative_path.split(os.sep)

            param_items = []
            valid = True
            for i, param in enumerate(params_name_list):
                expected_prefix = param + sep
                if subdir_parts[i].startswith(expected_prefix):
                    param_value = subdir_parts[i][len(expected_prefix):]
                    param_items.append((param, param_value))  # 将param和其值作为元组存储
                else:
                    valid = False
                    break
            
            # 如果目录结构有效，加载对应的pkl文件
            if valid:
                param_tuple = tuple(f"{param}{sep}{value}" for param, value in param_items)  # 生成包含param和value的元组
                pkl_path = os.path.join(root, f'{pkl_name}.{ext}')
                try:
                    all_pkl[param_tuple] = load_pkl(pkl_path)
                except Exception as e:
                    print(f"Error loading {pkl_path}: {e}")

    return all_pkl


def dict_exist(dict_data, basedir, pkl_name, ext='joblib', value_dir_key=None, both_dir_key=None, ignore_key=None, pop=True):
    '''
    比较参数,如果参数不同,则返回False

    使用方式:
    假设这次要保存的dict是dict_data,按照pop_dict_get_dir的规则,dict_data中的value_dir_key和both_dir_key对应的值会被弹出,并且会返回一个新的dict和一个路径;如果这个路径下的pkl文件存在,则会加载这个pkl文件,并且和新的dict进行比较,如果相同,则返回True,否则返回False

    适用场景:
    没有按照时间保存,而是按照pop_dict_get_dir保存的,那么这个函数可以用来判断是否已经保存过这个dict,如果保存过,则返回True,否则返回False
    '''
    local_dict_data = pop_dict_get_dir(dict_data, value_dir_key, both_dir_key, basedir)[0]
    if pop:
        pass
    else:
        local_dict_data = dict_data.copy()
    pkl_dir = os.path.join(basedir, f'{pkl_name}.{ext}')
    if not os.path.exists(pkl_dir):
        return False
    else:
        exist_dict_data = load_pkl(pkl_dir)
        return compare_dict(local_dict_data, exist_dict_data, ignore_key)


def search_dict_subdir(dict_data, basedir, pkl_name, ext='joblib', value_dir_key=None, both_dir_key=None, after_subdir='', ignore_key=None, pop=True):
    '''
    比较参数,如果参数不同,则返回False,如果参数相同,则返回对应文件夹,这个函数会遍历basedir下面的所有一级子文件夹,然后在每个一级子文件夹内的after_subdir文件夹下查找pkl文件

    适用场景:
    如果是pop_dict_get_dir保存,并且用时间作为文件夹,那么这个函数可以用来查找是否已经保存过这个dict,如果保存过,则返回对应的时间文件夹,和True,否则返回None和False

    返回:
    str: 时间文件夹的路径(没找到则为None)
    bool: 是否找到对应的文件
    '''
    local_dict_data, basedir = pop_dict_get_dir(dict_data, value_dir_key, both_dir_key, basedir)
    if pop:
        pass
    else:
        local_dict_data = dict_data.copy()
    if not os.path.exists(basedir):
        return None, False
    for time_dir in get_subdir(basedir):
        pkl_dir = os.path.join(time_dir, after_subdir, f'{pkl_name}.{ext}')
        if os.path.exists(pkl_dir):
            exist_dict_data = load_pkl(pkl_dir)
            if compare_dict(local_dict_data, exist_dict_data, ignore_key):
                return time_dir, True
    return None, False


def compare_dict(dict1, dict2, ignore_key=None):
    '''
    比较两个字典,包括嵌套字典和NumPy数组,可选择忽略某些键
    '''
    # 处理忽略键
    if ignore_key is None:
        d1 = dict1.copy()
        d2 = dict2.copy()
    else:
        d1 = {k: v for k, v in dict1.items() if k not in ignore_key}
        d2 = {k: v for k, v in dict2.items() if k not in ignore_key}

    # 递归比较值
    def compare_values(v1, v2):
        # 处理NumPy数组
        if isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray):
            return np.array_equal(v1, v2)
        
        # 处理嵌套字典
        elif isinstance(v1, dict) and isinstance(v2, dict):
            return compare_dict(v1, v2)
        
        # 处理其他类型,列表、标量等
        else:
            return v1 == v2

    # 比较所有键值对
    if set(d1.keys()) != set(d2.keys()):
        return False
    for key in d1:
        if not compare_values(d1[key], d2[key]):
            return False
    return True


def save_df(df, filename, index=True, format_list=None):
    '''
    Save a DataFrame to a CSV, Excel file, or Pickle, depending on the file extension or provided formats, with the option to include the index.

    Parameters:
    - df: DataFrame to save.
    - filename: Name of the base file to save the DataFrame to (without extension).
    - index: Boolean indicating whether to write the index to the file.
    - format_list: List of formats to save the DataFrame in.
    '''
    if format_list is None:
        format_list = ['xlsx', 'joblib']

    # 创建文件夹
    mkdir(os.path.dirname(filename))

    # 将文件名和格式分开,并且将文件名的后缀添加到format_list中
    if filename.endswith('.csv') or filename.endswith('.xlsx') or filename.endswith('.joblib'):
        format_list.append(filename.split('.')[-1])
        base_filename = os.path.splitext(filename)[0]
    else:
        base_filename = filename
    
    # 分别保存到不同的文件
    for format in format_list:
        full_filename = f"{base_filename}.{format}"
        if format == 'csv':
            df.to_csv(full_filename, index=index)
        elif format == 'xlsx':
            df.to_excel(full_filename, index=index, engine='openpyxl')
        elif format == 'joblib':
            df.to_pickle(full_filename)
        else:
            raise ValueError(
                "Unsupported file type. Please use csv, xlsx, or joblib")


def load_df(filename, index_col=0, index_dtype=str, col_dtype=str):
    '''
    Load a DataFrame from a CSV, Excel file, or Pickle file, depending on the file's extension.

    Parameters:
    - filename: Name of the file to load the DataFrame from (with extension).
    - index_col: Column(s) to set as index. If None, defaults to pandas' behavior.
    - index_dtype: Data type for the index column, defaults to str.
    '''
    # Check if the file exists with the provided extension
    if os.path.exists(filename):
        if filename.endswith('.pkl') or filename.endswith('.joblib') or filename.endswith('.pickle'):
            df = pd.read_pickle(filename)
        elif filename.endswith('.xlsx'):
            df = pd.read_excel(
                filename, index_col=index_col, engine='openpyxl')
        elif filename.endswith('.csv'):
            df = pd.read_csv(filename, index_col=index_col)

        df.index = df.index.astype(index_dtype)
        df.columns = df.columns.astype(col_dtype)

        # Convert all numerical columns to float
        df = df.apply(lambda x: x.astype(float)
                      if x.dtype.kind in 'iufc' else x)

        return df
    else:
        for ext in ['.xlsx', '.csv', '.pkl', '.joblib', '.pickle']:
            if os.path.exists(filename+ext):
                return load_df(filename+ext, index_col, index_dtype, col_dtype)
        raise ValueError(
            "File not found. Please check the file path and extension.")


def save_array(arr, filename):
    '''
    将数组保存到文件中。

    参数:
    arr: array
        要保存的数组对象。
    filename: str
        要保存的文件名，可以带有后缀或不带有后缀。

    返回:
    无返回值。
    '''
    # 创建文件夹
    mkdir(os.path.dirname(filename))

    # 添加后缀
    if not filename.endswith('.npy'):
        filename += '.npy'
    
    # 保存数组
    np.save(filename, arr)


def save_sps_array(arr, filename):
    '''
    将稀疏矩阵保存到文件中。

    参数:
    arr: sps.spmatrix
        要保存的稀疏矩阵对象。
    filename: str
        要保存的文件名，可以带有后缀或不带有后缀。

    返回:
    无返回值。
    '''
    # 创建文件夹
    mkdir(os.path.dirname(filename))

    # 添加后缀
    if not filename.endswith('.npz'):
        filename += '.npz'
    
    # 保存稀疏矩阵
    sps.save_npz(filename, arr)


def load_array(filename):
    '''
    从文件中加载数组。

    参数:
    filename: str
        要加载的文件名，可以带有后缀或不带有后缀。

    返回:
    array: array
        加载的数组对象。
    '''

    # 添加后缀
    if not filename.endswith('.npy'):
        filename += '.npy'
    
    # 加载数组
    return np.load(filename, allow_pickle=True)


def load_sps_array(filename):
    '''
    从文件中加载稀疏矩阵。

    参数:
    filename: str
        要加载的文件名，可以带有后缀或不带有后缀。

    返回:
    arr: sps.spmatrix
        加载的稀疏矩阵对象。
    '''
    # 添加后缀
    if not filename.endswith('.npz'):
        filename += '.npz'
    
    # 加载稀疏矩阵
    return sps.load_npz(filename)


def save_txt(data, filename):
    '''
    将数据保存到txt文件中
    '''
    # 创建文件夹
    mkdir(os.path.dirname(filename))

    # 添加后缀
    if not filename.endswith('.txt'):
        filename += '.txt'

    # 保存数据
    with open(filename, 'w') as f:
        f.write(str(data))


def save_list_txt(data, filename):
    '''
    将列表数据保存到txt文件中
    '''
    # 创建文件夹
    mkdir(os.path.dirname(filename))

    # 添加后缀
    if not filename.endswith('.txt'):
        filename += '.txt'

    # 保存数据
    with open(filename, 'w') as f:
        for item in data:
            f.write(str(item) + '\n')


def load_txt(filename):
    '''
    从txt文件中加载数据

    参数:
    filename: str要加载的文件名,可以带有后缀或不带有后缀
    '''
    if not filename.endswith('.txt'):
        filename += '.txt'
    with open(filename, 'r') as f:
        lines = f.readlines()
        return [line.strip() for line in lines]


# region lmdb
def get_size_in_bytes(size, unit):
    '''
    统一转换为bytes
    '''
    units = {'bytes':1, 'kb':1024, 'mb':1024**2, 'gb':1024**3, 'tb':1024**4}
    return int(size * units[unit.lower()])


def init_lmdb(filename, size=1, unit='gb'):
    """初始化 LMDB 环境，自动根据现有数据库调整大小"""

    initial_map_size = get_size_in_bytes(size, unit)
    
    if os.path.exists(filename):
        try:
            # 尝试打开现有环境获取实际 map_size
            existing_env = lmdb.open(filename, readonly=True, lock=False)
            existing_map_size = existing_env.info()['map_size']
            existing_env.close()
            initial_map_size = max(initial_map_size, existing_map_size)
        except lmdb.Error:
            # 处理无法打开的情况（如文件损坏）
            pass
    
    return lmdb.open(filename, 
                    map_size=initial_map_size,
                    subdir=True,
                    create=True,
                    readahead=True)


def _auto_expand(env, size, unit='gb', max_size=100, max_size_unit='gb'):
    """
    智能扩容机制,按几何倍增长
    """
    size_in_bytes = get_size_in_bytes(size, unit)
    max_size_in_bytes = get_size_in_bytes(max_size, max_size_unit)
    new_size_in_bytes = max(env.info()['map_size'], size_in_bytes) * 1.5
    new_size_in_unit = new_size_in_bytes / get_size_in_bytes(1, unit)
    if new_size_in_bytes > max_size_in_bytes:
        raise MemoryError(f'Need {new_size_in_unit:.2f}{unit} but max_size={max_size:.2f}{max_size_unit}, please set a larger size')
    env.set_mapsize(new_size_in_bytes)
    print(f"Auto-expanded to: {new_size_in_unit:.2f}{unit}")
    return new_size_in_unit


def _key_to_bytes(key):
    buffer = BytesIO()
    joblib.dump(key, buffer)
    return buffer.getvalue()


def _bytes_to_key(byte_key):
    buffer = BytesIO(byte_key)
    return joblib.load(buffer)


def save_dict_lmdb(data, filename, key_to_save=None, size=1, unit='gb', overwrite=True, max_size=100, max_size_unit='gb', remove_after_save=False, process_num=1):
    """根据 overwrite 决定是否覆盖"""
    if process_num != 1: # 为了与save_dict_separate兼容,这里支持输入process_num,但是不会生效
        print('LMDB has no multiprocess save')
    if key_to_save is None:
        key_to_save = list(data.keys())
    
    mkdir(os.path.dirname(filename))

    with init_lmdb(filename, size=size, unit=unit) as env:
        while size < max_size:
            try:
                with env.begin(write=True) as txn:
                    for key in key_to_save:
                        byte_key = _key_to_bytes(key)
                        buffer = BytesIO()
                        joblib.dump(data[key], buffer)
                        value = buffer.getvalue()
                        if overwrite or txn.get(byte_key) is None:
                            txn.put(byte_key, value)

                    # 保存所有key
                    all_keys = _get_all_keys_lmdb(txn)
                    save_list_txt(all_keys, os.path.join(filename, 'all_keys'))
                    save_joblib(all_keys, os.path.join(filename, 'all_keys'))
    
                    if remove_after_save:
                        for key in key_to_save:
                            del data[key]
                    return  # 成功保存后退出
            except lmdb.MapFullError:
                size = _auto_expand(env, size=size, unit=unit, max_size=max_size, max_size_unit=max_size_unit)
        raise lmdb.MapFullError(f"Auto expand to {size} {unit}, but exceed max_size={max_size} {max_size_unit}, please set a larger size")


def delete_saved_dict_lmdb(filename, key_to_delete=None):
    '''
    如果是OrderedDataContainer,删除key不会同步删去_param_map中的对应key,如果真的需要删除_param_map中的key,请手动删除
    '''
    if key_to_delete is None:
        rmdir(filename)
    else:
        all_keys = get_all_keys_lmdb_custom(filename)
        with init_lmdb(filename) as env:
            with env.begin(write=True) as txn:
                for key in key_to_delete:
                    byte_key = _key_to_bytes(key)
                    if key in all_keys:
                        txn.delete(byte_key)
                        all_keys.remove(key)
                    else:
                        print(f"{key} not found in {os.path.join(filename, 'all_keys')}")
        save_list_txt(all_keys, os.path.join(filename, 'all_keys'))
        save_joblib(all_keys, os.path.join(filename, 'all_keys'))


@_compatible_data_container
def _load_dict_lmdb(filename, key_to_load):
    """
    从数据库加载数据
    
    为了性能上的考虑,只有在必要的时候才会得到all_keys,因为对于一个key相当多的lmdb,get_all_keys_lmdb_custom是比较耗费时间的
    """
    with init_lmdb(filename) as env:
        with env.begin() as txn:
            try:
                return {
                    key: joblib.load(BytesIO(txn.get(_key_to_bytes(key))))
                    for key in key_to_load
                }
            except Exception as e:
                all_keys = get_all_keys_lmdb_custom(filename)
                if is_sublist(key_to_load, all_keys):
                    raise ValueError(e)
                else:
                    raise KeyError(f"key_to_load {key_to_load} is not a sub list of all_keys in {os.path.join(filename, 'all_keys')}")


@_compatible_data_container
def load_dict_lmdb(filename, key_to_load=None, ensure_config=True, process_num=1):
    """
    从数据库加载数据
    
    为了性能上的考虑,只有在必要的时候才会得到all_keys,因为对于一个key相当多的lmdb,get_all_keys_lmdb_custom是比较耗费时间的
    """
    all_keys = None

    if key_to_load is None:
        all_keys = get_all_keys_lmdb_custom(filename)
        key_to_load = all_keys.copy()

    if ensure_config:
        if all_keys is None:
            all_keys = get_all_keys_lmdb_custom(filename)
        # 确保'_config'在key_to_load中
        if '_config' in all_keys and '_config' not in key_to_load:
            key_to_load.append('_config')
    
    divided_key_to_load = split_list(key_to_load, process_num)
    kwargs_list = []
    for k_to_load in divided_key_to_load:
        kwargs_list.append({'filename': filename, 'key_to_load': k_to_load})
    result_list = multi_process(process_num=process_num, func=_load_dict_lmdb, kwargs_list=kwargs_list, func_name='load_dict_lmdb')
    r = {}
    for result in result_list:
        r.update(result)
    return r


def load_dict_lmdb_merge_to_exist(exist_dict, filename, key_to_load=None, ensure_config=True, process_num=1):
    '''
    见load_dict_separate_merge_to_exist
    '''
    loaded_data = load_dict_lmdb(filename, key_to_load=key_to_load, ensure_config=ensure_config, process_num=process_num)
    if '_config' in loaded_data.keys():
        config = loaded_data.pop('_config')
        exist_dict['_config']['param_map'].update(config['param_map'])
    exist_dict.update(loaded_data)
    return exist_dict


def _get_all_keys_lmdb(txn):
    return [_bytes_to_key(byte_key) for byte_key, _ in txn.cursor()]


def get_all_keys_lmdb(filename):
    """获取数据库所有键列表,较慢"""
    with init_lmdb(filename) as env:
        with env.begin() as txn:
            return _get_all_keys_lmdb(txn)


def get_all_keys_lmdb_custom(filename):
    '''
    从我保存的all_keys文件中获取所有键,极快
    '''
    all_keys = load_joblib(os.path.join(filename, 'all_keys'))
    return all_keys
# endregion
# endregion


# region 并行处理相关函数
def check_if_multiprocessing():
    current_process = multiprocessing.current_process()
    if current_process.name == 'MainProcess':
        return False
    else:
        return True


def split_list(lst, n):
    '''
    将列表尽量均等地分割为n个子列表。

    参数:
    - lst: list
        要分割的列表。
    - n: int
        子列表的数量。

    返回:
    list: 包含n个子列表的列表。
    '''
    # 计算每个子列表的长度
    length = len(lst)
    size = length // n
    remainder = length % n
    
    # 创建子列表
    divided_list = []
    start = 0
    for i in range(n):
        # 确定子列表的长度
        sublist_size = size + 1 if i < remainder else size
        # 添加子列表到结果列表中
        divided_list.append(lst[start:start + sublist_size])
        # 更新下一个子列表的起始位置
        start += sublist_size
    return divided_list


def split_array(arr, axis, n):
    '''
    将数组沿指定轴均等地分割为n个子数组。
    '''
    # Calculate sizes of chunks
    total_length = arr.shape[axis]
    chunk_sizes = [total_length // n + (1 if i < total_length % n else 0) for i in range(n)]

    # Compute slicing indices
    slices = []
    start = 0
    for size in chunk_sizes:
        stop = start + size
        slices.append(slice(start, stop))
        start = stop

    # Split the array using advanced indexing
    return [arr[(slice(None),) * axis + (s,)] for s in slices]


def _check_process_num_func_args_kwargs(process_num, func_list, args_list, kwargs_list):
    '''
    检查process_num, args_list, kwargs_list的长度是否符合要求
    注意:
    用户必须确保args_list,kwargs_list,func_list的长度相同
    process_num必须等于len(func_list)或者1
    '''
    len_set = {len(func_list), len(args_list), len(kwargs_list)}
    if len(len_set) > 1:
        raise ValueError("The length of func_list, args_list and kwargs_list must be the same.")
    if process_num != 1 and process_num != len(func_list):
        raise ValueError("The process_num must be equal to the length of func_list or 1.")


def basic_multi_process(process_num, func_list, args_list=None, kwargs_list=None, task_name=''):
    '''
    最基础的multi_process

    参数:
    - process_num: int, 并行处理的进程数
    - func_list: list, 要并行处理的函数列表
    - args_list: list, 函数的位置参数列表
    - kwargs_list: list, 函数的关键字参数列表
    - task_name: str, 任务的名称(也可以输入函数的名称等需要显示的信息)
    注意:
    用户必须确保args_list,kwargs_list,func_list的长度相同
    process_num必须等于len(func_list)或者1
    '''
    _check_process_num_func_args_kwargs(process_num, func_list, args_list, kwargs_list)

    if process_num > 1 and check_if_multiprocessing():
        print_title('unable to use multiprocessing inside a multiprocessing process, use single process instead')
        process_num = 1

    if process_num != 1:
        print_title(f"Start {task_name} with {process_num} processes")
        results = []
        # 使用 ProcessPoolExecutor 进行多进程处理
        with ProcessPoolExecutor(max_workers=process_num) as executor:
            # 提交任务
            futures = [executor.submit(func, *args, **kwargs) for func, args, kwargs in zip(func_list, args_list, kwargs_list)]

            # 等待所有future对象按照提交的顺序完成，并收集结果
            for future in futures:
                with FlexibleTry() as ft:
                    results.append(future.result())
                if not ft.success:
                    results.append(None)
        print_title(f"Finish {task_name}")
        return results
    elif process_num == 1:
        return [func(*args, **kwargs) for func, args, kwargs in zip(func_list, args_list, kwargs_list)]


def _prepare_for_multi_process(process_num, func_list, args_list, kwargs_list):
    '''
    注意:
    当func_list,args_list,kwargs_list的len为1时,其会被自动广播到最大的一方
    假如args_list = [(1), (2)]这样的写法是不对的,至少要让里面成为元组,即args_list = [(1,), (2,)]
    假如已经在multi_process中,继续使用multi_process会自动转为单进程运行(此时args_list和kwargs_list会被flatten)
    '''
    # 确认输入的都是list或者None
    if (not isinstance(func_list, list)):
        raise ValueError("func_list must be a list")
    if (not isinstance(args_list, list)) and (args_list is not None):
        raise ValueError("args_list must be a list or None")
    if (not isinstance(kwargs_list, list)) and (kwargs_list is not None):
        raise ValueError("kwargs_list must be a list or None")
    
    # 处理None(args_list和kwargs_list本身为None)
    if args_list is None:
        args_list = [()]
    if kwargs_list is None:
        kwargs_list = [{}]

    # 处理长度不一致的情况
    max_length = max(len(func_list), len(args_list), len(kwargs_list))
    if len(func_list) != max_length:
        if len(func_list) == 1:
            func_list = func_list * max_length
        else:
            raise ValueError("The length of func_list must be equal to max_length or 1.")
    if len(args_list) != max_length:
        if len(args_list) == 1:
            args_list = args_list * max_length
        else:
            raise ValueError("The length of args_list must be equal to max_length or 1.")
    if len(kwargs_list) != max_length:
        if len(kwargs_list) == 1:
            kwargs_list = kwargs_list * max_length
        else:
            raise ValueError("The length of kwargs_list must be equal to max_length or 1.")
    
    # 处理None(args_list和kwargs_list内部的None)
    for i, args in enumerate(args_list):
        if args is None:
            args_list[i] = ()
    for i, kwargs in enumerate(kwargs_list):
        if kwargs is None:
            kwargs_list[i] = {}

    return process_num, func_list, args_list, kwargs_list


def _run_multi_func(func_list, args_list, kwargs_list):
    '''
    运行多函数
    '''
    results = []
    for func, args, kwargs in zip(func_list, args_list, kwargs_list):
        results.append(func(*args, **kwargs))
    return results


def multi_process_for_different_func(process_num, func_list, args_list=None, kwargs_list=None, task_name=''):
    '''
    多进程并行处理不同的函数,相较basic_multi_process,增添了以下功能
    
    自动广播:如果func_list,args_list,kwargs_list的len为1时,其会被自动广播到最大的一方(如果为None,则也自动设置)
    自动切分:process_num不再被限制为只能是1或者len(func_list),而是可以是任意值,但是会自动切分func_list,args_list,kwargs_list
    '''
    process_num, func_list, args_list, kwargs_list = _prepare_for_multi_process(process_num, func_list, args_list, kwargs_list)

    divided_func_list = split_list(func_list, process_num)
    divided_args_list = split_list(args_list, process_num)
    divided_kwargs_list = split_list(kwargs_list, process_num)

    new_func_list = [_run_multi_func] * process_num
    new_args_list = [()] * process_num
    new_kwargs_list = [{'func_list': divided_func_list[i], 'args_list': divided_args_list[i], 'kwargs_list': divided_kwargs_list[i]} for i in range(process_num)]
    return flatten_list(basic_multi_process(process_num, func_list=new_func_list, args_list=new_args_list, kwargs_list=new_kwargs_list, task_name=task_name), level=1)


def multi_process(process_num, func, args_list=None, kwargs_list=None, func_name=''):
    '''
    多进程并行处理函数,内部使用multi_process_for_different_func

    参数:
    - process_num: int,并行处理的进程数(由于multi_process状态下,代码错误的提示难以看出错误位置,在测试时可以先把process_num设置为1,这时会按照正常默认方式运行和报错)
    - func: function,要并行处理的函数
    - args_list: list,函数的位置参数列表
    - kwargs_list: list,函数的关键字参数列表
    - func_name: str,函数的名称(也可以输入任务的名称等需要显示的信息)

    注意:
    假如args_list = [(1), (2)]这样的写法是不对的,至少要让里面成为元组,即args_list = [(1,), (2,)]
    假如已经在multi_process中,继续使用multi_process会自动转为单进程运行(此时args_list和kwargs_list会被flatten)
    
    假如输入的args_list和kwargs_list都不包含多个元素,最终会被广播到process_num的数量
    否则,process_num只代表分割的数量,而不是重复的数量

    multi_process下要尽量避免使用全局随机和输入同一个rng,如果函数一样,多个process会有一样的结果,推荐的使用的方式是输入不同的seed在内部产生rng
    '''
    def _get_input_l_len(l):
        if l is None:
            return 0
        elif isinstance(l, list):
            return len(l)
        else:
            raise ValueError("args_list and kwargs_list must be list or None")
    
    max_len = max(_get_input_l_len(args_list), _get_input_l_len(kwargs_list))

    if max_len <= 1:
        if process_num > 1:
            print_title(f'input function, args_list and kwargs_list are all not multiple, use {process_num} processes to repeat the function')
        func_list = [func] * process_num
    else:
        func_list = [func] * max_len
    return multi_process_for_different_func(process_num, func_list, args_list, kwargs_list, func_name)


def part_list_for(func, for_list, for_idx_name, use_tqdm, *args, **kwargs):
    if use_tqdm and len(for_list) != 1:
        iterable = tqdm.tqdm(for_list)
    else:
        iterable = for_list
    results = []
    for i in iterable:
        results.append(func(*args, **{**kwargs, for_idx_name: i}))
    return results


def multi_process_list_for(process_num, func, args=None, kwargs=None, for_list=None, for_idx_name='i', func_name='', use_tqdm=True):
    '''
    多进程并行处理for循环,for循环形式为for i in for_list

    参数:
    - process_num: int, 并行处理的进程数
    - func: function, 要并行处理的函数
    - args: 函数的位置参数(不推荐,因为idx在func中的位置不确定)
    - kwargs: 函数的关键字参数
    - func_name: str, 函数的名称(也可以输入任务的名称等需要显示的信息)

    注意:
    只有当for循环每个之间独立时才能使用这个函数
    如果需要使用items()方法,请使用multi_process_items_for;如果需要使用enumerate()方法,请使用multi_process_enumerate_for;此处尚未支持zip()方法,但是zip也可以通过普通for循环实现
    '''
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}
    for_list = list(for_list)   # 防止for_list是生成器,比如range(10)

    if process_num > len(for_list):
        print_title(f'Reduce process_num from {process_num} to {len(for_list)} as process_num > len(for_list)')
        process_num = len(for_list)

    divided_list = split_list(for_list, process_num)
    args_list = [(func, divided, for_idx_name, use_tqdm)+args for divided in divided_list]
    kwargs_list = [kwargs] * process_num
    return flatten_list(multi_process(process_num, part_list_for, args_list, kwargs_list, func_name), level=1)


def part_enumerate_for(func, idx_list, for_list, for_idx_name, for_item_name, use_tqdm, *args, **kwargs):
    if use_tqdm and len(for_list) != 1:
        iterable = tqdm.tqdm(zip(idx_list, for_list))
    else:
        iterable = zip(idx_list, for_list)
    results = []
    for i, item in iterable:
        results.append(func(*args, **{**kwargs, for_idx_name: i, for_item_name: item}))
    return results


def multi_process_enumerate_for(process_num, func, args=None, kwargs=None, for_list=None, for_idx_name='i', for_item_name='item', func_name='', use_tqdm=True):
    '''
    多进程并行处理for循环,for循环形式为for i, item in enumerate(for_list)

    参数:
    - process_num: int, 并行处理的进程数
    - func: function, 要并行处理的函数(必须把for_idx_name和for_item_name作为关键字参数传入(推荐)或者放在最后的位置参数(不推荐))
    - args: 函数的位置参数(不推荐,因为idx和item在func中的位置不确定)
    - kwargs: 函数的关键字参数
    - func_name: str, 函数的名称(也可以输入任务的名称等需要显示的信息)

    注意:
    只有当for循环每个之间独立时才能使用这个函数
    '''
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}
    
    if process_num > len(for_list):
        print_title(f'Reduce process_num from {process_num} to {len(for_list)} as process_num > len(for_list)')
        process_num = len(for_list)

    divided_idx_list = split_list(range(len(for_list)), process_num)
    divided_list = split_list(for_list, process_num)
    args_list = [(func, divided_idx, divided, for_idx_name, for_item_name, use_tqdm)+args for divided_idx, divided in zip(divided_idx_list, divided_list)]
    kwargs_list = [kwargs] * process_num
    return flatten_list(multi_process(process_num, part_enumerate_for, args_list, kwargs_list, func_name), level=1)


def part_items_for(func, key_list, value_list, for_key_name, for_value_name, use_tqdm, *args, **kwargs):
    if use_tqdm and len(key_list) != 1:
        iterable = tqdm.tqdm(zip(key_list, value_list))
    else:
        iterable = zip(key_list, value_list)
    results = []
    for key, value in iterable:
        results.append(func(*args, **{**kwargs, for_key_name: key, for_value_name: value}))
    return results


def multi_process_items_for(process_num, func, args=None, kwargs=None, for_dict=None, for_key_name='key', for_value_name='value', func_name='', use_tqdm=True):
    '''
    多进程并行处理for循环,for循环形式为for key, value in for_dict.items()

    参数:
    - process_num: int, 并行处理的进程数
    - func: function, 要并行处理的函数(其会接收for_key_name和for_value_name作为关键字参数)
    - args: 函数的位置参数(不推荐,因为key,value在func中的位置不确定)
    - kwargs: 函数的关键字参数
    - func_name: str, 函数的名称(也可以输入任务的名称等需要显示的信息)

    注意:
    只有当for循环每个之间独立时才能使用这个函数

    示例:
    def func(k, v, x, y):
        print(k, v, x, y)
        
    for_dict = {'a': 1, 'b': 2, 'c': 3}
    multi_process_items_for(2, func, args=(), kwargs={'x': 3, 'y': 3}, for_dict=for_dict, for_key_name='k', for_value_name='v')
    '''
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}
    
    if process_num > len(for_dict):
        print_title(f'Reduce process_num from {process_num} to {len(for_dict)} as process_num > len(for_dict)')
        process_num = len(for_dict)
    
    divided_key_list = split_list(list(for_dict.keys()), process_num)
    divided_value_list = split_list(list(for_dict.values()), process_num)
    args_list = [(func, divided_key, divided_value, for_key_name, for_value_name, use_tqdm)+args for divided_key, divided_value in zip(divided_key_list, divided_value_list)]
    kwargs_list = [kwargs] * process_num
    return flatten_list(multi_process(process_num, part_items_for, args_list, kwargs_list, func_name), level=1)
# endregion


# region 生成字母序列
def get_tag(n, case=TAG_CASE, parentheses=TAG_PARENTHESES):
    if case == 'lower':
        # 生成小写字母标签
        tags = [chr(i) for i in range(ord('a'), ord('a') + n)]
    elif case == 'upper':
        # 生成大写字母标签
        tags = [chr(i) for i in range(ord('A'), ord('A') + n)]
    else:
        raise ValueError('Unknown case: ' + case)

    if parentheses:
        # 在每个标签前后添加括号
        tags = ['(' + tag + ')' for tag in tags]

    return tags
# endregion


# region 浮点数处理相关函数
def get_decimal_num(number, abs_tol=1e-5, rel_tol=1e-5):
    """
    计算浮点数的有效小数位数.如果某一位的小数值相对于数本身小于tol,则从该位开始忽略.
    
    :param number: 输入的浮点数
    :param abs_tol: 容忍度,决定从哪一位开始忽略,默认值为 1e-5(如果不想检测abs_tol,可以设置为None)
    :param rel_tol: 相对容忍度,决定从哪一位开始忽略,默认值为 1e-5(如果不想检测rel_tol,可以设置为None)
    :return: 小数位数
    """
    # 记录初始的小数位数
    decimal_count = 0

    # 我们循环检查小数位,逐位扩大10倍,直到发现某一位可以忽略
    while True:
        rounded_number = round(number, decimal_count)
        
        abs_break = True
        rel_break = True
        if abs_tol is not None:
            if abs(rounded_number - number) > abs_tol:
                abs_break = False
        if rel_tol is not None:
            if abs(rounded_number - number) > rel_tol * abs(number):
                rel_break = False
        if abs_break and rel_break:
            break

        # 否则继续检查下一位
        decimal_count += 1
    
    return decimal_count


def get_max_decimal_num(number_list, **kwargs):
    decimal_count_list = [get_decimal_num(number, **kwargs) for number in number_list]
    return max(decimal_count_list)
# endregion


# region 字符串,filename处理相关函数
def camel_to_snake(name):
    """
    将驼峰命名法字符串转换为下划线命名法
    示例:
        camelCase -> camel_case
        CamelCase -> camel_case
        HTTPRequest -> http_request
        SimpleXMLParser -> simple_xml_parser
        version2Update -> version2_update
    """
    # 处理连续大写字母的情况（如HTTP）
    s1 = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', name)
    # 处理单个大写字母（或数字后的大写字母）的情况
    s2 = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s1)
    # 转换为全小写
    return s2.lower()


def format_text(text, text_process=None):
    '''
    格式化文本(主要是为了画图美观)
    '''
    text_process = update_dict(TEXT_PROCESS, text_process)

    capitalize = text_process['capitalize']
    replace_underscore = text_process['replace_underscore']
    ignore_equation_underscore = text_process['ignore_equation_underscore']

    if text is None:
        return None
    else:
        # 分割字符串，保留$$之间的内容
        parts = []
        if ignore_equation_underscore:
            # 使用标志位记录是否在$$内
            in_math = False
            temp_str = ""
            for char in text:
                if char == "$" and not in_math:  # 开始$$
                    in_math = True
                    if temp_str:  # 保存$$之前的内容
                        parts.append(temp_str)
                        temp_str = ""
                    parts.append(char)
                elif char == "$" and in_math:  # 结束$$
                    in_math = False
                    parts.append(temp_str + char)
                    temp_str = ""
                elif in_math:  # $$内的内容直接添加
                    temp_str += char
                else:  # $$外的内容
                    if char == "_":
                        temp_str += replace_underscore if replace_underscore else "_"
                    else:
                        temp_str += char
            if temp_str:  # 添加最后一部分
                parts.append(temp_str)
        else:
            parts = [text.replace('_', replace_underscore)
                     if replace_underscore else text]

        # 处理大写转换
        if capitalize:
            processed_parts = [part[0].capitalize(
            ) + part[1:] if part else "" for part in parts]
        else:
            processed_parts = parts

        return "".join(processed_parts)


def uppercase_text(text):
    '''
    将文本转换为大写
    '''
    return text.upper()


def sci_str_to_latex(sci_str):
    '''
    将科学计数法字符串转换为LaTeX字符串

    例子:
    sci_str_to_latex('1.23e-4')  # 返回 '$1.23 \\times 10^{-4}$'

    注意:
    此函数与sci_float_to_latex不同,不仅在于输入的不同(一个字符串一个浮点数),还在于内部的处理方式不同(此函数假定了浮点数已经被python原生的round scientific notation表示并以10为底,而sci_float_to_latex手动实现了这个过程,可以以其他数为底)
    '''
    match = re.match(r"([+-]?\d*\.\d+)e([+-]?\d+)", sci_str)
    if match:
        base, exponent = match.groups()
        # Format the LaTeX string
        latex_str = f"${base} \\times 10^{{{exponent}}}$"
        return latex_str
    else:
        raise ValueError('The input string is not in scientific notation.')


def round_float(number, digits=ROUND_DIGITS, format_type=ROUND_FORMAT, latex=False):
    '''
    Rounds a float to a given number of digits and returns it in specified format.

    Parameters:
    - number: float, the number to be rounded.
    - digits: int, the number of digits to round to.
    - format_type: str, the format type for the output ('standard', 'scientific', 'percent', 'general'). 'general' means choose standard or scientific based on the length of the string representation.
    - latex: bool, whether to return the rounded number as a LaTeX string for scientific notation.

    Returns:
    - str, the rounded number as a string in the specified format.
    '''
    if format_type == 'general':
        result = f"{number:.{digits}g}"
        if 'e' in result:
            return round_float(number, digits=digits, format_type='scientific', latex=latex)
        else:
            return result
    elif format_type == 'standard':
        return f"{number:.{digits}f}"
    elif format_type == 'scientific':
        if latex:
            return sci_str_to_latex(f"{number:.{digits}e}")
        else:
            return f"{number:.{digits}e}"
    elif format_type == 'percent':
        return f"{number:.{digits}%}"


def round_float_auto(number, format_type=ROUND_FORMAT, latex=False, **kwargs):
    '''
    当没有科学记数法时,利用get_decimal_num函数自动确定小数位数
    当科学记数法时(比如1.5*10^3),利用get_decimal_num确定1.5的小数位数

    注意:
    general的情形下,不会采取科学记数法,默认采取标准记数法(因为这个函数的特性在于自动确定小数位数,general无法判断采用哪种记数法)
    '''
    if format_type == 'scientific':
        coefficient, _ = get_coefficient_exponent(number)
        decimal_count = get_decimal_num(coefficient, **kwargs)
    else:
        decimal_count = get_decimal_num(number, **kwargs)
    return round_float(number, digits=decimal_count, format_type=format_type, latex=latex)


def get_coefficient_exponent(number, base=10):
    '''
    将浮点数转换为系数和指数的形式
    '''
    exponent = int(np.floor(np.log(abs(number)) / np.log(base)))
    coefficient = number / (base ** exponent)
    return coefficient, exponent


def sci_float_to_latex(number, base=10, base_str='10', round_digits=0):
    '''
    假设是coefficient乘以base^exp的形式,返回科学计数法的latex字符串

    注意:
    默认状态下,假设了number的形式,并非通用,只适用于整数乘以base^exp的形式,一般来说,是用在ticks的标签上
    当然,也可以用在其他地方,只要符合coefficient乘以base^exp的形式(但是这时需要调整round_digits)
    '''
    if number == 0:
        return r"$0$"  # 0 doesn't have an exponential form

    coefficient, exp = get_coefficient_exponent(number, base)

    # 当在整数模式下使用,良好的处理整数1和-1
    if np.allclose(round_digits, 0):
        if np.allclose(coefficient, 1):
            return fr"${base_str}^{{{exp}}}$"
        elif np.allclose(coefficient, -1):
            return fr"$-{base_str}^{{{exp}}}$"
    # 当在小数模式下使用,不处理整数1和-1,直接返回
    return fr"${coefficient:.{round_digits}f} \times {base_str}^{{{exp}}}$"


def format_float(number, format_type=ROUND_FORMAT, latex=False, replace_dot=REPLACE_DOT, **kwargs):
    '''
    将浮点数格式化为字符串,支持自动确定小数位数.并且可以替换小数点为指定字符.
    '''
    return round_float_auto(number, format_type=format_type, latex=latex, **kwargs).replace('.', replace_dot)


def extract_float_from_filename(filename, key, separator="d"):
    """
    从指定文件名中提取数值(整数或浮点数),支持自定义分隔符(如 d 或 .)
    
    参数:
        filename (str): 文件名
        key (str): 要匹配的 key
        separator (str): 用于分隔整数和小数部分的符号,默认为 'd'
    
    返回:
        float: 文件名中的数值,如果没有匹配则raise ValueError

    注意:
        尽管支持匹配int和float,但是返回值都是float

    使用例子:
        extract_number_from_filename("model_lr_1d0001.pkl", "lr")  # 返回 1.0001
        extract_number_from_filename("model_lr_1.0001.pkl", "lr", separator=".")  # 返回 1.0001
        extract_number_from_filename("./basedir/results/model_lr_1.pkl", "lr")  # 返回 1
    """
    # 正则表达式允许整数或浮点数匹配
    pattern = re.compile(rf"{re.escape(key)}_([+-]?\d+)(?:{re.escape(separator)}(\d+))?")
    match = pattern.search(filename)
    
    if match:
        # 如果有小数部分，拼接成浮点数；否则返回整数
        integer_part, decimal_part = match.groups()
        if decimal_part is not None:
            return float(f"{integer_part}.{decimal_part}")
        else:
            return float(integer_part)
    else:
        raise ValueError('failed to extract number')


def get_filename_and_extension(filename, remove_ext_dot=True):
    # 使用 os.path.splitext 分离文件名和后缀
    name, extension = os.path.splitext(filename)
    
    # 根据选项决定是否去掉后缀名的点
    if remove_ext_dot:
        extension = extension.lstrip('.')  # 去掉前导的点
    
    return name, extension


def align_decimal(number, reference_value):
    '''
    将数字的小数点位数调整为与参考值相匹配。

    参数:
    - number (float): 要对齐的数字。
    - reference_value (float): 用于确定所需小数位数的参考值。

    返回:
    - aligned_number (float): 小数点位数与参考值相匹配的数字列表。
    '''
    decimal_place = len(str(reference_value).split('.')[1]) if '.' in str(reference_value) else 0

    # 格式化数字，保留与参考值相同的小数位数
    aligned_number = "{:.{}f}".format(number, decimal_place)
    return aligned_number


def format_filename(filename, file_process=None):
    '''
    格式化文件名(目前是为了去除空格,格式化浮点数的小数点没有加入此函数)
    '''
    file_process = update_dict(FILENAME_PROCESS, file_process)
    if filename is None:
        return None
    else:
        return filename.replace(' ', file_process['replace_blank'])


def concat_str(strs, sep='_', rm_double_sep=True, ignore_none=True, ignore_empty=True, convert_non_str=True):
    '''
    连接字符串列表,并使用指定的分隔符连接
    
    Parameters:
    - strs: 要连接的字符串列表
    - sep: 用作分隔符的字符串,默认为'_'
    - rm_double_sep: 是否移除重复的分隔符,默认为True
    - ignore_none: 是否忽略None值,默认为True
    - ignore_empty: 是否忽略空字符串,默认为True
    - convert_non_str: 是否将非字符串类型转换为字符串,默认为True
    
    Returns:
    - 连接后的字符串
    '''
    # 如果 ignore_none 为 True,过滤掉 None 值
    if ignore_none:
        strs = [s for s in strs if s is not None]
    
    # 如果 ignore_empty 为 True,过滤掉空字符串
    if ignore_empty:
        strs = [s for s in strs if s != '']
    
    # 如果 convert_non_str 为 True,将非字符串类型转换为字符串
    if convert_non_str:
        strs = [str(s) for s in strs]

    # 连接字符串
    result = sep.join(strs)
    
    # 如果设置了 rm_double_sep 为 True，移除重复的分隔符
    if rm_double_sep:
        while sep*2 in result:
            result = result.replace(sep*2, sep)
    
    return result


def cat(*args, sep='_', rm_double_sep=True, ignore_none=True, ignore_empty=True, convert_non_str=True):
    '''
    连接字符串列表,并使用指定的分隔符连接(简化版)
    '''
    return concat_str(strs=args, sep=sep, rm_double_sep=rm_double_sep, ignore_none=ignore_none, ignore_empty=ignore_empty, convert_non_str=convert_non_str)


def hash_or_str(key, sep='_', replace_dot=REPLACE_DOT):
    """将键转换为字符串，如果复杂则生成哈希值。"""
    if isinstance(key, (str, int, float, bool)):
        if replace_dot:
            return str(key).replace('.', replace_dot)
        else:
            return str(key)
    if isinstance(key, tuple):
        if len(key) < 10:
            return str_tuple(key, sep=sep, replace_dot=replace_dot)
    return hashlib.md5(str(key).encode()).hexdigest()


def str_tuple(t, sep='_', replace_dot=REPLACE_DOT):
    '''
    将元组转换为字符串,并使用指定的分隔符连接

    replace_dot: 则将.替换为replace_dot以应对浮点数转换时出现的'.'
    
    有时候画图需要按照xlim,ylim等作为文件名,这个函数可以将xlim,ylim等转化为字符串
    '''
    result = sep.join(map(str, t))
    if replace_dot:
        result = result.replace('.', replace_dot)
    return result


def str_list(l, sep='_', replace_dot=REPLACE_DOT):
    '''
    将列表转换为字符串,并使用指定的分隔符连接

    replace_dot: 则将.替换为replace_dot以应对浮点数转换时出现的'.'

    有时候画图需要按照xlim,ylim等作为文件名,这个函数可以将xlim,ylim等转化为字符串
    '''
    result = sep.join(map(str, l))
    if replace_dot:
        result = result.replace('.', replace_dot)
    return result
# endregion


# region dict处理相关函数
@message_decorator('use dict(**kwargs) instead of get_dict(**kwargs)')
def get_dict(**kwargs):
    '''获取字典'''
    return kwargs


def update_dict_kwargs(d, **kwargs):
    '''根据kwargs更新字典'''
    d.update(kwargs)
    return d


def fill_missing_key(d, key_list, fill_value=None):
    '''
    处理d
    如果d为None,则返回一个key_list的字典,值为fill_value
    如果d是字典,如果d缺少了key_list中的key,则将key, fill_value添加到d中
    '''
    if d is None:
        return {k: fill_value for k in key_list}
    new_d = d.copy()
    for k in key_list:
        if k not in new_d:
            new_d[k] = fill_value
    return new_d


def fill_dict_kwargs(d, key_list):
    '''
    处理复杂的kwargs输入
    '''
    return fill_missing_key(d, key_list, fill_value={})


@not_recommend
def update_dict_by_name(dic, variable_names):
    '''根据变量名更新字典,不推荐'''
    for name in variable_names:
        dic.update(**{name: eval(name)})


def create_dict(keys, values=None, default_value=None, deep_copy=True):
    """
    创建字典
    
    参数:
    - keys: list, 字典的键
    - values: list, 字典的值
    - default_value: 任意类型, 默认值
    - deep_copy: bool, 是否深拷贝,默认为True

    注意:
    - 如果values为None,则使用default_value作为默认值
    - 如果values不为None,则使用values作为值
    - 用这个函数创建字典时,默认值和值都是深拷贝的,不会产生同时变化的问题
    """
    if deep_copy:
        if values is None:
            return {key: copy.deepcopy(default_value) for key in keys}
        else:
            return {key: copy.deepcopy(value) for key, value in zip(keys, values)}
    else:
        if values is None:
            return {key: default_value for key in keys}
        else:
            return {key: value for key, value in zip(keys, values)}


def union_dict(*dicts):
    '''合并多个字典'''
    # 假如有重复的key,则raise ValueError
    if len(set([k for d in dicts for k in d.keys()])) != sum([len(d) for d in dicts]):
        raise ValueError('Duplicated keys found in the input dictionaries.')
    else:
        return {k: v for d in dicts for k, v in d.items()}


def filter_dict(d, keys):
    '''过滤字典'''
    return {k: d[k] for k in keys if k in d}


def get_sub_dict(d, keys):
    '''获取字典的子集,仅仅是filter_dict的别名'''
    return filter_dict(d, keys)


def update_dict(original_dict, new_dict):
    '''更新字典,保护原始字典'''
    if original_dict is None:
        original_dict = {}
    if new_dict is None:
        return original_dict.copy()
    else:
        return {**original_dict, **new_dict}


def update_dict_ignore(original_dict, new_dict, ignore_value_list=None):
    '''更新字典,忽略某些值,保护原始字典'''
    if ignore_value_list is None:
        ignore_value_list = [None]
    local_new_dict = new_dict.copy()
    for key, value in new_dict.items():
        if value in ignore_value_list:
            local_new_dict.pop(key)
    return update_dict(original_dict, local_new_dict)


def adjust_dict_key(d, key_map):
    '''
    调整字典的键名

    参数:
    - d: 原始字典
    - key_map: function, 键名映射关系
    '''
    new_dict = {}
    for k, v in d.items():
        new_key = key_map(k)
        new_dict[new_key] = v
    return new_dict


def is_subdict(sub_dict, main_dict):
    '''
    判断一个字典是否是另一个字典的子集
    '''
    for key, value in sub_dict.items():
        if key not in main_dict or main_dict[key] != value:
            return False
    return True


def all_elements_in_dict_keys(l, d):
    """检查列表所有元素是否都是字典的键"""
    return all(element in d for element in l)


def get_key_by_value(d, value):
    # 先检查整个字典的值是否重复
    values = list(d.values())
    if len(values) != len(set(values)):
        raise ValueError("字典的值存在重复！")
    
    # 查找匹配的键
    for k, v in d.items():
        if v == value:
            return k
    raise  KeyError(f"值 '{value}' 不存在于字典中")
# endregion


# region data container
class BuildKeyStrategy(abc.ABC):
    """生成键的通用策略基类"""
    def __init__(self, included_name_list=None):
        if included_name_list is None:
            included_name_list = []
        self.included_name_list = included_name_list

    def validate_params(self, kwargs, allowed_params=None):
        if not kwargs:
            raise ValueError("至少需要提供一个有效参数")
        
        if allowed_params is not None:
            invalid = [p for p in kwargs if p not in allowed_params]
            if invalid:
                raise ValueError(f"无效参数: {invalid},允许的参数为: {allowed_params}")

    def _append_to_parts(self, parts, name, value):
        if name in self.included_name_list:
            parts.append(name)
        parts.append(value)

    @abc.abstractmethod
    def build_key(self, **kwargs):
        pass

    def __call__(self, **kwargs):
        """直接调用时触发build_key"""
        return self.build_key(**kwargs)


class SortedBuildKey(BuildKeyStrategy):
    """按参数名首字母顺序生成键"""
    def build_key(self, **kwargs):
        self.validate_params(kwargs)
        sorted_params = sorted(kwargs.keys())
        parts = []
        for name in sorted_params:
            value = kwargs[name]
            self._append_to_parts(parts, name, value)
        return cat(*parts)


class OrderedBuildKey(BuildKeyStrategy):
    """按预定义顺序生成键"""
    def __init__(self, param_order, included_name_list=None):
        self.param_order = param_order
        super().__init__(included_name_list=included_name_list)

    def build_key(self, **kwargs):
        self.validate_params(kwargs, self.param_order)
        parts = []
        for param in self.param_order:
            if param in kwargs:
                value = kwargs[param]
                self._append_to_parts(parts, param, value)
        return cat(*parts)


class DataContainer(dict):
    """
    数据容器,内部是字典,但是可以使用kwargs访问value
    自动按参数名首字母顺序生成键的数据容器,无需指定param_order
    """
    def __init__(self, included_name_list=None):
        super().__init__()
        self.key_builder = SortedBuildKey(included_name_list=included_name_list)
        self['_config'] = {}
        self['_config']['included_name_list'] = self.key_builder.included_name_list
        self['_config']['param_map'] = {}
        self.ignore_key_list = ['_config']

    def build_key(self, **kwargs):
        return self.key_builder.build_key(**kwargs)

    def decompose_key(self, key):
        '''
        将键分解为参数名和参数值
        '''
        return self['_config']['param_map'][key]

    def set_value(self, value, **kwargs):
        key = self.build_key(**kwargs)
        self[key] = value  # 这里没有copy,因为value可能要在内部和外部同时变动
        self['_config']['param_map'][key] = kwargs.copy()  # 保存参数信息
    
    def get_value(self, **kwargs):
        key = self.build_key(**kwargs)
        if key in self:
            return self[key]
        else:
            raise KeyError(f"Key '{key}' not found in DataContainer")

    def get_partial_match_key_list(self, **kwargs):
        '''
        获得包含指定参数的所有键
        '''
        key_list = []
        for k, params in self['_config']['param_map'].items():
            if is_subdict(kwargs, params):
                key_list.append(k)
        return key_list

    def _set_subcontainer(self, subcontainer, remove_matched_params=True, **kwargs):
        '''
        根据参数条件设置子数据容器
        '''
        for key in self:
            if key in self.ignore_key_list:
                continue
            params = self['_config']['param_map'][key]
            if is_subdict(kwargs, params):
                if remove_matched_params:
                    # 获取params除去kwargs的部分
                    new_params = {k: v for k, v in params.items() if k not in kwargs}
                    # new_params有可能为空
                    if not new_params:
                        print(f"Warning: {key} has no remaining parameters after removing matched params, it will be ignored, use remove_matched_params=False to keep it")
                        continue
                else:
                    new_params = params.copy()
                subcontainer.set_value(self[key], **new_params)

    def get_subcontainer(self, remove_matched_params=True, **kwargs):
        """
        根据参数条件获取子数据容器
        
        remove_matched_params: bool, 是否移除匹配的参数
        kwargs: 关键字参数, 用于匹配子数据容器的参数

        示例:
        sub = dc.get_subcontainer(area='V1', prop='spike', remove_matched_params=True)
        则返回一个新的DataContainer,其中包含所有匹配的参数,并且移除匹配的参数
        sub = dc.get_subcontainer(area='V1', prop='spike', remove_matched_params=False)
        则返回一个新的DataContainer,其中包含所有匹配的参数,并且保留匹配的参数
        """
        subcontainer = DataContainer(included_name_list=self['_config']['included_name_list'])
        self._set_subcontainer(subcontainer, remove_matched_params=remove_matched_params, **kwargs)
        return subcontainer

    def delete(self, exact_match=True, **kwargs):
        '''
        删除指定键的值
        
        exact_match: bool, 是否精确匹配, 默认为True, 只有完全匹配的键才会被删除, 如果为False, 则会删除所有包含指定参数的键
        '''
        if exact_match:
            key = self.build_key(**kwargs)
            if key in self:
                del self[key]
                del self['_config']['param_map'][key]
            else:
                raise KeyError(f"Key '{key}' not found in DataContainer")
        else:
            keys_to_delete = []
            for key in self:
                if key in self.ignore_key_list:
                    continue
                params = self['_config']['param_map'][key]
                if is_subdict(kwargs, params):
                    keys_to_delete.append(key)
            for key in keys_to_delete:
                del self[key]
                del self['_config']['param_map'][key]

    def absorb(self, other_container):
        """
        吸收另一个DataContainer的数据(有可能覆盖)
        并根据当前容器的参数顺序重新生成键
        """
        self.absorb_with_new_params(other_container)

    def absorb_from_dict(self, other_dict, dict_key_name, **kwargs):
        '''
        吸收另一个字典的数据(有可能覆盖)
        '''
        for key, value in other_dict.items():
            param = {dict_key_name: key}
            new_param = update_dict(param, kwargs)
            new_key = self.build_key(**new_param)
            self[new_key] = value
            self['_config']['param_map'][new_key] = new_param.copy()  # 保存参数信息

    def absorb_with_new_params(self, other_container, **kwargs):
        '''
        吸收另一个DataContainer的数据(有可能覆盖)
        可以添加新的参数或者覆盖已有的参数

        示例:
        dc.absorb_with_new_params(other_container, new_param='new_value')
        '''
        for key in other_container:
            if key in self.ignore_key_list:
                continue  # 跳过内部参数顺序标记
            # 获取原键对应的参数信息
            if key not in other_container['_config']['param_map']:
                raise KeyError(f"Key '{key}' 不在输入的 data_container 的 param_map 中, 这可能是因为它不是通过 set_value 方法设置的")
            else:
                params_from_other = other_container['_config']['param_map'][key]
            # 生成新键并存储
            new_params_from_other = update_dict(params_from_other, kwargs)
            new_key = self.build_key(**new_params_from_other)
            self[new_key] = other_container[key]
            self['_config']['param_map'][new_key] = new_params_from_other.copy()  # 保存参数信息

    def copy_with_adjusted_params(self, original_params, adjustments):
        '''
        复制数据,并调整键名

        例如:
        original_params = {'area': 'V1', 'prop': 'spike'}
        adjustments = {'prop': 'spike_new'}
        dc.copy_with_adjusted_params(original_params, adjustments)
        '''
        # 合并参数
        new_params = original_params.copy()
        new_params.update(adjustments)
        
        # 获取原始值
        original_value = self.get_value(**original_params)
        
        # 设置新值
        self.set_value(value=original_value, **new_params)

    def move_with_adjusted_params(self, original_params, adjustments):
        '''
        移动数据,并调整键名
        '''
        self.copy_with_adjusted_params(original_params, adjustments)

        original_key = self.build_key(**original_params)
        del self[original_key]
        del self['_config']['param_map'][original_key]

    def save(self):
        '''
        按照各类dict的函数存即可
        '''
        pass

    def load(self):
        '''
        按照各类dict的函数读取即可
        '''
        pass


class OrderedDataContainer(DataContainer):
    '''
    数据容器,内部是字典,但是可以使用kwargs访问value

    使用示例:
    dc = DataContainer(param_order=['area', 'prop', 'time_period'])
    dc.set_value(area='V1', prop='spike', time_period='pre_stimulus_period', value=1)
    dc.get_value(area='V1', prop='spike', time_period='pre_stimulus_period')  # 返回 1

    # 可以不输入完全的参数
    dc.set_value(area='V1', prop='spike', value=1)
    dc.get_value(area='V1', prop='spike')  # 返回 1

    # 可以从另一个DataContainer中更新数据
    dc2 = DataContainer(param_order=['area', 'prop', 'time_period'])
    dc2.set_value(area='V1', prop='spike', time_period='stimulus_period', value=2)
    dc.absorb(dc2)

    # 可以在吸收时添加新的参数
    dc.absorb_with_new_params(dc2, new_param='new_value')

    # 可以从字典中吸收数据
    dc.absorb_from_dict()
    
    # 可以获取子数据容器
    subcontainer = dc.get_subcontainer(area='V1', prop='spike')
    '''
    def __init__(self, param_order, included_name_list=None):
        super().__init__(included_name_list=included_name_list)
        self.key_builder = OrderedBuildKey(param_order=param_order, included_name_list=included_name_list)
        self['_config']['param_order'] = param_order

    def get_subcontainer(self, remove_matched_params=True, **kwargs):
        # 保持原有的参数顺序
        subcontainer = OrderedDataContainer(param_order=self['_config']['param_order'], included_name_list=self['_config']['included_name_list'])
        # 直接调用父类的方法
        self._set_subcontainer(subcontainer, remove_matched_params=remove_matched_params, **kwargs)
        return subcontainer


def _dict_to_data_container(d):
    '''
    将字典转换为DataContainer对象(需要自带_config和其中的param_map)

    示例:
    读取了一个字典,并将其转换为DataContainer对象
    '''
    config = d.pop('_config')
    param_order = config.pop('param_order', None)
    param_map = config.pop('param_map')
    included_name_list = config.pop('included_name_list', None)
    if param_order is None:
        data_container = DataContainer(included_name_list=included_name_list)
    else:
        data_container = OrderedDataContainer(param_order=param_order, included_name_list=included_name_list)
    data_container['_config']['param_map'] = param_map
    for key, value in d.items():
        data_container[key] = value
    return data_container
# endregion


# region data keeper
class DataKeeper:
    """
    数据管理器：灵活管理不同类型的数据存储和读取
    
    注意:
    如果需要,在保存后可调用mark_all_saved()方法来标记所有数据已保存
    """
    def __init__(self, name, basedir, data_type='dict', save_load_method='separate', param_order=None, included_name_list=None):
        self.read_only = False  # 默认可读写
        self.process_num = 1  # 默认单process

        self.name = name
        self.basedir = basedir
        self.data_type = data_type
        self.save_load_method = save_load_method
        self.param_order = param_order
        self.included_name_list = included_name_list
        
        # 初始化结果容器
        if data_type == 'dict':
            self.data = {}
        elif data_type == 'OrderedDataContainer':
            if not param_order:
                raise ValueError("OrderedDataContainer requires param_order")
            self.data = OrderedDataContainer(param_order=param_order, included_name_list=included_name_list)
        else:
            raise ValueError(f"Unknown data_type: {data_type}")
        
        # 配置存储加载方法
        self._config_save_load()
        
        # 尝试从已保存的数据中更新配置
        self.update_config_from_saved()

    def set_read_only(self, read_only):
        '''
        read_only为True时,阻止set_value和save方法的调用
        '''
        self.read_only = read_only

    def _config_save_load(self):
        """配置数据保存和加载方法"""
        valid_methods = ['separate', 'lmdb']
        if self.save_load_method not in valid_methods:
            raise ValueError(f"Invalid save_load_method. Must be one of {valid_methods}")
            
        if self.save_load_method == 'separate':
            self.save_func = save_dict_separate_merge_to_saved
            self.load_func = load_dict_separate_merge_to_exist
            self.delete_func = delete_saved_dict_separate
        elif self.save_load_method == 'lmdb':
            self.save_func = save_dict_lmdb
            self.load_func = load_dict_lmdb_merge_to_exist
            self.delete_func = delete_saved_dict_lmdb

    def _get_data_path(self):
        """获取数据的完整存储路径"""
        return pj(self.basedir, self.name)

    def load(self, **kwargs):
        """从磁盘加载数据"""
        self.load_func(self.data, self._get_data_path(), process_num=self.process_num, **kwargs)

    def update_config_from_saved(self):
        """从已保存的数据中更新配置"""
        if check_all_file_exist(self._get_data_path()) and self.data_type == 'OrderedDataContainer':
            try:
                self.load_func(self.data, self._get_data_path(), 
                            key_to_load=['_config'], ensure_config=False)
            except:
                print('Fail to update config from saved')

    def _get_key(self, key=None, **kwargs):
        '''
        有key时直接使用key,否则根据kwargs构建key(但只对OrderedDataContainer有效)
        '''
        if key is None:
            if self.data_type != 'OrderedDataContainer':
                raise ValueError("Key required for non-container types")
            key = self.data.build_key(**kwargs)
        return key

    def set_value(self, value, key=None, **kwargs):
        if self.read_only:
            raise PermissionError("DataKeeper is read-only, cannot set value")
        else:
            if self.data_type == 'OrderedDataContainer':
                self.data.set_value(value, **kwargs)
            else:
                self.data[key] = value  # 直接设置值

    def get_value(self, key=None, **kwargs):
        """
        获取单个数据值
        - 对于dict: 直接使用key
        - 对于OrderedDataContainer: 可使用关键字参数构建key
        """
        key = self._get_key(key, **kwargs)
        
        # 如果内存中没有则从磁盘加载
        if key not in self.data:
            self.load_func(self.data, self._get_data_path(), 
                          key_to_load=[key], ensure_config=False)
        return self.data[key]

    def get_subcontainer(self, remove_matched_params=True, **kwargs):
        """获取子数据集,仅支持OrderedDataContainer"""
        if self.data_type != 'OrderedDataContainer':
            raise ValueError("Subcontainers only available for OrderedDataContainer")
        
        # 确保所需key已加载
        keys_to_load = self.data.get_partial_match_key_list(**kwargs)
        existing_keys = set(self.data.keys())
        missing_keys = [k for k in keys_to_load if k not in existing_keys]
        
        if missing_keys:
            self.load_func(self.data, self._get_data_path(), 
                          key_to_load=missing_keys, ensure_config=False)
        
        return self.data.get_subcontainer(
            remove_matched_params=remove_matched_params, **kwargs
        )

    def save(self, **kwargs):
        """保存数据到磁盘"""
        if self.read_only:
            raise PermissionError("DataKeeper is read-only, cannot save data")
        else:
            self.save_func(self.data, self._get_data_path(), process_num=self.process_num, **kwargs)
            print_title(f"{self.name} saved")

    def mark_all_saved(self):
        """标记所有数据已保存"""
        save_dict({f"{self.name}_all_saved": True},
                  pj(self._get_data_path(), f"{self.name}_all_saved"),
                  format_list=['txt'])

    def check_all_saved(self):
        """检查保存状态"""
        flag_path = pj(self._get_data_path(), f"{self.name}_all_saved")
        if check_all_file_exist_with_any_extension(flag_path):
            return True
        else:
            return False

    def delete(self, **kwargs):
        """删除数据"""
        self.delete_func(self._get_data_path(), **kwargs)

    def release_memory(self, keys_to_keep=None, keys_to_release=None):
        """
        释放内存中的数据,提供细粒度控制
        
        参数:
        keys_to_keep - 指定要保留的键列表,优先于keys_to_release
        keys_to_release - 指定要释放的键列表,默认释放所有键
        
        默认行为:
        保留配置键'_config'
        """
        if keys_to_keep is None:
            keys_to_keep = []
        if keys_to_release is None:
            keys_to_release = list(self.data.keys())  # 默认释放所有键
        if '_config' not in keys_to_keep:
            keys_to_keep.append('_config')  # 确保配置键始终被保留
            
        # 安全释放指定键
        for key in keys_to_release:
            if key in self.data and key not in keys_to_keep:
                del self.data[key]
# endregion


# region list处理相关函数
def append_unique(lst, item):
    '''
    向列表中添加元素,如果元素已经存在,则不添加
    '''
    if item not in lst:
        lst.append(item)
    return lst


def flatten_list(input_list, level=None):
    '''
    展开嵌套列表(要求每一层都是列表)

    参数：
    - input_list: 嵌套列表
    - level: 展开的层级数,如果为1,则从外往内展开一层,如果为2,则从外往内展开两层,以此类推;如果为None,则展开所有层级
    '''
    if level is None:
        level = float('inf')  # 默认情况下展开所有层
    
    def flatten_recursive(lst, curr_level):
        flattened = []
        for item in lst:
            if isinstance(item, list) and curr_level < level:
                flattened.extend(flatten_recursive(item, curr_level + 1))
            else:
                flattened.append(item)
        return flattened
    
    return flatten_recursive(input_list, 0)


def union_list(*lists, rm_repeat=True):
    '''
    获取多个列表的并集(去重)
    
    参数:
    - lists: 一个或多个列表的可变参数
    
    返回:
    - 一个列表，包含所有输入列表的并集
    '''
    if rm_repeat:
        union_set = set()
        for lst in lists:
            union_set = union_set.union(set(lst))
        return list(union_set)
    else:
        merged_list = []
        for lst in lists:
            merged_list.extend(lst)
        return merged_list


def intersect_list(*lists):
    '''
    获取多个列表的交集。
    
    参数:
    - lists: 一个或多个列表的可变参数。
    
    返回:
    - 一个列表，包含所有输入列表的交集。
    '''
    intersection_set = set(lists[0])
    for lst in lists[1:]:
        intersection_set = intersection_set.intersection(set(lst))
    return list(intersection_set)


def remove_list_element(original_list, elements_to_remove, mode='skip'):
    """
    从列表中移除指定元素，可控制元素不存在时的处理模式。
    
    Args:
        original_list: 原始列表
        elements_to_remove: 需要移除的元素列表
        mode: 处理模式，'skip'（默认跳过不存在元素）或 'raise'（元素不存在时报错）
    
    Returns:
        list: 移除元素后的新列表
    
    Raises:
        ValueError: 当模式为'raise'且存在元素不在原始列表中时
    """
    elements_to_remove_set = set(elements_to_remove)
    
    if mode == 'raise':
        missing_elements = [elem for elem in elements_to_remove_set if elem not in original_list]
        if missing_elements:
            raise ValueError(f"Elements {missing_elements} not found in the list.")
    elif mode != 'skip':
        raise ValueError("Invalid mode. Use 'skip' or 'raise'.")

    return [item for item in original_list if item not in elements_to_remove_set]

@direct_use
def get_element_index_in_list(element, lst):
    return lst.index(element)


def rebuild_list_with_index(flattened, original, index=0):
    '''
    以original为模板，根据flattened中的元素重新构建一个列表。其括号结构与original相同，但元素顺序与flattened相同。这个函数必须要输出index才能不断调用自己，一般而言建议使用rebuild_list作为外部的接口。
    '''
    result = []
    for item in original:
        if isinstance(item, list):
            sub_list, index = rebuild_list_with_index(flattened, item, index)
            result.append(sub_list)
        else:
            result.append(flattened[index])
            index += 1
    return result, index


def rebuild_list(flattened, original):
    '''以original为模板，根据flattened中的元素重新构建一个列表。其括号结构与original相同，但元素顺序与flattened相同。'''
    return rebuild_list_with_index(flattened, original)[0]


def pure_list(l):
    '''遍历嵌套列表，将所有数组转换为列表，但保持嵌套结构不变。'''
    if isinstance(l, list):
        # 如果是列表，递归地对每个元素调用 pure_list
        return [pure_list(item) for item in l]
    elif isinstance(l, np.ndarray):
        # 如果是numpy数组，且其形状大于1（即不是单个数字），先将其转换为列表
        if l.shape:  # 检查数组是否不仅仅是单个元素
            return [pure_list(item) for item in l.tolist()]
        else:
            # 对于单个元素的numpy数组，直接返回它的Python类型
            return l.item()
    elif isinstance(l, tuple):
        # 如果是元组，先将其转换为列表
        return [pure_list(item) for item in l]
    else:
        # 对于其他类型的元素，直接返回
        return l


def list_shape(lst):
    '''
    假定list是和numpy的array类似的嵌套列表，返回list的形状。(不能处理不规则的嵌套列表)
    '''
    if not isinstance(lst, list):
        return ()
    if len(lst) == 0:
        return (0,)
    return (len(lst),) + list_shape(lst[0])


def is_sublist(sub_list, main_list):
    '''
    判断一个列表是否是另一个列表的子集
    '''
    main_counter = Counter(main_list)
    sub_counter = Counter(sub_list)
    for elem, count in sub_counter.items():
        if main_counter[elem] < count:
            return False
    return True
# endregion


# region instance container
class InstanceContainer:
    '''
    用来存储一组实例,并提供一些常用的操作方法,如过滤/提取信息等
    '''
    def __init__(self, items=None):
        self._instances = list(items) if items else []
    
    def append(self, item):
        self._instances.append(item)
    
    def extend(self, items):
        self._instances.extend(items)
    
    def get_filtered_by_func(self, func):
        '''
        注意这个操作会返回一个新的InstanceContainer,原来的不会变
        '''
        return InstanceContainer(item for item in self if func(item))
    
    def get_filtered_by_attr(self, **attributes):
        '''
        注意这个操作会返回一个新的InstanceContainer,原来的不会变
        '''
        def match(item):
            for attr, value in attributes.items():
                if not hasattr(item, attr) or getattr(item, attr) != value:
                    return False
            return True
            
        return InstanceContainer(filter(match, self._instances))
    
    def inplace_filter_by_func(self, func):
        '''
        原地过滤,使用函数过滤,直接修改当前容器
        保留满足条件的实例,移除不满足条件的
        返回自身以支持链式调用
        '''
        self._instances = [item for item in self._instances if func(item)]
        return self
    
    def inplace_filter_by_attr(self, **attributes):
        '''
        原地过滤,使用属性条件过滤,直接修改当前容器
        保留满足所有属性条件的实例,移除不满足的
        返回自身以支持链式调用
        '''
        def match(item):
            for attr, value in attributes.items():
                if not hasattr(item, attr) or getattr(item, attr) != value:
                    return False
            return True
            
        self._instances = list(filter(match, self._instances))
        return self

    def get_info_by_attr(self, attribute):
        return [getattr(item, attribute) for item in self._instances]
    
    def get_info_by_func(self, func):
        return [func(item) for item in self._instances]
    
    def __len__(self):
        return len(self._instances)
    
    def __iter__(self):
        return iter(self._instances)
    
    def __getitem__(self, index):
        return self._instances[index]
# endregion


# region tuple处理相关函数
def flatten_tuple(tpl, level=None):
    '''
    展开嵌套元组(要求每一层都是元组)

    参数：
    - tpl: 嵌套元组
    - level: 展开的层级数,如果为1,则从外往内展开一层,如果为2,则从外往内展开两层,以此类推;如果为None,则展开所有层级
    '''
    if level is None:
        level = float('inf')  # 默认情况下展开所有层

    def flatten_recursive(tpl, curr_level):
        flattened = []
        for item in tpl:
            if isinstance(item, tuple) and curr_level < level:
                flattened.extend(flatten_recursive(item, curr_level + 1))
            else:
                flattened.append(item)
        return flattened
    
    return tuple(flatten_recursive(tpl, 0))


def union_tuple(*tuples):
    '''
    获取多个元组的并集。
    
    参数:
    - tuples: 一个或多个元组的可变参数。
    
    返回:
    - 一个元组，包含所有输入元组的并集。
    '''
    union_set = set()
    for tpl in tuples:
        union_set = union_set.union(set(tpl))
    return tuple(union_set)


def intersect_tuple(*tuples):
    '''
    获取多个元组的交集。
    
    参数:
    - tuples: 一个或多个元组的可变参数。
    
    返回:
    - 一个元组，包含所有输入元组的交集。
    '''
    intersection_set = set(tuples[0])
    for tpl in tuples[1:]:
        intersection_set = intersection_set.intersection(set(tpl))
    return tuple(intersection_set)


def rebuild_tuple_with_index(flattened, original, index=0):
    '''
    以original为模板，根据flattened中的元素重新构建一个元组。其括号结构与original相同，但元素顺序与flattened相同。这个函数必须要输出index才能不断调用自己，一般而言建议使用rebuild_tuple作为外部的接口。
    '''
    result = []
    for item in original:
        if isinstance(item, tuple):
            sub_tuple, index = rebuild_tuple_with_index(flattened, item, index)
            result.append(sub_tuple)
        else:
            result.append(flattened[index])
            index += 1
    return tuple(result), index


def rebuild_tuple(flattened, original):
    '''以original为模板，根据flattened中的元素重新构建一个元组。其括号结构与original相同，但元素顺序与flattened相同。'''
    return rebuild_tuple_with_index(flattened, original)[0]


def pure_tuple(tpl):
    '''遍历嵌套元组，将所有数组转换为元组，但保持嵌套结构不变。'''
    if isinstance(tpl, tuple):
        # 如果是元组，递归地对每个元素调用 pure_tuple
        return tuple([pure_tuple(item) for item in tpl])
    elif isinstance(tpl, list):
        # 如果是列表，先将其转换为元组
        return tuple([pure_tuple(item) for item in tpl])
    elif isinstance(tpl, np.ndarray):
        # 如果是numpy数组，且其形状大于1（即不是单个数字），先将其转换为元组
        if tpl.shape:  # 检查数组是否不仅仅是单个元素
            return tuple([pure_tuple(item) for item in tpl.tolist()])
        else:
            # 对于单个元素的numpy数组，直接返回它的Python类型
            return tpl.item()
    else:
        # 对于其他类型的元素，直接返回
        return tpl


def tuple_shape(tpl):
    '''
    假定tuple是和numpy的array类似的嵌套元组，返回tuple的形状。(不能处理不规则的嵌套元组)
    '''
    if not isinstance(tpl, tuple):
        return ()
    if len(tpl) == 0:
        return (0,)
    return (len(tpl),) + tuple_shape(tpl[0])
# endregion


# region DataFrame处理相关函数
def reorder_df(df, new_row=None, new_col=None):
    '''
    按照指定顺序对DataFrame的行和列进行排序
    '''
    if new_row is not None:
        df = df.loc[new_row]
    if new_col is not None:
        df = df[new_col]
    return df


def align_df_row_col_order(df, keep_which='row'):
    '''
    确保df的行和列顺序一致
    '''
    if keep_which == 'row':
        return df.sort_index(axis=1).sort_index(axis=0)
    elif keep_which == 'col':
        return df.sort_index(axis=0).sort_index(axis=1)


def get_sub_df(df, row_list=None, col_list=None):
    '''
    获取DataFrame的子集
    '''
    if row_list is not None:
        new_df = df.loc[row_list]
    else:
        new_df = df.copy()
    if col_list is not None:
        new_df = new_df[col_list]
    return new_df


def get_pd_triangle_value(df, triangle_type='lower', diagonal=False):
    '''
    从Pandas DataFrame中提取三角形部分的值
    df: 输入的DataFrame
    triangle_type: 'lower' 或 'upper',表示提取下三角还是上三角部分,默认为None,表示提取全部
    diagonal: 是否包括对角线
    返回：提取的三角形部分的值
    '''
    # 确保行和列顺序一致
    df_sorted = align_df_row_col_order(df)
    
    arr = df_sorted.to_numpy()  # 转换为Numpy数组
    
    # 使用get_triangle_value函数提取三角形部分
    return get_arr_triangle_value(arr, triangle_type, diagonal)


def replace_df_diagonal(df, replace_value=0):
    '''
    将DataFrame对角线上的元素替换为指定值
    '''
    df_copy = df.copy()
    for row in df_copy.index:
        for col in df_copy.columns:
            if row == col:
                df_copy.loc[row, col] = replace_value
    return df_copy


def symmetrize_df(df):
    '''
    将DataFrame转换为对称矩阵
    '''
    # 确保df的row和col集合是一样的(顺序可以不一样)
    if not set(df.index) == set(df.columns):
        raise ValueError('The DataFrame must have the same row and column names.')
    new_df = df.copy()
    for row in new_df.index:
        for col in new_df.columns:
            if row != col:
                new_df.loc[col, row] = (df.loc[row, col] + df.loc[col, row] ) / 2
    return new_df


def get_zero_df(index, columns):
    '''创建一个DataFrame'''
    df = pd.DataFrame(np.zeros((len(index), len(columns))), index=index, columns=columns)
    index_col_str(df)
    return df


def index_col_str(df):
    '''将df的index和columns转换为字符串'''
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)


def normalize_df_with_sum(df, row_or_col):
    '''
    正则化df,使得行或列的和为1

    注意:
    nan值会被当作0处理
    '''
    if row_or_col == 'row':
        axis = 1
    elif row_or_col == 'col':
        axis = 0
    
    return df.div(df.sum(axis=axis), axis=1-axis)


def row_sum_df(df):
    '''计算DataFrame每行的和'''
    return df.sum(axis=1)


def row_mean_df(df):
    '''计算DataFrame每行的均值'''
    return df.mean(axis=1)


def row_std_df(df):
    '''计算DataFrame每行的标准差'''
    return df.std(axis=1)


def row_median_df(df):
    '''计算DataFrame每行的中位数'''
    return df.median(axis=1)


def col_sum_df(df):
    '''计算DataFrame每列的和'''
    return df.sum(axis=0)


def col_mean_df(df):
    '''计算DataFrame每列的均值'''
    return df.mean(axis=0)


def col_std_df(df):
    '''计算DataFrame每列的标准差'''
    return df.std(axis=0)


def col_median_df(df):
    '''计算DataFrame每列的中位数'''
    return df.median(axis=0)


def double_sum_df(df, sum_type='intersect'):
    if sum_type == 'intersect':
        sum_list = intersect_list(df.columns, df.index)
    elif sum_type == 'union':
        sum_list = union_list(df.columns, df.index)
    
    sum_series = pd.Series(np.zeros(len(sum_list)), index=sum_list)

    for sum_name in sum_list:
        for row in df.index:
            for col in df.columns:
                if row == sum_name or col == sum_name:
                    sum_series[sum_name] += df.loc[row, col]
    return sum_series
# endregion


# region array处理相关函数
def flatten_array(arr):
    '''扁平化数组'''
    return arr.flatten()


def get_arr_triangle_value(arr, triangle_type=None, diagonal=False):
    """
    获取数组的三角形部分的值
    arr: 输入的二维数组或矩阵
    triangle_type: 'lower' 或 'upper',表示提取下三角还是上三角部分,默认为None,表示提取全部
    diagonal: 是否包括对角线
    返回：提取的三角形部分的值
    """
    if triangle_type not in ['lower', 'upper', None]:
        raise ValueError("triangle_type must be 'lower' or 'upper' or None")
    
    # 获取三角形的索引
    if triangle_type == 'lower':
        triangle_mask = np.tril(np.ones_like(arr, dtype=bool))  # 下三角
    elif triangle_type == 'upper':
        triangle_mask = np.triu(np.ones_like(arr, dtype=bool))  # 上三角
    else:
        triangle_mask = np.ones_like(arr, dtype=bool)  # 全部
    
    if not diagonal:
        # 如果不包括对角线，去掉对角线
        np.fill_diagonal(triangle_mask, False)
    
    return arr[triangle_mask]


def split_matrix_diag(M):
    """
    将矩阵 M 分成两个矩阵：
    - 一个矩阵包含对角线元素，其他位置为 0
    - 另一个矩阵包含非对角线元素，对角线位置为 0

    参数:
    M (numpy.ndarray): 输入矩阵

    返回:
    diag_matrix, non_diag_matrix (numpy.ndarray, numpy.ndarray): 对角线和非对角线矩阵
    """
    # 创建与 M 形状相同的全零矩阵
    diag_matrix = np.zeros_like(M)
    non_diag_matrix = np.zeros_like(M)

    # 获取对角线成分并填充到对角线矩阵中
    np.fill_diagonal(diag_matrix, np.diag(M))

    # 将对角线位置置为 0，保留非对角线元素
    non_diag_matrix = M - diag_matrix

    return diag_matrix, non_diag_matrix


def get_connection_idx_and_weight(weight_matrix):
    '''
    获取连接的索引和权重
    '''
    positions = np.argwhere(weight_matrix != 0)
    i_indices = positions[:, 0]
    j_indices = positions[:, 1]
    connection_strengths = weight_matrix[i_indices, j_indices]
    return i_indices, j_indices, connection_strengths


def get_nearby_idx(arr, index, nearby_index):
    """
    获取给定多维数组 arr 中某个索引 index 前后 nearby_index 范围内的所有索引。

    参数:
    arr (numpy.ndarray): 需要操作的多维数组
    index (tuple): 需要获取的索引,可以是多个维度
    nearby_index (int): 要获取的范围,即每个维度上索引 index 前后 nearby_index 个元素

    返回:
    list: 所有满足条件的索引组成的列表
    """
    # 获取数组的维度
    dims = len(index)

    # 计算需要获取的起始和结束索引
    starts = [max(0, index[i] - nearby_index) for i in range(dims)]
    ends = [min(arr.shape[i], index[i] + nearby_index + 1) for i in range(dims)]
    return [(start, end) for start, end in zip(starts, ends)]


def assign_nearby_value(arr, index, value, nearby_index):
    """
    将给定多维数组 arr 中某个索引 index 处的值,以及距离 index 在 nearby_index 范围内的所有值都赋为 value。
    
    参数:
    arr (numpy.ndarray): 需要操作的多维数组
    index (tuple): 需要赋值的索引,可以是多个维度
    value (any): 要赋的值
    nearby_index (int): 要赋值的范围,即每个维度上索引 index 前后 nearby_index 个元素
    
    返回:
    numpy.ndarray: 修改后的数组
    """
    # 获取数组的维度
    dims = len(index)
    
    # 计算需要赋值的起始和结束索引
    starts = [max(0, index[i] - nearby_index) for i in range(dims)]
    ends = [min(arr.shape[i], index[i] + nearby_index + 1) for i in range(dims)]

    # 创建一个切片对象
    slices = [slice(starts[i], ends[i]) for i in range(dims)]

    # 赋值
    arr[tuple(slices)] = value
    
    return arr


def step_linspace(start, stop, step, endpoint=True):
    '''等差数列'''
    arr = np.arange(start, stop, step)
    if endpoint:
        # 判断step是否能整除(start, stop)
        num = int((stop - start) / step)
        if np.allclose(start + num * step, stop) or np.allclose(start + num * step, stop - step):
            return arr
        else:
            raise ValueError('step不能整除(start, stop)')
    else:
        return arr


def gradient_step_linspace(start, end, num, endpoint=True, gradient='center'):
    """
    生成一个密度梯度的线性空间。

    参数:
    start : float
        线性空间的起始点。
    end : float
        线性空间的结束点。
    num : int
        生成的点的总数。
    endpoint : bool, 默认True
        是否包含结束点。
    gradient : str, 默认'center', 可选'center','edge','left','right'
        线性空间的密度梯度，'center'表示中心密度高两边低，'edge'表示两边密度高中心低, 'left'表示左边密度高右边低, 'right'表示右边密度高左边低。

    返回:
    numpy.ndarray
        生成的点组成的数组。
    """
    if gradient == 'center':
        gradient_func = lambda x: np.arcsin(x) / (np.pi / 2)
        return (gradient_func(np.linspace(-1, 1, num, endpoint=endpoint)) + 1)/2 * (end - start) + start
    elif gradient == 'edge':
        gradient_func = lambda x: np.sin(x * np.pi / 2)
        return (gradient_func(np.linspace(-1, 1, num, endpoint=endpoint)) + 1)/2 * (end - start) + start
    elif gradient == 'left':
        gradient_func = lambda x: 1 - np.sin(x * np.pi / 2)
        return gradient_func(np.linspace(0, 1, num, endpoint=endpoint)) * (end - start) + start
    elif gradient == 'right':
        gradient_func = lambda x: np.sin(x * np.pi / 2)
        return gradient_func(np.linspace(0, 1, num, endpoint=endpoint)) * (end - start) + start


def insert_mid(arr):
    '''在数组中间插入相邻元素的平均值'''
    new_arr = []
    for i in range(len(arr)):
        new_arr.append(arr[i])
        if i < len(arr) - 1:
            new_arr.append((arr[i] + arr[i+1]) / 2)
    
    if isinstance(arr, np.ndarray):
        return np.array(new_arr)
    else:
        return new_arr
# endregion


# region 切片,维数变换相关函数
def get_index(*slices):
    """
    根据输入的 slices 返回一个可以直接用于 numpy 数组的索引.
    支持 slice, ':', ..., None 和整型索引.
    
    参数:
        *slices: 可变参数，可以是 slice 对象, int, ':'(代表整个维度), None(代表整个维度) 或 ...(代表剩余维度)。
    返回:
        索引元组，可以直接用于 numpy 数组的索引。
    """
    # 将 ':' 和 None 处理为 slice(None) 以表示整个维度
    indices = tuple(slice(None) if s == ':' or s is None else s for s in slices)
    return indices


def slice_to_array(s):
    # 获取 start、stop、step，处理 step 为 None 的情况
    start = s.start
    stop = s.stop
    step = s.step if s.step is not None else 1

    # 检查 start 或 stop 是否为 None
    if start is None or stop is None:
        raise ValueError("slice的 start 和 stop 不能为 None")

    # 检查 step 是否为 0
    if step == 0:
        raise ValueError("slice的 step 不能为 0")

    # 生成索引数组
    return np.arange(start, stop, step)


def get_slice(start=None, stop=None, step=None):
    '''
    创建一个 slice 对象。
    
    参数:
    start (int, 可选): 切片的起始索引。如果不提供,默认为 None。
    stop (int, 可选): 切片的结束索引(不包含)。如果不提供,默认为 None。
    step (int, 可选): 切片的步长。如果不提供,默认为 None。
    
    返回:
    slice: 根据输入参数创建的 slice 对象。
    '''
    return slice(start, stop, step)


def get_slice_length(s):
    '''
    获取 slice 对象的长度(即切片的元素个数)
    '''
    return len(range(s.start, s.stop, s.step)) if s.step else len(range(s.start, s.stop))


def out_ensure_dim(arr, target_dim):
    '''
    通过在最外侧增加或者删除维数的方式确保输入数组具有指定的维度。
    
    参数:
    arr (list/numpy.ndarray): 输入数组。
    target_dims (int): 目标维度。
    
    返回值:
    numpy.ndarray: 具有目标维度的输入数组。
    
    说明:
    1. 如果输入数组的维度小于目标维度,则在外部添加新的维度,直到达到目标维度。
    2. 如果输入数组的维度已经等于或大于目标维度,则不进行任何操作。
    3. 该函数支持列表和NumPy数组作为输入。
    4. 返回值始终是NumPy数组。
    '''
    # 将输入转换为NumPy数组
    arr = np.array(arr)
    
    # 获取输入数组的维度
    curr_dims = arr.ndim
    
    # 如果输入数组的维度小于目标维度,则添加新的维度
    if curr_dims < target_dim:
        for _ in range(target_dim - curr_dims):
            arr = np.expand_dims(arr, axis=0)
    
    # 如果输入数组的维度大于目标维度
    elif curr_dims > target_dim:
        print('Warning: 输入数组的维度大于目标维度。将移除最外层维度以匹配目标维度。')
        # 如果最外层维度的长度为1,则将其移除
        while curr_dims > target_dim and arr.shape[0] == 1:
            arr = np.squeeze(arr, axis=0)
            curr_dims -= 1
        
        # 如果维度仍然大于目标维度,则引发异常
        if curr_dims > target_dim:
            raise ValueError("输入数组的维度大于目标维度,且无法通过移除最外层维度来解决。")
    return arr


def rm_out_dim(arr):
    '''
    移除输入数组中所有多余的外层括号(维度)。
    
    参数:
    arr (list/numpy.ndarray): 输入数组。
    
    返回值:
    numpy.ndarray: 移除了多余外层括号的输入数组。
    
    说明:
    1. 该函数会循环移除输入数组的最外层维度,直到该维度的长度不为1。
    2. 该函数支持列表和NumPy数组作为输入。
    3. 返回值始终是NumPy数组。
    '''
    # 将输入转换为NumPy数组
    arr = np.array(arr)
    
    # 循环移除最外层维度,直到该维度的长度不为1
    while arr.ndim > 0 and arr.shape[0] == 1:
        arr = np.squeeze(arr, axis=0)
    
    return arr


def rm_extra_dim(arr):
    '''
    移除输入数组中所有多余的括号(维度)。
    
    参数:
    arr (list/numpy.ndarray): 输入数组。
    
    返回值:
    numpy.ndarray: 移除了所有多余括号的输入数组。
    
    说明:
    1. 该函数会移除输入数组中所有长度为1的维度。
    2. 该函数支持列表和NumPy数组作为输入。
    3. 返回值始终是NumPy数组。
    '''
    # 将输入转换为NumPy数组
    arr = np.array(arr)
    
    # 获取包含长度不为1的维度的索引
    non_redundant_dims = [i for i, dim in enumerate(arr.shape) if dim != 1]
    
    # 使用上述索引从原始数组中提取非多余维度
    arr_reduced = arr.reshape(tuple(arr.shape[i] for i in non_redundant_dims))
    
    return arr_reduced
# endregion


# region 排序相关函数
def sort_dict(d, sort_by='value', reverse=False):
    '''
    Sort a dictionary by its keys or values.

    Parameters:
    - d: The dictionary to sort.
    - sort_by: Criteria for sorting ('value' or 'key').
    - reverse: Sort in descending order if True, else in ascending order.

    Returns:
    - A new sorted dictionary.
    '''
    if sort_by == 'key':
        return {k: d[k] for k in sorted(d.keys(), reverse=reverse)}
    elif sort_by == 'value':
        return {k: v for k, v in sorted(d.items(), key=lambda item: item[1], reverse=reverse)}
    else:
        raise ValueError("sort_by must be 'key' or 'value'")


def sort_dict_by_list(d, sort_by):
    '''
    Sort a dictionary's value by the order of elements in a list.

    Parameters:
    - d: The dictionary to sort.
    - sort_by: The list to use for sorting.

    Returns:
    - A new sorted dictionary.
    '''
    return {k: d[k] for k in sort_by if k in d}


def sort_list(l, reverse=False):
    '''
    Sort a list in ascending or descending order.

    Parameters:
    - l: The list to sort.
    - reverse: Sort in descending order if True, else in ascending order.

    Returns:
    - A new sorted list.
    '''
    return sorted(l, reverse=reverse)


def sort_list_as_subsequence(list_to_sort, sort_by, reverse=False):
    '''
    Sort a list (which is a subset of another list) based on the order of the other list.

    Parameters:
    - list_to_sort: The list to sort, which is a subset of 'sort_by'.
    - sort_by: The list to use for sorting.
    - reverse: Sort in descending order if True, else in ascending order.

    Returns:
    - A new list sorted according to the order of elements in 'sort_by'.
    '''
    # Create a dictionary that maps each value in 'sort_by' to its index
    index_map = {value: index for index, value in enumerate(sort_by)}

    # Filter 'list_to_sort' to include only those elements that are present in 'sort_by'
    # Then sort these elements based on their index in 'sort_by'
    sorted_list = sorted(filter(lambda x: x in index_map, list_to_sort),
                         key=lambda x: index_map[x],
                         reverse=reverse)

    return sorted_list


def sort_series(s, sort_by='value', reverse=False, ignore_capitalize=True):
    '''
    Sort a pandas Series by its index or values.

    Parameters:
    - s: The Series to sort.
    - sort_by: Criteria for sorting ('value' or 'index').
    - reverse: Sort in descending order if True, else in ascending order.
    - ignore_capitalize: Ignore capitalization when sorting by index.

    Returns:
    - A new sorted Series.
    '''
    if ignore_capitalize and sort_by == 'index':
        # Convert index to lowercase for case-insensitive sorting
        temp_series = s.copy()
        temp_series.index = s.index.str.lower()
        sorted_series = temp_series.sort_index(ascending=not reverse)
        # Restore original index order based on sorted lower case index
        # This step involves matching the lowercase sorted index with the original index
        # and arranging the original Series according to this order
        original_order_index = dict(zip(temp_series.index, s.index))
        final_index_order = [original_order_index[i] for i in sorted_series.index]
        return s.loc[final_index_order]
    elif sort_by == 'index':
        return s.sort_index(ascending=not reverse)
    elif sort_by == 'value':
        return s.sort_values(ascending=not reverse)
    else:
        raise ValueError("sort_by must be 'index' or 'value'")


def sort_df(df, sort_by='index', axis=0, key=None, reverse=False):
    '''
    Sort a pandas DataFrame by its index, a specific column, or columns.

    Parameters:
    - df: The DataFrame to sort.
    - sort_by: Criteria for sorting ('index', 'values').
    - axis: The axis to sort (0 for rows, 1 for columns). 
             Note: Sorting by columns ('values' with axis=1) is not supported in this function.
    - key: The column name(s) to sort by if sort_by is 'values'. 
           Ignored if sort_by is 'index'. Can be a single label or list of labels.
    - reverse: Sort in descending order if True, else in ascending order.

    Returns:
    - A new sorted DataFrame.

    Raises:
    - ValueError: If `sort_by` is not 'index' or 'values', or if sorting by columns with 'values'.
    '''
    if sort_by == 'index':
        return df.sort_index(axis=axis, ascending=not reverse)
    elif sort_by == 'values':
        if axis == 1:
            raise ValueError(
                "Sorting by columns ('values' with axis=1) is not supported.")
        if key is None:
            raise ValueError("Key must be provided when sorting by 'values'.")
        return df.sort_values(by=key, ascending=not reverse)
    else:
        raise ValueError("sort_by must be 'index' or 'values'")
# endregion


# region 特殊函数
def identical_func(x):
    '''
    返回输入的参数，不做任何处理
    '''
    return x
# endregion


# region 函数处理相关函数
@direct_use
def fix_param(func, **kwargs):
    '''
    固定函数的部分参数，返回一个新的函数。
    '''
    return partial(func, **kwargs)


def globalize_func(func):
    global_func_name = func.__name__
    globals()[global_func_name] = func
    return func


def get_default_param(func):
    sig = inspect.signature(func)
    return {
        k: v.default
        for k, v in sig.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def get_param_value(func, args, kwargs, param_name):
    """
    获取函数指定参数的值，包括考虑默认值的情况。(适合在decorator内使用,因为需要获取某个变量的值并根据这个变量做一些操作)
    
    :param func: 被装饰的函数
    :param args: 位置参数
    :param kwargs: 关键字参数
    :param param_name: 需要获取值的参数名称
    :return: 参数的值（如果存在），否则为 None
    """
    signature = inspect.signature(func)
    bound_args = signature.bind_partial(*args, **kwargs)
    bound_args.apply_defaults()  # 应用默认值
    
    # 获取指定参数的值
    return bound_args.arguments.get(param_name, None)


def get_all_func(module, only_module=True):
    '''
    获取指定模块中的所有函数名。

    参数:
    module: Python模块对象，你想要获取函数名的模块。
    only_module: 布尔值，如果为True，只返回模块中自己定义的函数；
                 如果为False，返回模块中的所有函数，包括导入的函数。

    返回值:
    一个列表，包含了指定模块中的所有函数名。函数名按照在源文件中的位置排序。
    '''
    if only_module:
        functions = [(name, obj) for name, obj in inspect.getmembers(module) 
                     if isinstance(obj, types.FunctionType) and obj.__module__ == module.__name__]
    else:
        functions = [(name, obj) for name, obj in inspect.getmembers(module) 
                     if isinstance(obj, types.FunctionType)]
    functions.sort(key=lambda x: inspect.getsourcelines(x[1])[1])  # 按照函数在源文件中的位置排序
    return [name for name, _ in functions]


def print_func_source(module, function_name):
    '''
    打印指定模块中指定函数的源代码。

    参数:
    module: Python模块对象，你想要获取函数源代码的模块。
    function_name: 字符串，你想要获取源代码的函数的名字。

    返回值:
    无
    '''
    function = getattr(module, function_name, None)
    if function and inspect.isfunction(function):
        print(f"Function name: {function_name}")
        print(inspect.getsource(function))
    else:
        print(f"No function named '{function_name}' in module '{module.__name__}'")


def write_func_to_file(module, functions, filename):
    if isinstance(functions, str):
        functions = [functions]
    with open(filename, 'w') as f:
        for func_name in functions:
            func = getattr(module, func_name)  # 获取函数对象
            f.write(f"# Function: {func_name}\n")
            f.write(inspect.getsource(func))
            f.write("\n\n")


def get_func_name(function):
    return function.__name__


def is_func(function):
    return isinstance(function, types.FunctionType)


def run_func(func, *args, **kwargs):
    '''
    运行函数并返回结果。

    参数:
    - func: 要运行的函数。
    - args: 函数的位置参数。
    - kwargs: 函数的关键字参数。

    返回值:
    函数的返回值。
    '''
    return func(*args, **kwargs)


def call_func(func_name, *args, **kwargs):
    '''
    使用函数名调用函数。
    '''
    # 获取当前全局命名空间
    func = globals().get(func_name)
    # 如果找到了对应的函数，则调用它
    if func:
        return func(*args, **kwargs)
    else:
        print(f"No function named '{func_name}' found.")


def call_func_from_module(module_path, func_name, *args, **kwargs):
    # 动态导入指定模块
    module = importlib.import_module(module_path)
    
    # 从模块中获取函数并调用
    if hasattr(module, func_name):
        func = getattr(module, func_name)
        return func(*args, **kwargs)  # 调用函数
    else:
        print(f"No function named '{func_name}' found in module '{module_path}'.")
# endregion


# region class处理相关函数
def get_class_name(cls):
    return cls.__name__


def is_class(obj):
    return inspect.isclass(obj)


def add_method_to_instance(instance, method_func, method_name):
    """
    将方法动态添加到一个实例中。
    
    参数:
        instance: 目标实例
        method_func: 要添加的函数,需包含self参数
        method_name: 添加后的方法名称
    """
    bound_method = types.MethodType(method_func, instance)
    setattr(instance, method_name, bound_method)
# endregion


# region 排序
def sort_array(arr, ascending=True, nan_policy='warn'):
    """
    对数组或列表进行排序,如果包含 NaN 则发出警告。
    
    参数:
        arr (list or array-like):要排序的输入数据
        ascending (bool):如果为 True,按从小到大排序,否则按从大到小排序
        nan_policy (str): 可选'warn', 'raise', 'propagate'之一,默认为'warn'; 'warn'表示在数据中存在NaN时发出警告,'raise'表示在数据中存在NaN时引发异常,'propagate'表示不处理NaN
    
    返回:
        sorted_array (list or ndarray): 排序后的结果，保持输入类型。
    """
    is_list = isinstance(arr, list)
    arr = np.asarray(arr)
    
    if np.isnan(arr).any():
        if nan_policy == 'warn':
            print("Warning: NaN values are present in the input data.")
        elif nan_policy == 'raise':
            raise ValueError("NaN values are present in the input data.")
        elif nan_policy == 'propagate':
            pass
    
    if ascending:
        sorted_arr = np.sort(arr)
    else:
        sorted_arr = np.sort(arr)[::-1]
    
    return sorted_arr.tolist() if is_list else sorted_arr


def argsort_array(arr, ascending=True, nan_policy='warn'):
    """
    对数组或列表返回排序后的索引,如果包含 NaN 则根据 nan_policy 参数处理
    
    参数:
        arr (list or array-like): 要排序的输入数据
        ascending (bool): 如果为 True,按从小到大排序;否则按从大到小排序
        nan_policy (str): 可选 'warn', 'raise', 'propagate' 之一,默认为 'warn'
                         'warn' 表示在数据中存在 NaN 时发出警告
                         'raise' 表示在数据中存在 NaN 时引发异常
                         'propagate' 表示不处理 NaN
    
    返回:
        indices (ndarray): 排序后的索引
    """
    arr = np.asarray(arr)
    
    # 检查 NaN 并处理
    if np.isnan(arr).any():
        if nan_policy == 'warn':
            print("Warning: NaN values are present in the input data.")
        elif nan_policy == 'raise':
            raise ValueError("NaN values are present in the input data.")
        elif nan_policy == 'propagate':
            pass

    # 根据升序或降序返回排序索引
    if ascending:
        indices = np.argsort(arr)
    else:
        indices = np.argsort(arr)[::-1]
    
    return indices
# endregion


# region 数据清洗相关函数(nan,inf)
def process_inf(data, inf_policy=INF_POLICY):
    '''Processes inf values according to the specified policy.'''
    if inf_policy == 'to_nan':
        if isinstance(data, np.ndarray):
            data = np.where(np.isinf(data), np.nan, data)
        elif isinstance(data, list):
            data = np.where(np.isinf(data), np.nan, data).tolist()
        elif isinstance(data, (pd.Series, pd.DataFrame)):
            data = data.replace([np.inf, -np.inf], np.nan)
    elif inf_policy == 'ignore':
        pass
    return data


def process_nan(data, nan_policy=NAN_POLICY, fill_value=0):
    '''
    主要适用于一维数据,对于二维数据,drop interplate等操作不被允许,需要使用者分解为一维数据后再进行操作

    参数:
    - data: 输入的数据
    - nan_policy: 可选'propagate', 'fill', 'drop', 'interpolate'之一,默认为NAN_POLICY; propogate表示不处理NaN,fill表示用fill_value填充NaN,drop表示删除NaN,interpolate表示使用插值填充NaN
    - fill_value: 用于填充NaN的值, 默认为0
    '''
    def df_like_interpolate(data):
        if isinstance(data, np.ndarray):
            return pd.Series(data).interpolate().values
        elif isinstance(data, list):
            return pd.Series(data).interpolate().tolist()
    if isinstance(data, list):
        data = np.array(data)
        if nan_policy == 'propagate':
            result = data
        elif nan_policy == 'fill':
            result = np.where(np.isnan(data), fill_value, data)
        elif nan_policy == 'drop':
            result = data[~np.isnan(data)]
        elif nan_policy == 'interpolate':
            result = df_like_interpolate(data)  # 使用和Pandas一致的插值方法
        else:
            raise ValueError('nan_policy not supported')

        return result.tolist()

    if isinstance(data, np.ndarray):
        if nan_policy == 'propagate':
            result = data
        elif nan_policy == 'fill':
            result = np.where(np.isnan(data), fill_value, data)
        elif nan_policy == 'drop':
            if data.ndim > 1:
                raise ValueError('drop nan not supported for multi-dimensional arrays')
            result = data[~np.isnan(data)]
        elif nan_policy == 'interpolate':
            if data.ndim > 1:
                raise ValueError('interpolate nan not supported for multi-dimensional arrays')
            result = df_like_interpolate(data)  # 使用和Pandas一致的插值方法
        else:
            raise ValueError('nan_policy not supported')

        return result

    if isinstance(data, (pd.Series, pd.DataFrame)):
        if nan_policy == 'propagate':
            result = data
        elif nan_policy == 'fill':
            result = data.fillna(fill_value)
        elif nan_policy == 'drop':
            if isinstance(data, pd.DataFrame):
                if len(data.columns) > 1:
                    raise ValueError('drop nan not supported for multi-column DataFrames')
            result = data.dropna()
        elif nan_policy == 'interpolate':
            if isinstance(data, pd.DataFrame):
                if len(data.columns) > 1:
                    raise ValueError('interpolate nan not supported for multi-column DataFrames')
            result = data.interpolate()  # 线性插值填充NaN
        else:
            raise ValueError('nan_policy not supported')

        return result


def process_special_value(data, nan_policy=NAN_POLICY, fill_value=0, inf_policy=INF_POLICY):
    '''
    处理数据中的NaN和inf值
    '''
    data = process_inf(data, inf_policy=inf_policy)  # Process inf values first
    data = process_nan(data, nan_policy=nan_policy,
                       fill_value=fill_value)  # Process NaN values
    return data


def sync_special_value(*args, inf_policy=INF_POLICY):
    '''
    同步多个输入数据中的NaN和inf值
    '''
    # Verify that all inputs have the same shape
    shapes = [a.shape if isinstance(
        a, (np.ndarray, pd.Series, pd.DataFrame)) else len(a) for a in args]
    if len(set(shapes)) > 1:
        raise ValueError("All inputs must have the same shape")

    args = [process_inf(a, inf_policy=inf_policy)
            for a in args]  # Process inf values first

    # Convert all inputs to numpy arrays for processing
    args_np = [np.array(a) if not isinstance(
        a, np.ndarray) else a for a in args]

    # Find indices where any input has NaN
    nan_mask = np.any(np.isnan(args_np), axis=0)

    # Apply NaN where any NaNs are found across corresponding elements
    args_np_synced = [np.where(nan_mask, np.nan, arg) for arg in args_np]

    # Convert back to original data types
    args_synced = []
    for original_arg, synced_arg in zip(args, args_np_synced):
        if isinstance(original_arg, list):
            args_synced.append(synced_arg.tolist())
        elif isinstance(original_arg, pd.Series):
            args_synced.append(pd.Series(synced_arg, index=original_arg.index))
        elif isinstance(original_arg, pd.DataFrame):
            args_synced.append(pd.DataFrame(
                synced_arg, index=original_arg.index, columns=original_arg.columns))
        elif isinstance(original_arg, np.ndarray):
            args_synced.append(synced_arg)

    return args_synced if len(args_synced) > 1 else args_synced[0]


def sync_special_value_along_axis(data, sync_axis=0, inf_policy=INF_POLICY):
    '''
    同步多维数据中的NaN和inf值沿着指定的轴
    '''
    # Input verification
    if not isinstance(data, (np.ndarray, pd.DataFrame, list)):
        raise ValueError("Data must be a numpy array, a list or a pandas DataFrame")

    if isinstance(data, pd.DataFrame):
        if sync_axis == 'row':
            sync_axis = 1
        elif sync_axis == 'column' or sync_axis == 'col':
            sync_axis = 0

    if isinstance(data, np.ndarray):
        local_data = data.copy()
    elif isinstance(data, list):
        local_data = np.array(data)
    elif isinstance(data, pd.DataFrame):
        local_data = data.values

    if not 0 <= sync_axis < local_data.ndim:
        raise ValueError("Invalid sync_axis for the data shape")
    
    # Apply inf_policy
    data_processed = process_inf(local_data, inf_policy=inf_policy)

    # Sync nan and possibly inf values along the specified axis
    if local_data.ndim == 1:
        # For 1D data, simply apply the syncing across the whole array
        nan_mask = np.isnan(data_processed)
    else:
        # For multi-dimensional data, iterate over slices along the sync_axis
        nan_mask = np.any(np.isnan(data_processed), axis=sync_axis, keepdims=True)
    
    # Apply the mask
    data_synced = np.where(nan_mask, np.nan, data_processed)

    # Convert back to original data type if needed
    if isinstance(data, pd.DataFrame):
        data_synced = pd.DataFrame(data_synced, index=data.index, columns=data.columns)
    elif isinstance(data, list):
        data_synced = data_synced.tolist()

    return data_synced


@direct_use
def npnan_in_list(lst):
    return np.isnan(lst).any()
# endregion


# region 数据处理相关函数(缩放、标准化、裁剪、按比例分配、划分、分bin、卷积、平滑化)
def scale_range(min_val, max_val, prop):
    '''
    根据最小值和最大值计算扩展后的范围。

    :param min_val: 一个数值，代表最小值
    :param max_val: 一个数值，代表最大值
    :param prop: 一个数值，代表新的范围相对于原范围的比例
    :return: 一个包含两个元素的元组，代表扩展后的范围
    '''
    return min_val - (prop - 1) / 2 * (max_val - min_val), max_val + (prop - 1) / 2 * (max_val - min_val)


def scale_to_new_range(data, old_min, old_max, new_min, new_max):
    '''
    将数据缩放到新的范围。
    
    参数:
    - data: 输入数据
    - old_min: 原始数据的最小值。
    - old_max: 原始数据的最大值。
    - new_min: 新的最小值。
    - new_max: 新的最大值。
    '''
    return (data - old_min) / (old_max - old_min) * (new_max - new_min) + new_min


def get_z_score(data):
    '''
    计算并返回数据集的Z分数。
    '''
    if isinstance(data, list):
        return get_z_score_list(data)
    elif isinstance(data, np.ndarray):
        return get_z_score_arr(data)
    elif isinstance(data, dict):
        return get_z_score_dict(data)
    elif isinstance(data, pd.DataFrame):
        return get_z_score_df(data)
    elif isinstance(data, pd.Series):
        return get_z_score_series(data)
    else:
        raise TypeError("Unsupported data type.")


def get_z_score_arr(data):
    '''
    计算并返回np.array中所有数据的Z分数。
    '''
    mean = np.nanmean(data)
    std = np.nanstd(data)
    return (data - mean) / std


def get_z_score_list(data):
    '''
    计算并返回列表中所有数据的Z分数。
    '''
    data = np.array(data)
    mean = np.nanmean(data)
    std = np.nanstd(data)
    return list((data - mean) / std)


def get_z_score_dict(data):
    '''
    计算并返回字典中所有数据集的Z分数。
    '''
    return create_dict(data.keys(), get_z_score_list(list(data.values())))


def get_z_score_df(data):
    '''
    计算并返回DataFrame中所有数据的Z分数。
    '''
    return pd.DataFrame(get_z_score_arr(data.values), columns=data.columns, index=data.index)


def get_z_score_series(data):
    '''
    计算并返回Series中所有数据的Z分数。
    '''
    return pd.Series(get_z_score_arr(data.values), index=data.index)


def get_z_score_on_column(data, column_name):
    '''
    使用Z分数对数据集的df的指定列进行缩放。
    '''
    new_data = data.copy()
    new_data[column_name] = get_z_score_arr(data[column_name].values)
    return new_data


def get_z_score_on_column(df, column_name):
    '''
    使用Z分数对数据集的指定列进行缩放。
    '''
    new_df = df.copy()
    new_df[column_name] = get_z_score_arr(df[column_name].values)
    return new_df


def get_min_max_scaling(data, min_val=0, max_val=1):
    '''
    使用最小-最大缩放对数据集进行缩放。
    
    参数:
    - data: 输入数据，可以是列表,字典,np.array,Pandas DataFrame或Series。
    - min_val: 缩放后的最小值。
    - max_val: 缩放后的最大值。

    注意:
    - 这个函数是普适的，可以处理多种数据类型。但是如果已知数据类型，最好使用专门的函数, 可以提高效率。
    '''
    if isinstance(data, list):
        return get_min_max_scaling_list(data, min_val, max_val)
    elif isinstance(data, np.ndarray):
        return get_min_max_scaling_arr(data, min_val, max_val)
    elif isinstance(data, dict):
        return get_min_max_scaling_dict(data, min_val, max_val)
    elif isinstance(data, pd.DataFrame):
        return get_min_max_scaling_df(data, min_val, max_val)
    elif isinstance(data, pd.Series):
        return get_min_max_scaling_series(data, min_val, max_val)
    else:
        raise TypeError("Unsupported data type.")


def get_min_max_scaling_arr(data, min_val=0, max_val=1):
    '''
    将np.array中的所有数据进行最小-最大缩放。
    '''
    min_data = np.nanmin(data)
    max_data = np.nanmax(data)
    return (data - min_data) / (max_data - min_data) * (max_val - min_val) + min_val


def get_min_max_scaling_list(data, min_val=0, max_val=1):
    '''
    将list中的所有数据进行最小-最大缩放。
    '''
    data = np.array(data)
    min_data = np.nanmin(data)
    max_data = np.nanmax(data)
    return list((data - min_data) / (max_data - min_data) * (max_val - min_val) + min_val)


def get_min_max_scaling_df(data, min_val=0, max_val=1):
    '''
    将df中的所有数据进行最小-最大缩放。
    '''
    return pd.DataFrame(get_min_max_scaling_arr(data.values, min_val, max_val), columns=data.columns, index=data.index)


def get_min_max_scaling_series(data, min_val=0, max_val=1):
    return pd.Series(get_min_max_scaling_arr(data.values, min_val, max_val), index=data.index)


def get_min_max_scaling_dict(data, min_val=0, max_val=1):
    '''
    将字典中的所有数据进行最小-最大缩放。
    '''
    return create_dict(data.keys(), get_min_max_scaling_list(list(data.values()), min_val, max_val))


def get_min_max_scaling_on_column(data, column_name, min_val=0, max_val=1):
    '''
    使用最小-最大缩放对数据集的df的指定列进行缩放。
    '''
    new_data = data.copy()
    new_data[column_name] = get_min_max_scaling_arr(data[column_name].values, min_val, max_val)
    return new_data


def normalize(data, vmin, vmax):
    '''
    将输入数据标准化到[0, 1]范围内。

    参数:
    - data: 输入数据，可以是列表,字典,np.array,Pandas DataFrame或Series。
    - vmin: 标准化参考的最小值。等于vmin的数据会被标准化为0。
    - vmax: 标准化参考的最大值。等于vmax的数据会被标准化为1。

    返回:
    - 修改后的数据，类型与输入数据相同。

    注意:
    - 区别于get_min_max_scaling的标准化(在函数内部会考虑数据的最大值和最小值,所以对单点不适用)，这个函数是直接按照vmin, vmax计算出的范围进行标准化。
    '''
    if isinstance(data, (int, float)):
        return normalize_simple(data, vmin, vmax)
    elif isinstance(data, list):
        return normalize_list(data, vmin, vmax)
    elif isinstance(data, np.ndarray):
        return normalize_arr(data, vmin, vmax)
    elif isinstance(data, dict):
        return normalize_dict(data, vmin, vmax)
    elif isinstance(data, pd.DataFrame):
        return normalize_df(data, vmin, vmax)
    elif isinstance(data, pd.Series):
        return normalize_series(data, vmin, vmax)
    else:
        raise TypeError("Unsupported data type.")


def normalize_simple(data, vmin, vmax):
    '''
    将输入数据标准化到[0, 1]范围内。
    '''
    return (data - vmin) / (vmax - vmin)


def normalize_arr(data, vmin, vmax):
    '''
    将np.array中的所有数据标准化到[0, 1]范围内。
    '''
    return (data - vmin) / (vmax - vmin)


def normalize_list(data, vmin, vmax):
    '''
    将list中的所有数据标准化到[0, 1]范围内。
    '''
    return [(val - vmin) / (vmax - vmin) for val in data]


def normalize_dict(data, vmin, vmax):
    '''
    将字典中的所有数据标准化到[0, 1]范围内。
    '''
    return {k: (v - vmin) / (vmax - vmin) for k, v in data.items()}


def normalize_df(data, vmin, vmax):
    '''
    将Pandas DataFrame中的所有数据标准化到[0, 1]范围内。
    '''
    return (data - vmin) / (vmax - vmin)


def normalize_series(data, vmin, vmax):
    '''
    将Pandas Series中的所有数据标准化到[0, 1]范围内。
    '''
    return (data - vmin) / (vmax - vmin)


def normalize_on_column(data, column_name, vmin, vmax):
    '''
    使用normalize对数据集data的指定列进行标准化。
    '''
    new_data = data.copy()
    new_data[column_name] = normalize(data[column_name].values, vmin, vmax)
    return new_data


def clip(data, vmin, vmax):
    '''
    将输入数据中的值限制在vmin和vmax之间。支持列表、字典、Pandas DataFrame和Series。

    参数:
    - data: 输入数据，可以是列表,字典,np.array,Pandas DataFrame或Series。
    - vmin: 最小值阈值，数据中小于此值的将被设置为此值。
    - vmax: 最大值阈值，数据中大于此值的将被设置为此值。

    返回:
    - 修改后的数据，类型与输入数据相同。
    '''
    if isinstance(data, (int, float)):
        return clip_simple(data, vmin, vmax)
    elif isinstance(data, np.ndarray):
        return clip_arr(data, vmin, vmax)
    elif isinstance(data, list):
        return clip_list(data, vmin, vmax)
    elif isinstance(data, dict):
        return clip_dict(data, vmin, vmax)
    elif isinstance(data, pd.DataFrame):
        return clip_df(data, vmin, vmax)
    elif isinstance(data, pd.Series):
        return clip_series(data, vmin, vmax)
    else:
        raise TypeError("Unsupported data type.")


def clip_simple(data, vmin, vmax):
    '''
    将输入数据中的值限制在vmin和vmax之间
    '''
    return np.clip(data, vmin, vmax)


def clip_arr(data, vmin, vmax):
    '''
    将np.array中的值限制在vmin和vmax之间
    '''
    return np.clip(data, vmin, vmax)


def clip_list(data, vmin, vmax):
    '''
    将list中的值限制在vmin和vmax之间
    '''
    return np.clip(np.array(data), vmin, vmax).tolist()


def clip_dict(data, vmin, vmax):
    '''
    将字典中的值限制在vmin和vmax之间
    '''
    return {k: np.clip(v, vmin, vmax) for k, v in data.items()}


def clip_df(data, vmin, vmax):
    '''
    将Pandas DataFrame中的值限制在vmin和vmax之间
    '''
    return data.clip(lower=vmin, upper=vmax)


def clip_series(data, vmin, vmax):
    '''
    将Pandas Series中的值限制在vmin和vmax之间
    '''
    return data.clip(lower=vmin, upper=vmax)


def clip_on_column(data, column_name, vmin, vmax):
    '''
    使用clip对数据集data的指定列进行裁剪。
    '''
    new_data = data.copy()
    new_data[column_name] = clip(data[column_name].values, vmin, vmax)
    return new_data


def clip_normalize(data, vmin, vmax):
    '''
    将输入数据中的值限制在vmin和vmax之间，并将其标准化到[0, 1]范围内。

    参数:
    - data: 输入数据
    - vmin: 最小值阈值，数据中小于此值的将被设置为此值。
    - vmax: 最大值阈值，数据中大于此值的将被设置为此值。

    返回:
    - 修改后的数据，类型与输入数据相同。

    注意:
    - 区别于get_min_max_scaling的标准化(在函数内部会考虑数据的最大值和最小值,所以对单点不适用)，这个函数是直接按照vmin, vmax进行标准化。
    '''
    clipped_data = clip(data, vmin, vmax)
    normalized_data = normalize(clipped_data, vmin, vmax)
    return normalized_data


def cluster_matrix(data, metric='cosine', method='average', square=True):
    '''
    将数据矩阵进行层次聚类，并返回根据聚类结果重新排列的矩阵。

    参数:
    - data: 一个numpy数组或Pandas DataFrame，包含要聚类的数据。
    - metric: 用于计算距离的度量标准。
    - method: 用于聚类的方法。
    - square: 是否将聚类后的矩阵排序使得前面的行和列名相同。
    '''
    if isinstance(data, pd.DataFrame):
        data_df = data.copy()
    elif isinstance(data, np.ndarray):
        data_df = pd.DataFrame(data)

    # Compute the distance matrices for rows and columns
    row_distances = pdist(data_df, metric=metric)
    col_distances = pdist(data_df.T, metric=metric)

    # Apply hierarchical clustering
    row_clusters = linkage(row_distances, method=method)
    col_clusters = linkage(col_distances, method=method)

    # Get the order of rows and columns as per clustering
    ordered_row_indices = leaves_list(row_clusters)
    ordered_col_indices = leaves_list(col_clusters)

    ordered_row_names = data_df.index[ordered_row_indices].tolist()  # 获取聚类后的行名顺序
    ordered_col_names = data_df.columns[ordered_col_indices].tolist()  # 获取聚类后的列名顺序

    if square:
        if len(ordered_row_names) > len(ordered_col_names):
            new_ordered_row_names = ordered_col_names.copy()
            for row in ordered_row_names:
                if row not in ordered_col_names:
                    new_ordered_row_names.append(row)
            clustered_matrix = data_df.loc[new_ordered_row_names, ordered_col_names]
        else:
            new_ordered_col_names = ordered_row_names.copy()
            for col in ordered_col_names:
                if col not in ordered_row_names:
                    new_ordered_col_names.append(col)
            clustered_matrix = data_df.loc[ordered_row_names, new_ordered_col_names]
    else:
        clustered_matrix = data_df.loc[ordered_row_names, ordered_col_names]
    return clustered_matrix


def split_num_proportionally(total_num, proportion_dict, mode):
    '''
    根据给定的比例分配总数。

    参数:
    total_num (int): 待分配的总数。
    proportion_dict (dict): 一个字典，值是比例。
    mode (str): 分配模式。支持的模式包括：
                - 'optimal': 尽可能均匀地分配数字，同时尊重比例。
                - 'positive_prop_positive': 为每个有正比例的区域保证最小值为1，如果可能的话。
    
    返回:
    num_dict (dict): 一个字典，值是分配的数字。
    '''
    if not np.allclose(sum(proportion_dict.values()), 1):
        raise ValueError("比例的总和必须为1")

    # 计算理想中每个key应得的数量
    ideal_counts = {key: total_num * proportion for key, proportion in proportion_dict.items()}
    num_dict = {key: 0 for key in proportion_dict.keys()}
    remaining_num = total_num

    if mode == 'optimal':
        # 首先根据比例进行初步分配
        for key, ideal_count in sorted(ideal_counts.items(), key=lambda item: item[1]):
            count = min(int(ideal_count), remaining_num)
            num_dict[key] = count
            remaining_num -= count

        # 分配剩余的数字，优先考虑num_dict和ideal_counts的差值最大的区域(这里由于上面是int,所以ideal_counts[key]一定大于num_dict[key])
        while remaining_num > 0:
            for key, ideal_count in sorted(ideal_counts.items(), key=lambda item: item[1] - num_dict[item[0]], reverse=True):
                if remaining_num > 0:
                    num_dict[key] += 1
                    remaining_num -= 1
                else:
                    break

    elif mode == 'positive_prop_positive':
        # 确保所有有正比例的区域至少分配到1，前提是总数允许
        for key, ideal_count in sorted(ideal_counts.items(), key=lambda item: item[1]):
            if ideal_count > 0:
                count = max(1, round(ideal_count))
            else:
                count = 0
            num_dict[key] = count
            remaining_num -= count

        # 如果分配完后剩余的数字小于0，说明分配超出了总数，需要调整，优先考虑num_dict和ideal_counts的差值最大的区域(因为这边要减去数,要考虑超出的越多越优先)
        while remaining_num < 0:
            for key, ideal_count in sorted(ideal_counts.items(), key=lambda item: num_dict[item[0]] - item[1], reverse=True):
                if remaining_num < 0:
                    # 为了保证不会把proption是正的key分配的值变为0,所以只有当num_dict[key]大于1时才减1
                    if num_dict[key] > 1:
                        num_dict[key] -= 1
                        remaining_num += 1
                else:
                    break

        # 如果分配完后剩余的数字大于0，分配剩余的数字，优先考虑ideal_counts和num_dict的差值最大的区域(因为这边要加上数,要考虑差值越大越优先)
        while remaining_num > 0:
            for key, ideal_count in sorted(ideal_counts.items(), key=lambda item: item[1] - num_dict[item[0]], reverse=True):
                if remaining_num > 0:
                    num_dict[key] += 1
                    remaining_num -= 1
                else:
                    break

    return num_dict


def get_bin_idx(data, bins, right=False, left_most=True, right_most=True):
    '''
    获取数据在指定区间的索引。

    参数:
    - data: 输入数据，可以是单个数值、列表、numpy数组。
    - bin: 区间列表，例如[0, 10, 20, 30]。
    - right: 是否包含右边界。(只管内部的区间，不管最左和最右的区间)
    - left_most: 是否包含最左边界, 当数据为最左边界时，返回0。
    - right_most: 是否包含最右边界, 当数据为最右边界时，返回len(bins) - 2。

    返回:
    - idx: 数据在区间的索引，如果数据不在区间内则返回np.nan。

    注意:
    - 此函数不同于np.digitize，它返回的一定是从最开始的区间开始的索引, 而不是从无穷开始的索引。
    '''
    if isinstance(data, (np.ndarray, list)):
        # 处理多个输入
        idx = np.digitize(data, bins, right=right) - 1
        # 初始化掩码数组为False
        mask = np.zeros_like(data, dtype=bool)

        if left_most and right_most:
            mask = (data < bins[0]) | (data > bins[-1])
        elif left_most and not right_most:
            mask = (data < bins[0]) | (data >= bins[-1])
        elif not left_most and right_most:
            mask = (data <= bins[0]) | (data > bins[-1])
        elif not left_most and not right_most:
            mask = (data <= bins[0]) | (data >= bins[-1])
        
        # 将掩码位置的索引设置为np.nan
        idx = np.where(mask, np.nan, idx)

        # 处理最左和最右的区间
        if left_most:
            idx[data == bins[0]] = 0
        if right_most:
            idx[data == bins[-1]] = len(bins) - 2
    else:
        # 处理单个输入
        idx = np.digitize(data, bins, right=right) - 1
        if left_most and right_most:
            if data < bins[0] or data > bins[-1]:
                return np.nan
            if data == bins[0]:
                idx = 0
            elif data == bins[-1]:
                idx = len(bins) - 2
        elif left_most and not right_most:
            if data < bins[0] or data >= bins[-1]:
                return np.nan
            if data == bins[0]:
                idx = 0
        elif not left_most and right_most:
            if data <= bins[0] or data > bins[-1]:
                return np.nan
            if data == bins[-1]:
                idx = len(bins) - 2
        elif not left_most and not right_most:
            if data <= bins[0] or data >= bins[-1]:
                return np.nan
    return idx


def bin_timeseries(timeseries, bin_size, mode='mean'):
    """
    将时间序列转换为Bin后的时间序列,每个Bin包含固定数量的元素
    
    参数:
    timeseries (list or np.array): 时间序列数据
    bin_size (int): 每个Bin包含的元素个数
    mode (str): Bin后的时间序列的计算模式,可以是'mean','sum'.默认为'mean'
    
    返回:
    np.array: Bin后的时间序列,每个值为每个Bin的平均值

    注意:
    最后一点可能会被丢弃,以使时间序列的长度可以被bin_size整除
    """
    # 确保timeseries是一个numpy数组
    timeseries = np.array(timeseries)
    
    # 计算总的Bin数
    num_bins = len(timeseries) // bin_size
    
    # 截取时间序列,使其长度可以被bin_size整除
    trimmed_timeseries = timeseries[:num_bins * bin_size]
    
    # 将时间序列重塑为(num_bins, bin_size)的二维数组
    reshaped_timeseries = trimmed_timeseries.reshape((num_bins, bin_size))
    
    # 计算每个Bin的值
    if mode == 'mean':
        binned_timeseries = np.mean(reshaped_timeseries, axis=1)
    elif mode == 'sum':
        binned_timeseries = np.sum(reshaped_timeseries, axis=1)
    else:
        raise ValueError("Invalid mode. Expected 'mean' or 'sum'.")
    
    return binned_timeseries


def bin_multi_timeseries(multi_timeseries, bin_size, mode='mean'):
    '''
    multi_timeseries: (N, T)的二维数组,N为时间序列的数量,T为时间序列的长度
    bin_size: 每个Bin的大小(时间步数)
    mode: 'mean' 或 'sum',表示对每个Bin计算均值或求和
    '''
    # 输入检查
    if not isinstance(multi_timeseries, np.ndarray):
        multi_timeseries = np.array(multi_timeseries)
    
    if multi_timeseries.ndim != 2:
        raise ValueError("multi_timeseries should be a 2D array.")
    
    # 计算总的Bin数
    num_bins = multi_timeseries.shape[1] // bin_size
    
    # 截取时间序列，使其长度可以被bin_size整除
    trimmed_multi_timeseries = multi_timeseries[:, :num_bins * bin_size]
    
    # 将时间序列重塑为 (N, num_bins, bin_size)
    reshaped_multi_timeseries = trimmed_multi_timeseries.reshape((multi_timeseries.shape[0], num_bins, bin_size))
    
    # 计算每个Bin的值
    if mode == 'mean':
        binned_timeseries = np.mean(reshaped_multi_timeseries, axis=2)
    elif mode == 'sum':
        binned_timeseries = np.sum(reshaped_multi_timeseries, axis=2)
    else:
        raise ValueError("Invalid mode. Expected 'mean' or 'sum'.")
    
    return binned_timeseries


def convolve_timeseries(timeseries, kernel, mode='full'):
    """
    对时间序列进行卷积操作,并返回结果

    参数:
    timeseries (list or np.array): 时间序列数据
    kernel (list or np.array): 用于卷积的核
    mode (str): 卷积模式,可以是'full','valid'或'same'.默认为'full'.
                - 'full': 返回完整的卷积结果
                - 'valid': 仅返回完全重叠部分的卷积结果
                - 'same': 返回与输入时间序列长度相同的卷积结果

    返回:
    np.array: 卷积后的时间序列
    """
    # 确保timeseries和kernel是numpy数组
    timeseries = np.array(timeseries)
    kernel = np.array(kernel)

    # 进行卷积操作
    convolved_timeseries = scipy.signal.convolve(timeseries, kernel, mode=mode)

    return convolved_timeseries


def convolve_multi_timeseries(multi_timeseries, kernel, mode='full'):
    """
    对多个时间序列进行卷积操作,并返回结果

    参数:
    multi_timeseries (np.array): 一个形状为 (N, T) 的二维数组

    返回:
    np.array: 卷积后的时间序列,形状为 (N, T'),其中 T' 取决于 mode 选项和 kernel 的长度
    """
    # 确保 multi_timeseries 是一个 numpy 数组
    multi_timeseries = np.array(multi_timeseries)
    kernel = np.array(kernel)

    # 获取时间序列的数量 N 和序列长度 T
    N, T = multi_timeseries.shape

    # 计算输出序列的长度
    if mode == 'full':
        output_length = T + len(kernel) - 1
    elif mode == 'same':
        output_length = T
    elif mode == 'valid':
        output_length = T - len(kernel) + 1
    else:
        raise ValueError("Invalid mode. Expected 'full', 'same', or 'valid'.")

    # 初始化卷积结果数组
    convolved_timeseries = np.zeros((N, output_length))

    # 对每个时间序列进行卷积
    for i in range(N):
        convolved_timeseries[i] = scipy.signal.convolve(multi_timeseries[i], kernel, mode=mode)

    return convolved_timeseries


def get_normalized_rectangular_kernal(width, dt):
    width1 = int(width / 2 / dt) * 2 + 1
    kernel = np.ones(width1) / width1
    return kernel


def convolve_timeseries_with_rectangular_kernal(timeseries, width, dt, mode='full'):
    return convolve_timeseries(timeseries, get_normalized_rectangular_kernal(width, dt), mode=mode)


def convolve_multi_timeseries_with_rectangular_kernal(multi_timeseries, width, dt, mode='full'):
    return convolve_multi_timeseries(multi_timeseries, get_normalized_rectangular_kernal(width, dt), mode=mode)


def lowess_smooth(x, y, frac=0.2):
    """
    对给定的 x 和 y 数据使用 LOWESS 进行平滑处理。函数内部会确保 x 是有序的。
    
    参数:
    x (array-like): 自变量（输入）的数据。
    y (array-like): 因变量（输出）的数据。
    frac (float): 控制局部回归时使用的窗口大小，值越大，曲线越平滑。默认值为 0.2。
    
    返回:
    x_smooth (numpy array): 平滑后的 x 数据。
    y_smooth (numpy array): 平滑后的 y 数据。
    """
    # 确保是array
    x = np.array(x)
    y = np.array(y)
    
    # 确保 x 是有序的，首先对 x 和 y 进行排序
    sort_index = np.argsort(x)
    x_sorted = x[sort_index]
    y_sorted = y[sort_index]
    
    # 使用 LOWESS 进行平滑
    lowess_result = lowess(y_sorted, x_sorted, frac=frac)
    
    # 从结果中提取平滑后的 x 和 y 值
    x_smooth = lowess_result[:, 0]
    y_smooth = lowess_result[:, 1]
    
    return x_smooth, y_smooth


def get_envelope(signal):
    dc_offset = np.mean(signal)
    zero_mean_signal = signal - dc_offset
    analytic_signal = scipy.signal.hilbert(zero_mean_signal)
    envelope = np.abs(analytic_signal)
    envelope = envelope + dc_offset
    return envelope
# endregion


# region 数据降维相关函数
def get_pca(data, n_components=2):
    '''
    执行主成分分析 (PCA) 并返回降维后的数据
    自动处理NaN值,通过删除含有NaN的行

    :param data: DataFrame或二维数组,原始数据
    :param n_components: int, 保留的主成分数量
    :return: PCA转换后的DataFrame
    '''
    data_clean = data.dropna()
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(data_clean)
    principal_df = pd.DataFrame(data=principal_components,
                                columns=[f'Principal Component {i+1}' for i in range(n_components)])
    return principal_df


def get_nmf(X, n_components, init='nndsvda', random_state=0, max_iter=200, normalize_components=True, **kwargs):
    '''
    Perform dimensionality reduction using NMF. (对行向量进行的)

    Parameters:
    X - array, the original data.
    n_components - int, the number of components to keep.
    init - {'random', 'nndsvd', 'nndsvda', 'nndsvdar', 'custom'}, default 'random'.
    random_state - int, default 0.
    max_iter - int, default 200.
    normalize_components - bool, default True.

    Returns:
    W - array, the reduced dimensionality data.
    H - array, the components.
    '''
    model = NMF(n_components=n_components, init=init,
                random_state=random_state, max_iter=max_iter, **kwargs)
    W = model.fit_transform(X)
    H = model.components_

    if normalize_components:
        # Normalize each column in H to have unit L2 norm
        norms = np.linalg.norm(H, axis=1)
        H = H / norms[:, np.newaxis]

    return W, H


def get_tsne(data, n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=0):
    '''
    '''
    pass


def get_diffusion_map(data, n_components=2, n_neighbors=10, alpha=0.5, affinity='gaussian', random_state=0):
    '''
    '''
    pass
# endregion


# region 数据分析相关函数
def get_linear_score(points):
    '''
    计算一系列点的共线性得分。
    
    参数:
    points: 点坐标,shape=(n, d),n为点的数量,d为点的维度。
    
    返回:
    float: 共线性得分。值越接近1表示越共线,值越接近0表示越不共线。
    '''
    if len(points) < 3:
        return 0.0
    
    # 计算向量并进行单位化
    vectors = [(points[i+1] - points[i]) / np.linalg.norm(points[i+1] - points[i]) for i in range(len(points)-1)]
    
    # 计算点积
    return np.mean([np.abs(np.dot(vectors[i], vectors[i+1])) for i in range(len(vectors)-1)])


def get_distance(points):
    '''
    计算多个点的距离矩阵
    :param points: 二维数组,每行代表一个点的坐标
    :return: 距离矩阵
    '''
    return np.sqrt(np.sum((points[:, np.newaxis] - points[np.newaxis, :])**2, axis=-1))


def get_mutual_distance(x, y):
    '''
    计算两个数组的距离矩阵
    :param x: 第一个数组
    :param y: 第二个数组
    :return: 距离矩阵
    '''
    return np.sqrt(np.sum((np.array(x)[:, np.newaxis, :] - np.array(y)[np.newaxis, :, :]) ** 2, axis=-1))


def get_angle(source, target, angle_type='rad', angle_range='0:360'):
    '''
    计算source到target的向量的角度

    参数:
    source - 起始点(2D),支持同时输入多个点，形状为(N, 2)或者(2,)
    target - 终点(2D),支持同时输入多个点，形状为(N, 2)或者(2,)
    angle_type - 角度类型,支持'rad'和'deg'
    angle_range - 角度范围,支持'0:360'和'-180:180'

    返回:
    angle - 角度，如果输入是多个点，返回形状为(N,),如果输入是单个点,返回标量
    '''
    source = np.array(source)
    target = np.array(target)
    if len(source.shape) == 1:
        source = source.reshape(1, -1)
    if len(target.shape) == 1:
        target = target.reshape(1, -1)
    
    if source.shape[0] != target.shape[0] and source.shape[0] != 1 and target.shape[0] != 1:
        raise ValueError("The number of source and target points must be the same.")
    
    angle = np.arctan2(target[:, 1] - source[:, 1], target[:, 0] - source[:, 0])
    if angle_type == 'deg':
        angle = np.degrees(angle)
    if angle_range == '0:360':
        if angle_type == 'deg':
            angle = np.where(angle < 0, angle + 360, angle)
        if angle_type == 'rad':
            angle = np.where(angle < 0, angle + 2 * np.pi, angle)
        
    if len(angle) == 1:
        angle = angle[0]
    return angle


def get_angle_3d(source, target, angle_type='rad', azim_range='0:360', elev_range='-90:90'):
    '''
    计算source到target的向量的水平角(azimuth)和垂直角(elevation)

    参数:
    source - 起始点(3D),支持同时输入多个点,形状为(N, 3)或者(3,)
    target - 终点(3D),支持同时输入多个点,形状为(N, 3)或者(3,)
    angle_type - 角度类型,支持'rad'和'deg'
    azim_range - 水平角范围,支持'0:360'和'-180:180'
    elev_range - 垂直角范围,支持'0:180'和'-90:90'

    返回:
    azim - 水平角度,如果输入是多个点,返回形状为(N,),如果输入是单个点,返回标量
    elev - 垂直角度,如果输入是多个点,返回形状为(N,),如果输入是单个点,返回标量(数学中的垂直角是从z轴开始算起的,范围是0:180,如果从x-y平面开始算起,范围是-90:90)
    '''
    source = np.array(source)
    target = np.array(target)
    if len(source.shape) == 1:
        source = source.reshape(1, -1)
    if len(target.shape) == 1:
        target = target.reshape(1, -1)

    if source.shape[0] != target.shape[0] and source.shape[0] != 1 and target.shape[0] != 1:
        raise ValueError("The number of source and target points must be the same.")

    # 计算水平角(azimuth)和垂直角(elevation)
    dx = target[:, 0] - source[:, 0]
    dy = target[:, 1] - source[:, 1]
    dz = target[:, 2] - source[:, 2]
    azim = np.arctan2(dy, dx)
    elev = np.arctan2(dz, np.sqrt(dx**2 + dy**2))

    if angle_type == 'deg':
        azim = np.degrees(azim)
        elev = np.degrees(elev)

    if azim_range == '0:360':
        if angle_type == 'deg':
            azim = np.where(azim < 0, azim + 360, azim)
        if angle_type == 'rad':
            azim = np.where(azim < 0, azim + 2 * np.pi, azim)
    if elev_range == '0:180':
        if angle_type == 'deg':
            azim = np.where(azim < 0, azim + 180, azim)
        if angle_type == 'rad':
            azim = np.where(azim < 0, azim + np.pi, azim)

    if len(azim) == 1:
        azim = azim[0]
        elev = elev[0]

    return azim, elev


def get_corr(x, y, nan_policy='drop', fill_value=0, inf_policy=INF_POLICY, sync=True):
    '''
    计算两个数组的相关系数
    sync表示是否将两个数组的nan和inf位置同步处理(假如一个数据有nan,另一个没有,则将两个数据的nan位置都设置为nan)
    '''
    x, y = sync_special_value(x, y, inf_policy=inf_policy) if sync else (x, y)
    return np.corrcoef(process_special_value(x, nan_policy=nan_policy, fill_value=fill_value, inf_policy=inf_policy), process_special_value(y, nan_policy=nan_policy, fill_value=fill_value, inf_policy=inf_policy))[0, 1]


def get_matrix_corr(matrix_x, matrix_y, triangle_type=None, diagonal=True, get_corr_kwargs=None):
    '''
    计算两个矩阵flatten后的相关系数
    triangle_type: 'upper' or 'lower',表示只计算上三角或者下三角的相关系数,默认为None,表示计算全部
    diagonal: True or False,表示是否计算对角线上的相关系数,默认为True,表示计入对角线
    get_corr_kwargs: dict, get_corr函数的参数
    '''
    if get_corr_kwargs is None:
        get_corr_kwargs = {}
    matrix_x_value_to_compute = get_arr_triangle_value(matrix_x, triangle_type=triangle_type, diagonal=diagonal)
    matrix_y_value_to_compute = get_arr_triangle_value(matrix_y, triangle_type=triangle_type, diagonal=diagonal)
    return get_corr(matrix_x_value_to_compute, matrix_y_value_to_compute, **get_corr_kwargs)


def get_rectangle_matrix_corr(matrix_x, matrix_y, diagonal=True, get_corr_kwargs=None):
    '''
    计算两个矩阵flatten后的相关系数,非方阵
    diagonal: True or False,表示是否计算对角线上的相关系数,默认为True,表示计入对角线
    get_corr_kwargs: dict, get_corr函数的参数
    '''
    if get_corr_kwargs is None:
        get_corr_kwargs = {}
    
    if diagonal:
        matrix_x_flat = matrix_x.flatten()
        matrix_y_flat = matrix_y.flatten()
    else:
        # 创建非对角线元素的掩码
        rows, cols = matrix_x.shape
        mask = np.eye(rows, cols, dtype=bool)
        non_diag_mask = ~mask
        
        # 应用掩码并展平
        matrix_x_flat = matrix_x[non_diag_mask].flatten()
        matrix_y_flat = matrix_y[non_diag_mask].flatten()
    
    # 调用相关系数计算函数
    return get_corr(matrix_x_flat, matrix_y_flat, **get_corr_kwargs)


def get_pd_corr(df_x, df_y, triangle_type=None, diagonal=True, get_corr_kwargs=None):
    '''
    计算两个DataFrame的相关系数,类似于get_matrix_corr
    '''
    if get_corr_kwargs is None:
        get_corr_kwargs = {}
    df_x_value_to_compute = get_arr_triangle_value(df_x, triangle_type=triangle_type, diagonal=diagonal)
    df_y_value_to_compute = get_arr_triangle_value(df_y, triangle_type=triangle_type, diagonal=diagonal)
    return get_corr(df_x_value_to_compute, df_y_value_to_compute, **get_corr_kwargs)


def get_pd_rectangle_corr(df_x, df_y, diagonal=True, get_corr_kwargs=None):
    '''
    计算两个DataFrame的相关系数,类似于get_rectangle_matrix_corr
    '''
    df_x_aligned = align_df_row_col_order(df_x)
    df_y_aligned = align_df_row_col_order(df_y)
    return get_rectangle_matrix_corr(df_x_aligned.values, df_y_aligned.values, diagonal=diagonal, get_corr_kwargs=get_corr_kwargs)


def get_CV(data, nan_policy='drop', fill_value=0, inf_policy=INF_POLICY):
    '''
    计算数据的变异系数(Coefficient of Variation, CV)
    '''
    data = process_special_value(data, nan_policy=nan_policy, fill_value=fill_value, inf_policy=inf_policy)
    return np.std(data) / np.mean(data)


def get_FF(data, nan_policy='drop', fill_value=0, inf_policy=INF_POLICY):
    '''
    计算数据的Fano因子(Fano Factor, FF)
    '''
    data = process_special_value(data, nan_policy=nan_policy, fill_value=fill_value, inf_policy=inf_policy)
    return np.var(data) / np.mean(data)


def get_linregress(x, y, nan_policy='drop', fill_value=0, inf_policy=INF_POLICY, sync=True):
    '''
    对给定的数据进行线性回归拟合

    参数:
    x - x数据
    y - y数据

    返回:
    slope - 斜率
    intercept - 截距
    r - 相关系数
    p - p值
    std_err - 标准误差
    residual - 残差(y - y_pred)
    '''
    x, y = sync_special_value(x, y, inf_policy=inf_policy) if sync else (x, y)
    x = np.array(process_special_value(x, nan_policy=nan_policy,
                 fill_value=fill_value, inf_policy=inf_policy))
    y = np.array(process_special_value(y, nan_policy=nan_policy,
                 fill_value=fill_value, inf_policy=inf_policy))

    regress_dict = {}

    # 计算线性回归
    regress_dict['slope'], regress_dict['intercept'], regress_dict['r'], regress_dict['p'], regress_dict['std_err'] = st.linregress(
        x, y)
    regress_dict['residual'] = y - \
        (x * regress_dict['slope'] + regress_dict['intercept'])

    return regress_dict


def get_curvefit(x, y, func, p0=None, bounds=(-np.inf, np.inf), maxfev=1000, nan_policy='drop', fill_value=0, inf_policy=INF_POLICY, sync=True):
    '''
    对给定的数据进行曲线拟合

    输出:
    popt - 最优参数
    pcov - 参数协方差
    error - 残差平方和
    '''
    x, y = sync_special_value(x, y, inf_policy=inf_policy) if sync else (x, y)
    x = np.array(process_special_value(x, nan_policy=nan_policy, fill_value=fill_value, inf_policy=inf_policy))
    y = np.array(process_special_value(y, nan_policy=nan_policy, fill_value=fill_value, inf_policy=inf_policy))

    popt, pcov = scipy.optimize.curve_fit(func, x, y, p0=p0, bounds=bounds, maxfev=maxfev)
    error = []
    for xi, yi in zip(x, y):
        error.append((yi - func(xi, *popt))**2)
    error = np.sum(error)
    return popt, pcov, error


def get_ks(x, y, nan_policy='drop', fill_value=0, inf_policy=INF_POLICY):
    '''
    计算两个数组的Kolmogorov-Smirnov距离

    参数:
    x - 第一个数组
    y - 第二个数组或者CDF函数

    返回:
    ks_distance - 两个数组的KS距离
    '''
    return get_ks_result(x, y, nan_policy=nan_policy, fill_value=fill_value, inf_policy=inf_policy)['ks']


def get_ks_result(x, y, nan_policy='drop', fill_value=0, inf_policy=INF_POLICY):
    '''
    计算两个数组的Kolmogorov-Smirnov距离及其P值,注意当分布一样时,KS距离为0,而P值为1

    参数:
    x - 第一个数组
    y - 第二个数组或者CDF函数

    返回:
    ks_distance - 两个数组的KS距离
    p_value - 对应的P值
    '''
    # 处理输入x的特殊值
    x_clean = process_special_value(x, nan_policy=nan_policy, fill_value=fill_value, inf_policy=inf_policy)
    
    if callable(y):
        # 单样本检验：y是CDF函数
        ks_result = st.kstest(x_clean, y)
    else:
        # 双样本检验：处理y的特殊值后计算
        y_clean = process_special_value(y, nan_policy=nan_policy, fill_value=fill_value, inf_policy=inf_policy)
        ks_result = st.ks_2samp(x_clean, y_clean)
    
    r = {'ks': ks_result.statistic, 'p': ks_result.pvalue}
    return r


def get_kl_divergence(p, q, base=None, **kwargs):
    '''
    计算两个概率分布的KL散度
    两个概率分布的长度必须相同

    参数:
    p - 第一个概率分布
    q - 第二个概率分布

    返回:
    kl_divergence - 两个概率分布的KL散度
    '''
    return st.entropy(p, q, base=base, **kwargs)


def get_entropy(p, base=None, **kwargs):
    '''
    计算概率分布的熵

    参数:
    p - 概率分布

    返回:
    entropy - 概率分布的熵
    '''
    return st.entropy(p, base=base, **kwargs)


def get_mutual_information(p, q, joint_pq):
    '''
    计算两个概率分布的互信息
    两个概率分布的长度必须相同

    参数:
    p - 第一个概率分布
    q - 第二个概率分布
    joint_pq - 联合概率分布

    返回:
    mutual_information - 两个概率分布的互信息
    '''
    return get_entropy(p) + get_entropy(q) - get_entropy(joint_pq)


def get_jsd(p, q, base=None, **kwargs):
    '''
    计算两个概率分布的Jensen-Shannon散度
    两个概率分布的长度必须相同

    参数:
    p - 第一个概率分布
    q - 第二个概率分布

    返回:
    jsd - 两个概率分布的Jensen-Shannon散度
    '''
    m = (np.array(p) + np.array(q)) / 2
    return (get_kl_divergence(p, m, base=base, **kwargs) + get_kl_divergence(q, m, base=base, **kwargs)) / 2


def get_fft(timeseries, T=None, sample_rate=None, nan_policy='interpolate', fill_value=0, inf_policy=INF_POLICY):
    '''
    执行快速傅里叶变换 (FFT) 并返回变换后的信号
    自动处理NaN值,通过线性插值填充NaN

    注意: 这里的T需要以秒为单位,得到的结果才是以Hz为单位的
    '''
    if T is None and sample_rate is None:
        raise ValueError("Either T or sample_rate must be provided")
    if T is None:
        T = 1 / sample_rate
    clean_timeseries = process_special_value(
        timeseries, nan_policy=nan_policy, fill_value=fill_value, inf_policy=inf_policy)
    n = len(clean_timeseries)
    yf = scipy.fft.fft(clean_timeseries)
    xf = scipy.fft.fftfreq(n, T)

    xf = scipy.fft.fftshift(xf)
    yf = scipy.fft.fftshift(yf)
    return xf, yf


def get_ifft(s, T=None, sample_rate=None, nan_policy='interpolate', fill_value=0, inf_policy=INF_POLICY):
    ifft_s = scipy.fft.ifft(s)
    pass


def get_power_spectrum(timeseries, T=None, sample_rate=None, nan_policy='interpolate', fill_value=0, inf_policy=INF_POLICY):
    '''
    利用welch方法计算功率谱密度并返回结果

    注意: 这里的T需要以秒为单位,得到的结果才是以Hz为单位的;如果以ms为单位,注意统一为秒
    '''
    if T is None and sample_rate is None:
        raise ValueError("Either T or sample_rate must be provided")
    if T is None:
        T = 1 / sample_rate
    clean_timeseries = process_special_value(
        timeseries, nan_policy=nan_policy, fill_value=fill_value, inf_policy=inf_policy)
    f, Pxx = scipy.signal.welch(clean_timeseries, fs=1/T)
    return f, Pxx


def get_multi_power_spectrum(multi_timeseries, T=None, sample_rate=None, nan_policy='interpolate', fill_value=0, inf_policy=INF_POLICY, process_num=1):
    r = multi_process_list_for(process_num=process_num, func=get_power_spectrum, for_list=multi_timeseries, kwargs={
        'T': T, 'sample_rate': sample_rate, 'nan_policy': nan_policy, 'fill_value': fill_value, 'inf_policy': inf_policy}, for_idx_name='timeseries')
    f = r[0][0]
    multi_Pxx = np.array([i[1] for i in r])
    return f, multi_Pxx


def _pre_process_for_acf(timeseries, T=None, sample_rate=None, nlags=None, nan_policy='interpolate', fill_value=0, inf_policy=INF_POLICY, nlags_policy='raise'):
    if T is None and sample_rate is None:
        raise ValueError("Either T or sample_rate must be provided")
    if T is None:
        T = 1 / sample_rate
    if nlags > len(timeseries) - 1:
        if nlags_policy == 'clip':
            nlags = len(timeseries) - 1
        elif nlags_policy == 'raise':
            raise ValueError("nlags must be less than the length of the timeseries")
    clean_timeseries = process_special_value(
        timeseries, nan_policy=nan_policy, fill_value=fill_value, inf_policy=inf_policy)
    return T, nlags, clean_timeseries


def get_acf(timeseries, T=None, sample_rate=None, nlags=None, fft=True, nan_policy='interpolate', fill_value=0, inf_policy=INF_POLICY, nlags_policy='raise', **kwargs):
    '''
    计算自相关函数 (ACF) 并返回结果
    自动处理NaN值,通过线性插值填充NaN

    参数:
    timeseries: 一维数组,时间序列
    nlags_policy: 'raise'或'clip','raise'表示nlags超出时间序列长度时抛出异常,'clip'表示截断nlags到时间序列长度
    '''
    T, nlags, clean_timeseries = _pre_process_for_acf(timeseries=timeseries, T=T, sample_rate=sample_rate, nlags=nlags, nan_policy=nan_policy, fill_value=fill_value, inf_policy=inf_policy, nlags_policy=nlags_policy)
    return np.arange(nlags+1)*T, acf(clean_timeseries, nlags=nlags, fft=fft, **kwargs)


def get_acovf(timeseries, T=None, sample_rate=None, nlags=None, fft=True, nan_policy='interpolate', fill_value=0, inf_policy=INF_POLICY, nlags_policy='raise'):
    '''
    计算自协方差函数 (ACOVF) 并返回结果
    自动处理NaN值,通过线性插值填充NaN

    参数:
    timeseries: 一维数组,时间序列
    nlags_policy: 'raise'或'clip','raise'表示nlags超出时间序列长度时抛出异常,'clip'表示截断nlags到时间序列长度
    '''
    T, nlags, clean_timeseries = _pre_process_for_acf(timeseries=timeseries, T=T, sample_rate=sample_rate, nlags=nlags, nan_policy=nan_policy, fill_value=fill_value, inf_policy=inf_policy, nlags_policy=nlags_policy)
    return np.arange(nlags+1)*T, acovf(clean_timeseries, nlag=nlags, fft=fft)


def get_acovf_without_center(timeseries, T=None, sample_rate=None, nlags=None, fft=True, nan_policy='interpolate', fill_value=0, inf_policy=INF_POLICY, nlags_policy='raise'):
    '''
    E[x(t)x(t+tau)], 即ACOVF没有中心化
    '''
    T, nlags, clean_timeseries = _pre_process_for_acf(timeseries=timeseries, T=T, sample_rate=sample_rate, nlags=nlags, nan_policy=nan_policy, fill_value=fill_value, inf_policy=inf_policy, nlags_policy=nlags_policy)
    return np.arange(nlags+1)*T, acovf(clean_timeseries, nlag=nlags, fft=fft) + np.mean(clean_timeseries)**2


def get_ccf(x, y, T=None, sample_rate=None, nlags=None, nan_policy='interpolate', fill_value=0, inf_policy=INF_POLICY, nlags_policy='raise'):
    '''
    计算交叉相关函数 (CCF) 并返回结果
    自动处理NaN值,通过线性插值填充NaN
    
    参数:
    x - 第一个时间序列
    y - 第二个时间序列
    nlags - int, 最大滞后数
    nlags_policy: 'raise'或'clip','raise'表示nlags超出时间序列长度时抛出异常,'clip'表示截断nlags到时间序列长度

    注意:
    这里输入nlags后输出的是nlags+1的值(与acf的用法一样,包含0)
    '''
    if T is None and sample_rate is None:
        raise ValueError("Either T or sample_rate must be provided")
    if T is None:
        T = 1 / sample_rate
    if nlags is None:
        nlags = len(x) - 1
    if nlags > len(x) - 1 or nlags > len(y) - 1:
        if nlags_policy == 'clip':
            nlags = min(len(x) - 1, len(y) - 1)
        elif nlags_policy == 'raise':
            raise ValueError("nlags must be less than the length of the timeseries")
    local_nlags = nlags + 1
    x, y = sync_special_value(x, y, inf_policy=inf_policy)
    x = process_special_value(x, nan_policy=nan_policy, fill_value=fill_value, inf_policy=inf_policy)
    y = process_special_value(y, nan_policy=nan_policy, fill_value=fill_value, inf_policy=inf_policy)
    return np.arange(0, local_nlags)*T, ccf(x, y, nlags=local_nlags)


def get_ccovf(x, y, T=None, sample_rate=None, nlags=None, nan_policy='interpolate', fill_value=0, inf_policy=INF_POLICY, nlags_policy='raise'):
    '''
    计算交叉协方差函数 (CCOVF) 并返回结果
    自动处理NaN值,通过线性插值填充NaN
    
    参数:
    x - 第一个时间序列
    y - 第二个时间序列
    nlags - int, 最大滞后数
    nlags_policy: 'raise'或'clip','raise'表示nlags超出时间序列长度时抛出异常,'clip'表示截断nlags到时间序列长度

    注意:
    这里输入nlags后输出的是nlags+1的值(与acf的用法一样,包含0)
    '''
    lag_times, ccf_values = get_ccf(x, y, T=T, sample_rate=sample_rate, nlags=nlags, nan_policy=nan_policy, fill_value=fill_value, inf_policy=inf_policy, nlags_policy=nlags_policy)
    return lag_times, ccf_values * np.std(x) * np.std(y)


def get_multi_acf(multi_timeseries, T=None, sample_rate=None, nlags=None, fft=True, nan_policy='interpolate', fill_value=0, inf_policy=INF_POLICY, process_num=1, use_tqdm=True, **kwargs):
    '''
    处理多个时间序列的自相关函数 (ACF) 并返回结果,multi_timeseries的shape为(time_series_num, time_series_length)
    '''
    r = multi_process_list_for(process_num=process_num, func=get_acf, for_list=multi_timeseries, kwargs={'T': T, 'sample_rate': sample_rate, 'nlags': nlags, 'fft': fft, 'nan_policy': nan_policy, 'fill_value': fill_value, 'inf_policy': inf_policy, **kwargs}, for_idx_name='timeseries', use_tqdm=use_tqdm)
    lag_times = r[0][0]
    multi_acf = np.array([i[1] for i in r])
    return lag_times, multi_acf


def get_multi_acovf(multi_timeseries, T=None, sample_rate=None, nlags=None, fft=True, nan_policy='interpolate', fill_value=0, inf_policy=INF_POLICY, process_num=1):
    '''
    处理多个时间序列的自协方差函数 (ACOVF) 并返回结果,multi_timeseries的shape为(time_series_num, time_series_length)
    '''
    r = multi_process_list_for(process_num=process_num, func=get_acovf, for_list=multi_timeseries, kwargs={'T': T, 'sample_rate': sample_rate, 'nlags': nlags, 'fft': fft, 'nan_policy': nan_policy, 'fill_value': fill_value, 'inf_policy': inf_policy}, for_idx_name='timeseries')
    lag_times = r[0][0]
    multi_acovf = np.array([i[1] for i in r])
    return lag_times, multi_acovf


def get_multi_acovf_without_center(multi_timeseries, T=None, sample_rate=None, nlags=None, fft=True, nan_policy='interpolate', fill_value=0, inf_policy=INF_POLICY, process_num=1):
    r = multi_process_list_for(process_num=process_num, func=get_acovf_without_center, for_list=multi_timeseries, kwargs={'T': T, 'sample_rate': sample_rate, 'nlags': nlags, 'fft': fft, 'nan_policy': nan_policy, 'fill_value': fill_value, 'inf_policy': inf_policy}, for_idx_name='timeseries')
    lag_times = r[0][0]
    multi_acovf = np.array([i[1] for i in r])
    return lag_times, multi_acovf


def _get_multi_ccf_ccovf_single_process(i, x, multi_y, T=None, sample_rate=None, nlags=None, nan_policy='interpolate', fill_value=0, inf_policy=INF_POLICY, ccf_or_ccovf='ccf'):
    '''
    不使用multi_process的版本

    i: x属于multi_x的第几个时间序列
    '''
    if ccf_or_ccovf == 'ccf':
        f = get_ccf
    elif ccf_or_ccovf == 'ccovf':
        f = get_ccovf
    else:
        raise ValueError("ccf_or_ccovf must be 'ccf' or 'ccovf'")
    ccf_or_ccovf_dict = {}
    for j, y in enumerate(multi_y):
        lag_times, ccf_or_ccovf_dict[(i, j)] = f(x, y, T=T, sample_rate=sample_rate, nlags=nlags, nan_policy=nan_policy, fill_value=fill_value, inf_policy=inf_policy)
    return lag_times, ccf_or_ccovf_dict


def _get_multi_ccf_ccovf(multi_x, multi_y, T=None, sample_rate=None, nlags=None, nan_policy='interpolate', fill_value=0, inf_policy=INF_POLICY, process_num=1, ccf_or_ccovf='ccf'):
    '''
    multi_x和multi_y的shape为(time_series_num_x, time_series_length)和(time_series_num_y, time_series_length)

    返回的multi_ccf_or_ccovf为一个字典,键为(i, j),表示第i个时间序列和第j个时间序列的ccf或ccovf
    '''
    r = multi_process_enumerate_for(process_num=process_num, func=_get_multi_ccf_ccovf_single_process, for_list=multi_x, kwargs={'multi_y': multi_y, 'T': T, 'sample_rate': sample_rate, 'nlags': nlags, 'nan_policy': nan_policy, 'fill_value': fill_value, 'inf_policy': inf_policy, 'ccf_or_ccovf': ccf_or_ccovf}, for_idx_name='i', for_item_name='x')
    lag_times = r[0][0]
    multi_ccf_or_ccovf = {}
    for i in r:
        multi_ccf_or_ccovf.update(i[1])
    return lag_times, multi_ccf_or_ccovf


def get_multi_ccf(multi_x, multi_y, T=None, sample_rate=None, nlags=None, nan_policy='interpolate', fill_value=0, inf_policy=INF_POLICY, process_num=1):
    return _get_multi_ccf_ccovf(multi_x, multi_y, T=T, sample_rate=sample_rate, nlags=nlags, nan_policy=nan_policy, fill_value=fill_value, inf_policy=inf_policy, process_num=process_num, ccf_or_ccovf='ccf')


def get_multi_ccovf(multi_x, multi_y, T=None, sample_rate=None, nlags=None, nan_policy='interpolate', fill_value=0, inf_policy=INF_POLICY, process_num=1):
    return _get_multi_ccf_ccovf(multi_x, multi_y, T=T, sample_rate=sample_rate, nlags=nlags, nan_policy=nan_policy, fill_value=fill_value, inf_policy=inf_policy, process_num=process_num, ccf_or_ccovf='ccovf')


def get_hist(data, bins, stat='probability', nan_policy='drop', fill_value=0, inf_policy=INF_POLICY):
    '''
    Generates a histogram with various normalization options.

    Parameters:
    - data: array-like, the data to be histogrammed.
    - bins: int or array-like, the bin specification.
    - stat: str, the normalization method to use. One of ['count', 'frequency', 'probability', 'proportion', 'percent', 'density'] as sns.histplot.

    Returns:
    - values: array, the values corresponding to the normalization method chosen.
    - bin_edges: array, the edges of the bins.
    - midpoints: array, the midpoints of the bins.
    '''

    local_data = process_special_value(data, nan_policy=nan_policy, fill_value=fill_value, inf_policy=inf_policy)

    hist, bin_edges = np.histogram(local_data, bins=bins)

    # Calculate bin widths
    bin_widths = np.diff(bin_edges)

    if stat == 'count':
        # No change needed, hist already represents the count
        values = hist
    elif stat == 'frequency':
        # Frequency is count divided by bin width
        values = hist / bin_widths
    elif stat == 'probability' or stat == 'proportion':
        # Probability is count divided by total number of observations
        values = hist / hist.sum()
    elif stat == 'percent':
        # Percent is probability times 100
        values = (hist / hist.sum()) * 100
    elif stat == 'density':
        # Density normalizes count by total observations and bin width
        values = hist / (hist.sum() * bin_widths)
    else:
        raise ValueError("Unsupported stat method.")

    return values, bin_edges, get_midpoint(bin_edges)


def get_hist_2d(data_x, data_y, bins_x, bins_y, stat='probability', nan_policy='drop', fill_value=0, inf_policy=INF_POLICY, sync=True):
    '''
    Generates a 2D histogram with various normalization options for two-dimensional data,
    allowing separate bin specifications for each axis.

    Parameters:
    - data_x: array-like, the data for the x-axis to be histogrammed.
    - data_y: array-like, the data for the y-axis to be histogrammed.
    - bins_x: int or array-like, the bin specification for the x-axis.
    - bins_y: int or array-like, the bin specification for the y-axis.
    - stat: str, the normalization method to use. One of ['count', 'frequency', 'probability', 'proportion', 'percent', 'density'].

    Returns:
    - values: 2D array, the values corresponding to the normalization method chosen.
    - bin_edges_x: array, the edges of the bins for the x-axis.
    - bin_edges_y: array, the edges of the bins for the y-axis.
    - midpoints_x: array, the midpoints of the bins for the x-axis.
    - midpoints_y: array, the midpoints of the bins for the y-axis.
    '''

    if sync:
        local_data_x, local_data_y = sync_special_value(data_x, data_y, inf_policy=inf_policy)
    else:
        local_data_x, local_data_y = data_x.copy(), data_y.copy()
    local_data_x = process_special_value(local_data_x, nan_policy=nan_policy, fill_value=fill_value, inf_policy=inf_policy)
    local_data_y = process_special_value(local_data_y, nan_policy=nan_policy, fill_value=fill_value, inf_policy=inf_policy)

    hist, bin_edges_x, bin_edges_y = np.histogram2d(
        local_data_x, local_data_y, bins=[bins_x, bins_y])

    # Calculate bin areas for density calculation
    bin_widths_x = np.diff(bin_edges_x)
    bin_widths_y = np.diff(bin_edges_y)
    bin_areas = np.outer(bin_widths_x, bin_widths_y)

    if stat == 'count':
        values = hist
    elif stat == 'frequency':
        values = hist / bin_areas
    elif stat == 'probability' or stat == 'proportion':
        values = hist / hist.sum()
    elif stat == 'percent':
        values = (hist / hist.sum()) * 100
    elif stat == 'density':
        values = hist / (hist.sum() * bin_areas)
    else:
        raise ValueError("Unsupported stat method.")

    return values, bin_edges_x, bin_edges_y, get_midpoint(bin_edges_x), get_midpoint(bin_edges_y)


def get_kde(data, bw_method='scott', nan_policy='drop', fill_value=0, inf_policy=INF_POLICY, sync=True, **kwargs):
    '''
    计算核密度估计 (KDE) 并返回结果
    :param data: array-like, shape (dims, points) or (points,), the input data.
    '''
    if sync:
        local_data = sync_special_value_along_axis(data=data, sync_axis=0, inf_policy=inf_policy)
    else:
        local_data = data.copy()
    local_data = pure_list(local_data)
    shape = list_shape(local_data)
    if len(shape) > 1:
        for i in range(shape[0]):
            local_data[i] = process_special_value(
                local_data[i], nan_policy=nan_policy, fill_value=fill_value, inf_policy=inf_policy)
    else:
        local_data = process_special_value(
            local_data, nan_policy=nan_policy, fill_value=fill_value, inf_policy=inf_policy)
    return st.gaussian_kde(local_data, bw_method=bw_method, **kwargs)


def get_cdf(data, nan_policy='drop', fill_value=0, inf_policy=INF_POLICY, **kwargs):
    '''
    计算累积分布函数 (CDF) 并返回结果
    :param data: 1-d array-like, the input data.
    '''
    clean_data = process_special_value(
        data, nan_policy=nan_policy, fill_value=fill_value, inf_policy=inf_policy)
    return ECDF(clean_data, **kwargs)


def get_midpoint(x):
    '''
    Calculate the midpoints of a sequence of values.

    Parameters:
    - x: array-like, the values to calculate midpoints for.

    Returns:
    - midpoints: array, the midpoints of the input values.
    '''
    return (x[1:] + x[:-1]) / 2


def get_mode_kde(data, bandwidth='scott', grid_size=1000):
    """
    使用核密度估计 (KDE) 找到数据的模态。

    参数:
    - data: 输入数据数组 (list, numpy array)
    - bandwidth: KDE的带宽，可以是'scott', 'silverman'，或者一个浮点数
    - grid_size: 用于计算密度的网格点数，默认是1000

    返回:
    - mode_estimate: 估计的模态值
    """
    try:
        # 创建核密度估计对象
        kde = gaussian_kde(data, bw_method=bandwidth)

        # 在数据范围内生成网格点
        x = np.linspace(min(data), max(data), grid_size)
        
        # 计算每个点的密度值
        kde_values = kde(x)
        
        # 找到密度值最高的点
        mode_estimate = x[np.argmax(kde_values)]
        
        return mode_estimate
    except:
        print_title('Error: Failed to estimate the mode using KDE, returning mean instead.')
        return np.nanmean(data)


def repeat_data(data, repeat_times):
    '''
    重复列表中的每个元素指定的次数。

    参数:
    - data: 要重复的数据,可以是列表,numpy数组,字典,Pandas Series或DataFrame
    - repeat_times: 重复的次数,可以是整数,列表,numpy数组,字典,Pandas Series或DataFrame

    返回:
    - 所有重复后的元素
    '''
    # For lists and numpy arrays
    if isinstance(data, list):
        if isinstance(repeat_times, int):
            return [item for item in data for _ in range(repeat_times)]
        elif isinstance(repeat_times, (list, np.ndarray)) and len(data) == len(repeat_times):
            return [item for item, count in zip(data, repeat_times) for _ in range(count)]
    elif isinstance(data, np.ndarray):
        if isinstance(repeat_times, (list, np.ndarray, int)):
            return np.repeat(data, repeat_times)

    # Handling dictionary data
    if isinstance(data, dict) and isinstance(repeat_times, (dict, int)):
        result = []
        if isinstance(repeat_times, int):
            for key, value in data.items():
                if isinstance(value, list):
                    result.extend(
                        [item for item in value for _ in range(repeat_times)])
                else:
                    result.extend([value for _ in range(repeat_times)])
        elif isinstance(repeat_times, dict):
            for key, value in data.items():
                if isinstance(value, list):
                    result.extend(
                        [item for item in value for _ in range(repeat_times[key])])
                else:
                    result.extend(
                        [value for _ in range(repeat_times[key])])
        return result

    # Convert Series and DataFrame outputs to list or np.array using pandas.concat
    if isinstance(data, pd.Series):
        if isinstance(repeat_times, int):
            return data.repeat(repeat_times).tolist()
        elif isinstance(repeat_times, pd.Series):
            repeated_values = [pd.Series(
                [value] * repeat_times.get(index, 1)) for index, value in data.items()]
            repeated_series = pd.concat(repeated_values).reset_index(drop=True)
            return repeated_series.tolist()
    if isinstance(data, pd.DataFrame):
        if isinstance(repeat_times, int):
            return data.loc[data.index.repeat(repeat_times)].to_numpy()
        elif isinstance(repeat_times, pd.DataFrame):
            data_flat = data.to_numpy().flatten()
            repeats_flat = repeat_times.to_numpy().flatten().astype(int)
            return np.repeat(data_flat, repeats_flat)
# endregion


# region 基础模型
class MetaResultsMixin:
    '''
    将某个f'{results_prefix}_results'的结果设置为特定的类型,并且灵活的保存和读取
    '''
    def _set_results(self, results_prefix, results_type='dict', param_order=None):
        '''
        设置结果的类型并初始化
        '''
        setattr(self, f'{results_prefix}_results_type', results_type)
        if results_type == 'dict':
            setattr(self, f'{results_prefix}_results', {})
        elif results_type == 'OrderedDataContainer':
            setattr(self, f'{results_prefix}_results', OrderedDataContainer(param_order=param_order))

    def _set_save_load_method(self, results_prefix, save_load_method='separate'):
        '''
        注意:
        这里对于separate使用了save_dict_separate_merge_to_saved,方便只增加部分新results
        '''
        # 验证保存加载方法
        acceptable_methods = ['separate', 'lmdb']
        if save_load_method not in acceptable_methods:
            raise ValueError(f'save_load_method must be one of {acceptable_methods}')
        
        # 设置基础属性名
        save_attr = f'save_{results_prefix}_func'
        load_attr = f'load_{results_prefix}_func'
        method_attr = f'{results_prefix}_save_load_method'
        
        # 设置方法类型
        setattr(self, method_attr, save_load_method)
        
        # 分配对应的函数
        if save_load_method == 'separate':
            setattr(self, save_attr, save_dict_separate_merge_to_saved)
            setattr(self, load_attr, load_dict_separate_merge_to_exist)
        elif save_load_method == 'lmdb':
            setattr(self, save_attr, save_dict_lmdb)
            setattr(self, load_attr, load_dict_lmdb_merge_to_exist)

    def _load_results(self, results_prefix, **kwargs):
        '''
        读取结果

        注意:
        如果{results_prefix}_results是datacontainer,这个函数在读取已经保存的部分结果时,可能会覆盖_config
        只建议在项目的一开始使用,后面建议使用get_{results_prefix}_value
        '''
        results_attr = f'{results_prefix}_results'
        results_dict = getattr(self, results_attr)
        load_func = getattr(self, f'load_{results_prefix}_func')
        load_func(results_dict, pj(self.outcomes_dir, f'{results_prefix}_results'), **kwargs)

    def _update_config_from_saved_results(self, results_prefix):
        '''
        从已经保存的结果中更新_config
        '''
        results_attr = f'{results_prefix}_results'
        results_dict = getattr(self, results_attr)
        if check_all_file_exist(pj(self.outcomes_dir, f'{results_prefix}_results')):
            try:
                load_func = getattr(self, f'load_{results_prefix}_func')
                load_func(results_dict, pj(self.outcomes_dir, f'{results_prefix}_results'), key_to_load=['_config'], ensure_config=False)
            except Exception as e:
                # print(f'Error loading config from {results_prefix}_results: {e}')
                pass
        self.__setattr__(f'{results_attr}_config_updated', True)

    def _get_results_value_by_key(self, results_prefix, key):
        '''
        从结果中获取值,但是只支持直接输入key
        
        此实现方式避免了一次性读取所有分析结果,节省内存
        '''
        results_attr = f'{results_prefix}_results'
        results_dict = getattr(self, results_attr)
        if key in results_dict:
            return results_dict[key]
        else:
            load_func = getattr(self, f'load_{results_prefix}_func')
            partial_results = load_func(results_dict, pj(self.outcomes_dir, f'{results_prefix}_results'), key_to_load=[key], ensure_config=False)
            return results_dict[key]

    def _get_results_value(self, results_prefix, key=None, **kwargs):
        '''
        兼容dict和datacontainer的情况
        '''
        results_attr = f'{results_prefix}_results'
        results_dict = getattr(self, results_attr)
        results_type = getattr(self, f'{results_prefix}_results_type')

        if results_type == 'dict':
            return self._get_results_value_by_key(results_prefix=results_prefix, key=key)
        elif results_type == 'OrderedDataContainer':
            if key is None:
                key = results_dict.build_key(**kwargs)
            return self._get_results_value_by_key(results_prefix=results_prefix, key=key)

    def _get_results_subcontainer(self, results_prefix, remove_matched_params=True, **kwargs):
        '''
        从结果中获取子容器

        注意:
        不能直接使用self.{results_prefix}_results.get_subcontainer(**kwargs),因为有可能self.{results_prefix}_results没有读取所有的元素,直接使用会缺失这些元素
        '''
        results_attr = f'{results_prefix}_results'
        results_dict = getattr(self, results_attr)
        
        results_type = getattr(self, f'{results_prefix}_results_type')
        if results_type == 'OrderedDataContainer':
            # 保证读取需要的所有元素
            key_to_load = results_dict.get_partial_match_key_list(**kwargs)
            key_to_load = remove_list_element(key_to_load, list(results_dict.keys()))
            
            if key_to_load:
                getattr(self, f'load_{results_prefix}_results')(key_to_load=key_to_load)
            
            # 利用get_subcontainer方法获取子容器
            subcontainer = results_dict.get_subcontainer(remove_matched_params=remove_matched_params, **kwargs)
            return subcontainer
        else:
            raise ValueError('Only OrderedDataContainer supports subcontainer')

    def _save_results(self, results_prefix, filename, **kwargs):
        results_attr = f'{results_prefix}_results'
        results_dict = getattr(self, results_attr)
        filename = pj(self.outcomes_dir, filename)
        getattr(self, f'save_{results_prefix}_func')(results_dict, filename, **kwargs)
        print_title(f'{self.model_name}: {results_prefix} results saved')

    def _mark_all_results_saved(self, results_prefix):
        '''
        标志着所有结果都已经保存完毕
        '''
        save_dict({f'{results_prefix}_results_saved': True}, pj(self.outcomes_dir, f'{results_prefix}_results', f'{results_prefix}_results_saved'))

    def _check_all_results_saved(self, results_prefix):
        '''
        确认是否所有结果都已经保存完毕
        '''
        if check_all_file_exist_with_any_extension(pj(self.outcomes_dir, f'{results_prefix}_results', f'{results_prefix}_results_saved')):
            setattr(self, f'{results_prefix}_results_saved', True)
            return True
        else:
            setattr(self, f'{results_prefix}_results_saved', False)
            return False

    def _delete_results(self, results_prefix):
        '''
        删除文件夹中的所有结果

        注意:
        一般不要对simulation使用
        '''
        rmdir(pj(self.outcomes_dir, f'{results_prefix}_results'))


class MetaAnalyzerMixin(MetaResultsMixin):
    def _analysis_params_setted(self):
        self.analysis_params_setted = True

    def set_analysis_params(self, analysis_params, set_to_attr=True):
        '''
        设置分析参数,子类如果修改,则注意不要放到params中,因为分析参数不需要search_params

        如果子类忘记写_analysis_params_setted,后面analyze会因为没有self.analysis_params_setted而raise
        '''
        self.analysis_params = analysis_params
        if set_to_attr:
            for k, v in analysis_params.items():
                if hasattr(self, k):
                    raise ValueError(f'Attribute {k} already exists, please use another name')
                setattr(self, k, v)
        save_dict(analysis_params, pj(self.params_dir, 'analysis_params'))
        self._analysis_params_setted()
    
    def set_analysis_task(self, analysis_task_list):
        '''
        设置分析结果的task列表
        '''
        self.analysis_task_list = analysis_task_list

    def extend_analysis_task(self, *new_analysis_task):
        if not hasattr(self, 'analysis_task_list'):
            self.analysis_task_list = []
        self.analysis_task_list.extend(new_analysis_task)
    
    def set_analysis_results(self, results_type='dict', param_order=None):
        self._set_results(results_prefix='analysis', results_type=results_type, param_order=param_order)

    def set_analysis_save_load_method(self, save_load_method='separate'):
        self._set_save_load_method(results_prefix='analysis', save_load_method=save_load_method)

    def whether_run_this_analysis_task(self, task):
        if task in self.analysis_task_list or 'all' in self.analysis_task_list:
            print_title(f'{self.model_name}: Analysis task {task} will be run')
            return True
        else:
            print_title(f'{self.model_name}: Analysis task {task} will be skipped')
            return False

    def _check_analysis_params(self):
        if not hasattr(self, 'analysis_params_setted') or not self.analysis_params_setted:
            raise ValueError('Please set analysis params first! Also please check if you have set self._analysis_params_setted() in your subclass!')

    def analyze_detail(self):
        '''
        子类可实现
        '''
        pass

    def analyze(self):
        '''
        运行分析
        '''
        self._check_analysis_params()
        analysis_timer = Timer(title=f'{self.model_name}: Analyze model')
        analysis_timer.start()
        self.analyze_detail()
        self.save_analysis_results()
        analysis_timer.end()

    def load_analysis_results(self, **kwargs):
        self._load_results(results_prefix='analysis', **kwargs)
    
    def get_analysis_value(self, key=None, **kwargs):
        return self._get_results_value(results_prefix='analysis', key=key, **kwargs)
    
    def get_analysis_subcontainer(self, remove_matched_params=True, **kwargs):
        return self._get_results_subcontainer(results_prefix='analysis', remove_matched_params=remove_matched_params, **kwargs)
    
    def save_analysis_results(self, filename='analysis_results', **kwargs):
        self._save_results(results_prefix='analysis', filename=filename, **kwargs)

    def delete_analysis_results(self):
        self._delete_results(results_prefix='analysis')


class MetaModel(abc.ABC, MetaAnalyzerMixin):
    '''
    一个元模型的基类，用于封装模型的参数设置,读取,保存;cpu,gpu设置;备份代码;设置logger等功能

    注意:
    当模型分为多个阶段,下一阶段依赖上一阶段结果时,可以嵌套Metamodel
    这么做的好处是,上一阶段在不被修改的时候不会被频繁运行,而是可以读取到已有结果
    例如,SNN生成连接可以单独作为一个model,运行snn也是一个model,运行SNN的model内部使用connection的model

    警告:
    当模型运算速度非常快,并且串行跑了多个,有可能导致timedir是一样的,这将导致后面的模型覆盖前面的模型
    当使用multiprocess运行时也有可能发生这种情况,一般不会,但是记录在此
    '''
    def __init__(self):
        self.force_run = False
        self.set_simulation_results() # 直接结果(simulation得到的),如果需要可以在子类改成DataContainer
        self.set_analysis_results() # 分析结果(analysis得到的),如果需要可以在子类改成DataContainer
        self.pop = False
        self.value_dir_key_before = []
        self.both_dir_key_before = []
        self.value_dir_key_after = []
        self.both_dir_key_after = []
        self.ignore_key_list = []
        self.code_file_list = []
        self.set_current_time()
        self.set_model_name()
        self.set_optional_params_default()

    # region set things
    def set_up(self, params, basedir, code_file_list, value_dir_key_before=None, both_dir_key_before=None, value_dir_key_after=None, both_dir_key_after=None, ignore_key_list=None, force_run=None):
        '''
        总体设置

        注意:
        不输入的参数会使用默认值
        '''
        if value_dir_key_before is None:
            value_dir_key_before = self.value_dir_key_before
        if both_dir_key_before is None:
            both_dir_key_before = self.both_dir_key_before
        if value_dir_key_after is None:
            value_dir_key_after = self.value_dir_key_after
        if both_dir_key_after is None:
            both_dir_key_after = self.both_dir_key_after
        if ignore_key_list is None:
            ignore_key_list = self.ignore_key_list
        if force_run is None:
            force_run = self.force_run
        self.set_params(params=params)
        self.set_basedir(basedir=basedir)
        self.set_value_dir_both_dir_key(value_dir_key_before=value_dir_key_before, both_dir_key_before=both_dir_key_before, value_dir_key_after=value_dir_key_after, both_dir_key_after=both_dir_key_after)
        self.set_ignore_key_for_search_params(ignore_key_list=ignore_key_list)
        self.create_timedir()
        self.set_code_file_list(code_file_list=code_file_list)
        self.set_force_run(force_run=force_run)

    def set_optional_params_default(self):
        self.set_analysis_task(['all'])
        self.set_simulation_save_load_method(save_load_method='separate')
        self.set_analysis_save_load_method(save_load_method='separate')

    def set_visualizer_kwargs(self, **visualizer_kwargs):
        '''
        设置可视化器的参数,不放到params中,因为可视化器的参数不需要保存到文件中
        '''
        self.visualizer_kwargs = visualizer_kwargs

    def set_force_run(self, force_run=False):
        '''
        设置模型的强制运行标志,默认为False,会尝试读取已有结果,如果为True,则无条件运行模型
        '''
        self.force_run = force_run

    def set_basedir(self, basedir):
        '''
        设置模型的基础文件夹
        
        示例:
        basedir = '../../results'
        basedir = os.path.join(basedir, os.path.splitext(os.path.basename(cf.current_file()))[0])
        这样可以根据代码文件名自动创建文件夹

        注意:
        basedir和timedir是不同的
        '''
        self.basedir = basedir

    def set_timedir(self, timedir):
        '''
        设置模型的时间文件夹

        示例:
        当已经手动确定了模型的时间文件夹时,可以直接设置,方便之后运行analysis

        注意:
        只有在希望单独运行analysis时才需要设置
        timedir不一定以时间结尾,特殊情况下时间后面还会有value_dir_key_after,both_dir_key_after等产生的文件夹
        '''
        self.timedir = timedir
        self.load_params()
        self.load_info_container()
        self._update_config_from_saved_results('simulation')
        self._update_config_from_saved_results('analysis')

    def set_current_time(self, current_time=None):
        '''
        设置当前时间,如果不设置,则使用当前时间
        '''
        if current_time is None:
            self.current_time = get_time()
        else:
            self.current_time = current_time

    def set_model_name(self, model_name=None):
        if model_name is None:
            self.model_name = self.__class__.__name__
        else:
            self.model_name = model_name

    def set_value_dir_both_dir_key(self, value_dir_key_before=None, both_dir_key_before=None, value_dir_key_after=None, both_dir_key_after=None):
        '''
        设置模型的value_dir_key和both_dir_key

        注意:
        输入none则不更新
        '''
        if value_dir_key_before is not None:
            self.value_dir_key_before = value_dir_key_before
        if both_dir_key_before is not None:
            self.both_dir_key_before = both_dir_key_before
        if value_dir_key_after is not None:
            self.value_dir_key_after = value_dir_key_after
        if both_dir_key_after is not None:
            self.both_dir_key_after = both_dir_key_after

    def set_ignore_key_for_search_params(self, ignore_key_list=None):
        '''
        设置搜索参数时忽略的key

        注意:
        输入none则不更新
        '''
        if ignore_key_list is not None:
            self.ignore_key_list = ignore_key_list

    def set_pop(self, pop):
        '''
        设置pop(储存params时是否pop掉value_dir_key和both_dir_key再保存)
        '''
        self.pop = pop

    def set_params(self, params, set_to_attr=True):
        '''
        设置模型参数
        '''
        if set_to_attr:
            for key, value in params.items():
                setattr(self, key, value)
        self.params = params

    def set_info_container(self, info_order):
        '''
        设置info_container

        例子:
        当模型有比较复杂的信息时,可以使用OrderedDataContainer来存储
        info_container中的元素并不会被保存到params中,而是单独保存;并且info_container中的元素常常是params中的元素经过运算
        '''
        self.info_container = OrderedDataContainer(param_order=info_order)

    def set_gpu(self, id=None):
        '''
        设置模型在GPU上运行
        '''
        if id is None:
            set_least_used_gpu()
        else:
            set_gpu(id)

    def set_code_file_list(self, code_file_list):
        '''
        设置模型代码文件列表

        示例:
        code_file_list = [cf.current_file(), '../../utils_function/utils_function.py']
        '''
        self.code_file_list = code_file_list

    def set_logger(self, filename=None, filemode='a'):
        '''
        设置logger
        '''
        self.logger = Logger()
        self.logger.get_py_logger(basedir=self.logs_dir, filename=filename, filemode=filemode, name=self.timedir)
        print_title(f'{self.model_name}: Logger initialized')

    def set_simulation_results(self, results_type='dict', param_order=None):
        '''
        设置模拟结果的类型并初始化
        '''
        self._set_results(results_prefix='simulation', results_type=results_type, param_order=param_order)

    def set_simulation_save_load_method(self, save_load_method='separate'):
        self._set_save_load_method(results_prefix='simulation', save_load_method=save_load_method)

    def extend_ignore_key_list(self, *new_ignore_key):
        '''
        向ignore_key_list中添加新的key
        '''
        self.ignore_key_list.extend(new_ignore_key)

    def extend_value_dir_key_before(self, *new_value_dir_key):
        '''
        向value_dir_key_before中添加新的key
        '''
        self.value_dir_key_before.extend(new_value_dir_key)

    def extend_both_dir_key_before(self, *new_both_dir_key):
        '''
        向both_dir_key_before中添加新的key
        '''
        self.both_dir_key_before.extend(new_both_dir_key)
    
    def extend_value_dir_key_after(self, *new_value_dir_key):
        '''
        向value_dir_key_after中添加新的key
        '''
        self.value_dir_key_after.extend(new_value_dir_key)
    
    def extend_both_dir_key_after(self, *new_both_dir_key):
        '''
        向both_dir_key_after中添加新的key
        '''
        self.both_dir_key_after.extend(new_both_dir_key)
    # endregion

    # region run
    @property
    def should_run(self):
        """
        判断是否需要执行run_detail的逻辑聚合
        """
        if not self.params_exist:
            # 当参数不存在时，强制标记结果不存在
            self.simulation_results_exist = False
            return True
        return not self.simulation_results_exist

    def run(self):
        '''
        运行模型
        '''
        self.prepare_run()

        # 通过属性判断执行
        if self.should_run:
            initialize_timer = Timer(title=f'{self.model_name}: Initialize model')
            initialize_timer.start()
            self.initialize_model()
            initialize_timer.end()

            run_timer = Timer(title=f'{self.model_name}: Run model')
            run_timer.start()
            self.run_detail()
            run_timer.end()

            self.finalize_run()

    def simulate(self):
        '''
        run的别名
        '''
        self.run()

    def prepare_run(self):
        '''
        run_detail的准备工作,包括search_params,check_simulation_results_exist,set_logger等
        '''
        self.search_params()
        
        # 自动触发结果存在性检查(当参数存在时)
        if self.params_exist:
            self.check_simulation_results_exist()

        # 统一设置日志(此时timedir已确定)
        self.set_logger()
        self.save_code()

    def initialize_model(self):
        '''
        部分模型需要在run前初始化,例如brainpy需要先构建net,monitor,runner才可以run
        '''
        self.save_info_container()
        self.save_params()

    @abc.abstractmethod
    def run_detail(self):
        """
        子类必须实现核心逻辑,需要将结果更新到self.simulation_results
        """
        pass

    def finalize_run_detail(self):
        '''
        子类可以更改
        '''
        if not self.simulation_results_exist:
            self.save_simulation_results()

    def finalize_run(self):
        """
        run的收尾操作,子类最好不要修改,而是修改finalize_run_detail
        """
        self.finalize_run_detail()
        self._mark_all_results_saved('simulation')
    # endregion

    # region visualize
    def get_visualizer(self):
        '''
        子类可实现,赋值到self.visualizer
        '''
        pass
    # endregion

    # region dir
    def create_timedir(self):
        '''
        获取模型的时间文件夹
        '''
        _, self.before_timedir = pop_dict_get_dir(self.params, self.value_dir_key_before, self.both_dir_key_before, self.basedir)
        _, self.after_timedir = pop_dict_get_dir(self.params, self.value_dir_key_after, self.both_dir_key_after, '')
        self.timedir = pj(self.before_timedir, self.current_time, self.after_timedir)

    def get_current_time_from_timedir(self):
        '''
        从timedir中获取当前时间
        '''
        return get_first_subdir(self.timedir, self.before_timedir)

    @property
    def logs_dir(self):
        return pj(self.timedir, 'logs')

    @property
    def figs_dir(self):
        return pj(self.timedir, 'figs')
    
    @property
    def models_dir(self):
        return pj(self.timedir, 'models')
    
    @property
    def outcomes_dir(self):
        return pj(self.timedir, 'outcomes')

    @property
    def params_dir(self):
        return pj(self.timedir, 'params')

    @property
    def codes_dir(self):
        return pj(self.timedir, 'code')

    @property
    def code_dir(self):
        '''保留之前的没有加s的版本'''
        return pj(self.timedir, 'code')
    # endregion

    # region load
    def search_params(self):
        '''
        搜索运行过相同参数的文件夹
        '''
        if self.force_run:
            print_title(f'{self.model_name}: Force run')
            self.params_exist = False
        else:
            self.params_exist_timedir, self.params_exist = search_dict_subdir(dict_data=self.params, basedir=self.basedir, pkl_name='params', value_dir_key=self.value_dir_key_before, both_dir_key=self.both_dir_key_before, after_subdir=pj(self.after_timedir, 'params'), ignore_key=self.ignore_key_list, pop=self.pop)
            if self.params_exist:
                print_title(f'{self.model_name}: Params exist')
                self.original_timedir = self.timedir
                self.timedir = pj(self.params_exist_timedir, self.after_timedir)
            else:
                print_title(f'{self.model_name}: Params do not exist')

    def check_simulation_results_exist(self):
        '''
        检查simulation结果是否存在

        注意:
        如果params_exist但是simulation_results不存在,说明之前的运行被中断了,需要重新运行,此时会新开一个timedir,而由于params_exist,timedir被设置为params_exist_timedir,所以需要在这里恢复timedir
        '''
        self.simulation_results_exist = self._check_all_results_saved('simulation')
        if self.simulation_results_exist:
            print_title(f'{self.model_name}: Simulation results exist')
        else:
            print_title(f'{self.model_name}: Simulation results do not exist')
            self.timedir = self.original_timedir # 恢复timedir

    def load_params(self):
        '''
        读取模型参数
        '''
        self.params = load_pkl(pj(self.params_dir, 'params'))
        self.set_params(self.params)

    def load_simulation_results(self, **kwargs):
        self.simulation_results.update(load_dict_separate(pj(self.outcomes_dir, 'simulation_results'), **kwargs))

    def get_simulation_value(self, key=None, **kwargs):
        return self._get_results_value(results_prefix='simulation', key=key, **kwargs)

    def get_simulation_subcontainer(self, remove_matched_params=True, **kwargs):
        return self._get_results_subcontainer(results_type='simulation', remove_matched_params=remove_matched_params, **kwargs)

    def load_info_container(self):
        '''
        info container不一定存在,所以使用try except
        '''
        try:
            self.info_container = load_dict(pj(self.params_dir, 'info_container'))
            self.key_builder = self.info_container.key_builder
        except:
            pass
    # endregion

    # region save
    def save_code(self):
        '''
        备份模型代码
        '''
        # 保存code_file
        for code_file in self.code_file_list:
            save_code_copy(self.code_dir, code_file)
        # 保存CF_DIR下的所有函数
        for sub_file in get_subfile(CF_DIR):
            if sub_file.endswith('.py'):
                save_code_copy(self.code_dir, sub_file)
        print_title(f'{self.model_name}: Code saved')

    def save_params(self):
        '''
        保存模型参数
        '''
        if hasattr(self, 'params'):
            if check_all_file_exist_with_any_extension(pj(self.params_dir, 'params')):
                print_title(f'{self.model_name}: Params already saved')
            else:
                save_timed_dir_dict(dict_data=self.params, basedir=self.basedir, dict_name='params', value_dir_key=self.value_dir_key_before, both_dir_key=self.both_dir_key_before, after_timedir=pj(self.after_timedir, 'params'), current_time=self.current_time, pop=self.pop)
                print_title(f'{self.model_name}: Params saved')
        else:
            print_title(f'{self.model_name}: No params to save')

    def save_info_container(self):
        '''
        保存info_container
        '''
        if hasattr(self, 'info_container'):
            save_dict(self.info_container, pj(self.params_dir, 'info_container')) # info_container一般比较小但是包含的元素可能比较多,所以直接保存更好
            print_title(f'{self.model_name}: Info container saved')

    def save_simulation_results(self, filename='simulation_results', **kwargs):
        self._save_results(results_prefix='simulation', filename=filename, **kwargs)
    # endregion


def find_incomplete_model(dir_before_timedir, dir_after_timedir=None):
    '''
    寻找不完整的模型文件夹,以'simulation_results_saved'是否存在为判断标准
    '''
    incomplete_model_list = []

    sub_dir_list = get_subdir(dir_before_timedir)
    for sub_dir in sub_dir_list:
        mark_file = pj(sub_dir, dir_after_timedir, 'outcomes', 'simulation_results', 'simulation_results_saved')
        if check_all_file_exist_with_any_extension(mark_file):
            pass
        else:
            model_dir = pj(sub_dir, dir_after_timedir)
            print(f'Found incomplete model directory: {model_dir}')
            incomplete_model_list.append(model_dir)
    return incomplete_model_list


def clean_incomplete_model(dir_before_timedir, dir_after_timedir=None):
    '''
    删除不完整的模型文件夹,以'simulation_results_saved'是否存在为判断标准
    '''
    print('Warning: The following incomplete model directories will be removed')
    incomplete_model_list = find_incomplete_model(dir_before_timedir, dir_after_timedir)
    print('You still have 1 minute to cancel this operation')
    time.sleep(60)
    for incomplete_model in incomplete_model_list:
        rmdir(incomplete_model)


class MetaModelContainer(InstanceContainer):
    '''
    获取一个目录下的所有模型,功能如下:
    初始化时,自动略过不完整的模型
    统计参数
    '''
    def __init__(self, model_class, dir_before_timedir, dir_after_timedir=None):
        incomplete_model_list = find_incomplete_model(dir_before_timedir, dir_after_timedir)
        sub_dir_list = get_subdir(dir_before_timedir)
        instance_list = []
        for sub_dir in sub_dir_list:
            timedir = pj(sub_dir, dir_after_timedir)
            if timedir in incomplete_model_list:
                continue
            else:
                instance = model_class()
                instance.set_timedir(timedir)
                instance_list.append(instance)
        super().__init__(instance_list)

    def count_params_by_key(self, key):
        count_dict = defaultdict(int)
        for instance in self.instance_list:
            value = instance.params[key]
            count_dict[repr(value)] += 1
        return count_dict


def get_sorted_keys_and_mean_variance_arrays_from_dict_of_list(data_dict):
    '''
    输入一个字典,value是list,返回排序后的key数组,均值数组,方差数组
    例如:
    data_dict = {
        '0.1': [1, 2, 3],
        '0.2': [2, 3, 4],
        '0.3': [3, 4, 5]
    }
    x, y_mean, y_var = get_sorted_keys_mean_variance_arrays_from_dict_of_list(data_dict)
    print(x)
    print(y_mean)
    print(y_var)
    '''
    # 提取并排序x轴数据（参数值）
    sorted_keys = sorted(data_dict.keys(), key=float)
    x = np.array(sorted_keys)
    
    # 计算每个参数对应的均值和方差
    means = []
    variances = []
    for key in sorted_keys:
        values = data_dict[key]
        mean = np.mean(values)
        variance = np.var(values)
        means.append(mean)
        variances.append(variance)
    
    # 转换为NumPy数组
    y_mean = np.array(means)
    y_var = np.array(variances)
    
    return x, y_mean, y_var
# endregion


# region 基础模型(compose version)
class ParamBasedDirManager:
    def __init__(self):
        self.timedir_injected = False
        self.kwargs = {
            'value_dir_key_before': [],
            'both_dir_key_before': [],
            'value_dir_key_after': [],
            'both_dir_key_after': [],
            'ignore_key_list': [],
            'pop': False
        }

    def update_kwargs(self, **kwargs):
        '''
        因为不同的参数可能会在不同的时间段输入,如basedir(experiment给了接口),params(pipeline的第一个将params输入),所以这里必须用kwargs.update()的方式来更新参数
        
        示例:
        basedir = '../../results'
        basedir = os.path.join(basedir, os.path.splitext(os.path.basename(cf.current_file()))[0])
        这样可以根据代码文件名自动创建文件夹

        注意:
        basedir和timedir是不同的
        '''
        self.kwargs.update(kwargs)

    def finalize_init(self):
        self.params = self.kwargs['params']
        self.basedir = self.kwargs['basedir']
        self.value_dir_key_before = self.kwargs['value_dir_key_before']
        self.both_dir_key_before = self.kwargs['both_dir_key_before']
        self.value_dir_key_after = self.kwargs['value_dir_key_after']
        self.both_dir_key_after = self.kwargs['both_dir_key_after']
        self.ignore_key_list = self.kwargs['ignore_key_list']
        self.pop = self.kwargs['pop']

        _, self.before_timedir = pop_dict_get_dir(self.params, self.value_dir_key_before, self.both_dir_key_before, self.basedir)
        _, self.after_timedir = pop_dict_get_dir(self.params, self.value_dir_key_after, self.both_dir_key_after, '')

        if hasattr(self, 'current_time'): 
            self.create_timedir()  # 如果已经有current_time,则创建timedir
        else:
            # 如果没有current_time,则使用当前时间
            self.set_current_time(current_time='now', update_timedir=True)

    def set_timedir(self, timedir, prefix=''):
        '''
        设置模型的时间文件夹

        示例:
        当已经手动确定了模型的时间文件夹时,可以直接设置,方便之后运行analysis

        注意:
        只有在希望单独运行analysis时才需要设置
        timedir不一定以时间结尾,特殊情况下时间后面还会有value_dir_key_after,both_dir_key_after等产生的文件夹
        '''
        self.timedir = timedir
        self.timedir_injected = True
        self.load_params(prefix=prefix)

    def set_current_time(self, current_time='now', update_timedir=True):
        '''
        current_time:
        - 'now': 使用当前时间
        - 其他字符串: 将此字符串作为当前时间
        '''
        if current_time == 'now':
            self.current_time = get_time()
        else:
            self.current_time = current_time
        if update_timedir:
            self.create_timedir()  # 重新创建timedir

    def create_timedir(self):
        '''
        获取模型的时间文件夹
        '''
        self.timedir = pj(self.before_timedir, self.current_time, self.after_timedir)

    def get_current_time_from_timedir(self):
        '''
        从timedir中获取当前时间
        '''
        return get_first_subdir(self.timedir, self.before_timedir)

    @property
    def logs_dir(self):
        return pj(self.timedir, 'logs')

    @property
    def figs_dir(self):
        return pj(self.timedir, 'figs')
    
    @property
    def models_dir(self):
        return pj(self.timedir, 'models')
    
    @property
    def outcomes_dir(self):
        return pj(self.timedir, 'outcomes')

    @property
    def params_dir(self):
        return pj(self.timedir, 'params')

    @property
    def codes_dir(self):
        return pj(self.timedir, 'code')

    def search_params(self, prefix=''):
        '''
        搜索运行过相同参数的文件夹
        '''
        self.params_exist_timedir, self.params_exist = search_dict_subdir(dict_data=self.params, basedir=self.basedir, pkl_name=cat(prefix, 'params'), value_dir_key=self.value_dir_key_before, both_dir_key=self.both_dir_key_before, after_subdir=pj(self.after_timedir, 'params'), ignore_key=self.ignore_key_list, pop=self.pop)
        if self.params_exist:
            print_title('Params exist')
            self.timedir = pj(self.params_exist_timedir, self.after_timedir)
            return True
        else:
            print_title('Params do not exist')
            return False

    def load_params(self, prefix='', set_to_self=True):
        '''
        读取模型参数(适用于手动设置了timedir的情况)
        '''
        params = load_pkl(pj(self.params_dir, cat(prefix, 'params')))
        if set_to_self:
            self.params = params
        return params


class CodeSaver:
    def __init__(self, dir_manager):
        self.dir_manager = dir_manager

    def set_code_file_list(self, code_file_list):
        '''
        设置模型代码文件列表

        示例:
        code_file_list = [cf.current_file(), '../../utils_function/utils_function.py']
        '''
        self.code_file_list = code_file_list

    def save_code(self):
        '''
        备份模型代码
        '''
        codes_dir = self.dir_manager.codes_dir

        # 保存code_file
        for code_file in self.code_file_list:
            save_code_copy(codes_dir, code_file)
        # 保存CF_DIR下的所有函数
        for sub_file in get_subfile(CF_DIR):
            if sub_file.endswith('.py'):
                save_code_copy(codes_dir, sub_file)
        print_title(f'Code saved')


class AbstractTool(abc.ABC):
    '''
    如果希望tools_pipeline中的tool可以被替换为其他tool,需要保证替换的tool之间name一致

    注意:
    和结果相关的参数,利用set_params输入
    和结果无关的设定,利用experiment中的set_tool_config_dict输入(例如process_num等)

    示例:
    class NewTool(AbstractTool):
        def _set_name(self):
            self.name = 'tool_name'
        
        def _set_required_key_list(self):
            super()._set_required_key_list()
            self.required_key_list.extend(['key1', 'key2'])

        def _set_optional_key_value_dict(self):
            super()._set_optional_key_value_dict()
            self.optional_key_value_dict.update({'key3': default_value3, 'key4': default_value4})
    
        def _config_dir_manager(self):
            super()._config_dir_manager()
            self.dir_manager_kwargs['value_dir_key_before'].extend(['animal', 'dataset_key'])
        
        def _config_data_keeper(self):
            super()._config_data_keeper()
            self.data_keeper_kwargs = {'data_type': 'dict', 'save_load_method': 'separate'}

        def _config_info_container(self):
            super()._config_info_container()
            self.info_container_kwargs = {'data_type': 'dict', 'save_load_method': 'lmdb'}

        def add_info_to_info_container(self):
            pass

        def run_detail(self):
            pass
    '''
    def __init__(self):
        self.already_done = False
        self._set_name() # 子类必须实现_set_name方法,用于设置name属性
        self._config_info_container() # 子类可以修改_config_info_container方法,用于设置info_container
        self._config_data_keeper() # 子类可以修改_config_data_keeper方法,用于设置data_keeper
        self._config_dir_manager() # 子类可以修改_config_dir_manager方法,用于设置dir_manager
        self.enable_search = False # search 有两步,第一步为寻找参数,第二步为check_all_saved
        self.enable_skip = False # 是否跳过已经完成的任务,默认是False,子类可以修改
        self.enable_try = False # 是否允许尝试运行,如果运行失败则不报错,默认是False,子类可以修改
        self.release_memory = False # 是否在运行结束后释放内存,默认是False,子类可以修改
        self.release_memory_kwargs = {} # 用于设置释放内存的kwargs,子类可以修改
        self.set_params_to_attr = True # 是否将params中的参数设置到实例属性中,默认是True,子类可以修改
        self._set_required_key_list() # 子类必须实现_set_required_key_list方法,用于设置required_key_list属性
        self._set_optional_key_value_dict() # 子类必须实现_set_optional_key_value_dict方法,用于设置optional_key_value_dict属性
        self.enable_delete_after_pipeline = False # 是否在pipeline结束后删除结果,默认是False,子类可以修改
        self.process_num = 1 # datakeeper读取和保存的process数量

    @abc.abstractmethod
    def _set_name(self):
        pass

    @abc.abstractmethod
    def _set_required_key_list(self):
        self.required_key_list = []

    @abc.abstractmethod
    def _set_optional_key_value_dict(self):
        self.optional_key_value_dict = {}

    def _check_required_keys(self):
        for key in self.required_key_list:
            if key not in self.params:
                raise ValueError(f'{key} is required but not found in params')

    def _check_optional_keys(self):
        for key, value in self.optional_key_value_dict.items():
            if key not in self.params:
                self.params[key] = value

    def set_params(self, params):
        self.params = params
        self._check_required_keys()
        self._check_optional_keys()
        if self.set_params_to_attr:
            for key, value in params.items():
                setattr(self, key, value)
        self.dir_manager_kwargs['params'] = self.params

    def input_dir_manager(self, dir_manager):
        '''
        不同tool之间共享dir_manager,所以从外部传入dir_manager
        '''
        self.dir_manager = dir_manager

    def _config_dir_manager(self):
        self.dir_manager_kwargs = {
            'value_dir_key_before': [],
            'both_dir_key_before': [],
            'value_dir_key_after': [],
            'both_dir_key_after': [],
            'ignore_key_list': [],
            'pop': False
        }

    def finalize_init_dir_manager(self):
        self.dir_manager.update_kwargs(**self.dir_manager_kwargs)
        self.dir_manager.finalize_init()

    def input_previous_info(self, info):
        '''
        一般情况下,只允许从前一个tool传入data_keeper,params
        '''
        name = info['name']
        for k, v in info.items():
            if k == 'name':
                pass
            else:
                setattr(self, f'{name}_{k}', v)

    def propagate_info(self):
        self.data_keeper.set_read_only(True)  # 确保后续的tool只能读取数据
        info = {
            'data_keeper': self.data_keeper,
            'params': self.params,
            'info_container': self.info_container,
            'name': self.name,
        }
        return info

    def _config_info_container(self):
        self.info_container_name = cat(self.name, 'info_container')
        self.info_container_kwargs = {'data_type': 'dict', 'save_load_method': 'lmdb'}
        # self.info_container_kwargs = {'data_type': 'OrderedDataContainer', 'save_load_method': 'lmdb', 'param_order': ..., 'included_name_list': ...} 注意设置param_order等

    def init_info_container(self):
        self.info_container = DataKeeper(name=self.info_container_name, basedir=self.dir_manager.params_dir, **self.info_container_kwargs)

    def _config_data_keeper(self):
        self.data_keeper_name = self.name
        self.data_keeper_kwargs = {'data_type': 'dict', 'save_load_method': 'separate'}
        # self.data_keeper_kwargs = {'data_type': 'OrderedDataContainer', 'save_load_method': 'lmdb', 'param_order': ..., 'included_name_list': ...} 注意设置param_order等

    def init_data_keeper(self):
        self.data_keeper = DataKeeper(name=self.data_keeper_name, basedir=self.dir_manager.outcomes_dir, **self.data_keeper_kwargs)
        self.data_keeper.process_num = self.process_num

    def search_params_when_enabled(self):
        if self.enable_search:
            params_exist = self.dir_manager.search_params(prefix=self.name)
            self.init_data_keeper() # 切换了dir_manager,需要重新初始化data_keeper
            self.check_all_saved()
            if params_exist and (not self.already_done):
                print_title(f'{self.name}: Not all results saved, create new dir')
                self.dir_manager.set_current_time(current_time='now', update_timedir=True)
        else:
            print_title(f'{self.name}: Search params is disabled, skip search')

    def check_all_saved(self):
        '''要把状态从data_keeper中获取出来,所以不要轻易移除这个接口'''
        self.already_done = self.data_keeper.check_all_saved()

    def save_params(self):
        if not check_all_file_exist_with_any_extension(pj(self.dir_manager.params_dir, cat(self.name, 'params'))):
            save_dict(self.params, pj(self.dir_manager.params_dir, cat(self.name, 'params')))

    def before_run(self):
        self.save_params()
        self.add_info_to_info_container()
        self.info_container.save()

    @abc.abstractmethod
    def run_detail(self):
        pass

    def after_run(self):
        self.data_keeper.save()
        if self.enable_skip:
            self.data_keeper.mark_all_saved()
        if self.release_memory:
            self.data_keeper.release_memory(**self.release_memory_kwargs)

    def run(self):
        if self.already_done:
            print_title(f'{self.name}: already done, skip')
        else:
            timer = Timer(title=f'{self.name}: running')
            timer.start()
            
            with FlexibleTry(enable_try=self.enable_try):
                self.before_run()
                self.run_detail()
                self.after_run()

            timer.end()
    
    def delete_after_pipeline_if_enabled(self):
        if self.enable_delete_after_pipeline:
            if self.data_keeper.check_all_saved():
                self.data_keeper.delete()
                self.data_keeper.mark_all_saved()  # 确保运行完成的标志仍然存在
            else:
                raise ValueError(f'{self.name}: Cannot delete results, only after all results are saved can you delete them')

    def load_params(self):
        self.params = self.dir_manager.load_params(prefix=self.name, set_to_self=False)
        self.set_params(self.params)

    def add_info_to_info_container(self):
        '''子类可按需要添加信息到info_container'''
        pass

    @property
    def logs_dir(self):
        return self.dir_manager.logs_dir

    @property
    def figs_dir(self):
        return self.dir_manager.figs_dir

    @property
    def models_dir(self):
        return self.dir_manager.models_dir

    @property
    def outcomes_dir(self):
        return self.dir_manager.outcomes_dir
    
    @property
    def params_dir(self):
        return self.dir_manager.params_dir
    
    @property
    def codes_dir(self):
        return self.dir_manager.codes_dir


class NoSkipTool(AbstractTool):
    '''
    用来传递无法保存的对象,所以无法跳过,不管什么情形都需要重新运行

    AbstractTool中有一个示例代码
    '''
    @property
    def already_done(self):
        return False
    
    @already_done.setter
    def already_done(self, value):
        pass  # 有意忽略所有赋值操作

    @property
    def enable_skip(self):
        return False

    @enable_skip.setter
    def enable_skip(self, value):
        pass

    @property
    def enable_search(self):
        return False
    
    @enable_search.setter
    def enable_search(self, value):
        pass


class Trainer(AbstractTool):
    '''
    AbstractTool中有一个示例代码
    '''
    def __init__(self):
        super().__init__()
        self.enable_search = True
        self.enable_skip = True


class Simulator(AbstractTool):
    '''
    AbstractTool中有一个示例代码
    '''
    def __init__(self):
        super().__init__()
        self.enable_search = True
        self.enable_skip = True


class FlexibleTool(AbstractTool):
    '''
    适用于零散的,失败了问题不大的任务

    功能:
    -可以选择性地执行某些任务,通过task_list控制(需要config进来,见Experiment中的tool_config_dict)
    -允许失败,通过enable_try控制

    AbstractTool中有一个示例代码
    '''
    def __init__(self):
        super().__init__()
        self.enable_try = True # 失败了问题不大,先try尽可能往后运行
        self.task_list = ['all']  # 默认是全部任务

    def whether_run_this_task(self, task):
        if task in self.task_list or 'all' in self.task_list:
            print_title(f'{self.name}: task {task} will be run')
            return True
        else:
            print_title(f'{self.name}: task {task} will be skipped')
            return False
    
    @staticmethod
    def whether_run_decorator(func):
        """
        装饰器：根据函数名自动判断是否执行
        
        子类中的使用方式为:
        @FlexibleTool.whether_run_decorator
        def some_task(self, ...):
            ...
        """
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if self.whether_run_this_task(func.__name__):
                return func(self, *args, **kwargs)
            return None
        return wrapper


class Analyzer(FlexibleTool):
    '''
    AbstractTool中有一个示例代码
    '''
    def __init__(self):
        super().__init__()


class Visualizer(FlexibleTool):
    '''
    Visualizer也可以持有data_keeper,万一需要存一些画图时候产生的指标等;虽然不建议在画图的时候运算,而是分离到analyzer,但这么做提供了可能性

    区别于single visualizer等,这里的visualizer是pipeline中的一个tool,其可以持有single visualizer等
    
    AbstractTool中有一个示例代码
    '''
    def __init__(self):
        super().__init__()
        self.save_fig_kwargs = {}
    
    def auto_save_fig(self, filename=None, save_fig_kwargs=None, fig=None):
        '''
        自动按照调用者的函数名保存fig
        '''
        if filename is None:
            filename = pj(self.figs_dir, sys._getframe(1).f_code.co_name)
        local_save_fig_kwargs = self.save_fig_kwargs.copy()
        local_save_fig_kwargs = update_dict(local_save_fig_kwargs, save_fig_kwargs)
        save_fig(fig, filename, **local_save_fig_kwargs)


class ToolsPipeLine:
    '''
    只允许self._pipeline中的第一个tool寻找参数和确定timedir
    所有tool共享dir_manager

    而tools_pipeline一般不会有子类,也就不需要name
    '''
    def __init__(self):
        self._pipeline = []
        self.timedir_injected = False

    def set_timedir(self, timedir):
        self.timedir = timedir
        self.timedir_injected = True

    def inject_tools(self, dir_manager, code_saver, *tools):
        self.dir_manager = dir_manager
        self.code_saver = code_saver
        name_list = []
        for tool in tools:
            tool.input_dir_manager(dir_manager)
            self._pipeline.append(tool)
            name_list.append(tool.name)
        if len(set(name_list)) != len(name_list):
            raise ValueError('Tool names must be unique, found duplicates: {}'.format(name_list))

    def set_dir_manager_timedir_by_first_tool(self):
        tool = self._pipeline[0]
        self.dir_manager.set_timedir(self.timedir, prefix=tool.name)

    def search_params_for_first_tool(self):
        tool = self._pipeline[0]
        tool.finalize_init_dir_manager() # 只有第一个有资格以他的信息来init_dir_manager
        tool.search_params_when_enabled()

    def init_info_container_for_each_tool(self):
        for tool in self._pipeline:
            tool.init_info_container()

    def init_data_keeper_for_each_tool(self):
        for tool in self._pipeline:
            tool.init_data_keeper()

    def load_params_for_each_tool(self):
        for tool in self._pipeline:
            tool.load_params()

    def load_info_container_for_each_tool(self):
        for tool in self._pipeline:
            tool.info_container.load()

    def mark_already_done_for_each_tool(self):
        for tool in self._pipeline:
            tool.already_done = True

    def check_results_for_each_tool(self):
        for tool in self._pipeline:
            if tool.already_done:
                pass # 有可能已经手动设置了already_done,所以不需要再检查,再检查反而有可能得到False结果,因为检查取决于标志文件是否存在,而对于enable_skip的tool,才会保存标志文件
            else:
                tool.check_all_saved()

    def before_run(self):
        if self.timedir_injected:
            self.set_dir_manager_timedir_by_first_tool()
            self.load_params_for_each_tool()
            self.init_info_container_for_each_tool()
            self.load_info_container_for_each_tool()
            self.init_data_keeper_for_each_tool()
            self.mark_already_done_for_each_tool()
        else:
            self.search_params_for_first_tool()
            self.init_info_container_for_each_tool()
            self.init_data_keeper_for_each_tool()
            self.check_results_for_each_tool()
            self.code_saver.save_code()

    def run_detail(self):
        for i, tool in enumerate(self._pipeline):
            tool.run()
            if i < len(self._pipeline) - 1:
                for downstream_tool in self._pipeline[i + 1:]:
                    downstream_tool.input_previous_info(tool.propagate_info())

    def after_run(self):
        # 收集所有tool的data_keeper和params
        self.data_keeper_dict = {}
        self.params_dict = {}
        for tool in self._pipeline:
            self.data_keeper_dict[tool.name] = tool.data_keeper
            self.params_dict[tool.name] = tool.params

        # 删除设定为enable_delete_after_pipeline的tool的结果
        for tool in self._pipeline:
            tool.delete_after_pipeline_if_enabled()

    def run(self):
        self.before_run()
        self.run_detail()
        self.after_run()


class Experiment(abc.ABC):
    '''
    使用场景:
    1. 创建新文件夹并运行所有实验
        basedir = '../../results'
        basedir = os.path.join(basedir, os.path.splitext(os.path.basename(cf.current_file()))[0])

        experiment = Experiment()
        experiment.set_tool_params_dict(tool_params_dict)
        experiment.set_basedir(basedir)
        experiment.set_code_file_list([cf.current_file()])
        experiment.run()
    2. 寻找已有的文件夹,并运行实验中尚未完成的部分(注意第一个tool需要开启enable_search)
        basedir = '../../results'
        basedir = os.path.join(basedir, os.path.splitext(os.path.basename(cf.current_file()))[0])

        experiment = Experiment()
        experiment.set_tool_params_dict(tool_params_dict)
        experiment.set_basedir(basedir)
        experiment.set_code_file_list([cf.current_file()])
        experiment.run()
    3. 设置current_time,将运行在一个指定time的文件夹中(当你不想要current_time这个子目录时,设置为'')
        basedir = '../../results'
        basedir = os.path.join(basedir, os.path.splitext(os.path.basename(cf.current_file()))[0])

        experiment = Experiment()
        experiment.set_tool_params_dict(tool_params_dict)
        experiment.set_basedir(basedir)
        experiment.set_code_file_list([cf.current_file()])
        experiment.set_current_time('2025_11_11_12_00_00') # 甚至可以是experiment.set_current_time('best')这种模式,消除每次时间戳的差异
        experiment.run()
    4. 只手动设置了运行完成的文件夹,不运行,但表现和运行结束完全一致(自动获取params和data_keeper等)
        experiment = Experiment()
        experiment.load('../../results/2025_11_11_12_00_00')
        experiment.tool1.data_keeper.get_value('some_key') # 后续可随意获取tool的params和data_keeper等

    注意:
    子类一般要结合特定的tool,这些新tool在设计时需要继承AbstractTool,并实现其run_detail方法;可以修改data_keeper的储存方式;dir_manager的设置方式等

    如果希望用于experiment pipeline,并希望可以替换某个experiment,需要保证替换的experiment之间name一致
    '''
    def __init__(self):
        self.timedir_injected = False
        self._set_name()  # 子类必须实现_set_name方法,用于设置name属性
        self.code_file_list = []
        self.tool_config_dict = {}  # 不必须输入,所以设置默认值
        self.previous_experiment_name_list = []  # 用于compose多个experiment,记录其名称

    @abc.abstractmethod
    def _set_name(self):
        pass

    def set_tool_params_dict(self, tool_params_dict):
        self.tool_params_dict = tool_params_dict

    def set_tool_config_dict(self, tool_config_dict):
        '''
        key: tool_name
        value: config_dict

        示例:
        config_dict = {
            'enable_delete_after_pipeline': True,  # 是否在pipeline结束后删除结果
            }
        tool_config_dict = {
            'simulator': config_dict,
            }
        示例:
        config_dict = {
            'already_done': True,  # 手动设置已经完成,不需要再运行
            }
        tool_config_dict = {
            'simulator': config_dict,
            }
        将会在_finalize_init_tools方法中将config_dict中的配置参数设置到tool实例中
        '''
        self.tool_config_dict = tool_config_dict

    def set_timedir(self, timedir):
        self.timedir = timedir
        self.timedir_injected = True

    def set_current_time(self, current_time):
        self.current_time = current_time

    def set_previous_timedir(self, previous_timedir):
        '''
        用于compose多个experiment,可以从后往前寻找timedir
        '''
        save_pkl(previous_timedir, pj(self.dir_manager.params_dir, 'previous_timedir'))
        save_txt(previous_timedir, pj(self.dir_manager.params_dir, 'previous_timedir'))
        self.previous_timedir = previous_timedir

    def get_previous_timedir(self):
        self._init_tools()
        self._init_dir_manager()
        self.previous_timedir = load_pkl(pj(self.dir_manager.params_dir, 'previous_timedir'))
        return self.previous_timedir

    def input_previous_info(self, info):
        name = info['name']
        for k, v in info.items():
            if k == 'name':
                self.previous_experiment_name_list.append(v)
            else:
                setattr(self, f'{name}_{k}', v)

    def propagate_info(self):
        info = {'data_keeper_dict': self.pipeline.data_keeper_dict,
                'params_dict': self.pipeline.params_dict,
                'name': self.name}
        return info

    def load(self, timedir):
        self.set_timedir(timedir)
        self.run()

    def set_basedir(self, basedir):
        self.basedir = basedir

    def _init_dir_manager(self):
        self.dir_manager = ParamBasedDirManager()
        if self.timedir_injected:
            # 根据pipeline的第一个tool获取params和prefix来设置timedir
            self.dir_manager.set_timedir(self.timedir, prefix=self.tools[0].name)
        else:
            self.dir_manager.update_kwargs(basedir=self.basedir)
        if hasattr(self, 'current_time'):
            self.dir_manager.set_current_time(self.current_time, update_timedir=False)  # 不更新timedir,因为dir_manager还没有拿到params,不能产生before_timedir和after_timedir
            for tool in self.tools:
                tool.enable_search = False  # 如果手动设置了current_time,则不需要搜索参数
            print_title(f'{self.name}: Set current time to {self.current_time}, disable search_params for all tools')

    def set_code_file_list(self, code_file_list):
        self.code_file_list = code_file_list

    def _init_code_saver(self):
        self.code_saver = CodeSaver(self.dir_manager)
        self.code_saver.set_code_file_list(self.code_file_list)

    @abc.abstractmethod
    def _minimal_init_tools(self):
        '''
        比如,创建Trainer,Simulator,Analyzer等工具实例,并将它们添加到self.tools中

        注意:
        1. 工具名称应与实例变量名一致(如工具名'trainer'对应实例 tool.name=='trainer')
        2. 不推荐将工具实例直接赋值给self属性,如避免 self.trainer = Trainer(), _finalize_init_tools 方法后续会自动将每个工具注入为当前实验对象的属性
        3. 只需关注工具实例创建和列表添加操作,参数配置将由后续流程处理
        4. 注意要放入的是实例,而不是类,如Simulator(),而不是Simulator
        '''
        self.tools = []

    def _finalize_init_tools(self):
        for tool in self.tools:
            if self.timedir_injected:
                pass # 无需从外部获取params
            else:
                tool.set_params(self.tool_params_dict[tool.name])
                config_dict = self.tool_config_dict.get(tool.name, {})
                for key, value in config_dict.items():
                    setattr(tool, key, value)  # 设置tool的配置参数
            tool._config_data_keeper()
            setattr(self, tool.name, tool)  # 将tool设置为experiment的属性,方便后续访问
        
            for previous_experiment_name in self.previous_experiment_name_list:
                setattr(tool, f'{previous_experiment_name}_data_keeper_dict', getattr(self, f'{previous_experiment_name}_data_keeper_dict', {}))
                setattr(tool, f'{previous_experiment_name}_params_dict', getattr(self, f'{previous_experiment_name}_params_dict', {}))

    def _init_tools(self):
        self._minimal_init_tools()
        self._finalize_init_tools()

    def _minimal_init_pipeline(self):
        self.pipeline = ToolsPipeLine()

    def _finalize_init_pipeline(self):
        self.pipeline.inject_tools(
            self.dir_manager,
            self.code_saver,
            *self.tools
        )
        if self.timedir_injected:
            self.pipeline.set_timedir(self.timedir)

    def _init_pipeline(self):
        self._minimal_init_pipeline()
        self._finalize_init_pipeline()

    def before_run(self):
        self._init_tools()
        self._init_dir_manager() # dir_manager依赖第一个tool的params和prefix
        self._init_code_saver()
        self._init_pipeline()
        self.pipeline.before_run()

    def run_detail(self):
        self.pipeline.run_detail()

    def after_run(self):
        self.pipeline.after_run()

    def run(self):
        self.before_run()
        self.run_detail()
        self.after_run()


class ExperimentPipeLine:
    '''
    包含多个experiment,后面的experiment必须依赖前面的experiment的结果(否则,可创建多个experiment实例).此时,手动设置timedir会设置最后一个experiment,前面的experiment会自动获取timedir
    '''
    def __init__(self):
        self.experiments = []
        self.timedir_injected = False

    def set_timedir(self, timedir):
        self.timedir = timedir
        self.timedir_injected = True

    def inject_experiments(self, *experiments):
        name_list = []
        for experiment in experiments:
            self.experiments.append(experiment)
            name_list.append(experiment.name)
        if len(set(name_list)) != len(name_list):
            raise ValueError('Experiment names must be unique, found duplicates: {}'.format(name_list))

    def set_dir_manager_timedir_in_reverse_order(self):
        for i, experiment in enumerate(self.experiments[::-1]):
            if i == 0:
                experiment.set_timedir(self.timedir)
            else:
                experiment.set_timedir(previous_timedir)
            if i < len(self.experiments) - 1:
                previous_timedir = experiment.get_previous_timedir()

    def before_run(self):
        if self.timedir_injected:
            self.set_dir_manager_timedir_in_reverse_order()
        else:
            pass

    def run_detail(self):
        for i, experiment in enumerate(self.experiments):
            experiment.run()
            if i > 0: # 必须从前面获取,而不是向后面传递,后面的experiment尚未创建dir_manager
                experiment.set_previous_timedir(self.experiments[i-1].dir_manager.timedir)
            if i < len(self.experiments) - 1:
                for downstream_experiment in self.experiments[i + 1:]:
                    downstream_experiment.input_previous_info(
                        experiment.propagate_info()
                    )

    def after_run(self):
        pass

    def run(self):
        self.before_run()
        self.run_detail()
        self.after_run()


class ComposedExperiment(abc.ABC):
    '''
    使用场景:
    1. 创建新文件夹并运行所有实验
        basedir = '../../results'
        basedir = os.path.join(basedir, os.path.splitext(os.path.basename(cf.current_file()))[0])

        composed_experiment = ComposedExperiment()
        composed_experiment.set_experiment_params_dict(experiment_params_dict)
        composed_experiment.set_basedir(basedir)
        composed_experiment.set_code_file_list([cf.current_file()])
        composed_experiment.run()
    2. 寻找已有的文件夹,并运行实验中尚未完成的部分
        basedir = '../../results'
        basedir = os.path.join(basedir, os.path.splitext(os.path.basename(cf.current_file()))[0])

        composed_experiment = ComposedExperiment()
        composed_experiment.set_experiment_params_dict(experiment_params_dict)
        composed_experiment.set_basedir(basedir)
        composed_experiment.set_code_file_list([cf.current_file()])
        composed_experiment.run()
    3. 设置current_time,将运行在一个指定time的文件夹中
        basedir = '../../results'
        basedir = os.path.join(basedir, os.path.splitext(os.path.basename(cf.current_file()))[0])

        composed_experiment = ComposedExperiment()
        composed_experiment.set_experiment_params_dict(experiment_params_dict)
        composed_experiment.set_basedir(basedir)
        composed_experiment.set_code_file_list([cf.current_file()])
        composed_experiment.set_current_time('2025_11_11_12_00_00') # 甚至可以是composed_experiment.set_current_time('best')这种模式,消除每次时间戳的差异
        composed_experiment.run()
    4. 只手动设置了运行完成的文件夹,不运行,但表现和运行结束完全一致(自动获取params和data_keeper等)
        composed_experiment = ComposedExperiment()
        composed_experiment.load('../../results/2025_11_11_12_00_00') # 注意这里输入的文件夹是最后一个experiment的timedir
        composed_experiment.experiment1.tool1.data_keeper.get_value('some_key') # 后续可随意获取experiment的属性
    '''
    def __init__(self):
        self.experiments = []
        self.timedir_injected = False
        self.experiment_config_dict = {}  # 不必须输入,所以设置默认值

    def set_experiment_params_dict(self, experiment_params_dict):
        self.experiment_params_dict = experiment_params_dict

    def set_experiment_config_dict(self, experiment_config_dict):
        self.experiment_config_dict = experiment_config_dict

    def set_timedir(self, timedir):
        self.timedir = timedir
        self.timedir_injected = True

    def set_current_time(self, current_time):
        self.current_time = current_time

    def load(self, timedir):
        self.set_timedir(timedir)
        self.run()

    def set_basedir(self, basedir):
        self.basedir = basedir

    def set_code_file_list(self, code_file_list):
        self.code_file_list = code_file_list

    @abc.abstractmethod
    def _minimal_init_experiments(self):
        '''
        与_minimal_init_tools同理
        '''
        self.experiments = []

    def _finalize_init_experiments(self):
        for experiment in self.experiments:
            if self.timedir_injected:
                pass
            else:
                experiment.set_tool_params_dict(self.experiment_params_dict[experiment.name])
                experiment.set_tool_config_dict(self.experiment_config_dict.get(experiment.name, {}))
                experiment.set_basedir(pj(self.basedir, experiment.name))
                experiment.set_code_file_list(self.code_file_list)
                if hasattr(self, 'current_time'):
                    experiment.set_current_time(self.current_time)
            setattr(self, experiment.name, experiment)  # 将experiment设置为composed_experiment的属性,方便后续访问

    def _init_experiments(self):
        self._minimal_init_experiments()
        self._finalize_init_experiments()

    def _minimal_init_pipeline(self):
        self.pipeline = ExperimentPipeLine()

    def _finalize_init_pipeline(self):
        self.pipeline.inject_experiments(*self.experiments)
        if self.timedir_injected:
            self.pipeline.set_timedir(self.timedir)

    def _init_pipeline(self):
        self._minimal_init_pipeline()
        self._finalize_init_pipeline()

    def before_run(self):
        self._init_experiments()
        self._init_pipeline()
        self.pipeline.before_run()

    def run_detail(self):
        self.pipeline.run_detail()

    def after_run(self):
        self.pipeline.after_run()

    def run(self):
        self.before_run()
        self.run_detail()
        self.after_run()


def find_incomplete_experiment(dir_before_timedir, tool_name, dir_after_timedir=None):
    '''
    寻找不完整的模型文件夹,以输入的tool_name对应的tool的f'{tool_name}_results_saved'是否存在为判断标准
    '''
    incomplete_experiment_list = []

    sub_dir_list = get_subdir(dir_before_timedir)
    for sub_dir in sub_dir_list:
        mark_file = pj(sub_dir, dir_after_timedir, 'outcomes', tool_name, f'{tool_name}_results_saved')
        if check_all_file_exist_with_any_extension(mark_file):
            pass
        else:
            experiment_dir = pj(sub_dir, dir_after_timedir)
            print(f'Found incomplete experiment directory: {experiment_dir}')
            incomplete_experiment_list.append(experiment_dir)
    return incomplete_experiment_list


def clean_incomplete_experiment(dir_before_timedir, tool_name, dir_after_timedir=None):
    '''
    删除不完整的模型文件夹,以输入的tool_name对应的tool的f'{tool_name}_results_saved'是否存在为判断标准
    '''
    print('Warning: The following incomplete experiment directories will be removed')
    incomplete_experiment_list = find_incomplete_experiment(dir_before_timedir, tool_name, dir_after_timedir)
    print('You still have 1 minute to cancel this operation')
    time.sleep(60)
    for incomplete_experiment in incomplete_experiment_list:
        rmdir(incomplete_experiment)


class ExperimentContainer(InstanceContainer):
    '''
    获取一个目录下的所有实验,功能如下:
    统计参数

    注意:
    兼容Experiment和ComposedExperiment
    '''
    def __init__(self, experiment_class, dir_before_timedir, dir_after_timedir=None):
        sub_dir_list = get_subdir(dir_before_timedir)
        instance_list = []
        for sub_dir in sub_dir_list:
            timedir = pj(sub_dir, dir_after_timedir)
            with FlexibleTry(enable_try=True):
                instance = experiment_class()
                instance.load(timedir)
                instance_list.append(instance)
        super().__init__(instance_list)

    def count_params_by_key(self, key, tool_name=None, experiment_name=None):
        count_dict = defaultdict(int)
        for instance in self.instance_list:
            if isinstance(instance, ComposedExperiment):
                experiment = instance.experiments[0] if experiment_name is None else getattr(instance, experiment_name)
            elif isinstance(instance, Experiment):
                experiment = instance
            tool = experiment.tools[0] if tool_name is None else getattr(experiment, tool_name)
            value = tool.params[key]
            count_dict[repr(value)] += 1
        return count_dict
# endregion


# region 参数优化相关函数
def search_optimal_param():
    pass
# endregion


# region 稀疏矩阵相关函数
# region coo
def binary_coo(row_indices, col_indices, shape):
    '''
    Create a binary COO matrix based on the specified row and column indices.
    :param row_indices: The row indices of the non-zero elements
    :param col_indices: The column indices of the non-zero elements
    :param shape: The shape of the COO matrix
    :return: The binary COO matrix
    '''
    return coo_matrix((np.ones_like(row_indices), (row_indices, col_indices)), shape=shape)


def rm_coo(coo, row_indices, col_indices):
    '''
    Remove the elements that are both in the specified row and column indices from a COO matrix.
    :param coo: The original COO matrix
    :param row_indices: The row indices to be considered
    :param col_indices: The column indices to be considered
    :return: The COO matrix after removing the specified elements
    '''
    mask = ~np.isin(coo.row, row_indices) | ~np.isin(coo.col, col_indices)
    return coo_matrix((coo.data[mask], (coo.row[mask], coo.col[mask])), shape=coo.shape)


def rm_row_coo(coo, row_indices):
    '''
    Remove the specified rows from a COO matrix.
    :param coo: The original COO matrix
    :param row_indices: The row indices to be removed
    :return: The COO matrix after removing the specified rows
    '''
    mask = ~np.isin(coo.row, row_indices)
    return coo_matrix((coo.data[mask], (coo.row[mask], coo.col[mask])), shape=coo.shape)


def rm_col_coo(coo, col_indices):
    '''
    Remove the specified columns from a COO matrix.
    :param coo: The original COO matrix
    :param col_indices: The column indices to be removed
    :return: The COO matrix after removing the specified columns
    '''
    mask = ~np.isin(coo.col, col_indices)
    return coo_matrix((coo.data[mask], (coo.row[mask], coo.col[mask])), shape=coo.shape)


def rm_intersect_row_col_coo(coo, row_indices, col_indices):
    '''
    Remove the elements that are both in the specified row and column indices from a COO matrix.
    :param coo: The original COO matrix
    :param row_indices: The row indices to be considered
    :param col_indices: The column indices to be considered
    :return: The COO matrix after removing the specified elements
    '''
    mask = ~(np.isin(coo.row, row_indices) & np.isin(coo.col, col_indices))
    return coo_matrix((coo.data[mask], (coo.row[mask], coo.col[mask])), shape=coo.shape)
# endregion


# region csr
def array_to_csr(array):
    '''
    Convert a 2D array to a CSR matrix.
    :param array: The 2D array to be converted
    :return: The CSR matrix
    '''
    return csr_matrix(array)


def get_csr_idx(csr):
    '''
    获取CSR矩阵的行索引和列索引

    注意:
        这个函数只会返回非零元素的行列索引, 比较符合直观。
    '''
    row_indices, col_indices = csr.nonzero()
    return row_indices, col_indices


def get_csr_indices_indprt(csr):
    """
    获取CSR矩阵的indices和indptr数组。

    参数:
        csr_matrix: 输入的CSR矩阵。

    返回值:
        indices和indptr数组的元组。

    注意:
        对于某些矩阵中看起来是零的元素, 他也有可能在indices和indptr中出现。
    """
    return csr.indices, csr.indptr


def binary_csr(row_indices, col_indices, shape):
    '''
    基于row_indices和col_indices创建一个CSR矩阵,每个连接将在零的基础上加1。

    注意:如果row_indices和col_indices中有重复的元素,则这些元素将被累加。
    '''
    return csr_matrix((np.ones_like(row_indices), (row_indices, col_indices)), shape=shape)


def rm_csr(csr, row_indices, col_indices):
    '''
    Remove the elements that are both in the specified row and column indices from a CSR matrix.
    :param csr: The original CSR matrix
    :param row_indices: The row indices to be considered
    :param col_indices: The column indices to be considered
    :return: The CSR matrix after removing the specified elements
    '''
    coo = csr.tocoo()
    mask = ~np.isin(coo.row, row_indices) | ~np.isin(coo.col, col_indices)
    return coo_matrix((coo.data[mask], (coo.row[mask], coo.col[mask])), shape=coo.shape).tocsr()


def rm_row_csr(csr, row_indices):
    '''
    Remove the specified rows from a CSR matrix.
    :param csr: The original CSR matrix
    :param row_indices: The row indices to be removed
    :return: The CSR matrix after removing the specified rows
    '''
    coo = csr.tocoo()
    mask = ~np.isin(coo.row, row_indices)
    return coo_matrix((coo.data[mask], (coo.row[mask], coo.col[mask])), shape=coo.shape).tocsr()


def rm_col_csr(csr, col_indices):
    '''
    Remove the specified columns from a CSR matrix.
    :param csr: The original CSR matrix
    :param col_indices: The column indices to be removed
    :return: The CSR matrix after removing the specified columns
    '''
    coo = csr.tocoo()
    mask = ~np.isin(coo.col, col_indices)
    return coo_matrix((coo.data[mask], (coo.row[mask], coo.col[mask])), shape=coo.shape).tocsr()


def rm_intersect_row_col_csr(csr, row_indices, col_indices):
    '''
    Remove the elements that are both in the specified row and column indices from a CSR matrix.
    :param csr: The original CSR matrix
    :param row_indices: The row indices to be considered
    :param col_indices: The column indices to be considered
    :return: The CSR matrix after removing the specified elements
    '''
    coo = csr.tocoo()
    mask = ~(np.isin(coo.row, row_indices) & np.isin(coo.col, col_indices))
    return coo_matrix((coo.data[mask], (coo.row[mask], coo.col[mask])), shape=coo.shape).tocsr()
# endregion
# endregion


# region 作图相关函数
# region 泛用函数(ax iterable,decorator,squeeze,unsqueeze)
def get_iterable_ax(ax):
    '''
    不会改变原有输入的ax,返回一个可以迭代的ax
    '''
    if isinstance(ax, np.ndarray):
        return ax.flatten()
    elif isinstance(ax, list):
        return flatten_list(pure_list(ax))
    elif isinstance(ax, dict):
        return list(ax.values())
    else:
        return [ax]


def get_iterable_ax_for_decorator(ax):
    '''
    不会改变原有输入的ax,返回一个可以迭代的ax,但是当ax是单个ax时,返回本体(用于decorator时,可以保证输出不会因为这里搞了iterable而把单个ax的结果包装成list)
    '''
    if isinstance(ax, np.ndarray):
        return ax.flatten(), True
    elif isinstance(ax, list):
        return flatten_list(pure_list(ax)), True
    elif isinstance(ax, dict):
        return list(ax.values()), True
    else:
        return ax, False


def iterate_over_obj(func, obj_name):
    """
    装饰器：将 obj_name 参数转换为可迭代对象，并对每个元素调用原始函数。
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 获取 obj_name 参数的值
        obj = get_param_value(func, args, kwargs, obj_name)
        iterable_obj, is_iterable = get_iterable_ax_for_decorator(obj)  # 将 obj 转换为可迭代对象
        if is_iterable:
            results = []

            # 获取函数签名，并动态绑定位置参数和关键字参数
            signature = inspect.signature(func)

            # 遍历 iterable_obj 的每个元素，并替换 obj_name 参数
            for item in iterable_obj:
                # 重新绑定参数，将 obj_name 替换为 item
                bound_args = signature.bind(*args, **kwargs)
                bound_args.apply_defaults()  # 应用默认参数

                # 将 obj_name 参数替换为当前的 item
                bound_args.arguments[obj_name] = item

                # 调用原始函数，传入修改后的参数
                result = func(*bound_args.args, **bound_args.kwargs)
                results.append(result)

            # 重构结果，使其与原来的 obj 形状相匹配
            return rebuild_ax(results, obj)
        else:
            return func(*args, **kwargs)

    return wrapper


def iterate_over_axs(func):
    return iterate_over_obj(func, 'ax')


def iterate_over_subfigs(func):
    return iterate_over_obj(func, 'subfig')


def rebuild_ax(flatten_ax, original_ax):
    '''
    根据 original_ax 的结构，还原扁平化的 flatten_ax。
    '''
    if isinstance(original_ax, np.ndarray):
        # 将扁平化的 ax 重塑为原有 ndarray 的形状
        result = np.empty(original_ax.shape, dtype=object)
        if original_ax.ndim == 1:
            for i in range(len(flatten_ax)):
                result[i] = flatten_ax[i]
        elif original_ax.ndim == 2:
            for i in range(original_ax.shape[0]):
                for j in range(original_ax.shape[1]):
                    result[i, j] = flatten_ax[i * original_ax.shape[1] + j]
        return result
    elif isinstance(original_ax, list):
        # 根据 list 的结构，递归还原
        return rebuild_list(flatten_ax, original_ax)
    elif isinstance(original_ax, dict):
        # 将 flatten_ax 的值与 dict 的 keys 重新组合
        return dict(zip(original_ax.keys(), flatten_ax))
    else:
        # 返回原有的 ax
        return flatten_ax


def squeeze_ax(ax):
    '''
    将array的ax压缩

    当然,也可以用于subfig的压缩
    '''
    if isinstance(ax, np.ndarray):
        if ax.size == 1:
            if ax.ndim == 2:
                return ax[0, 0]
            elif ax.ndim == 1:
                return ax[0]
        else:
            return np.squeeze(ax)
    else:
        return ax


def unsqueeze_ax(ax, ncols=1, nrows=1):
    '''
    将ax变成二维的array

    当然,也可以用于subfig的还原
    '''
    if isinstance(ax, np.ndarray):
        return np.reshape(ax, (nrows, ncols))
    else:
        return np.array([[ax]])
# endregion


# region 初级作图函数(matplotlib系列,输入向量使用)
def plt_scatter(ax, x, y, label=None, color=BLUE, vert=True, rasterized=False, rasterized_threshold=10000, xlim=None, ylim=None, linewidths=0., **kwargs):
    '''
    使用x和y绘制散点图,可以接受plt.scatter的其他参数
    :param ax: matplotlib的轴对象,用于绘制图形
    :param x: x轴的数据
    :param y: y轴的数据
    :param label: 图例标签,默认为None
    :param color: 散点图的颜色,默认为BLUE
    :param vert: 是否为垂直散点图,默认为True,即纵向
    :param rasterized: 是否对图像进行栅格化处理,默认为False(如果点数特别多,导致图像过大时,可以考虑开启)
    :param rasterized_threshold: 栅格化处理的阈值,默认为10000,当点数超过这个值时,会进行栅格化处理(如果不需要这个自动处理,可以设置为None或者False)
    :param xlim: x轴的范围,默认为None (用于仅仅绘画xlim范围内的点,而不是设定x轴范围)
    :param ylim: y轴的范围,默认为None
    :param linewidths: 线宽,默认为0. (防止点的s值小于linewidths时,出现空心点的情况)
    :param kwargs: 其他plt.scatter支持的参数

    注意
    -s是marker_size的平方
    -s是圆的面积,radius按照points的单位
    '''
    # 根据xlim和ylim预处理数据
    if xlim or ylim:
        filtered_data = [
            (xi, yi) for xi, yi in zip(x, y)
            if (xlim is None or xlim[0] <= xi <= xlim[1]) and 
               (ylim is None or ylim[0] <= yi <= ylim[1])
        ]
        x, y = zip(*filtered_data) if filtered_data else ([], [])

    if not vert:
        x, y = y, x

    if (rasterized_threshold is not None) and (rasterized_threshold is not False):
        if isinstance(x, np.ndarray) or isinstance(x, list):
            if len(x) > rasterized_threshold:
                rasterized = True

    # 画图
    if 'c' in kwargs:
        local_kwargs = kwargs.copy()
        local_kwargs['c'] = pure_list(local_kwargs['c'])
        if len(list_shape(local_kwargs['c'])) == 1:
            local_kwargs['c'] = [local_kwargs['c']]
        # 不输入color参数
        return ax.scatter(x, y, label=label, rasterized=rasterized, linewidths=linewidths, **local_kwargs)
    else:
        return ax.scatter(x, y, label=label, rasterized=rasterized, color=color, linewidths=linewidths, **kwargs)


def plt_line(ax, x, y, label=None, color=BLUE, vert=True, xlim=None, ylim=None, **kwargs):
    '''
    使用x和y绘制折线图,可以接受plt.plot的其他参数
    :param ax: matplotlib的轴对象,用于绘制图形
    :param x: x轴的数据
    :param y: y轴的数据
    :param label: 图例标签,默认为None
    :param color: 折线图的颜色,默认为BLUE
    :param vert: 是否为垂直折线图,默认为True,即纵向
    :param xlim: x轴的范围,默认为None (用于仅仅绘画xlim范围内的点,而不是设定x轴范围)
    :param ylim: y轴的范围,默认为None
    :param kwargs: 其他plt.plot支持的参数
    '''
    # 根据xlim和ylim预处理数据
    if xlim or ylim:
        filtered_data = [
            (xi, yi) for xi, yi in zip(x, y)
            if (xlim is None or xlim[0] <= xi <= xlim[1]) and 
               (ylim is None or ylim[0] <= yi <= ylim[1])
        ]
        x, y = zip(*filtered_data) if filtered_data else ([], [])
    
    # 画图
    if not vert:
        x, y = y, x
    return ax.plot(x, y, label=label, color=color, **kwargs)


def plt_bar(ax, x, y, label=None, color=BLUE, vert=True, equal_space=False, err=None, capsize=PLT_CAP_SIZE, ecolor=BLACK, elabel=None, width=BAR_WIDTH, **kwargs):
    '''
    使用x和y绘制柱状图，可以接受plt.bar的其他参数,此函数的特性是会根据x的值作为bar的位置,当x包含字符串或者equal_space=True时,会自动变成等距离排列。
    :param ax: matplotlib的轴对象，用于绘制图形
    :param x: x轴的数据
    :param y: y轴的数据
    :param label: 图例标签，默认为None
    :param color: 柱状图的颜色，默认为BLUE
    :param vert: 是否为垂直柱状图，默认为True，即纵向
    :param equal_space: 是否将x的值作为字符串处理，这将使得柱子等距排列，默认为False
    :param err: 误差线的数据，默认为None
    :param capsize: 误差线帽大小，默认为PLT_CAP_SIZE
    :param ecolor: 误差线颜色，默认为BLACK
    :param elabel: 误差线图例标签，默认为None
    :param width: 柱子宽度，默认为None(当vert=False时,此参数将自动赋值给height)
    :param kwargs: 其他plt.bar或plt.barh支持的参数
    '''
    if equal_space:
        # 将x的每个元素变为字符串
        x = [str(i) for i in x]

    # 添加elabel
    if err is not None:
        add_errorbar(ax, x, y, err, vert=vert, label=elabel,
                     color=ecolor, capsize=capsize, **kwargs)
    
    if vert:
        # 绘制垂直柱状图
        return ax.bar(x, y, label=label, color=color, yerr=err, capsize=capsize, ecolor=ecolor, width=width, **kwargs)
    else:
        # 绘制水平柱状图
        return ax.barh(x, y, label=label, color=color, xerr=err, capsize=capsize, ecolor=ecolor, height=width, **kwargs)


def plt_stair(ax, x, y, label=None, color=BLUE, **kwargs):
    '''
    使用x和y绘制阶梯图,可以接受plt.stairs的其他参数(注意x的长度比y的长度多1,因为阶梯图是根据x的值作为边界绘制的)
    :param ax: matplotlib的轴对象,用于绘制图形
    :param x: x轴的数据
    :param y: y轴的数据
    :param label: 图例标签,默认为None
    :param color: 折线图的颜色,默认为BLUE
    :param kwargs: 其他plt.stairs支持的参数
    '''
    return ax.stairs(y, x, label=label, color=color, **kwargs)


def plt_stack(ax, x, y, labels=None, colors=None, **kwargs):
    '''
    使用x和y绘制堆叠图,可以接受plt.stackplot的其他参数
    :param ax: matplotlib的轴对象,用于绘制图形
    :param x: x轴的数据
    :param y: y轴的数据,可以是一个2D数组
    :param labels: 图例标签,默认为None
    :param colors: 颜色列表,默认为CMAP中取色
    :param kwargs: 其他plt.stackplot支持的参数
    '''
    if colors is None:
        colors = [CMAP(i/y.shape[0]) for i in range(y.shape[0])]
    return ax.stackplot(x, *y, labels=labels, colors=colors, **kwargs)


def plt_violin(ax, data, positions=None, labels=None, body_colors=None, line_colors=None, line_width=LINE_WIDTH, vert=True, text_process=None, **kwargs):
    '''
    使用 data 绘制 violin plot。

    Parameters:
    ax (matplotlib.axes.Axes): 用于绘制图形的轴对象。
    data (array-like): 用于绘制 violin plot 的数据,可以是一个列表或 numpy 数组。
    positions (array-like, optional): 每个 violin plot 在 x 轴上的位置,默认为 range(len(data))。
    labels (list, optional): 每个 violin plot 的标签,默认为 None。
    body_colors (list, optional): 每个 violin plot 的颜色,默认为 None。
    line_colors (list, optional): 每个 violin plot 的线条颜色,默认为 None。
    vert (bool, optional): 如果为 True,则 violin plot 垂直绘制,否则水平绘制。默认为 True。
    **kwargs: 其他参数传递给 ax.violinplot。

    Returns:
    dict: 包含 violin plot 的各个元素的字典。
    '''
    if isinstance(data, list):
        data = np.array(data)
    if positions is None:
        positions = range(len(data))
    if labels is None:
        labels = [None] * len(data)
    if body_colors is None:
        body_colors = [BLUE for i in range(len(data))]
    if line_colors is None:
        line_colors = body_colors
    text_process = update_dict(TEXT_PROCESS, text_process)

    vp = ax.violinplot(data, positions, vert=vert, **kwargs)

    # 设置 violin plot 的颜色和标签
    for i, body in enumerate(vp['bodies']):
        body.set_color(body_colors[i])
        body.set_alpha(0.9)
    
    new_label = [format_text(label, text_process=text_process) for label in labels]
    if vert:
        ax.set_xticks(positions)
        ax.set_xticklabels(new_label)
    else:
        ax.set_yticks(positions)
        ax.set_yticklabels(new_label)

    for partname in ('cbars','cmins','cmaxes'):
        vp[partname].set_edgecolor(line_colors)
        vp[partname].set_linewidth(line_width)
    return vp


def plt_stem(ax, x, y, label=None, linefmt='-', markerfmt='o', basefmt='-', linecolor=BLUE, markercolor=BLUE, basecolor=BLACK, **kwargs):
    '''
    使用x和y绘制棉棒图,可以接受plt.stem的其他参数
    :param ax: matplotlib的轴对象,用于绘制图形
    :param x: x轴的数据
    :param y: y轴的数据
    :param label: 图例标签,默认为None
    :param linefmt: 折线的格式,默认为'-'
    :param markerfmt: 点的格式,默认为'o'
    :param basefmt: 基线的格式,默认为'k-'
    :param kwargs: 其他plt.stem支持的参数
    '''
    # 画图
    stem_container = ax.stem(x, y, label=label, linefmt=linefmt,
                             markerfmt=markerfmt, basefmt=basefmt, **kwargs)

    stem_container.stemlines.set_color(linecolor)
    # 注意这里可能是markerlines，取决于您的matplotlib版本
    stem_container.markerline.set_color(markercolor)
    stem_container.baseline.set_color(basecolor)
    return stem_container


def plt_quiver(ax, X, Y, U, V, color=BLUE, angles='xy', scale_units='xy', scale=5, width=0.015, **kwargs):
    '''
    使用X,Y,U,V绘制矢量场,可以接受plt.quiver的其他参数
    :param ax: matplotlib的轴对象,用于绘制图形
    :param X: x轴的数据
    :param Y: y轴的数据
    :param U: x轴方向的矢量
    :param V: y轴方向的矢量
    :param color: 箭头的颜色,默认为BLUE
    :param angles: 角度模式,默认为'xy'
    :param scale_units: 比例单位,默认为'xy'
    :param scale: 比例系数,默认为5
    :param width: 箭头的线宽,默认为0.015
    :param kwargs: 其他plt.quiver支持的参数
    '''
    return ax.quiver(X, Y, U, V, color=color, angles=angles,
                     scale_units=scale_units, scale=scale, width=width, **kwargs)


def plt_stream(ax, x, y, u, v, label=None, color=BLUE, density=1, broken_streamlines=True, **kwargs):
    '''
    使用 x, y 和 u, v 绘制流场图
    
    :param ax: matplotlib 的轴对象,用于绘制图形
    :param x: x 轴的坐标数据
    :param y: y 轴的坐标数据
    :param u: x 方向的流速分量
    :param v: y 方向的流速分量
    :param label: 图例标签,默认为 None
    :param color: 流线的颜色,默认为 BLUE
    :param density: 流线的密度,默认为 1
    :param broken_streamlines: 是否绘制断裂的流线,默认为 True
    :param kwargs: 其他 plt.streamplot 支持的参数
    '''
    if label is not None:
        ax.plot([], [], color=color, label=label)
    # 绘制流场图
    return ax.streamplot(x, y, u, v, color=color, density=density, broken_streamlines=broken_streamlines, **kwargs)


def plt_speed_stream(ax, x, y, u, v, label=None, color=BLUE, density=1, broken_streamlines=True, cmap=DENSITY_CMAP, cbar_kwargs=None, **kwargs):
    '''
    使用 x, y 和 u, v 绘制流场图,并叠加颜色编码图表示流场大小
    
    :param ax: matplotlib 的轴对象,用于绘制图形
    :param x: x 轴的坐标数据
    :param y: y 轴的坐标数据
    :param u: x 方向的流速分量
    :param v: y 方向的流速分量
    :param label: 图例标签,默认为 None
    :param color: 流线的颜色,默认为 BLUE
    :param density: 流线的密度,默认为 1
    :param broken_streamlines: 是否绘制断裂的流线,默认为 True
    :param cmap: 流速的颜色映射表,默认为 DENSITY_CMAP
    :param kwargs: 其他 plt.streamplot 和 plt.pcolormesh 支持的参数
    '''
    if label is not None:
        ax.plot([], [], color=color, label=label)
    if cbar_kwargs is None:
        cbar_kwargs = {}
    # 绘制流场图和绘制颜色编码图
    speed_mesh = ax.pcolormesh(x, y, np.sqrt(u**2 + v**2), cmap=cmap)
    return ax.streamplot(x, y, u, v, color=color, density=density, broken_streamlines=broken_streamlines, **kwargs), speed_mesh, add_side_colorbar(ax, speed_mesh, cmap=cmap, cbar_label='speed', **cbar_kwargs)


def plt_box(ax, x, y, width=BAR_WIDTH, label=None, patch_artist=True, boxprops=None, vert=True, **kwargs):
    '''
    使用x和y绘制箱形图,可以接受plt.boxplot的其他参数(和sns_box的输入方式完全不同,请注意并参考示例)
    :param ax: matplotlib的轴对象,用于绘制图形
    :param x: x轴的数据,应为一个列表或数组,其长度与y中的数据集合数量相匹配.示例:x = [1, 2, 3, 4]
    :param y: y轴的数据,每个位置的数据应该是一个列表或数组.示例:y = [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]] (注意就算画单个box,y也要是二维数组)
    :param label: 图例标签,默认为None
    :param patch_artist: 是否使用补丁对象,默认为True
    :param boxprops: 箱形图的属性
    :param vert: 是否为垂直箱形图,默认为True,即纵向
    :param kwargs: 其他plt.boxplot支持的参数

    注意:
    x不可以是单个数值,必须是一个列表或数组
    y必须是二维的,即使只有一个box,也要是二维的
    '''
    if boxprops is None:
        boxprops = dict(facecolor=BLUE)

    # 添加bar，用于添加图例，调整以支持横向箱形图
    if vert:
        ax.bar(x[0], 0, color=boxprops['facecolor'], label=label, bottom=y[0][0], **kwargs)
    else:
        ax.barh(x[0], 0, color=boxprops['facecolor'], label=label, left=y[0][0], **kwargs)

    # 画图
    return ax.boxplot(list(y), positions=x, patch_artist=patch_artist, boxprops=boxprops, vert=vert, widths=width, **kwargs)


def plt_hist(ax, data, bins=BIN_NUM, label=None, color=BLUE, stat='probability', vert=True, **kwargs):
    '''
    使用数据绘制直方图,可以接受plt.hist的其他参数
    :param ax: matplotlib的轴对象,用于绘制图形
    :param data: 用于绘制直方图的数据集
    :param bins: 直方图的箱数,默认为None,自动确定箱数
    :param label: 图例标签,默认为None
    :param color: 直方图的颜色,默认为BLUE
    :param stat: 直方图的统计方法,默认为'probability'
    :param vert: 是否为垂直直方图,默认为True,即纵向
    :param kwargs: 其他plt.hist支持的参数
    '''
    plt_hist_instance = PltHist(ax, data, bins=bins, label=label, color=color, stat=stat, vert=vert, **kwargs)
    return plt_hist_instance.bar


class PltHist:
    '''
    用于绘制直方图的类(方便后续获取hist,edge,mid_point等)
    '''
    def __init__(self, ax, data, bins=BIN_NUM, label=None, color=BLUE, stat='probability', vert=True, **kwargs):
        '''
        使用数据绘制直方图,可以接受plt.hist的其他参数
        :param ax: matplotlib的轴对象,用于绘制图形
        :param data: 用于绘制直方图的数据集
        :param bins: 直方图的箱数,默认为None,自动确定箱数
        :param label: 图例标签,默认为None
        :param color: 直方图的颜色,默认为BLUE
        :param stat: 直方图的统计方法,默认为'probability'
        :param vert: 是否为垂直直方图,默认为True,即纵向
        :param kwargs: 其他plt.hist支持的参数
        '''
        self.hist, self.edge, self.mid_point = get_hist(data, bins, stat=stat)
        set_ax(ax, ylabel=stat)
        self.bar = plt_bar(ax, self.mid_point, self.hist, label=label, color=color, width=self.edge[1]-self.edge[0], vert=vert, **kwargs)


def plt_hist_2d(ax, x, y, x_bins=BIN_NUM, y_bins=BIN_NUM, cmap=DENSITY_CMAP, label=None, stat='probability', cbar=True, cbar_position=None, cbar_label='stat', cbar_kwargs=None, vmin=None, vmax=None, **kwargs):
    '''
    使用x和y绘制二维直方图
    :param ax: matplotlib的轴对象,用于绘制图形
    :param x: x轴的数据
    :param y: y轴的数据
    :param x_bins: x轴的箱数,默认为BIN_NUM
    :param y_bins: y轴的箱数,默认为BIN_NUM
    :param cmap: 二维直方图的颜色映射表,默认为DENSITY_CMAP
    :param label: 图例标签,默认为None
    :param stat: 二维直方图的统计方法,默认为'probability'
    :param cbar: 是否添加颜色条,默认为True
    :param cbar_position: 颜色条的位置,默认为None
    :param cbar_kwargs: 其他plt.colorbar支持的参数
    :param vmin: 最小值阈值,数据中小于此值的将被设置为此值,默认为None
    :param vmax: 最大值阈值,数据中大于此值的将被设置为此值,默认为None
    :param kwargs: 其他plt.pcolormesh支持的参数
    '''
    if cbar_kwargs is None:
        cbar_kwargs = {}
    if cbar_label == 'stat':
        cbar_label = stat
    cbar_position = update_dict(CBAR_POSITION, cbar_position)

    h, x_edges, y_edges, x_midpoints, y_midpoints = get_hist_2d(x, y, x_bins, y_bins, stat=stat)

    if vmin is not None:
        vmin = np.nanmin(h)
    if vmax is not None:
        vmax = np.nanmax(h)

    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(y_edges[0], y_edges[-1])

    pc = ax.pcolormesh(x_edges, y_edges, h.T, cmap=cmap, label=label, vmin=vmin, vmax=vmax, **kwargs)
    if cbar:
        cbars = add_side_colorbar(ax, pc, cmap=cmap, cbar_label=stat, cbar_position=cbar_position, **cbar_kwargs)
        return pc, cbars
    else:
        return pc


def plt_hexbin(ax, x, y, gridsize=BIN_NUM, cmap=DENSITY_CMAP, **kwargs):
    '''
    使用x和y绘制hexbin图,可以接受plt.hexbin的其他参数
    :param ax: matplotlib的轴对象,用于绘制图形
    :param x: x轴的数据
    :param y: y轴的数据
    :param gridsize: hexbin的网格大小,默认为BIN_NUM
    :param cmap: 颜色映射,默认为DENSITY_CMAP
    :param kwargs: 其他plt.hexbin支持的参数

    注意:
    hexbin is only for count data, if you want to plot density data, use plt_hist_2d instead
    '''
    return ax.hexbin(x, y, gridsize=gridsize, cmap=cmap, **kwargs)


def plt_contour(ax, x, y, z, levels=None, color=BLUE, cmap=None, norm_mode='linear', vmin=None, vmax=None, norm_kwargs=None, label=None, label_cmap_float=1.0, clabel=True, clabel_kwargs=None, cbar=None, cbar_kwargs=None, **kwargs):
    '''
    使用x, y坐标网格和对应的z值绘制等高线图。
    :param ax: matplotlib的轴对象，用于绘制图形。
    :param x, Y: 定义等高线图网格的二维坐标数组。
    :param z: 在每个(x, y)坐标点上的z值。
    :param levels: 等高线的数量或具体的等高线级别列表，默认为自动确定。
    :param color: 等高线的颜色，默认为BLUE。
    :param cmap: 等高线的颜色映射表，默认为None。如果指定了cmap，则不使用color，并默认添加颜色条。
    :param label: 如果不为None，添加一个颜色条并使用该标签。
    :param label_cmap_float: 默认为1.0，用于指定标签的颜色。
    :param clabel: 是否添加等高线标签，默认为True。
    :param clabel_kwargs: 用于自定义等高线标签的参数，例如字体大小和颜色。
    :param cbar: 是否添加颜色条，默认为True。
    :param cbar_kwargs: 用于自定义颜色条的参数，例如位置和标签。
    :param kwargs: 其他plt.contour支持的参数。
    '''
    clabel_kwargs = update_dict(CLABEL_KWARGS, clabel_kwargs)
    cbar_kwargs = update_dict({}, cbar_kwargs)
    # 如果指定了cmap，则不使用color
    if cmap is not None:
        colors = None
        cbar = True
    else:
        colors = [color]
    if vmin is None:
        vmin = np.nanmin(z)
    if vmax is None:
        vmax = np.nanmax(z)
    
    # 获取norm
    norm = get_norm(vmin=vmin, vmax=vmax, norm_mode=norm_mode, norm_kwargs=norm_kwargs)

    # 画图
    contour =  ax.contour(x, y, z, levels=levels, colors=colors, cmap=cmap, norm=norm, **kwargs)
    if clabel:
        plt.clabel(contour, **clabel_kwargs)
    if label is not None:
        if cmap is not None:
            label_color = cmap(label_cmap_float)
            plt_line(ax, [], [], label=label, color=label_color)
        else:
            plt_line(ax, [], [], label=label, color=color)
    if cbar:
        add_side_colorbar(ax, norm_mode=norm_mode, vmin=vmin, vmax=vmax, norm_kwargs=norm_kwargs, cmap=cmap, **cbar_kwargs)
    return contour


def plt_contourf(ax, x, y, z, levels=None, cmap=DENSITY_CMAP, contour=True, contour_color=BLACK, clabel=True, clabel_kwargs=None, cbar=True, cbar_kwargs=None, **kwargs):
    '''
    使用X, Y坐标网格和对应的Z值绘制填充等高线图。
    :param ax: matplotlib的轴对象，用于绘制图形。
    :param x, y: 定义等高线图网格的二维坐标数组。
    :param z: 在每个(x, y)坐标点上的z值。
    :param levels: 等高线的数量或具体的等高线级别列表，默认为自动确定。
    :param cmap: 等高线的颜色映射表，默认为None。
    :param contour: 是否绘制等高线，默认为True。
    :param contour_color: 等高线的颜色，默认为BLACK。
    :param clabel: 是否添加等高线标签，默认为True。
    :param clabel_kwargs: 用于自定义等高线标签的参数，例如字体大小和颜色。
    :param cbar: 是否添加颜色条，默认为True。
    :param cbar_kwargs: 用于自定义颜色条的参数，例如位置和标签。
    :param kwargs: 其他plt.contourf支持的参数。
    '''
    cbar_kwargs = update_dict({}, cbar_kwargs)
    clabel_kwargs = update_dict(CLABEL_KWARGS, clabel_kwargs)
    contourf =  ax.contourf(x, y, z, levels=levels, cmap=cmap, **kwargs)
    if contour or clabel:
        plt_contour(ax, x, y, z, levels=levels, color=contour_color, cmap=None, clabel=clabel, **kwargs)
    if cbar:
        add_side_colorbar(ax, contourf, cmap=cmap, **cbar_kwargs)
    return contourf


def plt_pie(ax, data, labels, colors=CMAP, explode=None, autopct='%1.1f%%', startangle=90, shadow=False, textprops=None, **kwargs):
    '''
    在指定的轴上绘制饼图。
    :param ax: matplotlib的轴对象，用于绘制图形。
    :param data: 数据列表，每个元素对应饼图的一个部分。示例:pie_data = [30, 15, 45, 10]
    :param labels: 标签列表，与数据一一对应。示例:pie_labels = ['Category A', 'Category B', 'Category C', 'Category D']
    :param colors: 每个饼块的颜色列表，示例:pie_colors = ['blue', 'green', 'red', 'purple'];也可以指定cmap,默认为CMAP,然后自动生成颜色序列
    :param explode: 用于强调饼块的偏移量列表，可选。示例:pie_explode = (0, 0.1, 0, 0)
    :param autopct: 自动百分比显示格式。
    :param startangle: 饼图的起始角度。
    :param shadow: 是否显示阴影。
    :param textprops: 用于自定义文本样式的字典，例如字体大小和颜色。
    :param kwargs: 其他matplotlib.pie支持的参数。
    ax.pie()方法绘制饼图，支持通过textprops自定义文本样式。
    '''
    if textprops is None:
        textprops = {'fontsize': LABEL_SIZE, 'color': BLACK}
    if not isinstance(colors, list):
        colors = colors(np.linspace(0, 1, len(data)))

    ax.axis('equal')  # 保持圆形，确保饼图是正圆形。
    return ax.pie(data, labels=labels, colors=colors, explode=explode, autopct=autopct, startangle=startangle, shadow=shadow, textprops=textprops, **kwargs)


def plt_donut(ax, data, labels, colors=CMAP, explode=None, autopct='%1.1f%%', startangle=90, shadow=False, wedgeprops=None, textprops=None, **kwargs):
    '''
    在指定的轴上绘制环形图。
    :param ax: matplotlib的轴对象，用于绘制图形。
    :param data: 数据列表，每个元素对应环形图的一个部分。
    :param labels: 标签列表，与数据一一对应。
    :param colors: 每个环块的颜色列表;也可以指定cmap,默认为CMAP,然后自动生成颜色序列
    :param explode: 用于强调环块的偏移量列表，可选。
    :param autopct: 自动百分比显示格式。
    :param startangle: 环形图的起始角度。
    :param shadow: 是否显示阴影。
    :param wedgeprops: 环形图中心空白部分的属性，如宽度和边缘颜色。
    :param textprops: 用于自定义文本样式的字典，例如字体大小和颜色。
    :param kwargs: 其他matplotlib.pie支持的参数。
    ax.pie()方法绘制环形图，wedgeprops定义环的宽度和边缘颜色，支持通过textprops自定义文本样式。
    '''
    if wedgeprops is None:
        wedgeprops = {'width': 0.3, 'edgecolor': 'w'}
    if textprops is None:
        textprops = {'fontsize': LABEL_SIZE, 'color': BLACK}
    ax.axis('equal')  # 保持圆形，确保环形图是正圆形。
    return plt_pie(ax, data, labels=labels, colors=colors, explode=explode, autopct=autopct, startangle=startangle, shadow=shadow, wedgeprops=wedgeprops, textprops=textprops, **kwargs)


def plt_circle(ax, center, radius, color=BLUE, fill=True, adjust_lim=True, **kwargs):
    '''
    在指定的轴上绘制圆形。
    :param ax: matplotlib的轴对象，用于绘制图形。
    :param center: 圆心的坐标。
    :param radius: 圆的半径。
    :param color: 圆的颜色。
    :param fill: 是否填充圆。(如果为False,则为圆的边框,这时可以通过linewidth参数控制边框宽度)
    :param adjust_lim: 是否根据圆的位置和半径调整轴的限制。
    :param kwargs: 其他matplotlib.Circle支持的参数。
    ax.add_patch()方法绘制圆形，支持通过fill参数控制是否填充。
    '''
    if adjust_lim:
        ax.scatter(center[0], center[1], s=0, zorder=-1)
    circle = plt.Circle(center, radius, color=color, fill=fill, **kwargs)
    return ax.add_patch(circle)


def plt_rectangle(ax, xy, width, height, color=BLUE, fill=True, adjust_lim=True, **kwargs):
    '''
    在指定的轴上绘制矩形。
    :param ax: matplotlib的轴对象，用于绘制图形。
    :param xy: 矩形左下角的坐标。
    :param width: 矩形的宽度。
    :param height: 矩形的高度。
    :param color: 矩形的颜色。
    :param fill: 是否填充矩形。(如果为False,则为矩形的边框,这时可以通过linewidth参数控制边框宽度)
    :param adjust_lim: 是否根据矩形的位置和大小调整轴的限制。
    :param kwargs: 其他matplotlib.Rectangle支持的参数。
    ax.add_patch()方法绘制矩形，支持通过fill参数控制是否填充。
    '''
    rectangle = plt.Rectangle(xy, width, height, color=color, fill=fill, **kwargs)
    if adjust_lim:
        ax.scatter(xy[0], xy[1], s=0, zorder=-1)
    return ax.add_patch(rectangle)


def plt_polygon(ax, xy, color=BLUE, fill=True, adjust_lim=True, **kwargs):
    '''
    在指定的轴上绘制多边形。
    :param ax: matplotlib的轴对象，用于绘制图形。
    :param xy: 多边形的顶点坐标。例子:xy = [(0, 0), (1, 1), (1, 0)]
    :param color: 多边形的颜色。
    :param fill: 是否填充多边形。(如果为False,则为多边形的边框,这时可以通过linewidth参数控制边框宽度)
    :param adjust_lim: 是否根据多边形的位置和大小调整轴的限制。
    :param kwargs: 其他matplotlib.Polygon支持的参数。
    ax.add_patch()方法绘制多边形，支持通过fill参数控制是否填充。
    '''
    polygon = plt.Polygon(xy, color=color, fill=fill, **kwargs)
    if adjust_lim:
        for sub_xy in xy:
            ax.scatter(sub_xy[0], sub_xy[1], s=0, zorder=-1)
    return ax.add_patch(polygon)


def plt_imshow(ax, data, cmap=CMAP, norm=None, vmin=None, vmax=None, **kwargs):
    '''
    imshow

    注意:
    按默认方式使用,0,0会在左上角,并且y轴是反向的(0在上,最大值在下)
    '''
    return ax.imshow(data, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, **kwargs)


def plt_pcolormesh(ax, vertical_line_pos, horizontal_line_pos, data, cmap=CMAP, norm=None, vmin=None, vmax=None, **kwargs):
    '''
    pcolormesh

    优势:
    可以自由控制 vertical_line_pos 和 horizontal_line_pos,从而实现对矩阵的不均匀展示(甚至 log_scale)

    注意:
    vertical_line_pos 和 horizontal_line_pos 可以按照字面意思理解,是竖线和横线的位置
    这个函数的用意就是控制 vertical_line_pos 和 horizontal_line_pos,所以使用时必须传入(没有设置默认值 None)
    按默认方式使用,0,0会在左下角
    假如data的shape为(N_row, N_col),那么vertical_line_pos的shape应该为(N_col+1,),horizontal_line_pos的shape应该为(N_row+1,);在需要的时候,也很有可能需要转置data
    '''
    return ax.pcolormesh(vertical_line_pos, horizontal_line_pos, data, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, **kwargs)
# endregion


# region 初级作图函数(matplotlib系列,输入dict)
def plt_bar_dict(ax, data, label=None, color=BLUE, width=BAR_WIDTH, vert=True, **kwargs):
    '''
    使用字典数据绘制柱状图
    '''
    x = range(len(data))
    y = list(data.values())
    set_ax(ax, xtick=x, xtick_label=list(data.keys()))
    return plt_bar(ax, x, y, label=label, color=color, width=width, vert=vert, **kwargs)
# endregion


# region 初级作图函数(matplotlib系列,三维作图)
def plt_scatter_3d(ax, x, y, z, label=None, color=BLUE, **kwargs):
    '''
    使用x、y和z绘制3D散点图,可以接受ax.scatter的其他参数
    :param ax: matplotlib的3D轴对象,用于绘制图形
    :param x: x轴的数据
    :param y: y轴的数据
    :param z: z轴的数据
    :param label: 图例标签,默认为None
    :param color: 散点图的颜色,默认为BLUE
    :param kwargs: 其他ax.scatter支持的参数
    '''
    # 画图
    if 'c' in kwargs:
        return ax.scatter(x, y, z, label=label, **kwargs)
    else:
        return ax.scatter(x, y, z, label=label, color=color, **kwargs)


def plt_line_3d(ax, x, y, z, label=None, color=BLUE, **kwargs):
    '''
    使用x、y和z绘制3D折线图,可以接受ax.plot的其他参数
    :param ax: matplotlib的3D轴对象,用于绘制图形
    :param x: x轴的数据
    :param y: y轴的数据
    :param z: z轴的数据
    :param label: 图例标签,默认为None
    :param color: 折线图的颜色,默认为BLUE
    :param kwargs: 其他ax.plot支持的参数
    '''
    # 画图
    return ax.plot(x, y, z, label=label, color=color, **kwargs)


def plt_bar_3d(ax, x, y, z, dx, dy, dz, label=None, color=BLUE, **kwargs):
    '''
    使用x、y和z绘制3D柱状图,可以接受ax.bar3d的其他参数
    :param ax: matplotlib的3D轴对象,用于绘制图形
    :param x: x轴的数据
    :param y: y轴的数据
    :param z: z轴的数据
    :param label: 图例标签,默认为None
    :param color: 柱状图的颜色,默认为BLUE
    :param kwargs: 其他ax.bar3d支持的参数
    '''
    return ax.bar3d(x, y, z, dx, dy, dz, color=color, label=label, **kwargs)


def plt_surface_3d(ax, x, y, z, label=None, color=BLUE, alpha=FAINT_ALPHA, **kwargs):
    '''
    使用x、y和z绘制3D曲面图,可以接受ax.plot_surface的其他参数
    :param ax: matplotlib的3D轴对象,用于绘制图形
    :param x: x轴的数据
    :param y: y轴的数据
    :param z: z轴的数据
    :param label: 图例标签,默认为None
    :param color: 曲面图的颜色,默认为BLUE
    :param kwargs: 其他ax.plot_surface支持的参数
    '''
    # 画图
    return ax.plot_surface(x, y, z, label=label, color=color, alpha=alpha, **kwargs)


def plt_wireframe_3d(ax, X, Y, Z, rstride=BIN_NUM, cstride=BIN_NUM, color=BLUE, **kwargs):
    '''
    使用X,Y,Z绘制3D线框图,可以接受plt.plot_wireframe的其他参数
    :param ax: matplotlib的轴对象,用于绘制图形
    :param X: x轴的数据
    :param Y: y轴的数据
    :param Z: z轴的数据
    :param rstride: 行步长,默认为BIN_NUM
    :param cstride: 列步长,默认为BIN_NUM
    :param color: 线框的颜色,默认为BLUE
    :param kwargs: 其他plt.plot_wireframe支持的参数
    '''
    return ax.plot_wireframe(X, Y, Z, rstride=rstride, cstride=cstride, color=color, **kwargs)


def plt_voxel_3d(ax, data, label=None, color=BLUE, facecolors=None, edgecolors=None, **kwargs):
    '''
    使用数据绘制3D体素图,可以接受ax.voxels的其他参数
    :param ax: matplotlib的3D轴对象,用于绘制图形
    :param data: 三维数组,用于绘制3D体素图
    :param label: 图例标签,默认为None
    :param color: 体素图的颜色,默认为BLUE
    :param kwargs: 其他ax.voxels支持的参数

    注意:
    如果需要设置facecolors和edgecolors,请使用facecolors和edgecolors参数,并且把color参数设置为None,因为在这里color的优先级更高
    如果设置label的时候产生问题,建议设置label为None,另行添加图例
    '''
    if color is not None:
        return ax.voxels(data, color=color, label=label, facecolors=facecolors, edgecolors=edgecolors, **kwargs)
    else:
        return ax.voxels(data, label=label, facecolors=facecolors, edgecolors=edgecolors, **kwargs)


def plt_stem_3d(ax, x, y, z, label=None, linefmt='-', markerfmt='o', basefmt='k-', **kwargs):
    '''
    使用x、y和z绘制3D棉棒图,可以接受ax.stem的其他参数
    :param ax: matplotlib的3D轴对象,用于绘制图形
    :param x: x轴的数据
    :param y: y轴的数据
    :param z: z轴的数据
    :param label: 图例标签,默认为None
    :param linefmt: 折线的格式,默认为'-'
    :param markerfmt: 点的格式,默认为'o'
    :param basefmt: 基线的格式,默认为'k-'
    :param kwargs: 其他ax.stem支持的参数
    '''
    # 画图
    return ax.stem(x, y, z, label=label, linefmt=linefmt, markerfmt=markerfmt, basefmt=basefmt, **kwargs)
# endregion


# region 初级作图函数(sns系列,输入向量使用)
def sns_scatter(ax, x, y, label=None, color=BLUE, linewidth=0, **kwargs):
    '''
    使用x和y绘制散点图,可以接受sns.scatterplot的其他参数
    :param ax: matplotlib的轴对象,用于绘制图形
    :param x: x轴的数据
    :param y: y轴的数据
    :param label: 图例标签,默认为None
    :param color: 散点图的颜色,默认为BLUE
    :param kwargs: 其他sns.scatterplot支持的参数
    '''
    # 画图
    sns.scatterplot(x=x, y=y, ax=ax, label=label, color=color,
                    linewidth=linewidth, **kwargs)


def sns_line(ax, x, y, label=None, color=BLUE, **kwargs):
    '''
    使用x和y绘制折线图,可以接受sns.lineplot的其他参数
    :param ax: matplotlib的轴对象,用于绘制图形
    :param x: x轴的数据
    :param y: y轴的数据
    :param label: 图例标签,默认为None
    :param color: 折线图的颜色,默认为BLUE
    :param kwargs: 其他sns.lineplot支持的参数
    '''
    # 画图
    sns.lineplot(x=x, y=y, ax=ax, label=label, color=color, **kwargs)


def sns_bar(ax, x, y, label=None, bar_width=BAR_WIDTH, color=BLUE, orient='v', **kwargs):
    '''
    使用x和y绘制柱状图,可以接受sns.barplot的其他参数,特别注意,如果x需要展现出数字大小而不是等距,需要使用plt_bar函数
    :param ax: matplotlib的轴对象,用于绘制图形
    :param x: x轴的数据
    :param y: y轴的数据
    :param label: 图例标签,默认为None
    :param bar_width: 柱状图的宽度,默认为0.4
    :param color: 柱状图的颜色,默认为BLUE
    :param orient: 柱状图的方向,默认为'v',即纵向
    :param kwargs: 其他sns.barplot支持的参数
    '''

    # 画图
    if orient == 'v':
        sns.barplot(x=x, y=y, ax=ax, label=label, width=bar_width,
                    color=color, orient=orient, **kwargs)
    if orient == 'h':
        sns.barplot(y=x, x=y, ax=ax, label=label, width=bar_width,
                    color=color, orient=orient, **kwargs)


def sns_box(ax, x, y, label=None, color=BLUE, **kwargs):
    '''
    使用x和y绘制箱形图,可以接受sns.boxplot的其他参数(和plt_box的输入方式完全不同,请注意并参考示例)
    :param ax: matplotlib的轴对象,用于绘制图形
    :param x: x轴的数据，应为一个列表或数组，其长度与y中的数据集合数量相匹配。示例:x = ['A', 'A', 'B', 'B', 'C', 'C', 'C']
    :param y: y轴的数据，每个位置的数据应该是一个列表或数组。示例:y = [1, 2, 1, 2, 1, 2, 3]
    :param label: 图例标签,默认为None
    :param color: 箱形图的颜色,默认为BLUE
    :param kwargs: 其他sns.boxplot支持的参数
    '''
    sns.boxplot(x=x, y=y, ax=ax, color=color, **kwargs)
    if label:
        # 添加一个高度为0的bar，用于添加图例
        ax.bar(x[0], 0, width=0, bottom=np.nanmean(y[0]),
               color=color, linewidth=0, label=label, **kwargs)


def sns_hist(ax, data, bins=BIN_NUM, label=None, color=BLUE, log_scale=False, stat='probability', vert=True, **kwargs):
    '''
    使用数据绘制直方图,可以接受sns.histplot的其他参数(推荐使用)
    :param ax: matplotlib的轴对象,用于绘制图形
    :param data: 用于绘制直方图的数据(一维数组或列表)
    :param bins: 直方图的箱数,默认为None,自动确定箱数
    :param label: 图例标签,默认为None
    :param color: 直方图的颜色,默认为BLUE
    :param log_scale: 是否使用对数刻度,默认为False
    :param stat: 统计类型,默认为'probability'.'count': show the number of observations in each bin;'frequency': show the number of observations divided by the bin width;'probability' or 'proportion': normalize such that bar heights sum to 1;'percent': normalize such that bar heights sum to 100;'density': normalize such that the total area of the histogram equals 1
    :param vert: 是否垂直显示,默认为True
    :param kwargs: 其他sns.histplot支持的参数
    '''
    # 画图
    if vert:
        sns.histplot(data, bins=bins, label=label, color=color, ax=ax, log_scale=log_scale, stat=stat, **kwargs)
    else:
        local_data = pd.DataFrame({'temp': data})
        original_ylabel = ax.get_ylabel()
        sns.histplot(local_data, y='temp', bins=bins, label=label, color=color, ax=ax, log_scale=log_scale, stat=stat, **kwargs)
        ax.set_ylabel(original_ylabel)
# endregion


# region 初级作图函数(sns系列,输入pd dataframe或series使用)
def sns_scatter_pd(ax, data, x=None, y=None, label=None, color=BLUE, **kwargs):
    '''
    使用data的x和y列绘制散点图,可以接受sns.scatterplot的其他参数;当x为'index'时,使用DataFrame的索引作为x轴;对于Series,使用索引作为x轴,值作为y轴；当x列有重复,则会自动合并重复的x列并在对应位置绘制多个散点图。
    :param ax: matplotlib的轴对象,用于绘制图形
    :param data: 用于绘制散点图的数据集
    :param x: x轴的列名
    :param y: y轴的列名
    :param label: 图例标签,默认为None
    :param color: 散点图的颜色,默认为BLUE
    :param kwargs: 其他sns.scatterplot支持的参数
    '''
    # 画图
    if isinstance(data, pd.Series):
        # 对于Series，使用索引作为x轴，值作为y轴
        sns.scatterplot(x=data.index, y=data.values, ax=ax,
                        label=label, color=color, **kwargs)
        ax.set_xlabel(data.index.name)
        ax.set_ylabel(data.name)
    elif isinstance(data, pd.DataFrame):
        if x == 'index':
            # 将索引作为一个列用于绘图
            sns.scatterplot(data=data.reset_index(), x='index',
                            y=y, ax=ax, label=label, color=color, **kwargs)
            ax.set_xlabel(data.index.name)
        else:
            sns.scatterplot(data=data, x=x, y=y, ax=ax,
                            label=label, color=color, **kwargs)


def sns_line_pd(ax, data, x=None, y=None, label=None, color=BLUE, **kwargs):
    '''
    使用data的x和y列绘制折线图,可以接受sns.lineplot的其他参数;当x为'index'时,使用DataFrame的索引作为x轴;对于Series,使用索引作为x轴,值作为y轴；当x列有重复,则会自动合并重复的x列并计算y的均值和标准误差，作为折线图的值和误差线。
    :param ax: matplotlib的轴对象,用于绘制图形。
    :param data: 用于绘制折线图的数据集，可以是pd.Series或pd.DataFrame。
    :param x: x轴的列名或'index'，对于pd.DataFrame有效。
    :param y: y轴的列名，对于pd.DataFrame有效。
    :param label: 图例标签，默认为None。
    :param color: 折线图的颜色，默认为BLUE。
    :param kwargs: 其他sns.lineplot支持的参数。
    '''
    if isinstance(data, pd.Series):
        # 对于Series，使用索引作为x轴，值作为y轴
        sns.lineplot(x=data.index, y=data.values, ax=ax,
                     label=label, color=color, **kwargs)
        ax.set_xlabel(data.index.name)
        ax.set_ylabel(data.name)
    elif isinstance(data, pd.DataFrame):
        if x == 'index':
            # 将索引作为一个列用于绘图
            sns.lineplot(data=data.reset_index(), x='index', y=y,
                         ax=ax, label=label, color=color, **kwargs)
            ax.set_xlabel(data.index.name)
        else:
            sns.lineplot(data=data, x=x, y=y, ax=ax,
                         label=label, color=color, **kwargs)


def sns_bar_pd(ax, data, x=None, y=None, label=None, bar_width=BAR_WIDTH, color=BLUE, capsize=SNS_CAP_SIZE, err_kws=None, orient='v', **kwargs):
    '''
    使用data的x和y列或索引和值绘制柱状图,可以接受sns.barplot的其他参数,现支持DataFrame和Series。对于dataframe,假如x列都不重复,则会自动变成等距离排列，假设x列有重复,则会自动合并重复的x列并计算y的均值和标准误差，作为柱状图的值和误差线。假如x为index,则会使用DataFrame的索引作为x轴，由于index是不重复的，所以会自动变成等距离排列。对于Series，自动使用series的index和值作图
    :param ax: matplotlib的轴对象,用于绘制图形
    :param data: 用于绘制柱状图的DataFrame或Series
    :param x: x轴的列名或为'index'时使用DataFrame的索引；对于Series，保持为None
    :param y: y轴的列名；对于Series，此参数不使用
    :param label: 图例标签,默认为None
    :param bar_width: 柱状图的宽度,默认为BAR_WIDTH
    :param color: 柱状图的颜色,默认为BLUE
    :param capsize: 误差线帽大小,默认为SNS_CAP_SIZE
    :param err_kws: 误差线的属性,默认为{'color': BLACK}
    :param kwargs: 其他sns.barplot支持的参数
    '''
    if err_kws is None:
        err_kws = {'color': BLACK}

    if isinstance(data, pd.Series):
        plot_data = data.reset_index()
        plot_data.columns = ['index', data.name or 'value']
        if orient == 'v':
            sns.barplot(x='index', y=data.name or 'value', ax=ax, data=plot_data, width=bar_width,
                        color=color, label=label, capsize=capsize, err_kws=err_kws, **kwargs)
            ax.set_xlabel(data.index.name)
            ax.set_ylabel(data.name)
        else:
            sns.barplot(x=data.name or 'value', y='index', ax=ax, data=plot_data, width=bar_width,
                        color=color, label=label, orient=orient, capsize=capsize, err_kws=err_kws, **kwargs)
            ax.set_ylabel(data.index.name)
            ax.set_xlabel(data.name)
    elif isinstance(data, pd.DataFrame):
        if x == 'index':
            plot_data = data.reset_index()
            if orient == 'v':
                sns.barplot(x='index', y=y, ax=ax, data=plot_data, width=bar_width,
                            color=color, label=label, capsize=capsize, err_kws=err_kws, **kwargs)
                ax.set_xticks(range(len(plot_data['index'])))
                ax.set_xticklabels(plot_data['index'])
                # 重置x轴标签
                ax.set_xlabel('')
            else:
                sns.barplot(x=y, y='index', ax=ax, data=plot_data, width=bar_width, color=color,
                            label=label, orient=orient, capsize=capsize, err_kws=err_kws, **kwargs)
                ax.set_yticks(range(len(plot_data['index'])))
                ax.set_yticklabels(plot_data['index'])
                # 重置y轴标签
                ax.set_ylabel('')
        else:
            if orient == 'v':
                sns.barplot(x=x, y=y, ax=ax, data=data, width=bar_width, color=color,
                            label=label, capsize=capsize, err_kws=err_kws, **kwargs)
            else:
                sns.barplot(x=y, y=x, ax=ax, data=data, width=bar_width, color=color,
                            label=label, orient=orient, capsize=capsize, err_kws=err_kws, **kwargs)


def sns_hist_pd(ax, data, x=None, bins=BIN_NUM, label=None, color=BLUE, **kwargs):
    '''
    使用data的x列绘制直方图,可以接受sns.histplot的其他参数,此函数自动兼容DataFrame和Series,但是series状态需要保证x为None
    :param ax: matplotlib的轴对象,用于绘制图形
    :param data: 用于绘制直方图的数据集
    :param x: x轴的列名
    :param bins: 直方图的箱数,默认为None,自动确定箱数
    :param label: 图例标签,默认为None
    :param color: 直方图的颜色,默认为BLUE
    :param kwargs: 其他sns.histplot支持的参数
    '''
    # 画图
    sns.histplot(data=data, x=x, bins=bins, label=label,
                 color=color, ax=ax, **kwargs)
    if isinstance(data, pd.Series):
        ax.set_xlabel('')


def sns_box_pd(ax, data, x, y, color=BLUE, orient='v', **kwargs):
    '''
    使用data的x和y列绘制箱形图,可以接受sns.boxplot的其他参数
    :param ax: matplotlib的轴对象,用于绘制图形
    :param data: 用于绘制箱形图的数据集
    :param x: x轴的列名
    :param y: y轴的列名
    :param color: 箱形图的颜色,默认为BLUE
    :param kwargs: 其他sns.boxplot支持的参数
    '''
    # 画图
    if orient == 'v':
        sns.boxplot(data=data, x=x, y=y, ax=ax, color=color, **kwargs)
    if orient == 'h':
        sns.boxplot(data=data, x=y, y=x, ax=ax, color=color, **kwargs)
# endregion


# region 初级作图函数(sns系列,可同时接受矩阵和dataframe)
def sns_heatmap(ax, data, cmap=HEATMAP_CMAP, square=True, cbar=True, cbar_position=None, cbar_label=None, discrete_label=None, xtick_rotation=XTICK_ROTATION, ytick_rotation=YTICK_ROTATION, show_xtick=True, show_ytick=True, show_all_xtick=True, show_all_ytick=True, xtick_fontsize=TICK_SIZE, ytick_fontsize=TICK_SIZE, mask=None, mask_color=MASK_COLOR, mask_tick='mask', norm_mode='linear', vmin=None, vmax=None, norm_kwargs=None, text_process=None, heatmap_kwargs=None, cbar_kwargs=None):
    '''
    使用数据绘制热图,可以接受sns.heatmap的其他参数;注意,如果要让heatmap按照ax的框架显示,需要将square设置为False(如果想要mask是透明的,需要将mask_color设置为None)
    :param ax: matplotlib的轴对象,用于绘制图形
    :param data: 用于绘制热图的数据矩阵
    :param cmap: 热图的颜色映射,默认为CMAP。可以是离散的或连续的,须与discrete参数相符合。
    :param square: 是否以正方形显示每个cell,默认为True
    :param cbar: 是否显示颜色条,默认为True
    :param cbar_position: 颜色条的位置,默认为None,即使用默认位置;position参数可选'left', 'right', 'top', 'bottom'
    :param discrete: 是否离散显示,默认为False
    :param discrete_num: 离散显示的颜色数,默认为None,即使用默认颜色数
    :param discrete_label: 离散显示的标签,默认为None
    :param xtick_rotation: x轴刻度标签的旋转角度
    :param ytick_rotation: y轴刻度标签的旋转角度
    :param show_xtick: 是否显示x轴的刻度标签
    :param show_ytick: 是否显示y轴的刻度标签
    :param show_all_xtick: 是否显示所有x轴的刻度标签
    :param show_all_ytick: 是否显示所有y轴的刻度标签
    :param xtick_fontsize: x轴刻度标签的字体大小
    :param ytick_fontsize: y轴刻度标签的字体大小
    :param mask: 用于遮盖的矩阵,默认为None
    :param mask_color: 遮盖矩阵的颜色,默认为MASK_COLOR
    :param mask_tick: 遮盖矩阵颜色条的标签,默认为'mask'
    :param vmin: 热图的最小值,默认为None,即使用数据的最小值
    :param vmax: 热图的最大值,默认为None,即使用数据的最大值
    :param text_process: 是否对颜色条标签进行文本处理,默认为TEXT_PROCESS
    :param heatmap_kwargs: 传递给sns.heatmap的其他参数
    :param cbar_label_kwargs: 传递给颜色条标签的其他参数

    注意:
    此函数会将y轴的方向反转,即y轴的0位置在最上方
    '''
    text_process = update_dict(TEXT_PROCESS, text_process)
    cbar_kwargs = update_dict({}, cbar_kwargs)
    # 如果data是array,则转换为DataFrame
    if isinstance(data, np.ndarray):
        local_data = pd.DataFrame(data)
    else:
        local_data = data.copy()
    if isinstance(mask, np.ndarray):
        mask = pd.DataFrame(mask)
    elif mask is not None:
        mask = mask.copy()
    if mask_color is None:
        local_data = np.where(mask, np.nan, data)
        print('mask color is none need to be imporved')
    if vmin is None:
        vmin = np.nanmin(local_data.values)
    elif vmin > np.nanmin(local_data.values):
        cbar_kwargs['add_leq'] = True
    if vmax is None:
        vmax = np.nanmax(local_data.values)
    elif vmax < np.nanmax(local_data.values):
        cbar_kwargs['add_geq'] = True
    if heatmap_kwargs is None:
        heatmap_kwargs = {}
    cbar_position = update_dict(CBAR_POSITION, cbar_position)

    # 获取norm
    norm = get_norm(vmin=vmin, vmax=vmax, norm_mode=norm_mode, norm_kwargs=norm_kwargs)

    # 绘制热图
    if np.allclose(vmin, vmax):
        local_cmap = get_cmap([cmap(0.5), cmap(0.5)])
        sns.heatmap(local_data, ax=ax, cmap=local_cmap,
                    square=square, cbar=False, norm=norm, **heatmap_kwargs)
    else:
        sns.heatmap(local_data, ax=ax, cmap=cmap,
                    square=square, cbar=False, norm=norm, **heatmap_kwargs)

    # 绘制mask矩阵
    if mask is not None and mask_color is not None:
        use_mask = True
        mask_cmap = get_cmap([mask_color, mask_color])
        sns.heatmap(mask, ax=ax, cmap=mask_cmap,
                    cbar=False, mask=~mask, square=square)
    else:
        use_mask = False

    # 设置x轴和y轴的刻度标签
    if show_xtick:
        if show_all_xtick:
            ax.set_xticks(np.arange(len(local_data.columns))+0.5)
            ax.set_xticklabels(
                local_data.columns, rotation=xtick_rotation, fontsize=xtick_fontsize)
        else:
            ax.set_xticklabels(ax.get_xticklabels(),
                               rotation=xtick_rotation, fontsize=xtick_fontsize)
        ax.xaxis.set_tick_params(labelbottom=show_xtick)
    else:
        ax.set_xticks([])

    if show_ytick:
        if show_all_ytick:
            ax.set_yticks(np.arange(len(local_data.index))+0.5)
            ax.set_yticklabels(
                local_data.index, rotation=ytick_rotation, fontsize=ytick_fontsize)
        else:
            ax.set_yticklabels(ax.get_yticklabels(),
                               rotation=ytick_rotation, fontsize=ytick_fontsize)
        ax.yaxis.set_tick_params(labelleft=show_ytick)
    else:
        ax.set_yticks([])

    # 显示颜色条
    if cbar:
        cbars = add_side_colorbar(ax, mappable=None, cmap=cmap, cbar_position=cbar_position, cbar_label=cbar_label, norm_mode=norm_mode, vmin=vmin, vmax=vmax, norm_kwargs=norm_kwargs, discrete_label=discrete_label, use_mask=use_mask, mask_color=mask_color, mask_tick=mask_tick, **cbar_kwargs)
        return cbars
# endregion


# region 初级作图函数(添加errorbar)
def add_errorbar(ax, x, y, err, label=None, color=BLACK, linestyle='None', capsize=PLT_CAP_SIZE, vert=True, equal_space=False, **kwargs):
    '''
    在指定位置添加误差线。当x包含字符串或者equal_space=True时,会自动变成等距离排列。
    :param ax: matplotlib的轴对象，用于绘制图形。
    :param x: 误差线的x坐标。也支持字符串，此时会自动转换为等距的。
    :param y: 误差线的y坐标。
    :param err: 误差线的数据。
    :param label: 误差线的标签，默认为None。
    :param color: 误差线的颜色，默认为BLACK。
    :param linestyle: 误差线的线型，默认为'None'。
    :param capsize: 误差线的线帽大小，默认为PLT_CAP_SIZE。capsize是相对于图的大小的,不会受到xlim和ylim的影响
    :param vert: 是否为垂直误差线，默认为True。
    :param equal_space: 是否将x的值作为字符串处理，这将使得柱子等距排列，默认为False
    :param kwargs: 传递给`ax.errorbar`的额外关键字参数。
    '''
    # 检查x中元素是否包含字符串
    if isinstance(x, (list, np.ndarray)):
        x = list(x)
        if any(isinstance(item, str) for item in x) or equal_space:
            # 如果x中包含字符串，则将其全部转换为字符串
            x = [str(item) for item in x]
            if vert:
                ax.set_xticks(np.arange(len(x)))
                ax.set_xticklabels(x)
            else:
                ax.set_yticks(np.arange(len(x)))
                ax.set_yticklabels(x)
            x = np.arange(len(x))
    if vert:
        return ax.errorbar(x, y, yerr=err, label=label, color=color, linestyle=linestyle, capsize=capsize, **kwargs)
    else:
        return ax.errorbar(y, x, xerr=err, label=label, color=color, linestyle=linestyle, capsize=capsize, **kwargs)
# endregion


# region 初级作图函数(添加subfig)
def add_subfig(fig, left, right, bottom, top):
    '''
    在指定位置添加一个新的subfig

    :param fig: matplotlib的图形对象，用于绘制图形。
    :param left: 新subfig的左边界位置。
    :param right: 新subfig的右边界位置。
    :param bottom: 新subfig的下边界位置。
    :param top: 新subfig的上边界位置。
    '''
    gs = GridSpec(nrows=1, ncols=1, figure=fig, left=left, right=right, bottom=bottom, top=top)
    return get_subfig_from_gs(gs=gs)
# endregion


# region 初级作图函数(获取subfig的bbox)
def get_subfig_bbox_inches(subfig):
    '''
    获取subfig的bbox_inches
    
    用途:
    保存图片时,只保存subfig的内容,将这个bbox_inches作为参数传递给savefig函数,就可以做到只保存subfig的内容
    '''
    fig = subfig.get_figure()
    return subfig.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
# endregion


# region 初级作图函数(添加ax, zoom_in, twin_ax)
@iterate_over_axs
def inset_ax(ax, left, right, bottom, top, label='inset', inset_mode='fig', **kwargs):
    '''
    在指定位置添加一个新的inset ax;位置坐标是相对于原ax的坐标

    注意:
    如果使用inset_mode='ax',则获得的ax和普通的ax不同:比如,随着原ax的移动,内嵌ax的位置也会移动;不能单独移动内嵌的ax;从fig中获取axes时,内嵌的ax不会被获取到;随着原ax的删除,内嵌的ax也会被删除;内嵌ax的删除不能通过fig.delaxes()来删除,而是通过ax.remove()来删除;内嵌的ax无法作为reparent_ax的parent_ax(当然这有可能是我实现reparent_ax的不足导致的);对于原ax,set_visible(False)会同时隐藏内嵌的ax
    '''
    if inset_mode == 'fig':
        ax_left, ax_bottom, ax_width, ax_height = ax.get_position().bounds
        new_left = ax_left + left * ax_width
        new_right = ax_left + right * ax_width
        new_bottom = ax_bottom + bottom * ax_height
        new_top = ax_bottom + top * ax_height
        return add_ax(ax.get_figure(), new_left, new_right, new_bottom, new_top, label=cat(ax.get_label(), label), **kwargs)
    elif inset_mode == 'ax':
        width = right - left
        height = top - bottom
        new_ax = ax.inset_axes([left, bottom, width, height], label=cat(ax.get_label(), label), **kwargs)

        # 将new_ax添加到ax的custom_children中
        add_custom_child_to_ax(ax, new_ax, label)
        return new_ax

@iterate_over_axs
def reparent_ax(ax, parent_ax, label='inset', **kwargs):
    """
    将 ax 的位置变换为 parent_ax 的子轴,删除原始轴,并在父轴中创建内嵌轴
    
    参数:
    ax : 要转换的 matplotlib 轴
    parent_ax : 包含内嵌轴的父轴(不可以是内嵌轴)
    label : str, 可选,内嵌轴的标签,默认为 'inset'
    **kwargs : inset_ax 的其他关键字参数
    
    返回:
    inset_ax : 在父轴中创建的新内嵌轴

    注意:
    返回的是新的,空白的内嵌轴,原始轴已经被删除
    """
    # 获取原始轴的共享关系
    shared_x_axes = ax.get_shared_x_axes().get_siblings(ax)
    shared_y_axes = ax.get_shared_y_axes().get_siblings(ax)
    
    # 获取原始位置
    left, right, bottom, top = get_ax_position_custom(ax)
    left, bottom = map_transform(left, bottom, ax.figure.transFigure, parent_ax.transAxes)
    right, top = map_transform(right, top, ax.figure.transFigure, parent_ax.transAxes)
    
    # 删除原始轴
    rm_ax(ax)
    
    # 创建内嵌轴
    sub_ax = inset_ax(parent_ax, left, right, bottom, top, label=label, inset_mode='ax', **kwargs)
    
    # 恢复共享 x 轴关系
    success_share_x = False
    for shared_ax in shared_x_axes:
        try:
            share_axis_to_target(sub_ax, shared_ax, sharex=True, sharey=False)
            success_share_x = True
        except:
            try:
                share_axis_to_target(shared_ax, sub_ax, sharex=True, sharey=False)
                success_share_x = True
            except:
                pass
    if not success_share_x:
        print('Warning: 无法恢复共享 x 轴关系,请手动恢复')
        
    
    # 恢复共享 y 轴关系
    success_share_y = False
    for shared_ax in shared_y_axes:
        try:
            share_axis_to_target(sub_ax, shared_ax, sharex=False, sharey=True)
            success_share_y = True
        except:
            try:
                share_axis_to_target(shared_ax, sub_ax, sharex=False, sharey=True)
                success_share_y = True
            except:
                pass
    if not success_share_y:
        print('Warning: 无法恢复共享 y 轴关系,请手动恢复')
    
    return sub_ax


def add_ax(fig, left, right, bottom, top, label='add', **kwargs):
    '''
    在指定位置添加一个新的ax。
    :param fig: matplotlib的图形对象，用于绘制图形。
    :param left: 新ax的左边界位置。
    :param right: 新ax的右边界位置。
    :param bottom: 新ax的下边界位置。
    :param top: 新ax的上边界位置。
    :param kwargs: 传递给`fig.add_axes`的其他参数。比如sharex=some_ax, sharey=some_ax
    '''
    return fig.add_axes([left, bottom, right-left, top-bottom], label=label, **kwargs)


def add_ax_3d(fig, left, right, bottom, top, label='add', **kwargs):
    '''
    在指定位置添加一个新的3D ax。
    :param fig: matplotlib的图形对象，用于绘制图形。
    :param left: 新ax的左边界位置。
    :param right: 新ax的右边界位置。
    :param bottom: 新ax的下边界位置。
    :param top: 新ax的上边界位置。
    :param kwargs: 传递给`fig.add_axes`的其他参数。
    '''
    # 确保传递给add_axes的参数中包含projection='3d'
    kwargs.update({'projection': '3d'})
    return add_ax(fig, left, right, bottom, top, label=label, **kwargs)

@iterate_over_axs
def add_side_ax(ax, position='right', relative_size=SIDE_PAD*2, pad=SIDE_PAD, sharex=None, sharey=None, hide_repeat_xaxis=False, hide_repeat_yaxis=False, label='side', inset_mode='fig', spine_mode='same', **kwargs):
    '''
    在指定位置添加一个新的ax，并可以选择共享x轴或y轴。
    :param ax: matplotlib的轴对象，用于绘制图形。
    :param position: 新ax的位置，可以是'right', 'left', 'top', 'bottom'。
    :param relative_size: 新ax的宽度或高度，相对于原ax的对应大小。
    :param pad: 新ax与原ax的间距，相对于原ax的对应大小。
    :param sharex: 与新ax共享x轴的ax对象。也可以是True，表示与原ax共享x轴。
    :param sharey: 与新ax共享y轴的ax对象。也可以是True，表示与原ax共享y轴。
    :param hide_repeat_xaxis: 如果共享x轴，是否隐藏重复的x轴标签，默认为True。
    :param hide_repeat_yaxis: 如果共享y轴，是否隐藏重复的y轴标签，默认为True。
    :param kwargs: 传递给
    '''
    if sharex is True:
        sharex = ax
    elif sharex is None or sharex is False:
        sharex = None
    if sharey is True:
        sharey = ax
    elif sharey is None or sharey is False:
        sharey = None

    new_pos = {'left': 0., 'right': 1., 'top': 1., 'bottom': 0.}
    if position == 'right':
        new_pos['left'] = 1. + pad
        new_pos['right'] = 1. + pad + relative_size
    elif position == 'left':
        new_pos['left'] = - relative_size - pad
        new_pos['right'] = - pad
    elif position == 'top':
        new_pos['top'] = 1. + pad + relative_size
        new_pos['bottom'] = 1. + pad
    elif position == 'bottom':
        new_pos['top'] = - pad
        new_pos['bottom'] = - relative_size - pad

    # 创建并返回新的ax，可能共享x轴或y轴
    new_ax = inset_ax(ax, left=new_pos['left'], right=new_pos['right'], bottom=new_pos['bottom'], top=new_pos['top'], sharex=sharex, sharey=sharey, label=label, inset_mode=inset_mode, **kwargs)

    # 设置新ax的spine
    if spine_mode == 'same':
        for location in ['top', 'right', 'bottom', 'left']:
            new_ax.spines[location].set_visible(ax.spines[location].get_visible())
    elif spine_mode == 'all':
        for location in ['top', 'right', 'bottom', 'left']:
            new_ax.spines[location].set_visible(True)
    elif spine_mode in ['none', 'hide'] or not spine_mode:
        for location in ['top', 'right', 'bottom', 'left']:
            new_ax.spines[location].set_visible(False)

    # 如果共享x轴且hide_repeat_xaxis为True，将特定的x轴标签设为不可见
    if sharex is not None and hide_repeat_xaxis:
        if position=='top':
            new_ax.xaxis.set_visible(False)
        elif position=='bottom':
            new_ax.xaxis.set_visible(False)

    # 如果共享y轴且hide_repeat_yaxis为True，将特定的y轴标签设为不可见
    if sharey is not None and hide_repeat_yaxis:
        if position=='right':
            new_ax.yaxis.set_visible(False)
        elif position=='left':
            new_ax.yaxis.set_visible(False)

    return new_ax

@iterate_over_axs
def add_zoom_in_ax(ax, bounds_custom, xlim, ylim, edgecolor=BLACK, ax_facecolor=(1., 1., 1., 0.), label='zoom_in', inset_mode='fig', **kwargs):
    '''
    放大指定轴的显示范围。
    :param ax: matplotlib的轴对象,用于绘制图形。
    # :param bounds: list, 新ax的范围,相对于ax,为一个四元组(left, bottom, width, height)。比如(0.5, 0.5, 0.4, 0.4)表示zoom_in_ax的左下角在ax的(0.5, 0.5)位置,宽度和高度都是0.4
    : param bounds_custom: 新ax的范围,为一个四元组(left, right, bottom, top)。比如(0.5, 0.9, 0.5, 0.9)表示zoom_in_ax的左下角在ax的(0.5, 0.5)位置,右上角在(0.9, 0.9)位置
    :param xlim: 将要放大的原图的x轴范围,一个二元组(x1, x2)。
    :param ylim: 将要放大的原图的y轴范围,一个二元组(y1, y2)。
    :param edgecolor: 放大框的边框颜色,默认为BLACK。
    :param ax_facecolor: 放大框的背景颜色,默认为(1., 1., 1., 0.), 即透明(防止遮挡原图)
    :param kwargs: 传递给`ax.inset_axes`的其他参数。

    注意:
    - 该函数会在原图上绘制一个放大框,并返回新的ax对象。
    - 获得新的ax后仍需要再在新的ax上绘制图形,否则为空白。
    '''
    # 创建新的ax
    zoom_in_ax = inset_ax(ax, bounds_custom[0], bounds_custom[1], bounds_custom[2], bounds_custom[3], label=label, inset_mode=inset_mode, **kwargs)
    zoom_in_ax.set_xlim(xlim)
    zoom_in_ax.set_ylim(ylim)
    zoom_in_ax.set_xticklabels([])
    zoom_in_ax.set_yticklabels([])

    for location in ['top', 'right', 'bottom', 'left']:
        # 保证zoom in的所有spine都在
        zoom_in_ax.spines[location].set_visible(True)

        # 设置zoom in的width使得美观
        zoom_in_ax.spines[location].set_linewidth(AXES_LINEWIDTH/2)
    
    # 设置zoom in的tick
    zoom_in_ax.xaxis.set_tick_params(width=TICK_MAJOR_WIDTH/2, length=TICK_MAJOR_SIZE/2, which='major')
    zoom_in_ax.yaxis.set_tick_params(width=TICK_MAJOR_WIDTH/2, length=TICK_MAJOR_SIZE/2, which='major')
    zoom_in_ax.xaxis.set_tick_params(width=TICK_MINOR_WIDTH/2, length=TICK_MINOR_SIZE/2, which='minor')
    zoom_in_ax.yaxis.set_tick_params(width=TICK_MINOR_WIDTH/2, length=TICK_MINOR_SIZE/2, which='minor')

    # 设置zoom in的facecolor
    zoom_in_ax.set_facecolor(ax_facecolor)

    # 绘制放大框
    ax.indicate_inset_zoom(zoom_in_ax, edgecolor=edgecolor)
    return zoom_in_ax


def zoom_in_xrange(ax, zoom_in_ax, xmin, xmax, color=GREEN, alpha=FAINT_ALPHA, connection_mode=None):
    """
    在主图上添加一个缩放框，并在缩放图上显示缩放区域。
    
    参数:
    ax (Axes): 主图的 Axes 对象。
    zoom_in_ax (Axes): 缩放图的 Axes 对象。
    xmin (float): 缩放区域的最小 x 值。
    xmax (float): 缩放区域的最大 x 值。
    color (str): 缩放框的颜色。默认为绿色。
    alpha (float): 缩放框的透明度。默认为淡色。
    connection_mode (str): 连接线的方向。可以是 'up' 或 'down'。
    """
    zoom_in_xrange_partial(ax, zoom_in_ax, xmin, xmax, xmin, xmax, color, alpha, connection_mode)


def zoom_in_xrange_partial(ax, zoom_in_ax, xmin, xmax, zoom_xmin, zoom_xmax, color=GREEN, alpha=FAINT_ALPHA, connection_mode=None):
    """
    在主图上添加一个缩放框，并在缩放图上显示缩放区域。(这里的缩放图不会占满整个图像)
    
    参数:
    ax (Axes): 主图的 Axes 对象。
    zoom_in_ax (Axes): 缩放图的 Axes 对象。
    xmin (float): 缩放区域的最小 x 值。
    xmax (float): 缩放区域的最大 x 值。
    zoom_xmin (float): zoom_in_ax的最小 x 值。
    zoom_xmax (float): zoom_in_ax的最大 x 值。
    color (str): 缩放框的颜色。默认为绿色。
    alpha (float): 缩放框的透明度。默认为淡色。
    connection_mode (str): 连接线的方向。可以是 'up' 或 'down'。
    """
    # 获取ax和zoom_in_ax的position
    ax_pos = ax.get_position()
    zoom_in_ax_pos = zoom_in_ax.get_position()
    
    # 判断缩放图的位置
    if zoom_in_ax_pos.y0 > ax_pos.y0:
        connection_mode = 'up'
    elif zoom_in_ax_pos.y0 < ax_pos.y0:
        connection_mode = 'down'

    # 设置缩放图的 x 轴和 y 轴范围
    zoom_in_ax.set_xlim(zoom_xmin, zoom_xmax)
    ymin, ymax = ax.get_ylim()
    zoom_in_ax.set_ylim(ymin, ymax)
    
    # 在主图和缩放图上添加缩放框
    add_vspan(ax=ax, xmin=xmin, xmax=xmax, color=color, alpha=alpha)
    add_vspan(ax=zoom_in_ax, xmin=xmin, xmax=xmax, color=color, alpha=alpha)
    
    # 添加连接线
    if connection_mode == 'up':
        add_connection(axA=ax, axB=zoom_in_ax, xA=xmin, xB=xmin, yA=ymax, yB=ymin)
        add_connection(axA=ax, axB=zoom_in_ax, xA=xmax, xB=xmax, yA=ymax, yB=ymin)
    elif connection_mode == 'down':
        add_connection(axA=ax, axB=zoom_in_ax, xA=xmin, xB=xmin, yA=ymin, yB=ymax)
        add_connection(axA=ax, axB=zoom_in_ax, xA=xmax, xB=xmax, yA=ymin, yB=ymax)
    else:
        raise ValueError('connection_mode must be either "up" or "down"')


def zoom_in_yrange(ax, zoom_in_ax, ymin, ymax, color=GREEN, alpha=FAINT_ALPHA, connection_mode=None):
    """
    在主图上添加一个缩放框,并在缩放图上显示缩放区域。
    
    参数:
    ax (Axes): 主图的 Axes 对象。
    zoom_in_ax (Axes): 缩放图的 Axes 对象。
    ymin (float): 缩放区域的最小 y 值。
    ymax (float): 缩放区域的最大 y 值。
    color (str): 缩放框的颜色。默认为绿色。
    alpha (float): 缩放框的透明度。默认为淡色。
    connection_mode (str): 连接线的方向。可以是 'left' 或 'right'。
    """
    zoom_in_yrange_partial(ax, zoom_in_ax, ymin, ymax, ymin, ymax, color, alpha, connection_mode)


def zoom_in_yrange_partial(ax, zoom_in_ax, ymin, ymax, zoom_ymin, zoom_ymax, color=GREEN, alpha=FAINT_ALPHA, connection_mode=None):
    """
    在主图上添加一个缩放框,并在缩放图上显示缩放区域。(这里的缩放图不会占满整个图像)
    
    参数:
    ax (Axes): 主图的 Axes 对象。
    zoom_in_ax (Axes): 缩放图的 Axes 对象。
    ymin (float): 缩放区域的最小 y 值。
    ymax (float): 缩放区域的最大 y 值。
    zoom_ymin (float): zoom_in_ax的最小 y 值。
    zoom_ymax (float): zoom_in_ax的最大 y 值。
    color (str): 缩放框的颜色。默认为绿色。
    alpha (float): 缩放框的透明度。默认为淡色。
    connection_mode (str): 连接线的方向。可以是 'left' 或 'right'。
    """
    # 获取ax和zoom_in_ax的position
    ax_pos = ax.get_position()
    zoom_in_ax_pos = zoom_in_ax.get_position()
    
    # 判断缩放图的位置
    if zoom_in_ax_pos.x0 > ax_pos.x0:
        connection_mode = 'right'
    elif zoom_in_ax_pos.x0 < ax_pos.x0:
        connection_mode = 'left'

    # 设置缩放图的 x 轴和 y 轴范围
    zoom_in_ax.set_ylim(zoom_ymin, zoom_ymax)
    xmin, xmax = ax.get_xlim()
    zoom_in_ax.set_xlim(xmin, xmax)
    
    # 在主图和缩放图上添加缩放框
    add_hspan(ax=ax, ymin=ymin, ymax=ymax, color=color, alpha=alpha)
    add_hspan(ax=zoom_in_ax, ymin=ymin, ymax=ymax, color=color, alpha=alpha)
    
    # 添加连接线
    if connection_mode == 'left':
        add_connection(axA=ax, axB=zoom_in_ax, xA=xmin, xB=xmax, yA=ymin, yB=ymin)
        add_connection(axA=ax, axB=zoom_in_ax, xA=xmin, xB=xmax, yA=ymax, yB=ymax)
    elif connection_mode == 'right':
        add_connection(axA=ax, axB=zoom_in_ax, xA=xmax, xB=xmin, yA=ymin, yB=ymin)
        add_connection(axA=ax, axB=zoom_in_ax, xA=xmax, xB=xmin, yA=ymax, yB=ymax)
    else:
        raise ValueError('connection_mode must be either "left" or "right"')

@iterate_over_axs
def add_twin_ax(ax, axis, color='black', label='twin', inset_mode='fig'):
    '''
    在指定轴上添加一个新的双轴。
    :param ax: matplotlib的轴对象,用于绘制图形
    :param axis: 新双轴的位置,'x'或'y'
    :param color: 新双轴的颜色,默认为黑色

    注意: 
    axis为'x'时,share x轴,新轴的y轴在右边;axis为'y'时,share y轴,新轴的x轴在上边
    假如后续更改了原ax的位置,需要同步设置twin_ax的位置
    '''
    if axis == 'x':
        twin_ax = ax.twinx()
        twin_ax.set_position(ax.get_position())
        twin_ax.spines['right'].set_visible(True)
        twin_ax.spines['right'].set_color(color)
        twin_ax.tick_params(axis='y', colors=color)
        twin_ax.yaxis.label.set_color(color)
    elif axis == 'y':
        twin_ax = ax.twiny()
        twin_ax.set_position(ax.get_position())
        twin_ax.spines['top'].set_visible(True)
        twin_ax.spines['top'].set_color(color)
        twin_ax.tick_params(axis='x', colors=color)
        twin_ax.xaxis.label.set_color(color)
    set_ax_label(twin_ax, cat(ax.get_label(), label))
    if inset_mode == 'ax':
        return reparent_ax(twin_ax, ax, label=label)
    else:
        return twin_ax
# endregion


# region 初级作图函数(删除ax, 设置ax不可见, 清除ax内容)
@iterate_over_axs
def rm_ax(ax):
    """删除指定的Axes对象"""
    try:
        fig = ax.figure  # 获取 Axes 所属的 Figure
        fig.delaxes(ax)  # 从 Figure 中删除 Axes
    except:
        ax.remove()  # 移除 Axes 对象

@iterate_over_axs
def set_ax_invisible(ax):
    """设置指定的Axes对象不可见"""
    ax.set_visible(False)

@iterate_over_axs
def clear_ax(ax):
    '''
    清除指定的Axes对象的内容
    '''
    ax.clear()
# endregion


# region 初级作图函数(添加连接不同ax的线)
def add_connection(axA, axB, xA, yA, xB, yB, coordsA='data', coordsB='data', **kwargs):
    """
    在两个子图之间绘制连接线。
    
    参数:
    xA, yA (float): 起始点的 x, y 坐标
    xB, yB (float): 终止点的 x, y 坐标
    axA, axB (Axes): 起始点和终止点所在的子图
    coordsA, coordsB (str): 坐标系,可以是'data', 'axes fraction'等
    **kwargs: ConnectionPatch 的其他参数,如 arrowstyle, shrinkA, shrinkB 等
    """
    # 创建 ConnectionPatch 对象
    con = mpatches.ConnectionPatch(xyA=(xA, yA), xyB=(xB, yB),
                         coordsA=coordsA, coordsB=coordsB,
                         axesA=axA, axesB=axB, **kwargs)
    
    # 将 ConnectionPatch 添加到图形中
    axA.figure.add_artist(con)
# endregion


# region 初级作图函数(ax视角相关)
@iterate_over_axs
def set_ax_view_3d(ax, elev=ELEV, azim=AZIM):
    '''
    对单个或多个3D子图Axes应用统一的视角设置。

    参数:
    - ax: 单个Axes,或者np.ndarray,list,dict
    - elev: 视角的高度。
    - azim: 视角的方位角。
    '''
    if isinstance(ax, Axes3D):
        ax.view_init(elev=elev, azim=azim)
# endregion


# region 初级作图函数(ax位置相关)
@direct_use
def get_ax_position(ax):
    """
    获取 Matplotlib Axes 对象的位置信息。

    参数:
    ax (matplotlib.axes.Axes): 需要获取位置信息的 Axes 对象。

    返回:
    tuple: 包含 Axes 对象位置信息的 4 元组, 格式为 (left, bottom, width, height)。
        其中 left 和 bottom 表示 Axes 对象在图像中的左下角坐标,
        width 和 height 表示 Axes 对象的宽度和高度。
        这些值都是相对于图像大小的比例值,范围在 0 到 1 之间。
    """
    return ax.get_position().bounds

@direct_use
def set_ax_position(ax, left, bottom, width, height):
    '''
    设置轴的位置。
    '''
    ax.set_position([left, bottom, width, height])


def get_ax_position_custom(ax):
    '''
    更加人性化,返回left, right, bottom, top
    '''
    left, bottom, width, height = ax.get_position().bounds
    right = left + width
    top = bottom + height
    return left, right, bottom, top


def set_ax_position_custom(ax, left, right, bottom, top):
    '''
    设置轴的位置。

    参数:
    - ax: matplotlib的Axes对象
    - left: 左边界的位置
    - right: 右边界的位置
    - bottom: 下边界的位置
    - top: 上边界的位置
    '''
    ax.set_position([left, bottom, right - left, top - bottom])


def set_relative_ax_position(ax, nrows=1, ncols=1, margin=None, squeeze=False):
    '''
    自动设置subplot的位置,使其在等分的图像中按照给定的比例占据空间。

    参数:
    - nrows: 子图的行数。
    - ncols: 子图的列数。
    - ax: 一个或一组matplotlib的Axes对象。
    - margin: 一个字典,定义了图像边缘的留白,包括left, right, bottom, top。
    - squeeze: ax是否被压缩,squeeze为True时,对于单个ax其就是axes对象;squeeze为False时,对于单个ax其是一个(1,1)的np.ndarray对象
    '''
    if margin is None:
        margin = MARGIN.copy()

    # 计算每个子图的宽度和高度(相对于整个图像的宽度和高度,所以是比例值)
    subplot_width = 1 / ncols
    subplot_height = 1 / nrows
    ax_width = (margin['right'] - margin['left']) * subplot_width
    ax_height = (margin['top'] - margin['bottom']) * subplot_height

    if nrows > 1 and ncols > 1:
        # 对于每个子图，计算其位置并设置
        for row in range(nrows):
            for col in range(ncols):
                left = margin['left'] / ncols + col * subplot_width
                bottom = margin['bottom'] / nrows + \
                    (nrows - row - 1) * subplot_height

                # 设置子图的位置
                ax[row, col].set_position(
                    [left, bottom, ax_width, ax_height])
    if nrows == 1 and ncols > 1:
        for col in range(ncols):
            left = margin['left'] / ncols + col * subplot_width
            bottom = margin['bottom']

            # 设置子图的位置
            if squeeze:
                ax[col].set_position([left, bottom, ax_width, ax_height])
            else:
                ax[0, col].set_position([left, bottom, ax_width, ax_height])
    if nrows > 1 and ncols == 1:
        for row in range(nrows):
            left = margin['left']
            bottom = margin['bottom'] / nrows + (nrows - row - 1) * subplot_height

            # 设置子图的位置
            if squeeze:
                ax[row].set_position([left, bottom, ax_width, ax_height])
            else:
                ax[row, 0].set_position([left, bottom, ax_width, ax_height])
    if nrows == 1 and ncols == 1:
        left = margin['left']
        bottom = margin['bottom']

        # 设置子图的位置
        if squeeze:
            ax.set_position([left, bottom, ax_width, ax_height])
        else:
            ax[0, 0].set_position([left, bottom, ax_width, ax_height])

    return ax

@iterate_over_axs
def align_ax(ax, ref_ax, align_mode='horizontal', keep_size=False):
    '''
    将axs中ax的position对齐到ref_ax的position
    align_mode:
        'horizontal', 'vertical' - 改变高度和宽度,horizontal则左右对齐(ax的上下框对齐),vertical则上下对齐(ax的左右框对齐)
        'left', 'right', 'top', 'bottom' - 不改变高度和宽度,只改变位置去对齐到需要的位置
        'all' - 改变高度和宽度,同时改变位置去对齐到需要的位置(即直接对齐到ref_ax的position)
    keep_size: 是否保持ax的大小不变,默认为False,即改变大小(其他点固定,拉伸至指定点);假如为True,则保持大小,只改变位置(平移)
    '''
    if align_mode == 'horizontal' or align_mode == 'vertical' or align_mode == 'all':
        if keep_size:
            print('may not be able to keep size when align_mode is horizontal or vertical')
    ref_pos_left, ref_pos_right, ref_pos_bottom, ref_pos_top = get_ax_position_custom(ref_ax)

    pos_left, pos_right, pos_bottom, pos_top = get_ax_position_custom(ax)
    pos_width = pos_right - pos_left
    pos_height = pos_top - pos_bottom
    if align_mode == 'horizontal':
        pos_top = ref_pos_top
        pos_bottom = ref_pos_bottom
    if align_mode == 'vertical':
        pos_left = ref_pos_left
        pos_right = ref_pos_right
    if align_mode == 'all':
        pos_left = ref_pos_left
        pos_right = ref_pos_right
        pos_bottom = ref_pos_bottom
        pos_top = ref_pos_top
    if align_mode == 'left':
        pos_left = ref_pos_left
        if keep_size:
            pos_right = pos_left + pos_width
    if align_mode == 'right':
        pos_right = ref_pos_right
        if keep_size:
            pos_left = pos_right - pos_width
    if align_mode == 'top':
        pos_top = ref_pos_top
        if keep_size:
            pos_bottom = pos_top - pos_height
    if align_mode == 'bottom':
        pos_bottom = ref_pos_bottom
        if keep_size:
            pos_top = pos_bottom + pos_height
    set_ax_position_custom(ax, pos_left, pos_right, pos_bottom, pos_top)

@iterate_over_axs
def move_ax(ax, d_left=None, d_right=None, d_bottom=None, d_top=None, keep_size=False):
    '''
    向某个方向移动ax的位置

    参数:
    -keep_size: 是否保持ax的大小不变,默认为False,即改变大小(其他点固定,拉伸至指定点);假如为True,则保持大小,只改变位置(平移)
    '''
    left, right, bottom, top = get_ax_position_custom(ax)
    if keep_size:
        position_dict = {'left': left, 'right': right, 'bottom': bottom, 'top': top}
        if d_left is not None:
            position_dict['left'] -= d_left
            position_dict['right'] -= d_left
        if d_right is not None:
            position_dict['left'] += d_right
            position_dict['right'] += d_right
        if d_bottom is not None:
            position_dict['bottom'] -= d_bottom
            position_dict['top'] -= d_bottom
        if d_top is not None:
            position_dict['bottom'] += d_top
            position_dict['top'] += d_top

        left, right, bottom, top = position_dict['left'], position_dict['right'], position_dict['bottom'], position_dict['top']
    else:
        if d_left is not None:
            left -= d_left
        if d_right is not None:
            right += d_right
        if d_bottom is not None:
            bottom -= d_bottom
        if d_top is not None:
            top += d_top
    set_ax_position_custom(ax, left, right, bottom, top)


def rm_end_ax_and_align(axes, row=None, col=None):
    '''
    移除指定行/列的最后一个ax,并重新对齐剩余的子图
    
    参数:
    axes : matplotlib Axes 对象数组 (nrows x ncols)
    row : 要处理的行索引 (从0开始)
    col : 要处理的列索引 (从0开始)
    
    返回:
    调整后的Axes数组
    '''
    if row is not None:
        ax = axes[row, -1]
        rm_ax(ax)

        ax_row = axes[row, :-1]
        pos_0, _, _, _ = get_ax_position_custom(ax_row[0])
        pos_1, _, _, _ = get_ax_position_custom(ax_row[1])
        d_right = np.abs((pos_1 - pos_0) / 2)
        move_ax(ax_row, d_right=d_right, keep_size=True)
    if col is not None:
        ax = axes[-1, col]
        rm_ax(ax)

        ax_col = axes[:-1, col]
        _, _, pos_0, _ = get_ax_position_custom(ax_col[0])
        _, _, pos_1, _ = get_ax_position_custom(ax_col[1])
        d_bottom = np.abs((pos_1 - pos_0) / 2)
        move_ax(ax_col, d_bottom=d_bottom, keep_size=True)
    return axes
# endregion


# region 初级作图函数(ax维数与维数变换)
def is_ax_3d(ax):
    return isinstance(ax, Axes3D)

@iterate_over_axs
def convert_ax_to_2d(ax):
    '''
    将一个ax转换为2D的ax。
    :param ax: matplotlib的3D轴对象，用于绘制图形。
    '''
    ax_position = ax.get_position()
    fig = ax.get_figure()
    fig.delaxes(ax)
    return fig.add_axes([ax_position.x0, ax_position.y0, ax_position.width, ax_position.height])

@iterate_over_axs
def convert_ax_to_3d(ax):
    '''
    将一个ax转换为3D的ax。
    :param ax: matplotlib的2D轴对象，用于绘制图形。
    '''
    ax_position = ax.get_position()
    fig = ax.get_figure()
    fig.delaxes(ax)
    return fig.add_axes([ax_position.x0, ax_position.y0, ax_position.width, ax_position.height], projection='3d')
# endregion


# region 初级作图函数(分割ax)
def calculate_gs_coordinates(ncols, nrows, left, right, bottom, top, wspace, hspace, width_ratios, height_ratios):
    '''
    根据给定的边界和比例计算GridSpec的坐标
    '''
    # 设定默认的比例和间距
    if width_ratios is None:
        width_ratios = [1] * ncols
    if height_ratios is None:
        height_ratios = [1] * nrows
    if wspace is None:
        wspace = plt.rcParams['figure.subplot.wspace']
    if hspace is None:
        hspace = plt.rcParams['figure.subplot.hspace']


    # Calculate total space in each direction
    width_total = right - left
    height_total = top - bottom

    # Calculate sum of ratios and effective space for grids
    width_ratios_sum = sum(width_ratios)
    height_ratios_sum = sum(height_ratios)

    # Calculate grid widths and heights without spacing
    cell_widths = [(r / width_ratios_sum) * (width_total - (len(width_ratios) - 1) * wspace) for r in width_ratios]
    cell_heights = [(r / height_ratios_sum) * (height_total - (len(height_ratios) - 1) * hspace) for r in height_ratios]

    # Initialize lists to store cell boundaries
    x_coords = [left]
    y_coords = [top]  # Starting from top for a top-down layout

    # Calculate x coordinates for each column
    for width in cell_widths:
        x_coords.append(x_coords[-1] + width + wspace)

    # Calculate y coordinates for each row
    for height in cell_heights:
        y_coords.append(y_coords[-1] - height - hspace)

    # Collect coordinates for each grid cell
    # coordinates = []
    # for i in range(len(height_ratios)):
    #     row = []
    #     for j in range(len(width_ratios)):
    #         cell_left = x_coords[j]
    #         cell_right = x_coords[j + 1] - wspace
    #         cell_bottom = y_coords[i + 1] + hspace
    #         cell_top = y_coords[i]
    #         row.append((cell_left, cell_right, cell_bottom, cell_top))
    #     coordinates.append(row)
    coordinates = np.empty((nrows, ncols), dtype=object)
    for i in range(nrows):
        for j in range(ncols):
            cell_left = x_coords[j]
            cell_right = x_coords[j + 1] - wspace
            cell_bottom = y_coords[i + 1] + hspace
            cell_top = y_coords[i]
            coordinates[i, j] = (cell_left, cell_right, cell_bottom, cell_top)

    return coordinates


@iterate_over_axs
def split_ax_by_gs(ax, nrows=1, ncols=1, wspace=None, hspace=None, width_ratios=None, height_ratios=None, sharex=False, sharey=False, squeeze=True, keep_original=False, label='split', inset_mode='fig', **kwargs):
    '''
    在ax的位置为基础获取一个GridSpec对象,然后根据GridSpec对象获取所有的ax对象。可以用于切分ax。

    参数:
    -keep_original: 是否保留原始的ax,默认为False,即不保留(将会被rm);如果为True,则会保留原始的ax;如果为index,则会将原先的ax放在返回的ax中的index位置
    '''
    if inset_mode == 'fig':
        gs = get_gs_inside_ax(ax, nrows=nrows, ncols=ncols, wspace=wspace, hspace=hspace, width_ratios=width_ratios, height_ratios=height_ratios)
        label = cat(label, ax.get_label())
        sub_ax = get_all_ax_from_gs(gs, squeeze=squeeze, label=label, **kwargs)
        if keep_original is True:
            # 不做处理
            pass
        elif keep_original is False:
            rm_ax(ax)
            sub_ax = share_axis(sub_ax, sharex=sharex, sharey=sharey)
        else:
            # 重新share_axis到original_ax(此时不支持row,col)
            if sharex in ['row', 'col'] or sharey in ['row', 'col']:
                print('sharex or sharey can not be row or col in keep_original mode')
            sub_ax = share_axis_to_target(sub_ax, ax, sharex=sharex, sharey=sharey)
            # 将orginal_ax在图中放到index的ax所处的位置
            align_ax(ax, sub_ax[keep_original], align_mode='all')
            # 将sub_ax[keep_original]移除
            rm_ax(sub_ax[keep_original])
            # 将original_ax替换到sub_ax中
            sub_ax[keep_original] = ax
    elif inset_mode == 'ax':
        coords = calculate_gs_coordinates(ncols=ncols, nrows=nrows, left=0., right=1., bottom=0., top=1., wspace=wspace, hspace=hspace, width_ratios=width_ratios, height_ratios=height_ratios)
        # 根据coords创建ax
        sub_ax = np.empty((nrows, ncols), dtype=object)
        for i in range(nrows):
            for j in range(ncols):
                local_label = cat(label, f'row{i}_col{j}')
                sub_ax[i, j] = inset_ax(ax, coords[i][j][0], coords[i][j][1], coords[i][j][2], coords[i][j][3], label=local_label, inset_mode='ax', **kwargs)
        if keep_original is True:
            # 不做处理
            pass
        elif keep_original is False:
            # 注意,这里不可以把原ax直接删除,因为这会导致sub_ax中的ax也被删除,只能尽量把原ax的属性删除,就设计属性而言,用户不应该在原ax上画任何图像(因为原ax是要被分割的,不是用来画图的)
            rm_ax_spine(ax)
            rm_ax_tick(ax)
            rm_ax_ticklabel(ax)
            sub_ax = share_axis(sub_ax, sharex=sharex, sharey=sharey)
        else:
            print('unsupport keep_original in inset_mode=ax')
        if squeeze:
            sub_ax = squeeze_ax(sub_ax)
    return sub_ax


def split_ax(ax, nrows=1, ncols=1, wspace=None, hspace=None, width_ratios=None, height_ratios=None, sharex=False, sharey=False, squeeze=True, keep_original=False, label='split', inset_mode='fig', **kwargs):
    '''
    split_ax_by_gs的别名
    '''
    return split_ax_by_gs(ax, nrows=nrows, ncols=ncols, wspace=wspace, hspace=hspace, width_ratios=width_ratios, height_ratios=height_ratios, sharex=sharex, sharey=sharey, squeeze=squeeze, keep_original=keep_original, label=label, inset_mode=inset_mode, **kwargs)


def split_with_double_marginal_ax(ax, x_side_ax_position='top', y_side_ax_position='right', x_side_ax_pad=SIDE_PAD, y_side_ax_pad=SIDE_PAD, x_side_ax_size=0.3, y_side_ax_size=0.3, label='marginal', inset_mode='fig'):
    '''
    添加双边缘轴
    '''
    if x_side_ax_position == 'top':
        height_ratios = [x_side_ax_size, 1-x_side_ax_size]
        x_side_index = 0 # 取第0行
    elif x_side_ax_position == 'bottom':
        height_ratios = [1-x_side_ax_size, x_side_ax_size]
        x_side_index = 1

    if y_side_ax_position == 'right':
        width_ratios = [1-y_side_ax_size, y_side_ax_size]
        y_side_index = 1
    elif y_side_ax_position == 'left':
        width_ratios = [y_side_ax_size, 1-y_side_ax_size]
        y_side_index = 0

    # 目前ax模式不支持keep_original为一个特定索引(后续可能会支持)
    if inset_mode == 'fig':
        keep_original = (1-x_side_index, 1-y_side_index)
        sub_ax = split_ax_by_gs(ax, nrows=2, ncols=2, hspace=x_side_ax_pad, wspace=y_side_ax_pad, height_ratios=height_ratios, width_ratios=width_ratios, keep_original=keep_original, label=label, inset_mode=inset_mode)
    elif inset_mode == 'ax':
        print_title('inset_mode=ax need to delete original ax and re-create an ax in the needed position, thus not recommended, please insure that other process on ax is done after this function')
        sub_ax = split_ax_by_gs(ax, nrows=2, ncols=2, hspace=x_side_ax_pad, wspace=y_side_ax_pad, height_ratios=height_ratios, width_ratios=width_ratios, keep_original=False, label=label, inset_mode=inset_mode)
    
    x_side_ax = sub_ax[x_side_index, 1-y_side_index]
    y_side_ax = sub_ax[1-x_side_index, y_side_index]
    ax = sub_ax[1-x_side_index, 1-y_side_index]

    x_side_ax = share_axis_to_target(x_side_ax, ax, sharex=True, sharey=False)
    y_side_ax = share_axis_to_target(y_side_ax, ax, sharex=False, sharey=True)

    rm_ax(sub_ax[x_side_index, y_side_index])
    return x_side_ax, y_side_ax, ax
# endregion


# region 初级作图函数(合并ax)
def merge_ax(axs, rm_mode='rm_axis', label='merge'):
    '''
        合并给定的轴对象列表或数组为一个轴对象。

        :param axs: 要合并的轴对象列表或数组
        :param rm_mode: 是否删除原始的轴对象.默认为'rm_axis',即删除原始的轴对象;假如为'rm_ax',则删除整个ax;如果是其他值,则不删除原始的轴对象
    '''
    axs = get_iterable_ax(axs)
    for ax in axs:
        if rm_mode == 'rm_axis':
            rm_ax_axis(ax)
        elif rm_mode == 'rm_ax':
            rm_ax(ax)
        else:
            pass
        label = cat(label, ax.get_label())
    left = get_extreme_ax_position(axs, 'left')
    right = get_extreme_ax_position(axs, 'right')
    bottom = get_extreme_ax_position(axs, 'bottom')
    top = get_extreme_ax_position(axs, 'top')
    return add_ax(fig=axs[0].get_figure(), left=left, right=right, bottom=bottom, top=top, label=label)
# endregion


# region 初级作图函数(ax的custom_child)
def add_custom_child_to_ax(ax, child, child_name):
    '''
    向ax中添加一个自定义的子元素(放置于ax.custom_children中)
    '''
    if not hasattr(ax, 'custom_children'):
        ax.custom_children = {}
    if child_name not in ax.custom_children:
        ax.custom_children[child_name] = child
    else:
        new_child_name = child_name + '_new'
        add_custom_child_to_ax(ax, child, new_child_name)


def get_custom_child_from_ax(ax, child_name):
    if not hasattr(ax, 'custom_children'):
        raise ValueError('ax没有custom_children属性')
    if child_name in ax.custom_children:
        return ax.custom_children[child_name]
    else:
        raise ValueError(f'ax的custom_children中没有{child_name}, 所有的custom_children为{ax.custom_children.keys()}')
# endregion


# region 初级作图函数(获取多个ax的位置的极值)
def get_extreme_ax_position(axs, position):
    '''
    获取多个ax的位置的极值。
    :param axs: 多个ax对象。
    :param position: 位置参数，可选'left', 'right', 'top', 'bottom'。
    '''
    axs = get_iterable_ax(axs)
    if position not in ['left', 'right', 'top', 'bottom']:
        raise ValueError('position参数错误')
    if position == 'left':
        return min([ax.get_position().x0 for ax in axs])
    elif position == 'right':
        return max([ax.get_position().x1 for ax in axs])
    elif position == 'top':
        return max([ax.get_position().y1 for ax in axs])
    elif position == 'bottom':
        return min([ax.get_position().y0 for ax in axs])
# endregion


# region 初级作图函数(locator)
@iterate_over_axs
def set_ax_locator(ax, locator, axis='both', locator_type='major'):
    if axis == 'both':
        axis = ['x', 'y']
    if isinstance(axis, str):
        axis = [axis]
    if locator_type == 'major':
        for tick in axis:
            if tick == 'x':
                ax.xaxis.set_major_locator(locator)
            elif tick == 'y':
                ax.yaxis.set_major_locator(locator)
    elif locator_type == 'minor':
        for tick in axis:
            if tick == 'x':
                ax.xaxis.set_minor_locator(locator)
            elif tick == 'y':
                ax.yaxis.set_minor_locator(locator)

@iterate_over_axs
def set_ax_max_n_locator(ax, nbins, axis='both', locator_type='major', **kwargs):
    locator = ticker.MaxNLocator(nbins, **kwargs)
    set_ax_locator(ax, locator, axis=axis, locator_type=locator_type)

@iterate_over_axs
def set_ax_linear_locator(ax, numticks, axis='both', locator_type='major', **kwargs):
    '''
    会从min到max均匀分布的设置刻度(不实用,因为min有可能带有很多小数位数)
    '''
    locator = ticker.LinearLocator(numticks, **kwargs)
    set_ax_locator(ax, locator, axis=axis, locator_type=locator_type)

@iterate_over_axs
def set_ax_multiple_locator(ax, base, offset=0., axis='both', locator_type='major', **kwargs):
    '''
    会从base的倍数开始设置刻度
    '''
    locator = ticker.MultipleLocator(base, offset=offset, **kwargs)
    set_ax_locator(ax, locator, axis=axis, locator_type=locator_type)
# endregion


# region 初级作图函数(formatter)
def get_linear_but_log_formatter(base='10', label_format='auto'):
    '''
    假如散点等元素是log后作画的,ax本身的scale是linear,则使用这个formatter可以把tick显示为log的形式。
    base: 指数的底数,默认是10。
    label_format: 控制指数部分的格式，例如 '%.0f' 表示整数形式,'%.2f' 表示保留两位小数。默认是'auto'，即自动选择。
    '''
    def linear_but_log_formatter(x, pos):
        if label_format == 'auto':
            decimal_num = get_decimal_num(x)
            local_label_format = '%.' + str(decimal_num) + 'f'
        else:
            local_label_format = label_format
        return r"${}^{{{}}}$".format(base, local_label_format % x)
    
    return FuncFormatter(linear_but_log_formatter)

@iterate_over_axs
def set_linear_but_log_axis(ax, axis=None, base='10', label_format='auto'):
    '''
    注意:最好放在set ax后,不然有可能小数点的估计会错误
    '''
    if axis is None:
        axis = ['x', 'y']
    if isinstance(axis, str):
        axis = [axis]
    for axi in axis:
        if axi == 'x':
            if label_format == 'auto':
                max_decimal_num = get_max_decimal_num(ax.get_xticks())
                local_label_format = '%.' + str(max_decimal_num) + 'f'
            else:
                local_label_format = label_format
            ax.xaxis.set_major_formatter(get_linear_but_log_formatter(base=base, label_format=local_label_format))
        if axi == 'y':
            if label_format == 'auto':
                max_decimal_num = get_max_decimal_num(ax.get_yticks())
                local_label_format = '%.' + str(max_decimal_num) + 'f'
            else:
                local_label_format = label_format
            ax.yaxis.set_major_formatter(get_linear_but_log_formatter(base=base, label_format=local_label_format))


def get_log_e_formatter(label_format='auto'):
    '''
    假设ax本身的scale是log的并且base为e,则使用这个formatter可以把tick显示为e的形式(否则会有很多小数点)
    label_format: 控制指数部分的格式，例如 '%.0f' 表示整数形式,'%.2f' 表示保留两位小数。默认是'auto'，即自动选择。
    '''
    def log_e_formatter(x, pos):
        if label_format == 'auto':
            decimal_num = get_decimal_num(np.log(x))
            local_label_format = '%.' + str(decimal_num) + 'f'
        else:
            local_label_format = label_format
        return r"$e^{{{}}}$".format(local_label_format % np.log(x))
    
    return FuncFormatter(log_e_formatter)

@iterate_over_axs
def set_log_e_axis(ax, axis=None, label_format='auto'):
    '''
    注意:最好放在set ax后,不然有可能小数点的估计会错误
    '''
    if axis is None:
        axis = ['x', 'y']
    if isinstance(axis, str):
        axis = [axis]
    for axi in axis:
        if axi == 'x':
            if label_format == 'auto':
                max_decimal_num = get_max_decimal_num(np.log(ax.get_xticks()))
                local_label_format = '%.' + str(max_decimal_num) + 'f'
            else:
                local_label_format = label_format
            ax.xaxis.set_major_formatter(get_log_e_formatter(label_format=local_label_format))
        if axi == 'y':
            if label_format == 'auto':
                max_decimal_num = get_max_decimal_num(np.log(ax.get_yticks()))
                local_label_format = '%.' + str(max_decimal_num) + 'f'
            else:
                local_label_format = label_format
            ax.yaxis.set_major_formatter(get_log_e_formatter(label_format=local_label_format))


def get_sym_positive_formatter(label_format='auto'):
    '''
    获取对称正数的formatter
    '''
    def sym_positive_formatter(x, pos):
        if label_format == 'auto':
            decimal_num = get_decimal_num(x)
            local_label_format = '%.' + str(decimal_num) + 'f'
        else:
            local_label_format = label_format
        return local_label_format % abs(x)
    return FuncFormatter(sym_positive_formatter)

@iterate_over_axs
def set_sym_positive_axis(ax, axis=None, bound=None, label_format='auto'):
    '''
    将轴设置为对称正数
    '''
    if axis is None:
        axis = ['x', 'y']
    if isinstance(axis, str):
        axis = [axis]
    for axi in axis:
        if axi == 'x':
            # 将x轴设置为对称
            try:
                if bound is None:
                    bound = max(abs(ax.get_xlim()))
                ax.set_xlim(-bound, bound)
            except:
                pass

            # 获取label_format
            if label_format == 'auto':
                max_decimal_num = get_max_decimal_num(ax.get_xticks())
                local_label_format = '%.' + str(max_decimal_num) + 'f'
            else:
                local_label_format = label_format

            ax.xaxis.set_major_formatter(get_sym_positive_formatter(label_format=local_label_format))
        if axi == 'y':
            # 将y轴设置为对称
            try:
                if bound is None:
                    bound = max(abs(ax.get_ylim()))
                ax.set_ylim(-bound, bound)
            except:
                pass

            # 获取label_format
            if label_format == 'auto':
                max_decimal_num = get_max_decimal_num(ax.get_yticks())
                local_label_format = '%.' + str(max_decimal_num) + 'f'
            else:
                local_label_format = label_format

            ax.yaxis.set_major_formatter(get_sym_positive_formatter(label_format=local_label_format))
# endregion


# region 初级作图函数(norm)
def get_norm(norm_mode='linear', vmin=None, vmax=None, norm_kwargs=None):
    '''
    根据给定的模式创建一个matplotlib颜色规范化对象。

    参数:
    - norm_mode (str): 规范化模式。可以是'linear'、'log'、'symlog'、'two_slope'、'boundary'。
    - vmin (float or None, optional): 规范化范围的最小值。
    - vmax (float or None, optional): 规范化范围的最大值。
    - norm_kwargs (dict or None, optional): 规范化函数的其他关键字参数。

    返回:
    - norm: 基于指定模式的matplotlib规范化对象。
    '''
    if norm_kwargs is None:
        norm_kwargs = {}
    if norm_mode == 'linear':
        local_norm_kwargs = norm_kwargs.copy()
        if vmin is None:
            if 'vmin' not in local_norm_kwargs:
                vmin = 0
            else:
                vmin = local_norm_kwargs.pop('vmin')
        if vmax is None:
            if 'vmax' not in local_norm_kwargs:
                vmax = 1
            else:
                vmax = local_norm_kwargs.pop('vmax')
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax, **local_norm_kwargs)
    elif norm_mode == 'log':
        local_norm_kwargs = norm_kwargs.copy()
        if vmin is None:
            if 'vmin' not in local_norm_kwargs:
                vmin = 1e-1
            else:
                vmin = local_norm_kwargs.pop('vmin')
        if vmax is None:
            if 'vmax' not in local_norm_kwargs:
                vmax = 1
            else:
                vmax = local_norm_kwargs.pop('vmax')
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax, **local_norm_kwargs)
    elif norm_mode == 'symlog':
        local_norm_kwargs = norm_kwargs.copy()
        if vmin is None:
            if 'vmin' not in local_norm_kwargs:
                vmin = -10
            else:
                vmin = local_norm_kwargs.pop('vmin')
        if vmax is None:
            if 'vmax' not in local_norm_kwargs:
                vmax = 10
            else:
                vmax = local_norm_kwargs.pop('vmax')
        if 'linthresh' not in local_norm_kwargs:
            linthresh = 1
        else:
            linthresh = local_norm_kwargs.pop('linthresh')
        norm = mcolors.SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax, **local_norm_kwargs)
    elif norm_mode == 'two_slope':
        local_norm_kwargs = norm_kwargs.copy()
        if vmin is None:
            if 'vmin' not in local_norm_kwargs:
                vmin = -1
            else:
                vmin = local_norm_kwargs.pop('vmin')
        if vmax is None:
            if 'vmax' not in local_norm_kwargs:
                vmax = 1
            else:
                vmax = local_norm_kwargs.pop('vmax')
        if 'vcenter' not in local_norm_kwargs:
            vcenter = 0
        else:
            vcenter = local_norm_kwargs.pop('vcenter')
        norm = mcolors.TwoSlopeNorm(vcenter=vcenter, vmin=vmin, vmax=vmax, **local_norm_kwargs)
    elif norm_mode == 'boundary':
        local_norm_kwargs = norm_kwargs.copy()
        if 'boundaries' not in local_norm_kwargs:
            local_norm_kwargs['boundaries'] = np.linspace(0, 1, 6, endpoint=True)
        if 'ncolors' not in local_norm_kwargs:
            local_norm_kwargs['ncolors'] = 256
        norm = mcolors.BoundaryNorm(**local_norm_kwargs)
    return norm
# endregion


# region 初级作图函数(cmap)
def get_cmap(colors, continuous=True):
    '''
    生成颜色映射。
    :param colors: 颜色列表。
    :param continuous: 是否为连续颜色映射,默认为True。
    '''
    if len(colors) == 1:
        local_colors = colors * 2
    else:
        local_colors = colors.copy()
    if continuous:
        cmap = mcolors.LinearSegmentedColormap.from_list('custom', local_colors)
    else:
        cmap = mcolors.ListedColormap(colors)
    return cmap


def reverse_cmap(cmap):
    '''
    反转颜色映射。
    :param cmap: 颜色映射。
    '''
    return cmap.reversed()
# endregion


# region 初级作图函数(添加colorbar)
@iterate_over_axs
def add_colorbar(ax, mappable=None, cmap=CMAP, ticks=None, tick_labels=None, discrete_label=None, display_edge_ticks=True, cbar_position=None, cbar_label=None, use_mask=False, mask_color=MASK_COLOR, mask_pos='start', mask_pad=0, mask_cbar_ratio=None, mask_tick='mask', mask_tick_loc=None, label_size=CBAR_LABEL_SIZE, tick_size=CBAR_TICK_SIZE, adjust_tick_size=True, tick_proportion=TICK_PROPORTION, label_kwargs=None, norm_mode='linear', vmin=None, vmax=None, norm_kwargs=None, text_process=None, formatter=None, formatter_kwargs=None, round_digits=ROUND_DIGITS, round_format_type=ROUND_FORMAT, add_leq=False, add_geq=False, inset_mode='fig'):
    '''
    在指定ax添加颜色条。目前设置norm_mode为'boundary'时，最好输入一个离散的cmap，否则在log模式下会出现问题。
    :param ax: matplotlib的轴对象，用于绘制图形。
    :param mappable: 用于绘制颜色条的对象，默认为None。
    :param cmap: 颜色条的颜色映射，默认为CMAP。可以是离散的或连续的，须与discrete参数相符合。
    :param ticks: 颜色条的刻度,默认为None,自动设置(当不想显示tick的时候可以输入[]);如果输入了,则会使用输入的刻度
    :param tick_labels: 颜色条的刻度标签，默认为None,自动设置;如果输入了,则会使用输入的刻度标签
    :param discrete: 是否为离散颜色条，默认为False。
    :param discrete_num: 离散颜色条的数量，默认为5。
    :param discrete_label: 离散颜色条的标签，默认为None。
    :param display_edge_ticks: 是否显示颜色条的边缘刻度，默认为True。(只在norm_mode为'boundary'时并且discrete_label不为None时有效)
    :param cbar_position: 颜色条的位置，默认为None,即使用默认位置;position参数可选'left', 'right', 'top', 'bottom'
    :param cbar_label: 颜色条的标签，默认为None。
    :param use_mask: 是否使用mask颜色条，默认为False。
    :param mask_color: mask颜色条的颜色，默认为MASK_COLOR。
    :param mask_pad: mask_colorbar和colorbar的间距，默认为0。
    :param mask_cbar_ratio: mask颜色条的比例，默认为None。对于连续会自动设置为0.2，对于离散会自动设置为1/(discrete+1)
    :param mask_tick: mask颜色条的标签，默认为'mask'。
    :param mask_tick_loc: mask颜色条的标签位置，默认为None。
    :param label_size: 颜色条标签的字体大小，默认为CBAR_LABEL_SIZE。(这个是指colorbar旁边的colorbar标签的大小)
    :param tick_size: 颜色条刻度标签的字体大小，默认为CBAR_TICK_SIZE。
    :param adjust_tick_size: 是否根据颜色条的数量自动调整刻度标签的大小，默认为True。
    :param tick_proportion: 调整时的比例，默认为TICK_PROPORTION。
    :param label_kwargs: 颜色条标签的其他参数，默认为None。
    :param norm_mode: 颜色条的归一化模式，默认为'linear'。可选'linear', 'log', 'symlog', 'twoslope'.
    :param vmin: 颜色条的最小值，默认为None。
    :param vmax: 颜色条的最大值，默认为None。
    :param norm_kwargs: 归一化的其他参数，默认为None。具体会根据不同的norm_mode在内部设置默认值。
    :param text_process: 文本处理函数，默认为TEXT_PROCESS。
    :param round_digits: 刻度标签的小数位数，默认为ROUND_DIGITS。
    :param add_leq: 是否在最小值处添加'<=',默认为False。
    :param add_geq: 是否在最大值处添加'>=',默认为False。
    :param inset_mode: 插入模式，默认为'fig'。可选'fig', 'ax'。(指的是如果分割ax获得mask_ax和cbar_ax时,是插入到fig中还是ax中)
    '''
    # 设定默认参数
    norm_kwargs = update_dict({}, norm_kwargs)
    if mappable is None:
        # 根据norm_mode设置norm
        norm = get_norm(norm_mode=norm_mode, vmin=vmin, vmax=vmax, norm_kwargs=norm_kwargs)
        mappable = ScalarMappable(norm=norm, cmap=cmap)
    if label_kwargs is None:
        label_kwargs = {}
    if vmin is None:
        vmin = mappable.norm.vmin
    if vmax is None:
        vmax = mappable.norm.vmax

    # 更新默认值
    cbar_position = update_dict(CBAR_POSITION, cbar_position)
    text_process = update_dict(TEXT_PROCESS, text_process)

    # 根据vmin和vmax来clip这个mappable的范围
    mappable.set_clim(vmin, vmax)

    # 根据vmin和vmax判断是否add_leq和add_geq
    if vmin < mappable.norm.vmin:
        add_leq = True
    if vmax > mappable.norm.vmax:
        add_geq = True

    # 设置mask_cbar_ratio
    if mask_cbar_ratio is None:
        mask_cbar_ratio = 0.2
        if norm_mode == 'boundary':
            mask_cbar_ratio = 1/mappable.norm.boundaries.size

    # 格式化cbar_label和mask_tick
    cbar_label = format_text(cbar_label, text_process=text_process)
    mask_tick = format_text(mask_tick, text_process=text_process)

    # 格式化discrete_label
    if discrete_label is not None:
        discrete_label = [format_text(label, text_process=text_process) for label in discrete_label]

    # 获得orientation
    if cbar_position['position'] in ['right', 'left']:
        orientation = 'vertical'
    if cbar_position['position'] in ['top', 'bottom']:
        orientation = 'horizontal'

    # 获取cbar_ax和mask_ax
    if use_mask:
        if cbar_position['position'] in ['right', 'left']:
            if mask_pos == 'start':
                cbar_ax, mask_ax = split_ax_by_gs(ax, nrows=2, ncols=1, hspace=mask_pad, height_ratios=[1-mask_cbar_ratio, mask_cbar_ratio], label='cbar_mask', inset_mode=inset_mode)
            if mask_pos == 'end':
                mask_ax, cbar_ax = split_ax_by_gs(ax, nrows=2, ncols=1, hspace=mask_pad, height_ratios=[mask_cbar_ratio, 1-mask_cbar_ratio], label='mask_cbar', inset_mode=inset_mode)
        if cbar_position['position'] in ['top', 'bottom']:
            if mask_pos == 'start':
                mask_ax, cbar_ax = split_ax_by_gs(ax, nrows=1, ncols=2, wspace=mask_pad, width_ratios=[mask_cbar_ratio, 1-mask_cbar_ratio], label='mask_cbar', inset_mode=inset_mode)
            if mask_pos == 'end':
                cbar_ax, mask_ax = split_ax_by_gs(ax, nrows=1, ncols=2, wspace=mask_pad, width_ratios=[1-mask_cbar_ratio, mask_cbar_ratio], label='cbar_mask', inset_mode=inset_mode)

        # 去掉ax的坐标轴
        ax.axis('off')

        # 创建一个ScalarMappable对象，使用一个简单的颜色映射和Normalize对象,因为我们要显示一个纯色的colorbar，所以颜色映射范围不重要，只需确保vmin和vmax不同即可
        norm = Normalize(vmin=0, vmax=1)
        sm = ScalarMappable(norm=norm, cmap=get_cmap([mask_color, mask_color]))

        # 创建colorbar，将ScalarMappable作为输入
        mask_cbar = plt.colorbar(sm, cax=mask_ax, orientation=orientation)

        cbar = plt.colorbar(mappable, cax=cbar_ax, orientation=orientation)
    else:
        cbar = plt.colorbar(mappable, cax=ax, orientation=orientation)

    # 设置mask的刻度位置
    if use_mask:
        if norm_mode == 'boundary' and discrete_label is None:
            mask_cbar.set_ticks([0.0])   # 设置刻度位置在最侧边
        else:
            mask_cbar.set_ticks([0.5])   # 设置刻度位置在正中间
        
        # 假如输入了mask_tick_loc，则设置mask_cbar的刻度位置
        if mask_tick_loc is not None:
            mask_cbar.set_ticks([mask_tick_loc])

    # 设置colorbar的标签
    cbar.set_label(cbar_label, fontsize=label_size, **label_kwargs)

    # 假如颜色映射是一个离散的，则根据boundaries来设置ticks(假如输入了discrete_label,则插入中间刻度)
    if norm_mode == 'boundary':
        if ticks is None:
            ticks = mappable.norm.boundaries
            if discrete_label is not None:
                ticks = insert_mid(ticks)
                if not display_edge_ticks:
                    # 去掉边缘的刻度,只保留中间的刻度
                    ticks = ticks[1::2]
    else:
        if ticks is None:
            ticks = cbar.get_ticks()
        # 得到ticks中位于vmin和vmax之间的刻度
        ticks = [tick for tick in ticks if vmin <= tick <= vmax]

        # 当add_leq和add_geq为True时，强行添加末端的刻度
        if add_leq:
            if not np.allclose(vmin, ticks[0]):
                ticks = [vmin] + ticks
        if add_geq:
            if not np.allclose(vmax, ticks[-1]):
                ticks = ticks + [vmax]
    
    # 设置tick
    cbar.set_ticks(ticks)
    
    # format cbar的刻度标签
    if tick_labels is None:
        if formatter is None:
            if norm_mode == 'linear' or norm_mode == 'two_slope' or norm_mode == 'boundary':
                tick_labels = [round_float(tick, round_digits, round_format_type, latex=True) for tick in ticks]
            elif norm_mode == 'log' or norm_mode == 'symlog':
                tick_labels = [sci_float_to_latex(tick) for tick in ticks]
        else:
            if formatter_kwargs is None:
                formatter_kwargs = {}
            tick_labels = [formatter(tick, **formatter_kwargs) for tick in ticks]

    # 根据discrete_label进一步调整tick_labels
    if norm_mode == 'boundary':
        if discrete_label is not None:
            if display_edge_ticks:
                # 将tick_labels中间的刻度替换为discrete_label
                for i in range(len(ticks)-1):
                    if i % 2 == 1:
                        tick_labels[i] = discrete_label[i//2]
            else:
                # 将tick_labels全部替换为discrete_label
                tick_labels = discrete_label.copy()

    # 调整tick_size
    if adjust_tick_size:
        num_ticks = len(cbar.get_ticks())
        width, height = get_ax_size(cbar.ax)
        if cbar_position['position'] in ['right', 'left']:
            plt_size = height
        elif cbar_position['position'] in ['top', 'bottom']:
            plt_size = width
            tick_size = suitable_tick_size(num_ticks, plt_size, tick_size, tick_proportion)
    
    # 设置<=和>=
    if add_leq:
        if np.allclose(vmin, ticks[0]):
            tick_labels[0] = '≤' + tick_labels[0]
    if add_geq:
        if np.allclose(vmax, ticks[-1]):
            tick_labels[-1] = '≥' + tick_labels[-1]

    # 设置cbar刻度标签,并调节字体大小
    cbar.set_ticklabels(tick_labels, fontsize=tick_size)
    
    # 设置mask刻度标签,并调节字体大小
    if use_mask:
        mask_cbar.set_ticklabels([mask_tick], fontsize=tick_size)

    # 调整tick的方向
    if use_mask:
        cbar_list = [cbar, mask_cbar]
    else:
        cbar_list = [cbar]
    for cb in cbar_list:
        if cbar_position['position']=='right':
            pass
        elif cbar_position['position']=='left':
            cb.ax.yaxis.set_ticks_position('left')
        elif cbar_position['position']=='top':
            cb.ax.xaxis.set_ticks_position('top')
            cb.ax.set_xticks(cb.ax.get_xticks())
            cb.ax.set_xticklabels(cb.ax.get_xticklabels(), rotation=90)
        elif cbar_position['position']=='bottom':
            cb.ax.xaxis.set_ticks_position('bottom')
            cb.ax.set_xticks(cb.ax.get_xticks())
            cb.ax.set_xticklabels(cb.ax.get_xticklabels(), rotation=90)
        # 将spine全部去掉
        cb.outline.set_visible(False)

        # 将minor ticks全部去掉
        cb.ax.xaxis.set_tick_params(which='minor', length=0)
        cb.ax.yaxis.set_tick_params(which='minor', length=0)

    # return cbar
    if use_mask:
        add_custom_child_to_ax(ax, mask_cbar, f'mask_cbar_{mask_tick}')
        add_custom_child_to_ax(ax, cbar, f'cbar_{cbar_label}')
        return [cbar, mask_cbar]
    else:
        add_custom_child_to_ax(ax, cbar, f'cbar_{cbar_label}')
        return [cbar]

@iterate_over_axs
def add_side_colorbar(ax, mappable=None, cmap=CMAP, ticks=None, tick_labels=None, discrete_label=None, display_edge_ticks=True, cbar_position=None, cbar_label=None, use_mask=False, mask_color=MASK_COLOR, mask_pos='start', mask_pad=0, mask_cbar_ratio=None, mask_tick='mask', mask_tick_loc=None, label_size=CBAR_LABEL_SIZE, tick_size=CBAR_TICK_SIZE, adjust_tick_size=True, tick_proportion=TICK_PROPORTION, label_kwargs=None, norm_mode='linear', vmin=None, vmax=None, norm_kwargs=None, text_process=None, formatter=None, formatter_kwargs=None, round_digits=ROUND_DIGITS, round_format_type=ROUND_FORMAT, add_leq=False, add_geq=False, inset_mode='fig'):
    '''
    在指定ax的旁边添加颜色条。特别注意，对于离散的cmap，用户一定要提供对应的discrete_num
    :param ax: matplotlib的轴对象，用于绘制图形。
    :param mappable: 用于绘制颜色条的对象，默认为None。
    :param cmap: 颜色条的颜色映射，默认为CMAP。可以是离散的或连续的，须与discrete参数相符合。
    :param continuous_tick_num: 连续颜色条的刻度数量，默认为None,不作设置。
    :param discrete: 是否为离散颜色条，默认为False。
    :param discrete_num: 离散颜色条的数量，默认为5。
    :param discrete_label: 离散颜色条的标签，默认为None。
    :param display_edge_ticks: 是否显示颜色条的边缘刻度，默认为True。(只在离散颜色条下有效)
    :param display_center_ticks: 是否显示颜色条的中心刻度，默认为False。(只在离散颜色条下有效,并且假如discrete_label被设置了,即使display_center_ticks为True,也不会显示中心刻度)
    :param cbar_position: 颜色条的位置，默认为None,即使用默认位置;position参数可选'left', 'right', 'top', 'bottom'
    :param cbar_label: 颜色条的标签，默认为None。
    :param use_mask: 是否使用mask颜色条，默认为False。
    :param mask_color: mask颜色条的颜色，默认为MASK_COLOR。
    :param mask_pad: mask_colorbar和colorbar的间距，默认为0。
    :param mask_cbar_ratio: mask颜色条的比例，默认为None。对于连续会自动设置为0.2，对于离散会自动设置为1/(discrete+1)
    :param mask_tick: mask颜色条的标签，默认为'mask'。
    :param mask_tick_loc: mask颜色条的标签位置，默认为None。
    :param label_size: 颜色条标签的字体大小，默认为CBAR_LABEL_SIZE。
    :param tick_size: 颜色条刻度标签的字体大小，默认为CBAR_TICK_SIZE。
    :param adjust_tick_size: 是否根据颜色条的数量自动调整刻度标签的大小，默认为True。
    :param tick_proportion: 调整时的比例，默认为TICK_PROPORTION。
    :param label_kwargs: 颜色条标签的其他参数，默认为None。
    :param norm_mode: 颜色条的归一化模式，默认为'linear'。可选'linear', 'log', 'symlog', 'twoslope'.
    :param vmin: 颜色条的最小值，默认为None。
    :param vmax: 颜色条的最大值，默认为None。
    :param norm_kwargs: 归一化的其他参数，默认为None。具体会根据不同的norm_mode在内部设置默认值。
    :param text_process: 文本处理函数，默认为TEXT_PROCESS。
    :param round_digits: 刻度标签的小数位数，默认为ROUND_DIGITS。
    :param add_leq: 是否在最小值处添加'<=',默认为False。
    :param add_geq: 是否在最大值处添加'>=',默认为False。
    '''
    # 更新默认值
    if isinstance(ax, Axes3D):
        cbar_position = update_dict(CBAR_POSITION_3D, cbar_position)
    else:
        cbar_position = update_dict(CBAR_POSITION, cbar_position)

    side_ax = add_side_ax(ax, cbar_position['position'], cbar_position['size'], cbar_position['pad'], inset_mode=inset_mode)
    return add_colorbar(side_ax, mappable=mappable, cmap=cmap, ticks=ticks, tick_labels=tick_labels, discrete_label=discrete_label, display_edge_ticks=display_edge_ticks, cbar_position=cbar_position, cbar_label=cbar_label, use_mask=use_mask, mask_color=mask_color, mask_pos=mask_pos, mask_pad=mask_pad, mask_cbar_ratio=mask_cbar_ratio, mask_tick=mask_tick, mask_tick_loc=mask_tick_loc, label_size=label_size, tick_size=tick_size, adjust_tick_size=adjust_tick_size, tick_proportion=tick_proportion, label_kwargs=label_kwargs, norm_mode=norm_mode, vmin=vmin, vmax=vmax, norm_kwargs=norm_kwargs, text_process=text_process, formatter=formatter, formatter_kwargs=formatter_kwargs, round_digits=round_digits, round_format_type=round_format_type, add_leq=add_leq, add_geq=add_geq, inset_mode=inset_mode)

@iterate_over_axs
def add_scatter_colorbar(ax, mappable=None, cmap=CMAP, edgecolor=BLACK, tick_labels=None, cbar_label=None, cbar_position=None, label_size=CBAR_LABEL_SIZE, tick_size=CBAR_TICK_SIZE, text_pad=1.0, label_pad=None, adjust_tick_size=True, tick_proportion=TICK_PROPORTION, label_kwargs=None, vnorm_mode='linear', vmin=None, vmax=None, vnorm_kwargs=None, snorm_mode='linear', smin=None, smax=None, snorm_kwargs=None, smap=partial(scale_to_new_range, old_min=0, old_max=1, new_min=0.05, new_max=0.95), use_mask=None, mask_marker='X', mask_smap_float=1.0, mask_color=MASK_COLOR, mask_text='mask', epsilon=1e-3, text_process=None, formatter=None, formatter_kwargs=None, round_digits=ROUND_DIGITS, round_format_type=ROUND_FORMAT, add_leq=False, add_geq=False):
    '''
    在指定ax添加圆形颜色条。如果输入了mappable的同时指定了vmin和vmax,则会按照vmin,vmax来clip这个mappable的范围。
    :param ax: matplotlib的轴对象，用于绘制图形。
    :param scatter_smin: 圆形颜色条的scatter的s的最小值。
    :param scatter_smax: 圆形颜色条的scatter的s的最大值。
    :param mappable: 用于绘制颜色条的对象，默认为None。
    :param cmap: 颜色条的颜色映射，默认为CMAP。
    :param edgecolor: 圆形颜色条的边缘颜色，默认为BLACK。
    :param tick_labels: 颜色条的刻度标签，默认为None。
    :param cbar_label: 颜色条的标签，默认为None。
    :param cbar_position: 颜色条的位置，默认为None,即使用默认位置;position参数可选'left', 'right', 'top', 'bottom'
    :param label_size: 颜色条标签的字体大小，默认为CBAR_LABEL_SIZE。
    :param tick_size: 颜色条刻度标签的字体大小，默认为CBAR_TICK_SIZE。
    :param text_pad: scatter一旁的文本的间距，默认为1.0。
    :param label_pad: 颜色条标签的间距，默认为None。
    :param adjust_tick_size: 是否根据颜色条的数量自动调整刻度标签的大小，默认为True。
    :param tick_proportion: 调整时的比例，默认为TICK_PROPORTION。
    :param label_kwargs: 颜色条标签的其他参数，默认为None。
    :param vmin: 颜色条的最小值，默认为None。
    :param vmax: 颜色条的最大值，默认为None。
    :param vnorm_mode: 颜色条的归一化模式，默认为'linear'。可选'linear', 'log', 'symlog', 'twoslope'.
    :param vnorm_kwargs: 归一化的其他参数，默认为None。具体会根据不同的norm_mode在内部设置默认值。
    :param smin: 圆形颜色条的scatter的s的最小值，默认为None。
    :param smax: 圆形颜色条的scatter的s的最大值，默认为None。
    :param snorm_mode: 圆形颜色条的scatter的归一化模式，默认为'linear'。可选'linear', 'log', 'symlog', 'twoslope'.
    :param snorm_kwargs: 圆形颜色条的scatter的归一化的其他参数，默认为None。具体会根据不同的norm_mode在内部设置默认值。
    :param smap: 圆形颜色条的scatter的归一化函数，默认为partial(scale_to_new_range, old_min=0, old_max=1, new_min=0.05, new_max=0.95)。(自己用的时候可以修改new_min和new_max)
    :param epsilon: 圆形颜色条的scatter为0时的半径，默认为1e-3。(会自动带入到smap中)
    :param text_process: 文本处理函数，默认为TEXT_PROCESS。
    :param round_digits: 刻度标签的小数位数，默认为ROUND_DIGITS。
    :param round_format_type: 刻度标签的格式，默认为ROUND_FORMAT。
    :param add_leq: 是否在最小值处添加'<=',默认为False。
    :param add_geq: 是否在最大值处添加'>=',默认为False。
    '''
    # 获取norm以更新vmin和vmax,smin和smax
    cnorm = get_norm(vnorm_mode, vmin=vmin, vmax=vmax, norm_kwargs=vnorm_kwargs)
    snorm = get_norm(snorm_mode, vmin=smin, vmax=smax, norm_kwargs=snorm_kwargs)
    vmin, vmax = cnorm.vmin, cnorm.vmax
    smin, smax = snorm.vmin, snorm.vmax

    # 更新默认值
    text_process = update_dict(TEXT_PROCESS, text_process)
    cbar_position = update_dict(CBAR_POSITION, cbar_position)
    label_kwargs = update_dict({}, label_kwargs)

    # 由于cbar已经非常完善,这里调用cbar来获得需要的scatter的tick_labels(注意要输入smin等)
    if tick_labels is None:
        cbars = add_side_colorbar(ax, mappable=mappable, cmap=cmap, norm_mode=snorm_mode, vmin=smin, vmax=smax, norm_kwargs=snorm_kwargs, add_leq=add_leq, add_geq=add_geq, formatter=formatter, formatter_kwargs=formatter_kwargs, round_digits=round_digits, round_format_type=round_format_type)
        # 获取cbar的tick_labels
        tick_labels = cbars[0].ax.get_yticklabels()
        tick_labels = [label.get_text() for label in tick_labels]
        # 删除cbar
        for cbar in cbars:
            ax.figure.delaxes(cbar.ax)
    
    # 通过tick_labels来获得scatter的数量
    scatter_num = len(tick_labels)

    # 设置方向
    if cbar_position['position'] in ['right', 'left']:
        orientation = 'vertical'
    if cbar_position['position'] in ['top', 'bottom']:
        orientation = 'horizontal'

    # 调转方向
    if cbar_position['position'] == 'right':
        ax.yaxis.set_label_position('right')
    if cbar_position['position'] == 'top':
        ax.xaxis.set_label_position('top')
    
    # 计算出合适的text大小
    if adjust_tick_size:
        if orientation == 'vertical':
            plt_size = get_ax_size(ax)[1]
        elif orientation == 'horizontal':
            plt_size = get_ax_size(ax)[0]
        tick_size = suitable_tick_size(scatter_num, plt_size, tick_size, tick_proportion)

    # 添加圆形颜色条
    radius_list = smap(np.linspace(0, 1, scatter_num, endpoint=True))
    color_list = cmap(np.linspace(0, 1, scatter_num, endpoint=True))
    if np.allclose(mask_smap_float, 0.0):
        mask_idx = -1
    elif np.allclose(mask_smap_float, 1.0):
        mask_idx = scatter_num
    else:
        print('have not implemented yet')
    for i in list(range(scatter_num))+[mask_idx]:
        if orientation == 'vertical':
            center = (0.5, i)
            ax.set_xlim(0, 1)
            ax.set_ylim(-1, scatter_num)
        elif orientation == 'horizontal':
            center = (i, 0.5)
            ax.set_ylim(0, 1)
            ax.set_xlim(-1, scatter_num)
        if use_mask and i == mask_idx:
            ax.scatter(center[0], center[1], s=smap(mask_smap_float), color=mask_color, edgecolor=edgecolor, marker=mask_marker, clip_on=False)
        else:
            pass
        if i != mask_idx:
            text = tick_labels[i]
            ax.scatter(center[0], center[1], s=radius_list[i], color=color_list[i], edgecolor=edgecolor, clip_on=False) # clip_on=False表示不裁剪,防止框太小圆被裁剪
            if radius_list[i] == 0:
                ax.scatter(center[0], center[1], s=smap(epsilon), color=color_list[i], edgecolor=edgecolor, clip_on=False) # 稍微放大一点,防止点消失
        else:
            text = mask_text
        if cbar_position['position'] == 'right':
            add_text(ax, text=text, x=center[0]+text_pad, y=center[1], ha='left', va='center', fontsize=tick_size, color=edgecolor, transform='data', text_process=text_process, rotation=0)
        elif cbar_position['position'] == 'left':
            add_text(ax, text=text, x=center[0]-text_pad, y=center[1], ha='right', va='center', fontsize=tick_size, color=edgecolor, transform='data', text_process=text_process, rotation=0)
        elif cbar_position['position'] == 'top':
            add_text(ax, text=text, x=center[0], y=center[1]+text_pad, ha='center', va='bottom', fontsize=tick_size, color=edgecolor, transform='data', text_process=text_process, rotation=90)
        elif cbar_position['position'] == 'bottom':
            add_text(ax, text=text, x=center[0], y=center[1]-text_pad, ha='center', va='top', fontsize=tick_size, color=edgecolor, transform='data', text_process=text_process, rotation=90)

    # 取消框
    rm_ax_spine(ax)
    ax.set_xticks([])
    ax.set_yticks([])

    # 添加colorbar标签
    cbar_label = format_text(cbar_label, text_process=text_process)
    if cbar_position['position'] in ['right', 'left']:
        ax.set_ylabel(cbar_label, fontsize=label_size, labelpad=label_pad, **label_kwargs)
    elif cbar_position['position'] in ['top', 'bottom']:
        ax.set_xlabel(cbar_label, fontsize=label_size, labelpad=label_pad, **label_kwargs)

@iterate_over_axs
def add_side_scatter_colorbar(ax, mappable=None, cmap=CMAP, edgecolor=BLACK, tick_labels=None, cbar_label=None, cbar_position=None, label_size=CBAR_LABEL_SIZE, tick_size=CBAR_TICK_SIZE, text_pad=1.0, label_pad=None, adjust_tick_size=True, tick_proportion=TICK_PROPORTION, label_kwargs=None, vnorm_mode='linear', vmin=None, vmax=None, vnorm_kwargs=None, snorm_mode='linear', smin=None, smax=None, snorm_kwargs=None, smap=partial(scale_to_new_range, old_min=0, old_max=1, new_min=0.05, new_max=0.95), use_mask=None, mask_marker='X', mask_smap_float=1.0, mask_color=MASK_COLOR, mask_text='mask', epsilon=1e-3, text_process=None, formatter=None, formatter_kwargs=None, round_digits=ROUND_DIGITS, round_format_type=ROUND_FORMAT, add_leq=False, add_geq=False, inset_mode='fig'):
    if isinstance(ax, Axes3D):
        cbar_position = update_dict(CBAR_POSITION_3D, cbar_position)
    else:
        cbar_position = update_dict(CBAR_POSITION, cbar_position)

    side_ax = add_side_ax(ax, cbar_position['position'], cbar_position['size'], cbar_position['pad'], inset_mode=inset_mode)
    return add_scatter_colorbar(side_ax, mappable=mappable, cmap=cmap, edgecolor=edgecolor, tick_labels=tick_labels, cbar_label=cbar_label, cbar_position=cbar_position, label_size=label_size, tick_size=tick_size, text_pad=text_pad, label_pad=label_pad, adjust_tick_size=adjust_tick_size, tick_proportion=tick_proportion, label_kwargs=label_kwargs, vnorm_mode=vnorm_mode, vmin=vmin, vmax=vmax, vnorm_kwargs=vnorm_kwargs, snorm_mode=snorm_mode, smin=smin, smax=smax, snorm_kwargs=snorm_kwargs, smap=smap, use_mask=use_mask, mask_marker=mask_marker, mask_smap_float=mask_smap_float, mask_color=mask_color, mask_text=mask_text, epsilon=epsilon, text_process=text_process, formatter=formatter, formatter_kwargs=formatter_kwargs, round_digits=round_digits, round_format_type=round_format_type, add_leq=add_leq, add_geq=add_geq)
# endregion


# region 初级作图函数(添加边缘分布)
@iterate_over_axs
def add_marginal_distribution(ax, data, side_ax=None, side_ax_position='right', side_ax_pad=SIDE_PAD, side_ax_size=0.3, outside=True, color=BLUE, hist=True, stat='density', bins=BIN_NUM, hist_kwargs=None, kde=True, kde_kwargs=None, rm_tick=True, rm_spine=True, rm_axis=True, inset_mode='fig'):
    '''
    在指定位置添加边缘分布。
    :param ax: matplotlib的轴对象，用于绘制图形。
    :param x: x轴的数据。
    :param y: y轴的数据。
    :param color: 边缘分布的颜色，默认为BLUE。
    :param hist_kwargs: 传递给`sns.histplot`的其他参数。
    '''
    if hist_kwargs is None:
        hist_kwargs = {}
    if kde_kwargs is None:
        kde_kwargs = {}
    
    if side_ax_position in ['right', 'left']:
        hist_vert = False
    elif side_ax_position in ['top', 'bottom']:
        hist_vert = True

    # 得到边缘分布需要的ax
    if side_ax is None:
        if outside:
            if side_ax_position in ['right', 'left']:
                side_ax = add_side_ax(ax, side_ax_position, side_ax_size, side_ax_pad, sharey=True, inset_mode=inset_mode, hide_repeat_xaxis=False, hide_repeat_yaxis=False)
            elif side_ax_position in ['top', 'bottom']:
                side_ax = add_side_ax(ax, side_ax_position, side_ax_size, side_ax_pad, sharex=True, inset_mode=inset_mode, hide_repeat_xaxis=False, hide_repeat_yaxis=False)
        else:
            if side_ax_position == 'right':
                # 目前ax模式不支持keep_original为一个特定索引(后续可能会支持)
                ax, side_ax = split_ax_by_gs(ax, nrows=1, ncols=2, wspace=side_ax_pad, width_ratios=[1-side_ax_size, side_ax_size], label='side_ax', inset_mode=inset_mode, sharey=True, keep_original=(0,) if inset_mode=='fig' else False)
            elif side_ax_position == 'bottom':
                ax, side_ax = split_ax_by_gs(ax, nrows=2, ncols=1, hspace=side_ax_pad, height_ratios=[1-side_ax_size, side_ax_size], label='side_ax', inset_mode=inset_mode, sharex=True, keep_original=(1,) if inset_mode=='fig' else False)
            elif side_ax_position == 'left':
                side_ax, ax = split_ax_by_gs(ax, nrows=1, ncols=2, wspace=side_ax_pad, width_ratios=[side_ax_size, 1-side_ax_size], label='side_ax', inset_mode=inset_mode, sharey=True, keep_original=(1,) if inset_mode=='fig' else False)
            elif side_ax_position == 'top':
                side_ax, ax = split_ax_by_gs(ax, nrows=2, ncols=1, hspace=side_ax_pad, height_ratios=[side_ax_size, 1-side_ax_size], label='side_ax', inset_mode=inset_mode, sharex=True, keep_original=(0,) if inset_mode=='fig' else False)

    # 绘制边缘分布
    if hist:
        plt_hist(ax=side_ax, data=data, bins=bins, color=color, stat=stat, vert=hist_vert, **hist_kwargs)
    if kde:
        plt_kde(ax=side_ax, data=data, color=color, vert=hist_vert, **kde_kwargs)
    side_ax.set_xlabel('')  # 隐藏x轴标签
    side_ax.set_ylabel('')  # 隐藏y轴标签

    # 隐藏边缘分布的刻度和坐标轴
    if rm_tick:
        rm_ax_tick(side_ax)
    if rm_spine:
        rm_ax_spine(side_ax)
    if rm_axis:
        rm_ax_axis(side_ax)

    # 调整边缘分布的方向
    if side_ax_position == 'left':
        side_ax.invert_xaxis()
    if side_ax_position == 'bottom':
        side_ax.invert_yaxis()
    return side_ax, ax

@iterate_over_axs
def add_double_marginal_distribution(ax, x, y, x_side_ax=None, y_side_ax=None, outside=True, x_side_ax_position='top', y_side_ax_position='right', x_side_ax_pad=SIDE_PAD, y_side_ax_pad=SIDE_PAD, x_side_ax_size=0.3, y_side_ax_size=0.3, x_color=BLUE, y_color=BLUE, hist=True, stat='density', x_bins=BIN_NUM, y_bins=BIN_NUM, x_hist_kwargs=None, y_hist_kwargs=None, kde=True, x_kde_kwargs=None, y_kde_kwargs=None, rm_tick=True, rm_spine=True, rm_axis=True, inset_mode='fig'):
    '''
    在指定位置添加两个方向的边缘分布。
    :param ax: matplotlib的轴对象，用于绘制图形。
    :param x: x轴的数据。
    :param y: y轴的数据。
    :param outside: 是否将边缘分布放在外部，默认为True。
    :param x_side_ax_position: x轴边缘分布的位置，默认为'top'。
    :param y_side_ax_position: y轴边缘分布的位置，默认为'right'。
    :param x_side_ax_pad: x轴边缘分布与原ax的间距，默认为0.05。
    :param y_side_ax_pad: y轴边缘分布与原ax的间距，默认为0.05。
    :param x_side_ax_size: x轴边缘分布的大小，默认为0.3。
    :param y_side_ax_size: y轴边缘分布的大小，默认为0.3。
    :param x_color: x轴边缘分布的颜色，默认为BLUE。
    :param y_color: y轴边缘分布的颜色，默认为BLUE。
    :param hist: 是否绘制直方图，默认为True。
    :param stat: 直方图的统计量，默认为'probability'。
    :param x_bins: x轴直方图的箱数，默认为BIN_NUM。
    :param y_bins: y轴直方图的箱数，默认为BIN_NUM。
    :param x_hist_kwargs: 传递给`plt_hist`的其他参数。
    :param y_hist_kwargs: 传递给`plt_hist`的其他参数。
    :param kde: 是否绘制核密度估计，默认为True。
    :param x_kde_kwargs: 传递给`plt_kde`的其他参数。
    :param y_kde_kwargs: 传递给`plt_kde`的其他参数。
    :param rm_tick: 是否隐藏刻度，默认为True。
    :param rm_spine: 是否隐藏坐标轴，默认为False。
    '''
    if x_hist_kwargs is None:
        x_hist_kwargs = {}
    if y_hist_kwargs is None:
        y_hist_kwargs = {}
    if x_kde_kwargs is None:
        x_kde_kwargs = {}
    if y_kde_kwargs is None:
        y_kde_kwargs = {}
    
    # 添加边缘分布
    if outside:
        x_side_ax, ax = add_marginal_distribution(ax, x, x_side_ax, x_side_ax_position, side_ax_pad=x_side_ax_pad, side_ax_size=x_side_ax_size, outside=outside, color=x_color, hist=hist, stat=stat, bins=x_bins, hist_kwargs=x_hist_kwargs, kde=kde, kde_kwargs=x_kde_kwargs, rm_tick=rm_tick, rm_spine=rm_spine, rm_axis=rm_axis, inset_mode=inset_mode)
        y_side_ax, ax = add_marginal_distribution(ax, y, y_side_ax, y_side_ax_position, side_ax_pad=y_side_ax_pad, side_ax_size=y_side_ax_size, outside=outside, color=y_color, hist=hist, stat=stat, bins=y_bins, hist_kwargs=y_hist_kwargs, kde=kde, kde_kwargs=y_kde_kwargs, rm_tick=rm_tick, rm_spine=rm_spine, rm_axis=rm_axis, inset_mode=inset_mode)
    else:
        if x_side_ax is None and y_side_ax is None:
            # 当边缘分布在内部时，需要手动分划ax然后获取需要的
            x_side_ax, y_side_ax, ax = split_with_double_marginal_ax(ax=ax, x_side_ax_position=x_side_ax_position, y_side_ax_position=y_side_ax_position, x_side_ax_pad=x_side_ax_pad, y_side_ax_pad=y_side_ax_pad, x_side_ax_size=x_side_ax_size, y_side_ax_size=y_side_ax_size, inset_mode=inset_mode)
        # 绘制边缘分布
        if hist:
            plt_hist(ax=x_side_ax, data=x, bins=x_bins, color=x_color, stat=stat, vert=True, **x_hist_kwargs)
            plt_hist(ax=y_side_ax, data=y, bins=y_bins, color=y_color, stat=stat, vert=False, **y_hist_kwargs)
        if kde:
            plt_kde(ax=x_side_ax, data=x, color=x_color, vert=True, **x_kde_kwargs)
            plt_kde(ax=y_side_ax, data=y, color=y_color, vert=False, **y_kde_kwargs)

        # 隐藏边缘分布的刻度和坐标轴
        if rm_tick:
            rm_ax_tick(x_side_ax)
            rm_ax_tick(y_side_ax)
        if rm_spine:
            rm_ax_spine(x_side_ax)
            rm_ax_spine(y_side_ax)
        if rm_axis:
            rm_ax_axis(x_side_ax)
            rm_ax_axis(y_side_ax)

        if x_side_ax_position == 'bottom':
            x_side_ax.invert_yaxis()
        if y_side_ax_position == 'left':
            y_side_ax.invert_xaxis()
    return x_side_ax, y_side_ax, ax
# endregion


# region 初级作图函数(添加star)
def add_star(ax, x, y, label=None, marker=STAR, color=RED, markersize=STAR_SIZE, linestyle='None', **kwargs):
    '''
    在指定位置添加五角星标记。
    :param ax: matplotlib的轴对象，用于绘制图形。
    :param x: 五角星的x坐标。
    :param y: 五角星的y坐标。
    :param label: 五角星的标签，默认为None。
    :param marker: 五角星的标记，默认为STAR。
    :param color: 五角星的颜色，默认为RED。
    :param markersize: 五角星的大小，默认为STAR_SIZE。
    :param linestyle: 连接五角星的线型，默认为'None'。
    :param kwargs: 传递给`plot`函数的额外关键字参数。
    '''
    # 画图
    return ax.plot(x, y, label=label, marker=marker, color=color, markersize=markersize, linestyle=linestyle, **kwargs)


def polygon_star(ax, center, start_angle=np.pi / 2, num_points=5, outer_radius=1, inner_radius=0.4, color=RED, fill=True, adjust_lim=True, **kwargs):
    """
    在指定的Axes对象上绘制一个星形。(相较于add_star, 此函数可以更加灵活地设置fill等来绘制不同的星形。)
    
    参数:
    - ax: matplotlib的Axes对象，用于绘制星形。
    - center: 星形的中心点。
    - start_angle: 星形的起始角度，默认为np.pi / 2。
    - num_points: 星形的点数，默认为5。
    - outer_radius: 星形外圈的半径。
    - inner_radius: 星形内圈的半径。
    - color: 星形的颜色，默认为RED。
    - fill: 是否填充星形，默认为True。
    - adjust_lim: 是否调整坐标轴范围，默认为True。
    - kwargs: 传递给`plt_polygon`的额外关键字参数。
    """
    angles = np.linspace(start_angle, start_angle + 2 * np.pi, num_points * 2, endpoint=False)
    radius = np.array([outer_radius if i % 2 == 0 else inner_radius for i in range(num_points * 2)])
    
    # 计算顶点坐标
    points = np.vstack((radius * np.cos(angles), radius * np.sin(angles))).T
    points += np.array(center)  # 调整中心
    
    # 绘制星形
    return plt_polygon(ax, points, color=color, fill=fill, adjust_lim=adjust_lim, **kwargs)


def add_star_heatmap(ax, i, j, label=None, marker=STAR, color=RED, markersize=STAR_SIZE, linestyle='None', **kwargs):
    '''
    在热图的指定位置添加五角星标记。

    参数:
    - ax: matplotlib的Axes对象，即热图的绘图区域。
    - i, j: 要在其上添加五角星的行和列索引。(支持同时添加多个五角星，i和j可以是数组)
    - label: 五角星的标签，默认为None。
    - marker: 五角星的标记，默认为STAR。
    - color: 五角星的颜色，默认为RED。
    - markersize: 五角星的大小，默认为STAR_SIZE。
    - linestyle: 连接五角星的线型，默认为'None'。
    - kwargs: 传递给`plot`函数的额外关键字参数。
    '''

    # 转换行列索引为坐标位置
    x, y = np.array(j)+0.5, np.array(i)+0.5  # 0.5是为了将五角星放在格子的中心

    # 画图
    return add_star(ax, x, y, label=label, marker=marker, color=color, markersize=markersize, linestyle=linestyle, **kwargs)
# endregion


# region 初级作图函数(添加辅助线)
def add_vline(ax, x, label=None, color=RED, linestyle=AUXILIARY_LINE_STYLE, linewidth=LINE_WIDTH, **kwargs):
    '''
    在指定位置添加垂直线。
    :param ax: matplotlib的轴对象,用于绘制图形。
    :param x: 垂直线的x坐标。
    :param label: 垂直线的标签,默认为None。
    :param color: 垂直线的颜色,默认为RED。
    :param linestyle: 垂直线的线型,默认为AUXILIARY_LINE_STYLE。
    :param linewidth: 垂直线的线宽,默认为LINE_WIDTH。
    :param kwargs: 传递给`ax.axvline`的额外关键字参数。
    '''
    # 画图
    return ax.axvline(x, label=label, color=color, linestyle=linestyle, linewidth=linewidth, **kwargs)


def add_hline(ax, y, label=None, color=RED, linestyle=AUXILIARY_LINE_STYLE, linewidth=LINE_WIDTH, **kwargs):
    '''
    在指定位置添加水平线。
    :param ax: matplotlib的轴对象，用于绘制图形。
    :param y: 水平线的y坐标。
    :param label: 水平线的标签，默认为None。
    :param color: 水平线的颜色，默认为RED。
    :param linestyle: 水平线的线型，默认为AUXILIARY_LINE_STYLE。
    :param linewidth: 水平线的线宽，默认为LINE_WIDTH。
    :param kwargs: 传递给`ax.axhline`的额外关键字参数。
    '''
    # 画图
    return ax.axhline(y, label=label, color=color, linestyle=linestyle, linewidth=linewidth, **kwargs)


def add_sline(ax, slope, intercept, reference_point='auto', label=None, color=RED, linestyle=AUXILIARY_LINE_STYLE, linewidth=LINE_WIDTH, **kwargs):
    '''
    按照斜率和截距添加直线。(贯穿整个ax)

    add_sline意为add_slope_line
    :param ax: matplotlib的轴对象，用于绘制图形。
    :param slope: 直线的斜率。
    :param intercept: 直线的截距。
    :param reference_point: 直线的参考点，默认为'auto'。可输入浮点数
    :param label: 直线的标签，默认为None。
    :param color: 直线的颜色，默认为RED。
    :param linestyle: 直线的线型，默认为AUXILIARY_LINE_STYLE。
    :param linewidth: 直线的线宽，默认为LINE_WIDTH。
    :param kwargs: 传递给`ax.axline`的额外关键字参数。
    '''
    if reference_point == 'auto':
        reference_point = (ax.get_xlim()[0] + ax.get_xlim()[1]) / 2
    reference_intercept = slope * reference_point + intercept
    return ax.axline((reference_point, reference_intercept), slope=slope, label=label, color=color, linestyle=linestyle, linewidth=linewidth, **kwargs)


def add_grid(ax, x_list=None, y_list=None, color=RANA, linestyle=AUXILIARY_LINE_STYLE, linewidth=LINE_WIDTH, **kwargs):
    '''
    在指定位置添加网格线。
    :param ax: matplotlib的轴对象，用于绘制图形。
    :param x: 网格线的x坐标，默认为None。
    :param y: 网格线的y坐标，默认为None。
    :param label: 网格线的标签，默认为None。
    :param color: 网格线的颜色，默认为RED。
    :param linestyle: 网格线的线型，默认为AUXILIARY_LINE_STYLE。
    :param linewidth: 网格线的线宽，默认为LINE_WIDTH。
    :param kwargs: 传递给`ax.grid`的额外关键字参数。
    '''
    # 画图
    for x in x_list:
        ax.axvline(x, color=color, linestyle=linestyle, linewidth=linewidth, **kwargs)
    for y in y_list:
        ax.axhline(y, color=color, linestyle=linestyle, linewidth=linewidth, **kwargs)
# endregion


# region 初级作图函数(添加圆角矩形)
def add_rounded_rectangle(ax, pad=0.05, corner_radius=0.02, edgecolor="black", facecolor="white", **kwargs):
    """
    在坐标轴周围添加圆角矩形框
    
    参数：
    ax: matplotlib坐标轴对象
    pad: 边距(归一化到画布尺寸,0-1之间)
    corner_radius: 圆角半径(归一化到画布尺寸)
    edgecolor: 边框颜色
    facecolor: 填充颜色
    **kwargs: 其他图形参数(颜色、线宽等)
    """
    # 获取坐标轴在画布中的位置信息
    bbox = ax.get_position()
    
    # 计算新边界框的位置和尺寸
    x0 = bbox.x0 - pad
    y0 = bbox.y0 - pad
    width = bbox.width + 2*pad
    height = bbox.height + 2*pad
    
    # 创建圆角矩形补丁
    rect = mpatches.FancyBboxPatch(
        (x0, y0), width, height,
        boxstyle=f"round,pad=0,rounding_size={corner_radius}",
        transform=ax.figure.transFigure,  # 使用画布坐标系
        zorder=ax.get_zorder()-1,         # 确保在坐标轴下方
        edgecolor=edgecolor,
        facecolor=facecolor,
        **kwargs
    )
    
    # 将补丁添加到画布
    ax.figure.patches.append(rect)
    return rect
# endregion


# region 初级作图函数(添加span)
def add_vspan(ax, xmin, xmax, label=None, color=GREEN, alpha=FAINT_ALPHA, **kwargs):
    '''
    在指定位置添加垂直跨度。
    '''
    # 画图
    return ax.axvspan(xmin, xmax, label=label, color=color, alpha=alpha, **kwargs)


def add_hspan(ax, ymin, ymax, label=None, color=GREEN, alpha=FAINT_ALPHA, **kwargs):
    '''
    在指定位置添加水平跨度。
    '''
    # 画图
    return ax.axhspan(ymin, ymax, label=label, color=color, alpha=alpha, **kwargs)


def add_span(ax, color=GREEN, alpha=FAINT_ALPHA, **kwargs):
    '''
    将整个图像区域涂色。
    '''
    # 画图
    color_a = rgb_to_rgba(color, alpha)
    return ax.set_facecolor(color_a)
# endregion


# region 初级作图函数(创建patch)
def add_gradient_patch(ax, patch, extent, transform='data', auto_scale=True, vert=True, cmap=DENSITY_CMAP, gradient=None, alpha=None, vmin=None, vmax=None, imshow_kwargs=None):
    '''
    创建一个渐变色的patch。

    参数:
    - ax: matplotlib的Axes对象,用于绘制图形
    - patch: matplotlib的Patch对象,用于裁剪渐变色
    - extent: patch的范围,为(xmin, xmax, ymin, ymax)
    - transform: patch的坐标系,默认为None即ax数据坐标系
    - auto_scale: 是否自动调整坐标轴范围,默认为True
    - vert: 渐变色的方向,默认为True即垂直方向
    - cmap: 渐变色的颜色映射,默认为DENSITY_CMAP
    - gradient: 渐变色的数组,默认为None即自动生成
    '''
    imshow_kwargs = update_dict({}, imshow_kwargs)
    if transform == 'data':
        transform = ax.transData
    elif transform == 'axes':
        transform = ax.transAxes
        print('user should create a new axes with the desired position, set the xylim of these new axes and add patch on this new axes by transform=ax.transData')
        print("this may be modified in a new version, with the code like# 创建数据data = np.random.random((10, 10))fig, ax = plt.subplots()# 创建一个 AxesImage 对象image = AxesImage(ax, interpolation='nearest', cmap='viridis')# 将图像数据与对象关联image.set_data(data)# 设置变换为 ax.transAxesimage.set_transform(ax.transAxes)# 设置 extent 为 [0.1, 0.4] 等比例坐标image.set_extent([0.1, 0.4, 0.1, 0.4])# 添加到轴中ax.add_image(image)但是实际使用的时候似乎会改变ax的xylim所以需要之后再看下")

    if gradient is None:
        # 创建渐变数组
        gradient = np.linspace(0, 1, 256).reshape(1, -1)
        gradient = np.vstack((gradient, gradient))  # 扩展到2行
    elif isinstance(gradient, (list, tuple)):
        gradient = np.array(gradient).reshape(1, -1)
        gradient = np.vstack((gradient, gradient))  # 扩展到2行
    elif isinstance(gradient, np.ndarray):
        if gradient.ndim == 1:
            gradient = gradient.reshape(1, -1)
            gradient = np.vstack((gradient, gradient))  # 扩展到2行
        elif gradient.ndim == 2:
            pass

    if vert:
        gradient = gradient.T  # 垂直方向上的渐变

    # 在 ax 中绘制渐变色
    im = ax.imshow(gradient, aspect='auto', cmap=cmap, extent=extent, transform=transform, alpha=alpha, vmin=vmin, vmax=vmax, **imshow_kwargs)

    # 裁剪渐变色
    im.set_clip_path(patch)

    # 自动缩放视图
    if auto_scale:
        ax.autoscale_view()


def add_patch(ax, patch, auto_scale=True):
    '''
    在ax上添加patch。

    参数:
    - ax: matplotlib的Axes对象,用于绘制图形。
    - patch: matplotlib的Patch对象,用于添加到ax。
    - auto_scale: 是否自动调整坐标轴范围,默认为True。(如果为False,xlim和ylim不会被影响)
    '''
    ax.add_patch(patch)
    if auto_scale:
        ax.autoscale_view()


def add_path_patch(ax, vertices, codes=None, facecolor='none', edgecolor=BLACK, auto_scale=True, lw=LINE_WIDTH, **kwargs):
    '''
    利用path创建patch。

    参数:
    - vertices: 顶点坐标。示例: [[0, 0], [1, 0], [1, 1], [0, 1]]
    - codes: 顶点代码。示例: [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY],默认为None,意为第一个是MOVETO,其余是LINETO。CLOSYPOLY表示闭合到MOVETO的点;比较高级的用法还有CURVE3, CURVE4等,可以作出贝塞尔曲线
    - facecolor: 填充颜色,默认为'none'
    - edgecolor: 边框颜色,默认为BLACK
    - auto_scale: 是否自动调整坐标轴范围,默认为True
    '''
    p = mpatches.PathPatch(mpl.path.Path(vertices, codes), facecolor=facecolor, edgecolor=edgecolor, lw=lw, **kwargs)
    add_patch(ax, p, auto_scale=auto_scale)
    return p


def add_polygon_patch(ax, xy, facecolor='none', edgecolor=BLACK, auto_scale=True, lw=LINE_WIDTH, **kwargs):
    '''
    创建多边形patch。

    参数:
    - xy: 顶点坐标。示例: [[0, 0], [1, 0], [1, 1], [0, 1]]
    - facecolor: 填充颜色,默认为'none'
    - edgecolor: 边框颜色,默认为BLACK
    - auto_scale: 是否自动调整坐标轴范围,默认为True
    '''
    p = mpatches.Polygon(xy, facecolor=facecolor, edgecolor=edgecolor, lw=lw, **kwargs)
    add_patch(ax, p, auto_scale=auto_scale)
    return p


def add_circle_patch(ax, center, radius, facecolor='none', edgecolor=BLACK, auto_scale=True, lw=LINE_WIDTH, **kwargs):
    '''
    创建圆形patch。

    参数:
    - center: 圆心坐标。
    - radius: 圆半径。
    - facecolor: 填充颜色,默认为'none'
    - edgecolor: 边框颜色,默认为BLACK
    - auto_scale: 是否自动调整坐标轴范围,默认为True
    '''
    p = mpatches.Circle(center, radius, facecolor=facecolor, edgecolor=edgecolor, lw=lw, **kwargs)
    add_patch(ax, p, auto_scale=auto_scale)
    return p


def add_arc_patch(ax, center, radius, theta1, theta2, facecolor='none', edgecolor=BLACK, auto_scale=True, lw=LINE_WIDTH, angle_type='rad', **kwargs):
    '''
    创建圆弧patch

    angle_type: 角度类型,默认为'rad',即弧度制.可选'rad'或'deg'
    '''
    if angle_type == 'rad':
        theta1 = np.rad2deg(theta1)
        theta2 = np.rad2deg(theta2)
    elif angle_type == 'deg':
        pass
    else:
        raise ValueError("angle_type must be 'rad' or 'deg'")
    p = mpatches.Arc(center, 2*radius, 2*radius, theta1=theta1, theta2=theta2, facecolor=facecolor, edgecolor=edgecolor, lw=lw, **kwargs)
    add_patch(ax, p, auto_scale=auto_scale)
    return p
# endregion


# region 初级作图函数(clip)
def clip_ax_by_patch(ax, patch):
    '''
    利用patch对ax进行裁剪。

    参数:
    - ax: matplotlib的Axes对象,用于绘制图形。
    - patch: matplotlib的Patch对象,用于裁剪ax。(可以使用Rectangle, Circle等)
    '''
    for artist in ax.get_children():
        artist.set_clip_path(patch)


def clip_ax_by_polygon(ax, xy, patch_kwargs=None):
    '''
    利用xy构成的polygon对ax进行裁剪。
    '''
    patch_kwargs = update_dict(dict(alpha=0., edgecolor='none'), patch_kwargs)
    patch = add_polygon_patch(ax, xy, **patch_kwargs)
    clip_ax_by_patch(ax, patch)


def clip_ax_by_circle(ax, center, radius, patch_kwargs=None):
    '''
    利用center和radius构成的circle对ax进行裁剪。
    '''
    patch_kwargs = update_dict(dict(alpha=0., edgecolor='none'), patch_kwargs)
    patch = add_circle_patch(ax, center, radius, **patch_kwargs)
    clip_ax_by_patch(ax, patch)


def clip_ax_by_path(ax, vertices, codes=None, patch_kwargs=None):
    '''
    利用vertices和codes构成的path对ax进行裁剪。
    '''
    patch_kwargs = update_dict(dict(alpha=0., edgecolor='none'), patch_kwargs)
    patch = add_path_patch(ax, vertices, codes, **patch_kwargs)
    clip_ax_by_patch(ax, patch)
# endregion


# region 初级作图函数(添加箭头)
def add_arrow(ax, x_start, y_start, x_end, y_end, xycoords='data', label=None, fc=RED, ec=RED, linewidth=LINE_WIDTH, arrow_sytle=ARROW_STYLE, head_width=ARROW_HEAD_WIDTH, head_length=ARROW_HEAD_LENGTH, alpha=1.0, **kwargs):
    '''
    在指定位置添加箭头。更换了使用方式,现在是指定起点和终点,而不是指定增量,并且内部实际调用ax.annotation来保证箭头的头大小相对于ax美观,箭头的终点严格对应xy_end。可以处理输入单个箭头或多个箭头的情况。支持list,array,单个数字并且不会改变原始数据类型。
    :param ax: matplotlib的轴对象，用于绘制图形。
    :param x_start: 箭头的起始x坐标。
    :param y_start: 箭头的起始y坐标。
    :param x_end: 箭头的终止x坐标。
    :param y_end: 箭头的终止y坐标。
    :param label: 箭头的标签，默认为None。
    :param fc: 箭头的填充颜色，默认为RED。
    :param ec: 箭头的边框颜色，默认为RED。
    :param linestyle: 箭头的线型，默认为'-'。
    :param linewidth: 箭头的线宽，默认为LINE_WIDTH。
    :param arrow_sytle: 箭头的样式，默认为ARROW_STYLE。
    :param head_width: 箭头的宽度，默认为ARROW_HEAD_WIDTH。
    :param head_length: 箭头的长度，默认为ARROW_HEAD_LENGTH。
    :param adjust_end_point: 是否调整箭头的终点,使箭头的end坐标为箭头三角形终点(True)或者箭头直线的终点(False)
    :param kwargs: 传递给`ax.arrow`的额外关键字参数。
    '''

    # 确保起点和终点坐标可以迭代
    x_start, y_start, x_end, y_end = map(
        np.atleast_1d, [x_start, y_start, x_end, y_end])
    n_arrows = len(x_start)

    for i in range(n_arrows):
        add_annotation(ax, '', (x_end[i], y_end[i]), (x_start[i], y_start[i]), xycoords=xycoords, arrowprops={
                       'fc': fc, 'ec': ec, 'linewidth': linewidth, 'arrowstyle': arrow_sytle, 'head_length': head_length, 'head_width': head_width, 'alpha': alpha}, **kwargs)

    # 为了添加label而使用的虚拟箭头
    ax.arrow([], [], [], [], label=label, fc=fc, ec=ec,
             linewidth=linewidth, head_width=0, head_length=0, **kwargs)


def add_mid_arrow(ax, x_start, y_start, x_end, y_end, xycoords='data', label=None, fc=RED, ec=RED, linewidth=LINE_WIDTH, arrow_sytle=ARROW_STYLE, head_width=ARROW_HEAD_WIDTH, head_length=ARROW_HEAD_LENGTH, alpha=1.0, **kwargs):
    '''
    添加一个从起点到终点的箭头，箭头的中间有一个箭头头。支持输入单个箭头或多个箭头的情况。
    '''
    x_mid = (x_start + x_end) / 2
    y_mid = (y_start + y_end) / 2
    add_arrow(ax, x_start, y_start, x_mid, y_mid, xycoords=xycoords, label=label, fc=fc, ec=ec, linewidth=linewidth, arrow_sytle=arrow_sytle, head_width=head_width, head_length=head_length, alpha=alpha, **kwargs)
    plt_line(ax, [x_start, x_end], [y_start, y_end], color=fc, linestyle='-', linewidth=linewidth, alpha=alpha, **kwargs)


def add_double_arrow(ax, x_start, y_start, x_end, y_end, xycoords='data', label=None, fc=RED, ec=RED, linewidth=LINE_WIDTH, arrow_sytle=ARROW_STYLE, head_width=ARROW_HEAD_WIDTH, head_length=ARROW_HEAD_LENGTH, alpha=1.0, **kwargs):
    '''
    添加一个从起点到终点的双箭头。支持输入单个箭头或多个箭头的情况。
    '''
    add_arrow(ax, x_start, y_start, x_end, y_end, xycoords=xycoords, label=label, fc=fc, ec=ec, linewidth=linewidth, arrow_sytle=arrow_sytle, head_width=head_width, head_length=head_length, alpha=alpha, **kwargs)
    add_arrow(ax, x_end, y_end, x_start, y_start, xycoords=xycoords, label=label, fc=fc, ec=ec, linewidth=linewidth, arrow_sytle=arrow_sytle, head_width=head_width, head_length=head_length, alpha=alpha, **kwargs)


def add_quiver_3d(ax, x_start, y_start, z_start, x_end, y_end, z_end, label=None, color=RED, linewidth=LINE_WIDTH, arrow_length_ratio=0.3, alpha=1.0, **kwargs):
    '''
    在3D图中添加箭头。https://matplotlib.org/stable/api/_as_gen/mpl_toolkits.mplot3d.axes3d.Axes3D.quiver.html#mpl_toolkits.mplot3d.axes3d.Axes3D.quiver
    注意: 如果设定length,这里的length会基于原先的箭头加倍得到长度,而不是直接设定长度。
    '''
    return ax.quiver(x_start, y_start, z_start, x_end-x_start, y_end-y_start, z_end-z_start, label=label, color=color, linewidth=linewidth, arrow_length_ratio=arrow_length_ratio, alpha=alpha, **kwargs)
# endregion


# region 初级作图函数(添加注释)
def add_annotation(ax, text, xy, xytext, xycoords='data', fontsize=FONT_SIZE, arrowprops=None, text_process=None, **kwargs):
    '''
    在指定位置添加注释。只支持一个注释。
    :param ax: matplotlib的轴对象，用于绘制图形。
    :param text: 注释的文本。
    :param xy: 注释的位置。即箭头的终止位置。
    :param xytext: 注释文本的位置。即箭头的起始位置。
    :param xycoords: 注释的坐标系，默认为'data'。也可以选择'axes fraction', 'subfigure fraction'等,见https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.annotate.html#matplotlib.axes.Axes.annotate。
    :param fontsize: 注释文本的字体大小，默认为FONT_SIZE。
    :param arrowprops: 箭头的属性
    :param text_process: 文本处理参数，默认为TEXT_PROCESS。
    :param kwargs: 传递给`ax.annotate`的额外关键字参数。
    '''
    text_process = update_dict(TEXT_PROCESS, text_process)
    arrowprops = update_dict(ARROW_PROPS, arrowprops)

    # 自动调整dict
    if 'head_length' in arrowprops.keys() and 'head_length' not in arrowprops['arrowstyle']:
        arrowprops['arrowstyle'] += ',head_length=' + \
            str(arrowprops['head_length'])
        del arrowprops['head_length']
    if 'head_width' in arrowprops.keys() and 'head_width' not in arrowprops['arrowstyle']:
        arrowprops['arrowstyle'] += ',head_width=' + \
            str(arrowprops['head_width'])
        del arrowprops['head_width']

    if xycoords == 'data':
        # 添加一个s=0的散点,因为annotation函数不会自动调整xy轴的坐标系让箭头可见
        ax.scatter([xy[0], xytext[0]], [xy[1], xytext[1]], s=0)

    # 处理文本
    text = format_text(text, text_process)
    # 画图
    return ax.annotate(text, xy, xytext=xytext, xycoords=xycoords, fontsize=fontsize, arrowprops=arrowprops, **kwargs)


def add_bar_label(ax, bars, labels, label_type='edge', padding=0., rotation=0., text_process=None, **kwargs):
    '''
    在柱状图上添加标签

    参数:
    - ax: matplotlib的Axes对象,用于绘制图形
    - bars: ax.bar或者ax.barh的返回值
    - labels: 标签的内容
    - label_type: 标签的位置,默认为'edge'(在柱子顶点),可选'center'(在柱子中间)
    - padding: 标签与柱子的距离,默认为0(可以为负数)
    - rotation: 标签的旋转角度,默认为0
    - text_process: 文本处理参数,默认为TEXT_PROCESS

    注意:
    不可以使用va和ha
    '''
    text_process = update_dict(TEXT_PROCESS, text_process)
    local_labels = [format_text(label, text_process) for label in labels]
    return ax.bar_label(bars, labels=local_labels, label_type=label_type, padding=padding, rotation=rotation, **kwargs)
# endregion


# region 初级作图函数(文字)
def _check_maybe_wrong_transform_for_text(x, y, transform):
    if transform != 'data':
        if x > 2 or x < -2 or y > 2 or y < -2:
            print(f"Warning: x={x}, y={y}, transform={transform} may be wrong.")


def add_text(ax, text, x=TEXT_X, y=TEXT_Y, text_process=None, transform='ax', va=TEXT_VA, ha=TEXT_HA, fontsize=FONT_SIZE, color=BLACK, **kwargs):
    '''

    在指定位置添加文字。
    :param ax: matplotlib的轴对象,用于绘制图形
    :param text: 文本内容
    :param x: 文本的x坐标,默认为0.05
    :param y: 文本的y坐标,默认为0.95
    :param text_process: 文本处理参数,默认为TEXT_PROCESS
    :param transform: 文本的坐标系,默认为'ax'.也可以选择'fig'或'data'
    :param kwargs: 传递给'ax.text'的额外关键字参数
    '''
    text_process = update_dict(TEXT_PROCESS, text_process)
    _check_maybe_wrong_transform_for_text(x, y, transform)
    # 处理文本
    text = format_text(text, text_process)

    if transform == 'ax':
        transform = ax.transAxes
    elif transform == 'fig':
        transform = ax.figure.transFigure
    elif transform == 'data':
        transform = ax.transData

    return ax.text(x, y, text, transform=transform, va=va, ha=ha, fontsize=fontsize, color=color, **kwargs)


def adjust_text(fig_or_ax, text, new_position=None, new_text=None, text_kwargs=None, text_process=None):
    '''
    调整指定文本对象的文本,位置和对齐方式。
    
    参数:
    - fig_or_ax: matplotlib的Figure或Axes对象,指定在哪里搜索文本对象
    - text: str, 要搜索和调整的文本内容
    - new_position: tuple, 新的位置,格式为(x, y)
    - new_text: str, 新的文本内容,默认为None,表示不修改文本内容
    - text_kwargs: dict, 传递给'text.update'的其他参数,默认为None

    注意:
    改变位置时,坐标的ha,va,transform等属性和原先一样(除非指定了新的text_kwargs)
    '''
    text_process = update_dict(TEXT_PROCESS, text_process)

    # 确定搜索范围是在Figure级别还是Axes级别
    if isinstance(fig_or_ax, plt.Figure):
        text_objects = fig_or_ax.findobj(plt.Text)
    else:
        text_objects = fig_or_ax.texts

    # 遍历找到的text对象
    for text_object in text_objects:
        if text_object.get_text() == text or text_object.get_text() == format_text(text, text_process):
            if new_position is not None:
                text_object.set_position(new_position)
            if new_text is not None:
                text_object.set_text(format_text(new_text, text_process))
            if text_kwargs is not None:
                text_object.update(text_kwargs)


def adjust_text_obj(text_obj, new_position=None, new_text=None, text_kwargs=None, text_process=None):
    '''
    调整指定文本对象的文本,位置和对齐方式。
    
    参数:
    - text_obj: matplotlib的Text对象，要调整的文本对象。
    - new_position: tuple, 新的位置，格式为(x, y)。
    - new_text: str, 新的文本内容，默认为None,表示不修改文本内容。
    - text_kwargs: dict, 传递给`text.update`的其他参数，默认为None。

    注意:
    改变位置时,坐标的ha,va,transform等属性和原先一样(除非指定了新的text_kwargs)
    '''
    text_process = update_dict(TEXT_PROCESS, text_process)

    if new_position is not None:
        text_obj.set_position(new_position)
    if new_text is not None:
        text_obj.set_text(format_text(new_text, text_process))
    if text_kwargs is not None:
        text_obj.update(text_kwargs)


def adjust_text_obj_transform(text_obj, new_transform):
    """
    将Matplotlib text对象的坐标系转换为新的坐标系，同时保持其在图像上的视觉位置不变。

    参数:
    - text_obj: 要更改transform的Text对象。
    - new_transform: 新的transform对象。
    """
    # 获取text当前的位置
    x_old, y_old = text_obj.get_position()
    
    # 获取text在figure坐标系中的位置
    coords_in_fig = text_obj.get_transform().transform((x_old, y_old))
    
    # 将figure坐标系中的位置转换为新坐标系中的位置
    coords_in_new_transform = new_transform.inverted().transform(coords_in_fig)
    
    # 更新text的位置和坐标系，保持视觉上的位置不变
    text_obj.set_position(coords_in_new_transform)
    text_obj.set_transform(new_transform)


def adjust_text_obj_alignment(text_obj, ax=None, ha=None, va=None):
    """
    调整文本对象的对齐方式，同时尽可能保持其在图上的视觉位置不变。

    参数:
    - text_obj: Matplotlib的Text对象。
    - ha: 新的水平对齐方式。
    - va: 新的垂直对齐方式。

    注意:
    只对transdata变换的文本对象有效。
    """
    # 获取文本对象当前的位置、对齐方式和变换参数
    x0, y0 = text_obj.get_position()
    
    # 计算原始边界框
    renderer = plt.gcf().canvas.get_renderer()
    bbox_before = text_obj.get_window_extent(renderer)
    
    # 更新对齐方式
    if va:
        text_obj.set_verticalalignment(va)
    if ha:
        text_obj.set_horizontalalignment(ha)
    
    # 计算更新对齐方式后的边界框
    bbox_after = text_obj.get_window_extent(renderer)
    
    # 计算边界框的差异，并调整文本位置
    dx = bbox_before.x0 - bbox_after.x0 + (bbox_before.width - bbox_after.width) * 0.5
    dy = bbox_before.y0 - bbox_after.y0 + (bbox_before.height - bbox_after.height) * 0.5
    
    # 将差异转换为数据坐标系中的值
    if ax is None:
        ax = text_obj.axes
    inv = ax.transData.inverted()
    dx_data, dy_data = inv.transform((dx, dy)) - inv.transform((0, 0))
    
    # 更新文本位置
    text_obj.set_position((x0 + dx_data, y0 + dy_data))


def align_text_obj(text_obj_list, ref_text_obj, ref_ax=None, ha_align_mode='left', va_align_mode='bottom'):
    """
    将文本对象列表中的文本对象与参考文本对象对齐。

    参数:
    - text_obj_list: 包含要对齐的文本对象的列表。
    - ref_text_obj: 参考文本对象，用于对齐。
    - ha_align_mode: 水平对齐方式，默认为'left'。假如设置为None，则不对齐。
    - va_align_mode: 垂直对齐方式，默认为'bottom'。假如设置为None，则不对齐。

    注意:
    - 对于xylabel,title等文本对象，这个函数不适用
    """
    # 获取参考文本的ax
    if ref_ax is None:
        ref_ax = ref_text_obj.axes

    # 将参考文本设置为left, bottom, transData
    adjust_text_obj_transform(ref_text_obj, ref_ax.transData)
    adjust_text_obj_alignment(ref_text_obj, ax=ref_ax, ha='left', va='bottom')

    # 获取新的位置
    new_ref_x, new_ref_y = ref_text_obj.get_position()

    for text_obj in text_obj_list:
        # 更新文本对象的对齐方式和变换
        adjust_text_obj_transform(text_obj, ref_ax.transData)
        adjust_text_obj_alignment(text_obj, ax=ref_ax, ha=ha_align_mode, va=va_align_mode)

        # 获取当前文本对象的宽度和高度
        bbox = text_obj.get_window_extent(renderer=plt.gcf().canvas.get_renderer())
        text_width = bbox.width
        text_height = bbox.height

        # 根据对齐方式计算新位置
        if ha_align_mode == 'left':
            new_x = new_ref_x
        elif ha_align_mode == 'right':
            new_x = new_ref_x + text_width
        elif ha_align_mode == 'center':
            new_x = new_ref_x - text_width / 2
        elif ha_align_mode is None:
            new_x = text_obj.get_position()[0]
        else:
            raise ValueError("Unsupported ha align mode: {}".format(ha_align_mode))
        if va_align_mode == 'top':
            new_y = new_ref_y + text_height
        elif va_align_mode == 'bottom':
            new_y = new_ref_y
        elif va_align_mode == 'center':
            new_y = new_ref_y - text_height / 2
        elif va_align_mode is None:
            new_y = text_obj.get_position()[1]
        else:
            raise ValueError("Unsupported va align mode: {}".format(va_align_mode))

        text_obj.set_position((new_x, new_y))


def _scale_font_size(text_obj, factor):
    if text_obj:                      # could be None
        text_obj.set_fontsize(text_obj.get_fontsize() * factor)


def scale_ax_font_size(ax, factor=1.0):
    """缩放单个 Axes 内所有文字"""

    # 标题与轴标签
    _scale_font_size(ax.title, factor)
    _scale_font_size(ax.xaxis.label, factor)
    _scale_font_size(ax.yaxis.label, factor)

    # 刻度
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        _scale_font_size(t, factor)

    # # legend（直接修改，保持位置等属性）
    lgd = ax.get_legend()
    if lgd:
        _scale_font_size(lgd.get_title(), factor)
        for t in lgd.get_texts():
            _scale_font_size(t, factor)

    # 通过 ax.text 添加的文字
    for t in ax.texts:
        if t not in (ax.title, ax.xaxis.label, ax.yaxis.label):
            _scale_font_size(t, factor)


def scale_fig_font_size(fig, factor=1.0):
    """缩放整张 Figure 内所有文字"""

    _scale_font_size(getattr(fig, "_suptitle", None), factor)  # suptitle
    for t in fig.texts:                              # fig.text
        _scale_font_size(t, factor)
    for ax in fig.axes:                              # 每个子图
        scale_ax_font_size(ax, factor)
# endregion


# region 初级作图函数(添加tag)
def add_fig_tag(fig, tag, x=FIG_TAG_POS[0], y=FIG_TAG_POS[1], fontsize=TAG_SIZE, va=TAG_VA, ha=TAG_HA, **kwargs):
    '''
    在指定位置添加图的标签
    :param fig: matplotlib的图对象,用于绘制图形
    :param tag: 标签的文本
    :param x: 标签的x坐标,默认为FIG_TAG_POS[0]
    :param y: 标签的y坐标,默认为FIG_TAG_POS[1]
    :param fontsize: 标签的字体大小,默认为SUP_TITLE_SIZE
    :param va: 标签的垂直对齐方式,默认为TAG_VA
    :param ha: 标签的水平对齐方式,默认为TAG_HA
    :param kwargs: 传递给'fig.text'的额外关键字参数
    '''
    # 画图
    return fig.text(x, y, tag, fontsize=fontsize, va=va, ha=ha, **kwargs)


def add_ax_tag(ax, tag, x=AX_TAG_POS[0], y=AX_TAG_POS[1], fontsize=TAG_SIZE, va=TAG_VA, ha=TAG_HA, **kwargs):
    '''
    在指定位置添加轴的标签
    :param ax: matplotlib的轴对象,用于绘制图形
    :param tag: 标签的文本
    :param x: 标签的x坐标,默认为AX_TAG_POS[0]
    :param y: 标签的y坐标,默认为AX_TAG_POS[1]
    :param fontsize: 标签的字体大小,默认为TITLE_SIZE
    :param va: 标签的垂直对齐方式,默认为TAG_VA
    :param ha: 标签的水平对齐方式,默认为TAG_HA
    :param kwargs: 传递给'ax.text'的额外关键字参数
    '''
    # 画图
    return add_text(ax, tag, x=x, y=y, fontsize=fontsize, va=va, ha=ha, **kwargs)


def add_ax_tag_by_ref_ax(ax, tag, ref_ax, x=AX_TAG_POS[0], y=AX_TAG_POS[1], fontsize=TAG_SIZE, va=TAG_VA, ha=TAG_HA, **kwargs):
    '''
    在指定位置添加轴的标签,参考另一个轴的位置
    '''
    if x > 0:
        raise ValueError("x should be less than 0")
    if y < 1:
        raise ValueError("y should be greater than 1")
    if ref_ax is not None:
        ref_ax_width, ref_ax_height = get_ax_size(ref_ax)
        ax_width, ax_height = get_ax_size(ax)
        x = x * ref_ax_width / ax_width
        y = (y - 1) * ref_ax_height / ax_height + 1
    return add_ax_tag(ax, tag, x=x, y=y, fontsize=fontsize, va=va, ha=ha, **kwargs)


def add_axes_dict_tag(axes_dict, tag_dict=None, ref_ax=None, **kwargs):
    '''
    按照tag_dict的键值对,给axes_dict中的每个轴添加标签

    参数:
    - axes_dict: dict, 包含matplotlib的Axes对象的字典
    - tag_dict: dict or None, 标签的字典,默认为None,表示使用axes_dict的键值对(即将axes_dict的key作为标签显示)
    - ref_ax: matplotlib的Axes对象,用于参考位置,默认为None,表示不参考位置
    '''
    if tag_dict is None:
        tag_dict = {ax_name: ax_name for ax_name in axes_dict.keys()}
    for ax_name, tag in tag_dict.items():
        add_ax_tag_by_ref_ax(axes_dict[ax_name], tag, ref_ax, **kwargs)


def add_axes_list_tag_by_order(axes_list, tag_kwargs=None, ref_ax=None, **kwargs):
    '''
    按照顺序加a,b,c...标签。
    '''
    tag_kwargs = update_dict({}, tag_kwargs)
    tag_list = get_tag(len(axes_list), **tag_kwargs)
    for i, ax in enumerate(axes_list):
        add_ax_tag_by_ref_ax(ax, tag_list[i], ref_ax, **kwargs)
# endregion


# region 复杂作图函数(matplotlib系列,输入向量使用)
def plt_edge_scatter(ax, x, y, color=BLUE, edge_color=BLACK, s=MARKER_SIZE**2, edge_prop=1.1, label=None, density=False, edge_kwargs=None, sc_kwargs=None):
    '''
    绘制带边界的散点图。

    参数:
    - ax (matplotlib.axes.Axes): matplotlib的轴对象，用于绘制图形。
    - x (numpy.ndarray or list): x轴的数据。
    - y (numpy.ndarray or list): y轴的数据。 
    - color (str or tuple, optional): 散点的颜色，默认为BLUE。
    - edge_color (str or tuple, optional): 散点的边界颜色，默认为BLACK。
    - s (float, optional): 散点的大小，默认为MARKER_SIZE**2。
    - edge_prop (float, optional): 散点的边界比例，默认为1.1。
    - label (str or None, optional): 散点的标签，默认为None。
    - density (bool, optional): 是否绘制密度图，默认为False。
    - edge_kwargs (dict or None, optional): 传递给plt_scatter的其他参数。
    - sc_kwargs (dict or None, optional): 传递给plt_density_scatter或者plt_scatter的其他参数，取决于是否使用density。

    返回:
    - matplotlib.collections.PathCollection or None: 如果density为True，则返回plt_density_scatter的返回值，否则返回plt_scatter的返回值。
    '''
    # 设置默认参数
    if edge_kwargs is None:
        edge_kwargs = {}
    if sc_kwargs is None:
        sc_kwargs = {}

    # 绘制边界
    edge_size = s * edge_prop
    plt_scatter(ax, x, y, color=edge_color, s=edge_size, **edge_kwargs)
    
    # 绘制散点
    if density:
        local_sc_kwargs = sc_kwargs.copy()
        if 'scatter_kwargs' in sc_kwargs:
            local_sc_kwargs['scatter_kwargs'] = update_dict(sc_kwargs['scatter_kwargs'], {'s': s})
        else:
            local_sc_kwargs = update_dict(sc_kwargs, {'scatter_kwargs': {'s': s}})
        return plt_density_scatter(ax, x, y, label=label, **local_sc_kwargs)
    else:
        local_sc_kwargs = update_dict(sc_kwargs, {'s': s})
        return plt_scatter(ax, x, y, color=color, label=label, **local_sc_kwargs)


def plt_colorful_scatter(ax, x, y, c, cmap=CMAP, norm_mode='linear', vmin=None, vmax=None, norm_kwargs=None, s=MARKER_SIZE**2, label=None, label_cmap_float=1.0, scatter_kwargs=None, cbar=True, cbar_postion=None, cbar_kwargs=None):
    '''
    绘制颜色关于c值变化的散点图。

    参数:
    - ax (matplotlib.axes.Axes): matplotlib的轴对象, 用于绘制图形。
    - x (numpy.ndarray or list): x轴的数据。
    - y (numpy.ndarray or list): y轴的数据。 
    - c (numpy.ndarray or list): 颜色的数据。
    - cmap (matplotlib.colors.Colormap, optional): 颜色映射, 默认为CMAP。
    - norm_mode (str, optional): 颜色映射的规范化模式, 可选 'linear', 'log', 'symlog', 'two_slope'等, 默认为 'linear'
    - vmin (float, optional): 颜色映射的最小值, 默认为None。
    - vmax (float, optional): 颜色映射的最大值, 默认为None。
    - norm_kwargs (dict or None, optional): 颜色映射规范化的其他参数
    - s (float, optional): 散点的大小, 默认为MARKER_SIZE**2。
    - label (str or None, optional): 散点的标签, 默认为None。
    - label_cmap_float (float, optional): 代表性点的颜色映射模式, 默认为1.0。如果输入整数则会raise ValueError。
    - scatter_kwargs (dict or None, optional): 传递给plt_scatter的其他参数。
    - cbar (bool, optional): 是否添加颜色条,默认为True。
    - cbar_postion (str or None, optional): 颜色条的位置,默认为None。
    - cbar_kwargs (dict or None, optional): 传递给add_side_colorbar的其他参数。
    '''
    # 设置默认参数
    if scatter_kwargs is None:
        scatter_kwargs = {}
    if cbar_kwargs is None:
        cbar_kwargs = {}
    if vmin is None:
        vmin = np.nanmin(c)
    if vmax is None:
        vmax = np.nanmax(c)

    # 获取颜色映射的规范化对象
    norm = get_norm(norm_mode=norm_mode, vmin=vmin, vmax=vmax, norm_kwargs=norm_kwargs)

    # 绘制散点(暂时设置label为None)
    local_scatter_kwargs = update_dict(scatter_kwargs, {'s': s})
    sc = plt_scatter(ax, x, y, color=None, c=c, cmap=cmap, label=None, norm=norm, **local_scatter_kwargs)

    # 通过label_cmap_float来绘制代表性点并作出label
    if isinstance(label_cmap_float, int):
        print('warning: "label_cmap_float" is an integer, so it will be transformed to float')
    ax.scatter([], [], color=cmap(float(label_cmap_float)), label=label, s=s)

    # 添加颜色条
    if cbar:
        if vmin > np.nanmin(c):
            cbar_kwargs['add_leq'] = True
        if vmax < np.nanmax(c):
            cbar_kwargs['add_geq'] = True
        cbars = add_side_colorbar(ax, sc, vmin=vmin, vmax=vmax, norm_mode=norm_mode, norm_kwargs=norm_kwargs, cmap=cmap, cbar_position=cbar_postion, **cbar_kwargs)
        return sc, cbars
    else:
        return sc


def plt_colorful_line(ax, x, y, c, cmap=CMAP, norm_mode='linear', vmin=None, vmax=None, norm_kwargs=None, cbar=True, cbar_postion=None, cbar_kwargs=None, line_kwargs=None, adjust_lim=True):
    '''
    绘制颜色关于c值变化的线图。
    
    参数:
    - ax (Axes): 绘图轴
    - x (numpy.ndarray): x 坐标数据
    - y (numpy.ndarray): y 坐标数据
    - c (numpy.ndarray): 用于颜色映射的数据
    - cmap (str, optional): 颜色映射方案, 默认为 CMAP
    - norm_mode (str, optional): 颜色映射的规范化模式, 可选 'linear', 'log', 'symlog', 'two_slope'等, 默认为 'linear'
    - vmin (float, optional): 颜色映射的最小值, 默认为 c 的最小值
    - vmax (float, optional): 颜色映射的最大值, 默认为 c 的最大值
    - norm_kwargs (dict, optional): 颜色映射规范化的其他参数
    - cbar (bool, optional): 是否绘制颜色条, 默认为 True
    - cbar_postion (str, optional): 颜色条的位置
    - cbar_kwargs (dict, optional): 颜色条的其他参数
    - line_kwargs (dict, optional): mcoll.LineCollection的其他参数
    - adjust_lim (bool, optional): 是否调整坐标轴范围, 默认为 True
    '''
    # 设置默认参数
    if cbar_kwargs is None:
        cbar_kwargs = {}
    if line_kwargs is None:
        line_kwargs = {}

    # 设置 vmin 和 vmax 的默认值
    if vmin is None:
        vmin = np.nanmin(c)
    if vmax is None:
        vmax = np.nanmax(c)

    # 构建线条段
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # 获取颜色映射的规范化对象
    norm = get_norm(norm_mode=norm_mode, vmin=vmin, vmax=vmax, norm_kwargs=norm_kwargs)

    # 创建 LineCollection 对象并设置属性
    lc = mcoll.LineCollection(segments, cmap=cmap, norm=norm, **line_kwargs)
    lc.set_array(c)
    line = ax.add_collection(lc)

    # 调整坐标轴范围
    if adjust_lim:
        ax.plot(x, y, linewidth=0)

    # 添加颜色条
    if cbar:
        if vmin > np.nanmin(c):
            cbar_kwargs['add_leq'] = True
        if vmax < np.nanmax(c):
            cbar_kwargs['add_geq'] = True
        cbars = add_side_colorbar(ax, line, vmin=vmin, vmax=vmax, norm_mode=norm_mode, norm_kwargs=norm_kwargs, cmap=cmap, cbar_position=cbar_postion, **cbar_kwargs)
        return line, cbars
    else:
        return line


def plt_group_bar(ax, x, y, label_list, width=None, colors=CMAP, vert=True, **kwargs):
    '''
    绘制分组的柱状图。
    :param ax: matplotlib的轴对象,用于绘制图形
    :param x: x轴的分组标签,大组,每个组包含多个柱子(比如['A', 'B'])
    :param y: 一个二维列表或数组,表示每组中柱子的高度(shape=(len(x), len(label_list)),例如[[1, 2, 3], [4, 5, 6]])
    :param label_list: 每个柱子的标签,例如['x', 'y', 'z']
    :param bar_width: 单个柱子的宽度,默认为None,自动确定宽度
    :param colors: 柱状图的颜色序列,应与label_list的长度相匹配;也可以指定cmap,默认为CMAP,然后根据label_list的长度生成颜色序列
    :param kwargs: 其他plt.bar支持的参数
    '''
    # 假如colors不是一个list,则用colors对应的cmap生成颜色
    if not isinstance(colors, list):
        colors = colors(np.linspace(0, 1, len(label_list)))

    num_groups = len(y)  # 组的数量
    num_bars = len(y[0])  # 每组中柱子的数量，假设每组柱子数量相同

    if width is None:
        width = BAR_WIDTH / num_bars

    # 为每个柱子计算中心位置
    indices = np.arange(num_groups)
    for i, lbl in enumerate(label_list):
        offsets = (np.arange(num_bars) -
                   np.arange(num_bars).mean()) * width
        plt_bar(ax, indices + offsets[i], [y[j][i] for j in range(
            num_groups)], label=lbl, color=colors[i], width=width, vert=vert, **kwargs)

    if vert:
        ax.set_xticks(indices)
        ax.set_xticklabels(x)
    else:
        ax.set_yticks(indices)
        ax.set_yticklabels(x)


def plt_two_side_bar(ax, x, y1, y2, label1=None, label2=None, width=BAR_WIDTH, color1=BLUE, color2=RED, yerr1=None, yerr2=None, capsize=PLT_CAP_SIZE, ecolor1=BLACK, ecolor2=BLACK, elabel1=None, elabel2=None, vert=True, equal_space=False, **kwargs):
    '''
    使用x和y1,y2绘制两组柱状图
    '''
    plt_bar(ax, x, np.array(y1), label=label1, color=color1, vert=vert, equal_space=equal_space,
                 err=yerr1, capsize=capsize, ecolor=ecolor1, elabel=elabel1, width=width, **kwargs)
    plt_bar(ax, x, -np.array(y2), label=label2, color=color2, vert=vert, equal_space=equal_space,
                 err=yerr2, capsize=capsize, ecolor=ecolor2, elabel=elabel2, width=width, **kwargs)

    local_max = np.nanmax([np.nanmax(y1), np.nanmax(y2)]) * 1.2
    if vert:
        axis = 'y'
    else:
        axis = 'x'
    set_sym_positive_axis(ax, axis, local_max)


def plt_group_box(ax, x, y, label_list, width=None, colors=CMAP, vert=True, **kwargs):
    '''
    绘制分组的箱线图。
    :param ax: matplotlib的轴对象,用于绘制图形
    :param x: x轴的分组标签,大组,每个组包含多个箱线图(比如['A', 'B'])
    :param y: 一个三维列表或数组,表示每组中每个箱线图的所有数据点
              (shape=(len(x), len(label_list), num_points), 例如[[[1,2], [2,3], [3,4]], [[4,5], [5,6], [6,7]]])
    :param label_list: 每个箱线图的标签,例如['x', 'y', 'z']
    :param width: 单个箱线图的宽度,默认为None,自动确定宽度
    :param colors: 箱线图的颜色序列,应与label_list的长度相匹配;也可以指定cmap,默认为CMAP,然后根据label_list的长度生成颜色序列
    :param kwargs: 其他plt.boxplot支持的参数
    '''
    # 假如colors不是一个list,则用colors对应的cmap生成颜色
    if not isinstance(colors, list):
        colors = colors(np.linspace(0, 1, len(label_list)))

    num_groups = len(y)  # 组的数量
    num_boxes = len(y[0])  # 每组中箱线图的数量，假设每组箱线图数量相同

    if width is None:
        width = BAR_WIDTH / num_boxes

    # 为每个箱线图计算中心位置
    indices = np.arange(num_groups)
    for i, lbl in enumerate(label_list):
        offsets = (np.arange(num_boxes) - np.arange(num_boxes).mean()) * width
        group_data = [y[j][i] for j in range(num_groups)]
        pos = indices + offsets[i]
        plt_box(ax=ax, x=pos, y=group_data, label=lbl, vert=vert, width=width, boxprops={'facecolor': colors[i]}, **kwargs)

    # Set labels for the x or y axis based on orientation
    if vert:
        ax.set_xticks(indices)
        ax.set_xticklabels(x)
    else:
        ax.set_yticks(indices)
        ax.set_yticklabels(x)


def plt_linregress(ax, x, y, xlog=False, ylog=False, xlog_base=10, ylog_base=10, xlog_base_str=None, ylog_base_str=None, linear_but_log=False, label=None, scatter_color=BLUE, line_color=RED, line_bound='data', show_list=None, round_digit_dict=None, round_format_dict=None, text_size=FONT_SIZE, scatter_kwargs=None, line_kwargs=None, text_kwargs=None, regress_kwargs=None, show_scatter=True, scatter_mode='original'):
    '''
    使用线性回归结果绘制散点图和回归线,可以输入scatter的其他参数

    参数:
    ax - 绘图的Axes对象
    x, y - 数据
    xlog, ylog - 是否对 x 和 y 取对数
    xlog_base, ylog_base - 对数的底数
    xlog_base_str, ylog_base_str - 对数的底数的字符串形式(如果不输入则会自动从xlog_base和ylog_base中获取)
    linear_but_log - 如果为True,当x/ylog时直接将原数据log,并在线性尺度上绘制回归线,同时手动调整tick等变为x/ylog_base**k;如果为False,则在对数尺度上绘制回归线,此时ax本身是logscale的;两种方法的区别在于,对于10^0.1,10^0.2,直接使用logscale,将会显示不出来什么tick(因为logscale的tick都是整数),而手动调整tick则可以显示出来
    label - 数据标签
    x_label, y_label - 坐标轴标签
    title - 图标题
    fontsize - 标签字体大小
    x_sci, y_sci - 是否启用科学记数法
    scatter_color - 散点图颜色
    line_color - 线性回归线颜色
    show_corr - 是否显示相关系数
    show_corr_round - 相关系数小数点位数
    show_p - 是否显示 P 值
    show_p_round - P 值小数点位数
    show_scatter - 是否显示散点
    scatter_mode - 散点模式,默认为'original',可选'density'(此时scatter_kwargs中的参数会传递给plt_density_scatter)或者'edge'(此时scatter_kwargs中的参数会传递给plt_edge_scatter)
    '''
    if scatter_kwargs is None:
        scatter_kwargs = {}
    if line_kwargs is None:
        line_kwargs = {}
    if regress_kwargs is None:
        regress_kwargs = {}
    if text_kwargs is None:
        text_kwargs = {}
    if show_list is None:
        show_list = ['r', 'p']

    x = np.array(x)
    y = np.array(y)

    # 预处理数据(如果需要对数)
    if xlog:
        x_local = np.log(x) / np.log(xlog_base)
    else:
        x_local = x.copy()
    if ylog:
        y_local = np.log(y) / np.log(ylog_base)
    else:
        y_local = y.copy()

    # 绘制散点图
    if show_scatter:
        if linear_but_log:
            if scatter_mode == 'density':
                plt_density_scatter(ax, x_local, y_local, label=label, **scatter_kwargs)
            elif scatter_mode == 'edge':
                plt_edge_scatter(ax, x_local, y_local, color=scatter_color, label=label, **scatter_kwargs)
            else:
                plt_scatter(ax, x_local, y_local, color=scatter_color, label=label, **scatter_kwargs)
        else:
            if scatter_mode == 'density':
                plt_density_scatter(ax, x, y, label=label, **scatter_kwargs)
            elif scatter_mode == 'edge':
                plt_edge_scatter(ax, x, y, color=scatter_color, label=label, **scatter_kwargs)
            else:
                plt_scatter(ax, x, y, color=scatter_color, label=label, **scatter_kwargs)

    # 设置坐标轴和tick
    if xlog:
        if linear_but_log:
            if xlog_base_str is None: # 自动获取对数底数的字符串形式
                if np.allclose(xlog_base, np.e):
                    xlog_base_str = 'e'
                else:
                    xlog_base_str = str(xlog_base)
            set_linear_but_log_axis(ax, 'x', xlog_base_str)
        else:
            ax.set_xscale('log', base=xlog_base)
            if np.allclose(xlog_base, np.e):
                set_log_e_axis(ax, 'x')
    if ylog:
        if linear_but_log:
            if ylog_base_str is None: # 自动获取对数底数的字符串形式
                if np.allclose(ylog_base, np.e):
                    ylog_base_str = 'e'
                else:
                    ylog_base_str = str(ylog_base)
            set_linear_but_log_axis(ax, 'y', ylog_base_str)
        else:
            ax.set_yscale('log', base=ylog_base)
            if np.allclose(ylog_base, np.e):
                set_log_e_axis(ax, 'y')

    if x.size > 1 and y.size > 1:
        # 计算线性回归
        regress_dict = get_linregress(x_local, y_local, **regress_kwargs)

        # 绘制回归线
        if line_bound == 'ax':
            if xlog or ylog:
                raise ValueError('xlog and ylog cannot be True when line_bound is "ax"')
            add_sline(ax, regress_dict['slope'], regress_dict['intercept'], reference_point='auto', color=line_color, **line_kwargs)
        elif line_bound == 'data':
            x_local_min, x_local_max = np.nanmin(x_local), np.nanmax(x_local)
            # y_min不一定真的是min,这里只是为了对应x_min
            y_local_min = x_local_min * regress_dict['slope'] + regress_dict['intercept']
            y_local_max = x_local_max * regress_dict['slope'] + regress_dict['intercept']
            
            # 如果使用了对数坐标，将线还原到实际的 x 和 y 值
            if xlog and not linear_but_log:
                x_min, x_max = xlog_base ** x_local_min, xlog_base ** x_local_max
            else:
                x_min, x_max = x_local_min, x_local_max
            if ylog and not linear_but_log:
                y_min, y_max = ylog_base ** y_local_min, ylog_base ** y_local_max
            else:
                y_min, y_max = y_local_min, y_local_max
            plt_line(ax, [x_min, x_max], [y_min, y_max], color=line_color, **line_kwargs)

        add_text_by_dict(ax, text_dict=regress_dict, show_list=show_list, round_digit_dict=round_digit_dict, round_format_dict=round_format_dict, fontsize=text_size, **text_kwargs)
        return regress_dict


def plt_density_scatter(ax, x, y, label=None, label_cmap_float=1.0, estimate_type='kde', bw_method='auto', bins_x=BIN_NUM, bins_y=BIN_NUM, cmap=DENSITY_CMAP, norm_mode='linear', vmin=None, vmax=None, norm_kwargs=None, cbar=True, cbar_label='density', cbar_position=None, scatter_kwargs=None, cbar_kwargs=None):
    '''
    绘制密度散点图。
    :param ax: matplotlib的轴对象,用于绘制图形
    :param x: x轴的数据
    :param y: y轴的数据
    :param label: 图例标签,默认为None
    :param label_cmap_float: 图例颜色,默认为1.0
    :param estimate_type: 密度估计类型,默认为'kde',可选'hist'
    :param bw_method: 密度估计的带宽方法,默认为'auto'
    :param bins_x: x轴的bins,默认为BIN_NUM
    :param bins_y: y轴的bins,默认为BIN_NUM
    :param cmap: 颜色映射,默认为DENSITY_CMAP
    :param cbar: 是否显示颜色条,默认为True
    :param scatter_kwargs: 散点图的其他参数
    :param cbar_kwargs: 颜色条的其他参数

    注意:
    如果log scale,则不太适合使用hist,因为hist的bins是线性的,而log scale的bins是对数的
    '''
    if scatter_kwargs is None:
        scatter_kwargs = {}
    cbar_kwargs = update_dict(cbar_kwargs, {'cbar_label': cbar_label})
    cbar_position = update_dict(CBAR_POSITION, cbar_position)

    x_array = np.array(x)
    y_array = np.array(y)

    xy = np.vstack([x_array, y_array])

    if estimate_type == 'kde':
        if bw_method == 'auto':
            z = gaussian_kde(xy)(xy)
        else:
            z = gaussian_kde(xy, bw_method=bw_method)(xy)
    if estimate_type == 'hist':
        hist, xedges, yedges, _, _ = get_hist_2d(
            x_array, y_array, bins_x=bins_x, bins_y=bins_y, stat='density')
        ix = np.digitize(x_array, xedges, right=False) - 1
        iy = np.digitize(y_array, yedges, right=False) - 1

        ix = np.clip(ix, 0, hist.shape[0] - 1)
        iy = np.clip(iy, 0, hist.shape[1] - 1)
        z = hist[ix, iy]

    return plt_colorful_scatter(ax, x, y, c=z, cmap=cmap, norm_mode=norm_mode, vmin=vmin, vmax=vmax, norm_kwargs=norm_kwargs, label=label, label_cmap_float=label_cmap_float, scatter_kwargs=scatter_kwargs, cbar=cbar, cbar_postion=cbar_position, cbar_kwargs=cbar_kwargs)


def plt_density_line(ax, x, y, stat='density', bins_x=None, bins_y=None, cmap=DENSITY_CMAP, norm_mode='linear', vmin=None, vmax=None, norm_kwargs=None, cbar=True, cbar_label=None, cbar_position=None, cbar_kwargs=None, imshow_kwargs=None):
    '''
    将线按照密度,使用imshow

    x: (n, ) or (n, m), m为线的个数
    y: (n, ) or (n, m), m为线的个数
    '''
    if cbar_label is None:
        cbar_label = 'density'
    if cbar_kwargs is None:
        cbar_kwargs = {}
    if imshow_kwargs is None:
        imshow_kwargs = {}
    x = np.array(x)
    y = np.array(y)
    if x.ndim == 1 and y.ndim == 1:
        pass
    elif x.ndim == 2 and y.ndim == 2:
        pass
    elif x.ndim == 1 and y.ndim == 2:
        # 将x重复到y的形状
        x = np.repeat(x, y.shape[1]).reshape(y.shape)
    elif x.ndim == 2 and y.ndim == 1:
        # 将y重复到x的形状
        y = np.repeat(y, x.shape[1]).reshape(x.shape)
    
    if x.shape != y.shape:
        raise ValueError(f'After some processing, x and y still have different shapes: {x.shape} and {y.shape}. Please check your input data.')

    if bins_x is None:
        bins_x = x.shape[0] // 10
    if bins_y is None:
        bins_y = y.shape[0] // 10
    value, bin_edges_x, bin_edges_y, _, _ = get_hist_2d(x.flatten(), y.flatten(), bins_x=bins_x, bins_y=bins_y, stat=stat)
    
    if vmin is None:
        vmin = np.nanmin(value)
    if vmax is None:
        vmax = np.nanmax(value)
    norm = get_norm(norm_mode=norm_mode, vmin=vmin, vmax=vmax, norm_kwargs=norm_kwargs)
    im = plt_imshow(ax, value.T, cmap=cmap, norm=norm, extent=[bin_edges_x[0], bin_edges_x[-1], bin_edges_y[0], bin_edges_y[-1]], origin='lower', aspect='auto', **imshow_kwargs) # no need to pass vmin and vmax to imshow, because norm has been set
    cbar = add_side_colorbar(ax, im, cmap=cmap, cbar_label=cbar_label, cbar_position=cbar_position, vmin=vmin, vmax=vmax, norm_kwargs=norm_kwargs, norm_mode=norm_mode, **cbar_kwargs)
    return im, cbar, bin_edges_x, bin_edges_y


def plt_marginal_density_scatter(ax, x, y, x_side_ax=None, y_side_ax=None, density_scatter_kwargs=None, marginal_kwargs=None):
    '''
    绘制密度散点图和边缘分布。
    :param ax: matplotlib的轴对象,用于绘制图形
    :param x: x轴的数据
    :param y: y轴的数据
    :param x_side_ax: x轴的边缘分布轴,默认为None
    :param y_side_ax: y轴的边缘分布轴,默认为None
    :param density_scatter_kwargs: 传递给plt_density_scatter的额外关键字参数
    :param marginal_kwargs: 传递给add_double_marginal_distribution的额外关键字参数
    '''
    density_scatter_kwargs = update_dict(get_default_param(plt_density_scatter), density_scatter_kwargs)
    density_scatter_kwargs['cbar_position'] = update_dict(CBAR_POSITION, density_scatter_kwargs['cbar_position'])

    if x_side_ax is None and y_side_ax is None:
        marginal_kwargs = update_dict(get_default_param(add_double_marginal_distribution), marginal_kwargs)
        x_side_ax, y_side_ax, ax = add_double_marginal_distribution(ax, x, y, **marginal_kwargs)

    # 此处先不添加cbar
    sc = plt_density_scatter(ax, x, y, **update_dict(density_scatter_kwargs, {'cbar': False}))
    # 由于density scatter有概率输出多个结果,所以需要判断,而需要的sc一直是第一个
    if isinstance(sc, tuple):
        sc = sc[0]

    if density_scatter_kwargs['cbar']:
        if density_scatter_kwargs['cbar_position']['position'] == marginal_kwargs['x_side_ax_position']:
            density_scatter_kwargs['cbar_position']['pad'] += marginal_kwargs['x_side_ax_size'] + marginal_kwargs['x_side_ax_pad']
            if not marginal_kwargs['outside']:
                density_scatter_kwargs['cbar_position']['pad'] /= (1 - marginal_kwargs['x_side_ax_size'])
        if density_scatter_kwargs['cbar_position']['position'] == marginal_kwargs['y_side_ax_position']:
            density_scatter_kwargs['cbar_position']['pad'] += marginal_kwargs['y_side_ax_size'] + marginal_kwargs['y_side_ax_pad']
            if not marginal_kwargs['outside']:
                density_scatter_kwargs['cbar_position']['pad'] /= (1 - marginal_kwargs['y_side_ax_size'])

        cbars = add_side_colorbar(ax, mappable=sc, cmap=density_scatter_kwargs['cmap'], cbar_label=density_scatter_kwargs['cbar_label'], cbar_position=density_scatter_kwargs['cbar_position'], **update_dict({}, density_scatter_kwargs['cbar_kwargs']))
        return sc, x_side_ax, y_side_ax, cbars
    else:
        return sc, x_side_ax, y_side_ax


def plt_errorbar_line(ax, x, y, err, line_label=None, elabel=None, line_color=BLUE, ecolor=BLACK, capsize=PLT_CAP_SIZE, vert=True, line_kwargs=None, error_kwargs=None):
    '''
    使用x和y绘制折线图,并添加误差线
    :param ax: matplotlib的轴对象,用于绘制图形
    :param x: x轴的数据
    :param y: y轴的数据
    :param err: 误差线的数据
    :param line_label: 折线图的标签,默认为None
    :param elabel: 误差线的标签,默认为None
    :param line_color: 折线图的颜色,默认为BLUE
    :param ecolor: 误差线的颜色,默认为BLACK
    :param capsize: 误差线的线帽大小,默认为PLT_CAP_SIZE
    :param vert: 是否是垂直的误差线,默认为True
    :param line_kwargs: 传递给plt_line的额外关键字参数
    :param error_kwargs: 传递给add_errorbar的额外关键字参数
    '''
    if line_kwargs is None:
        line_kwargs = {}
    if error_kwargs is None:
        error_kwargs = {}
    # 画图
    if vert:
        return plt_line(ax, x, y, label=line_label, color=line_color, **line_kwargs), add_errorbar(ax, x, y, err, label=elabel, color=ecolor, capsize=capsize, vert=vert, **error_kwargs)
    else:
        return plt_line(ax, y, x, label=line_label, color=line_color, **line_kwargs), add_errorbar(ax, x, y, err, label=elabel, color=ecolor, capsize=capsize, vert=vert, **error_kwargs)


def plt_fill_between_line(ax, x, y1, y2, label=None, color=BLUE, alpha=FAINT_ALPHA, vert=True, **kwargs):
    '''
    使用x和y1,y2绘制填充区域
    :param ax: matplotlib的轴对象,用于绘制图形
    :param x: x轴的数据
    :param y1: y轴的数据
    :param y2: y轴的数据
    :param label: 填充区域的标签,默认为None
    :param color: 填充区域的颜色,默认为BLUE
    :param alpha: 填充区域的透明度,默认为FAINT_ALPHA
    :param kwargs: 传递给plt.fill_between的额外关键字参数
    '''
    # 画图
    if vert:
        return ax.fill_between(x, y1, y2, label=label, color=color, alpha=alpha, **kwargs)
    else:
        return ax.fill_betweenx(x, y1, y2, label=label, color=color, alpha=alpha, **kwargs)


def plt_band_line(ax, x, y, bandwidth, line_label=None, line_color=BLUE, fill_label=None, fill_color=BLUE, alpha=FAINT_ALPHA, vert=True, line_kwargs=None, fill_kwargs=None):
    '''
    使用x和y绘制折线图，并添加一个表示误差带的区域
    :param ax: matplotlib的轴对象,用于绘制图形
    :param x: x轴的数据
    :param y: y轴的数据
    :param bandwidth: 误差带的数据
    :param label: 图例标签,默认为None
    :param color: 折线图的颜色,默认为BLUE
    :param alpha: 误差带的透明度,默认为FAINT_ALPHA
    :param line_kwargs: 传递给plt_line的额外关键字参数
    :param fill_kwargs: 传递给plt_fill_between_line的额外关键字参数
    '''
    if line_kwargs is None:
        line_kwargs = {}
    if fill_kwargs is None:
        fill_kwargs = {}
    # 画图
    if vert:
        return plt_line(ax, x, y, label=line_label, color=line_color, **line_kwargs), plt_fill_between_line(ax, x, np.array(y)-np.array(bandwidth), np.array(y)+np.array(bandwidth), label=fill_label, color=fill_color, alpha=alpha, vert=vert, **fill_kwargs)
    else:
        return plt_line(ax, y, x, label=line_label, color=line_color, **line_kwargs), plt_fill_between_line(ax, x, np.array(y)-np.array(bandwidth), np.array(y)+np.array(bandwidth), label=fill_label, color=fill_color, alpha=alpha, vert=vert, **fill_kwargs)


def plt_smooth_scatter_line(ax, x, y, frac=0.2, scatter_label=None, line_label='LOWESS', scatter_color=BLUE, scatter_alpha=FAINT_ALPHA, line_color=RED, scatter_kwargs=None, line_kwargs=None):
    '''
    利用x和y绘制散点图和平滑线
    '''
    if scatter_kwargs is None:
        scatter_kwargs = {}
    if line_kwargs is None:
        line_kwargs = {}
    x_smooth, y_smooth = lowess_smooth(x, y, frac=frac)
    return plt_scatter(ax, x, y, color=scatter_color, label=scatter_label, alpha=scatter_alpha, **scatter_kwargs), plt_line(ax, x_smooth, y_smooth, color=line_color, label=line_label, **line_kwargs)


def plt_kde(ax, data, label=None, color=BLUE, vert=True, x=None, kde_kwargs=None, **kwargs):
    '''
    使用数据绘制核密度估计图,可以接受plt.plot的其他参数
    :param ax: matplotlib的轴对象,用于绘制图形
    :param data: 用于绘制核密度估计图的数据集
    :param label: 图例标签,默认为None
    :param color: 核密度估计图的颜色,默认为BLUE
    :param bw_method: 核密度估计的带宽,默认为None
    :param kwargs: 其他plt.plot支持的参数
    '''
    if kde_kwargs is None:
        kde_kwargs = {}
    kde = get_kde(data, **kde_kwargs)
    if x is None:
        x = np.linspace(np.nanmin(data), np.nanmax(data), 1000, endpoint=True)
    return plt_line(ax, x, kde(x), label=label, color=color, vert=vert, **kwargs)


def plt_kde_contour(ax, x, y, label=None, color=BLUE, cmap=None, levels=None, gridsize=100, xmin=None, xmax=None, ymin=None, ymax=None, kde_kwargs=None, contour_kwargs=None):
    '''
    使用x和y绘制核密度估计的轮廓图
    :param ax: matplotlib的轴对象,用于绘制图形
    :param x: x轴的数据
    :param y: y轴的数据
    :param label: 图例标签,默认为None
    :param color: 轮廓图的颜色,默认为BLUE
    :param cmap: 轮廓图的颜色映射,默认为None
    :param levels: 轮廓图的等高线数量,默认为None
    :param kde_kwargs: 传递给get_kde的额外关键字参数
    :param contour_kwargs: 传递给plt.contour的额外关键字参数
    '''
    if kde_kwargs is None:
        kde_kwargs = {}
    if contour_kwargs is None:
        contour_kwargs = {}

    kde = get_kde(np.vstack([x, y]), **kde_kwargs)

    # Generate a Grid
    if xmin is None:
        xmin = np.nanmin(x) - (np.nanmax(x) - np.nanmin(x)) * 0.1
    if xmax is None:
        xmax = np.nanmax(x) + (np.nanmax(x) - np.nanmin(x)) * 0.1
    if ymin is None:
        ymin = np.nanmin(y) - (np.nanmax(y) - np.nanmin(y)) * 0.1
    if ymax is None:
        ymax = np.nanmax(y) + (np.nanmax(y) - np.nanmin(y)) * 0.1
    xx, yy = np.mgrid[xmin:xmax:complex(gridsize), ymin:ymax:complex(gridsize)]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    
    # Evaluate the KDE
    z = np.reshape(kde(positions).T, xx.shape)

    return plt_contour(ax, xx, yy, z, label=label, color=color, cmap=cmap, levels=levels, **contour_kwargs)


def plt_kde_contourf(ax, x, y, cmap=DENSITY_CMAP, levels=None, gridsize=100, xmin=None, xmax=None, ymin=None, ymax=None, kde_kwargs=None, contourf_kwargs=None):
    '''
    使用x和y绘制核密度估计的填充轮廓图
    :param ax: matplotlib的轴对象,用于绘制图形
    :param x: x轴的数据
    :param y: y轴的数据
    :param label: 图例标签,默认为None
    :param cmap: 轮廓图的颜色映射,默认为DENSITY_CMAP
    :param levels: 轮廓图的等高线数量,默认为None
    :param kde_kwargs: 传递给get_kde的额外关键字参数
    :param contourf_kwargs: 传递给plt.contourf的额外关键字参数
    '''
    if kde_kwargs is None:
        kde_kwargs = {}
    if contourf_kwargs is None:
        contourf_kwargs = {}

    kde = get_kde(np.vstack([x, y]), **kde_kwargs)

    # Generate a Grid
    if xmin is None:
        xmin = np.nanmin(x) - (np.nanmax(x) - np.nanmin(x)) * 0.1
    if xmax is None:
        xmax = np.nanmax(x) + (np.nanmax(x) - np.nanmin(x)) * 0.1
    if ymin is None:
        ymin = np.nanmin(y) - (np.nanmax(y) - np.nanmin(y)) * 0.1
    if ymax is None:
        ymax = np.nanmax(y) + (np.nanmax(y) - np.nanmin(y)) * 0.1
    xx, yy = np.mgrid[xmin:xmax:complex(gridsize), ymin:ymax:complex(gridsize)]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    
    # Evaluate the KDE
    z = np.reshape(kde(positions).T, xx.shape)

    return plt_contourf(ax, xx, yy, z, cmap=cmap, levels=levels, **contourf_kwargs)


def plt_cdf(ax, data, label=None, color=BLUE, vert=True, **kwargs):
    '''
    使用数据绘制累积分布函数图,可以接受plt.plot的其他参数
    :param ax: matplotlib的轴对象,用于绘制图形
    :param data: 用于绘制累积分布函数图的数据集
    :param label: 图例标签,默认为None
    :param color: 累积分布函数图的颜色,默认为BLUE
    :param kwargs: 其他plt.plot支持的参数
    '''
    cdf = get_cdf(data)
    return plt_line(ax, cdf.x, cdf.y, label=label, color=color, vert=vert, **kwargs)


def plt_compare_cdf_ks(ax, data1, data2, label1=None, label2=None, color1=BLUE, color2=RED, vert=True, text_x=TEXT_X, text_y=TEXT_Y, fontsize=FONT_SIZE, round_digit_dict=None, round_format_dict=None, show_list=None, text_kwargs=None, cdf_kwargs=None):
    '''
    使用两组数据绘制累积分布函数图,并计算K-S检验的p值
    :param ax: matplotlib的轴对象,用于绘制图形
    :param data1: 第一组数据
    :param data2: 第二组数据
    :param label1: 第一组数据的标签
    :param label2: 第二组数据的标签
    :param color1: 第一组数据的颜色
    :param color2: 第二组数据的颜色
    :param kwargs: 传递给plt_cdf的额外关键字参数
    '''
    if cdf_kwargs is None:
        cdf_kwargs = {}

    plt_cdf(ax, data1, label=label1, color=color1, vert=vert, **cdf_kwargs)
    plt_cdf(ax, data2, label=label2, color=color2, vert=vert, **cdf_kwargs)

    r = get_ks_result(data1, data2)
    text_dict = {'KS': r['ks'], 'P': r['p']}
    add_text_by_dict(ax, text_dict=text_dict, show_list=show_list, round_digit_dict=round_digit_dict, round_format_dict=round_format_dict, fontsize=fontsize, text_x=text_x, text_y=text_y, text_kwargs=text_kwargs)


def plt_marginal_hist_2d(ax, x, y, x_side_ax=None, y_side_ax=None, stat='probability', cbar_label=None, hist_kwargs=None, marginal_kwargs=None):
    '''
    带有边缘分布的二维直方图
    :param ax: matplotlib的轴对象,用于绘制图形
    :param x: x轴的数据
    :param y: y轴的数据
    :param x_side_ax: x轴的边缘分布轴,默认为None
    :param y_side_ax: y轴的边缘分布轴,默认为None
    :param stat: 统计量,默认为'probability'
    :param cbar_label: 颜色条的标签,默认和stat相同
    :param hist_kwargs: 传递给plt_hist_2d的额外关键字参数
    :param marginal_kwargs: 传递给add_double_marginal_distribution的额外关键字参数
    '''
    if cbar_label is None:
        cbar_label = stat
    hist_kwargs = update_dict(get_default_param(plt_hist_2d), hist_kwargs)
    hist_kwargs['cbar_position'] = update_dict(CBAR_POSITION, hist_kwargs['cbar_position'])

    if x_side_ax is None and y_side_ax is None:
        marginal_kwargs = update_dict(get_default_param(add_double_marginal_distribution), marginal_kwargs)
        x_side_ax, y_side_ax, ax = add_double_marginal_distribution(ax, x, y, **marginal_kwargs)

    # 此处先不添加cbar
    hist = plt_hist_2d(ax, x, y, **update_dict(hist_kwargs, {'cbar': False, 'stat': stat}))


    if hist_kwargs['cbar']:
        if hist_kwargs['cbar_position']['position'] == marginal_kwargs['x_side_ax_position']:
            hist_kwargs['cbar_position']['pad'] += marginal_kwargs['x_side_ax_size'] + marginal_kwargs['x_side_ax_pad']
            if not marginal_kwargs['outside']:
                hist_kwargs['cbar_position']['pad'] /= (1 - marginal_kwargs['x_side_ax_size'])
        if hist_kwargs['cbar_position']['position'] == marginal_kwargs['y_side_ax_position']:
            hist_kwargs['cbar_position']['pad'] += marginal_kwargs['y_side_ax_size'] + marginal_kwargs['y_side_ax_pad']
            if not marginal_kwargs['outside']:
                hist_kwargs['cbar_position']['pad'] /= (1 - marginal_kwargs['y_side_ax_size'])

        cbars = add_side_colorbar(ax, mappable=hist, cmap=hist_kwargs['cmap'], cbar_label=cbar_label, cbar_position=hist_kwargs['cbar_position'], **update_dict({}, hist_kwargs['cbar_kwargs']))
        return hist, x_side_ax, y_side_ax, cbars
    else:
        return hist, x_side_ax, y_side_ax


def plt_vector_input_bar(ax, x, y, label=None, color=BLUE, vert=True, equal_space=False, err=None, capsize=PLT_CAP_SIZE, ecolor=BLACK, elabel=None, width=BAR_WIDTH, bar_kwargs=None, ebar_kwargs=None):
    '''
    使用x和y绘制柱状图，可以接受plt.bar的其他参数,此函数的特性是会根据x的值作为bar的位置,当x包含字符串或者equal_space=True时,会自动变成等距离排列。这个函数的特殊性是可以接受向量值的width，color，capsize，ecolor
    :param ax: matplotlib的轴对象，用于绘制图形
    :param x: x轴的数据
    :param y: y轴的数据
    :param label: 图例标签，默认为None
    :param color: 柱状图的颜色，默认为BLUE
    :param vert: 是否为垂直柱状图，默认为True，即纵向
    :param equal_space: 是否将x的值作为字符串处理，这将使得柱子等距排列，默认为False
    :param err: 误差线的数据，默认为None
    :param capsize: 误差线帽大小，默认为PLT_CAP_SIZE
    :param ecolor: 误差线颜色，默认为BLACK
    :param elabel: 误差线图例标签，默认为None
    :param width: 柱子宽度，默认为None(当vert=False时,此参数将自动赋值给height)
    :param kwargs: 其他plt.bar或plt.barh支持的参数
    '''
    if bar_kwargs is None:
        bar_kwargs = {}
    if ebar_kwargs is None:
        ebar_kwargs = {}

    if equal_space:
        # 将x的每个元素变为字符串
        xtick_label = [str(i) for i in x]
        xtick = np.arange(len(xtick))
        ax.set_xticklabels(xtick_label)
    else:
        xtick = x.copy()

    if not isinstance(color, (list, np.ndarray)):
        color = [color] * len(x)
    if not isinstance(width, (list, np.ndarray)):
        width = [width] * len(x)
    if not isinstance(err, (list, np.ndarray)):
        err = [err] * len(x)
    if not isinstance(capsize, (list, np.ndarray)):
        capsize = [capsize] * len(x)
    if not isinstance(ecolor, (list, np.ndarray)):
        ecolor = [ecolor] * len(x)

    for i in range(len(x)):
        if i==0:
            local_label = label
        else:
            local_label = None
        plt_bar(ax, xtick[i], y[i], label=local_label, color=color[i], vert=vert, equal_space=False, width=width[i], **bar_kwargs)
        add_errorbar(ax, xtick[i], y[i], err[i], vert=vert, label=elabel, color=color[i], capsize=capsize[i], ecolor=ecolor[i], **ebar_kwargs)


def plt_xlog_bar(*args, **kwargs):
    print('use sns bar and log_scale=True instead, or use linear_but_log to set the fake log scale')


def plt_polygon_heatmap(ax, xy_dict, value_dict, mask=None, mask_color=MASK_COLOR, norm_mode='linear', vmin=None, vmax=None, norm_kwargs=None, cmap=CMAP, edgecolor=BLACK, cbar=True, cbar_position=None, cbar_label=None, fill=True, adjust_lim=True, polygon_kwargs=None, cbar_kwargs=None):
    '''
    绘制多边形热力图。

    参数:
    - ax (matplotlib.axes.Axes): matplotlib的轴对象,用于绘制图形
    - xy_dict (dict): 包含多个区域的顶点坐标的字典。示例: {'region1': [(0, 0), (1, 1), (1, 0)], 'region2': [(1, 1), (2, 2), (2, 1)]}
    - value_dict (dict): 包含多个区域的值的字典。示例: {'region1': 1, 'region2': 2}
    - mask (dict or None, optional): 包含多个区域的掩码的字典,默认为None
    - mask_color (str or tuple, optional): 掩码的颜色,默认为MASK_COLOR
    - norm_mode (str, optional): 归一化模式，默认为'linear'
    - vmin (float or None, optional): 区域值的最小值,默认为None
    - vmax (float or None, optional): 区域值的最大值,默认为None
    - norm_kwargs (dict or None, optional): 传递给get_norm的额外关键字参数
    - cmap (str or matplotlib.colors.Colormap, optional): 区域值的颜色映射,默认为CMAP
    - edgecolor (str or tuple, optional): 区域的边框颜色,默认为BLACK
    - cbar (bool, optional): 是否显示颜色条,默认为True
    - cbar_position (str or None, optional): 颜色条的位置,默认为None
    - cbar_label (str or None, optional): 颜色条的标签,默认为None
    - fill (bool, optional): 是否填充区域(如果为False,则为区域的边框,这时可以通过linewidth参数控制边框宽度)
    - adjust_lim (bool, optional): 是否根据区域的位置和大小调整轴的限制
    - polygon_kwargs (dict or None, optional): 传递给plt_polygon的额外关键字参数
    - cbar_kwargs (dict or None, optional): 传递给add_side_colorbar的额外关键字参数

    返回:
    - add_side_colorbar的返回值,如果cbar为False,则返回None。
    '''
    # 设置默认参数
    if vmin is None:
        vmin = np.nanmin(list(value_dict.values()))
    if vmax is None:
        vmax = np.nanmax(list(value_dict.values()))
    polygon_kwargs = update_dict({}, polygon_kwargs)
    cbar_kwargs = update_dict({}, cbar_kwargs)
    
    # 获取use_mask
    if mask is None:
        use_mask = False
    elif isinstance(mask, dict):
        # 判断是否存在掩码
        if any(mask.values()):
            use_mask = True
        else:
            use_mask = False

    # 得到norm
    norm = get_norm(norm_mode, vmin=vmin, vmax=vmax, norm_kwargs=norm_kwargs)

    # 根据norm后的data值获取颜色
    for region, xy in xy_dict.items():
        if use_mask:
            if mask[region]:
                plt_polygon(ax, xy, color=None, facecolor=mask_color, edgecolor=edgecolor, fill=fill, adjust_lim=adjust_lim, **polygon_kwargs)
            else:
                plt_polygon(ax, xy, color=None, facecolor=cmap(norm(value_dict[region])), edgecolor=edgecolor, fill=fill, adjust_lim=adjust_lim, **polygon_kwargs)
        else:
            plt_polygon(ax, xy, color=None, facecolor=cmap(norm(value_dict[region])), edgecolor=edgecolor, fill=fill, adjust_lim=adjust_lim, **polygon_kwargs)
        
    # 添加colorbar
    if cbar:
        return add_side_colorbar(ax, cmap=cmap, norm_mode=norm_mode, vmin=vmin, vmax=vmax, norm_kwargs=norm_kwargs, cbar_label=cbar_label, cbar_position=cbar_position, use_mask=use_mask, mask_color=mask_color, **cbar_kwargs)


def plt_qq_plot(ax, data_x, data_y, n_quantiles=None, scatter_color=BLUE, line_color=RED, scatter_kwargs=None, line_kwargs=None, text_x=TEXT_X, text_y=TEXT_Y, show_list=None, round_digit_dict=None, round_format_dict=None, fontsize=FONT_SIZE, text_kwargs=None):
    """
    绘制QQ图
    :param ax: matplotlib的轴对象
    :param data_x: 数据集x
    :param data_y: 数据集y
    :param n_quantiles: 分位数的数量,如果为None,则根据较小的数据集长度自动确定
    """
    # 如果未指定分位数数量,则取两个数据集中较小的长度
    if n_quantiles is None:
        n_quantiles = min(len(data_x), len(data_y))

    # 计算分位数级别
    quantile_levels = np.linspace(0., 1., n_quantiles)

    # 计算两组数据在这些分位数级别上的值
    quantiles_x = np.quantile(data_x, quantile_levels)
    quantiles_y = np.quantile(data_y, quantile_levels)

    # 绘制QQ图
    if scatter_kwargs is None:
        scatter_kwargs = {}
    if line_kwargs is None:
        line_kwargs = {}
    plt_scatter(ax, quantiles_x, quantiles_y, color=scatter_color, **scatter_kwargs)
    plt_line(ax, [quantiles_x.min(), quantiles_x.max()], [quantiles_y.min(), quantiles_y.max()], color=line_color, **line_kwargs)

    # 添加ks检验的结果
    r = get_ks_result(data_x, data_y)
    text_dict = {'KS': r['ks'], 'P': r['p']}
    add_text_by_dict(ax, text_dict=text_dict, show_list=show_list, round_digit_dict=round_digit_dict, round_format_dict=round_format_dict, fontsize=fontsize, text_x=text_x, text_y=text_y, text_kwargs=text_kwargs)

    # 添加标题
    ax.set_title('QQ Plot')


def plt_pp_plot(ax, data_x, data_y, n_points=None, scatter_color=BLUE, line_color=RED, scatter_kwargs=None, line_kwargs=None, text_x=TEXT_X, text_y=TEXT_Y, show_list=None, round_digit_dict=None, round_format_dict=None, fontsize=FONT_SIZE, text_kwargs=None):
    """
    绘制PP图
    :param ax: matplotlib的轴对象
    :param data_x: 数据集x
    :param data_y: 数据集y
    :param n_points: 用于绘制CDF的点的数量,如果为None,则根据较小的数据集长度自动确定
    """
    # 如果未指定点的数量,根据数据集中较小的长度自动确定
    if n_points is None:
        n_points = min(len(data_x), len(data_y))

    # 对数据进行排序
    sorted_data_x = np.sort(data_x)
    sorted_data_y = np.sort(data_y)

    # 计算累计概率级别
    cdf_levels = np.linspace(0., 1., n_points)

    # 根据累积概率级别计算CDF的值
    cdf_x = np.interp(cdf_levels, np.linspace(0., 1., len(sorted_data_x)), sorted_data_x)
    cdf_y = np.interp(cdf_levels, np.linspace(0., 1., len(sorted_data_y)), sorted_data_y)

    # 绘制PP图
    if scatter_kwargs is None:
        scatter_kwargs = {}
    if line_kwargs is None:
        line_kwargs = {}
    
    # 绘制散点图
    plt_scatter(ax, cdf_x, cdf_y, color=scatter_color, **scatter_kwargs)
    
    # 绘制对角线,表示完美匹配的情况
    min_max_range = [min(sorted_data_x.min(), sorted_data_y.min()), max(sorted_data_x.max(), sorted_data_y.max())]
    plt_line(ax, min_max_range, min_max_range, color=line_color, **line_kwargs)

    # 添加ks检验的结果
    r = get_ks_result(data_x, data_y)
    text_dict = {'KS': r['ks'], 'P': r['p']}
    add_text_by_dict(ax, text_dict=text_dict, show_list=show_list, round_digit_dict=round_digit_dict, round_format_dict=round_format_dict, fontsize=fontsize, text_x=text_x, text_y=text_y, text_kwargs=text_kwargs)

    # 添加标题
    ax.set_title('PP Plot')


def plt_advance_quiver(ax):
    'https://matplotlib.org/stable/gallery/images_contours_and_fields/quiver_demo.html#sphx-glr-gallery-images-contours-and-fields-quiver-demo-py'
    pass
# endregion


# region 复杂作图函数(matplotlib系列,输入dict使用)
def plt_group_bar_dict(ax, data, width=None, colors=CMAP, vert=True, **kwargs):
    """
    使用字典数据绘制分组的柱状图。
    
    :param ax: matplotlib的轴对象
    :param data: 字典格式输入,结构为: {'A': {'x': 1, 'y': 2, 'z': 3}, 'B': {'x': 4, 'y': 5, 'z': 6}}
    :param width: 柱子宽度 (逻辑同原函数)
    :param colors: 颜色配置 (逻辑同原函数)
    :param vert: 是否垂直柱状图
    :param kwargs: 其他plt.bar参数
    """
    # 提取分组标签 (x轴标签)
    x_labels = list(data.keys())
    
    # 提取子组标签并验证一致性
    subgroup_labels = list(next(iter(data.values())).keys())  # 取第一个分组的子组作为标准
    for group, subgroups in data.items():
        if set(subgroups.keys()) != set(subgroup_labels):
            raise ValueError(f"分组 '{group}' 的子组标签与首分组不一致")
    
    # 构造二维y值数组
    y_values = [
        [subgroups[sub] for sub in subgroup_labels]
        for groups, subgroups in data.items()
    ]
    
    # 调用原始函数绘图
    return plt_group_bar(ax=ax, x=x_labels, y=y_values, label_list=subgroup_labels, width=width, colors=colors, vert=vert, **kwargs)
# endregion


# region 复杂作图函数(matplotlib系列,三维作图)
def plt_colorful_scatter_3d(ax, x, y, z, c, cmap=CMAP, norm_mode='linear', vmin=None, vmax=None, norm_kwargs=None, s=MARKER_SIZE**2, label=None, label_cmap_float=1.0, scatter_kwargs=None, cbar=True, cbar_postion=None, cbar_kwargs=None):
    '''
    绘制颜色关于c值变化的散点图。

    参数:
    - ax (matplotlib.axes.Axes): matplotlib的轴对象, 用于绘制图形。
    - x (numpy.ndarray or list): x轴的数据。
    - y (numpy.ndarray or list): y轴的数据。 
    - z (numpy.ndarray or list): z轴的数据。
    - c (numpy.ndarray or list): 颜色的数据。
    - cmap (matplotlib.colors.Colormap, optional): 颜色映射, 默认为CMAP。
    - norm_mode (str, optional): 颜色映射的规范化模式, 可选 'linear', 'log', 'symlog', 'two_slope'等, 默认为 'linear'
    - vmin (float, optional): 颜色映射的最小值, 默认为None。
    - vmax (float, optional): 颜色映射的最大值, 默认为None。
    - norm_kwargs (dict or None, optional): 颜色映射规范化的其他参数
    - s (float, optional): 散点的大小, 默认为MARKER_SIZE**2。
    - label (str or None, optional): 散点的标签, 默认为None。
    - label_cmap_float (float, optional): 代表性点的颜色映射模式, 默认为1.0。如果输入整数则会raise ValueError。
    - scatter_kwargs (dict or None, optional): 传递给plt_scatter的其他参数。
    - cbar (bool, optional): 是否添加颜色条,默认为True。
    - cbar_postion (str or None, optional): 颜色条的位置,默认为None。
    - cbar_kwargs (dict or None, optional): 传递给add_side_colorbar的其他参数。
    '''
    # 设置默认参数
    if scatter_kwargs is None:
        scatter_kwargs = {}
    if cbar_kwargs is None:
        cbar_kwargs = {}
    if vmin is None:
        vmin = np.nanmin(c)
    if vmax is None:
        vmax = np.nanmax(c)

    # 获取颜色映射的规范化对象
    norm = get_norm(norm_mode=norm_mode, vmin=vmin, vmax=vmax, norm_kwargs=norm_kwargs)

    # 绘制散点(暂时设置label为None)
    local_scatter_kwargs = update_dict(scatter_kwargs, {'s': s})
    sc = plt_scatter_3d(ax, x, y, z, color=None, c=c, cmap=cmap, label=None, norm=norm, **local_scatter_kwargs)

    # 通过label_cmap_float来绘制代表性点并作出label
    if isinstance(label_cmap_float, int):
        print('warning: "label_cmap_float" is an integer, so it will be transformed to float')
    ax.scatter([], [], [], color=cmap(float(label_cmap_float)), label=label, s=s)

    # 添加颜色条
    if cbar:
        if vmin > np.nanmin(c):
            cbar_kwargs['add_leq'] = True
        if vmax < np.nanmax(c):
            cbar_kwargs['add_geq'] = True
        cbars = add_side_colorbar(ax, sc, vmin=vmin, vmax=vmax, norm_mode=norm_mode, norm_kwargs=norm_kwargs, cmap=cmap, cbar_position=cbar_postion, **cbar_kwargs)
        return sc, cbars
    else:
        return sc


def plt_voxel_heatmap(ax, data, cmap=CMAP, norm_mode='linear', vmin=None, vmax=None, norm_kwargs=None, edgecolors=BLACK, cbar=True, cbar_position=None, cbar_label=None, mask=None, mask_color=MASK_COLOR, voxel_kwargs=None, cbar_kwargs=None):
    '''
    根据data中的值和指定的颜色映射绘制体素图。(data中的nan值将不会绘制,mask中的True值将会用mask_color绘制。)
    
    参数:
    - ax: matplotlib的Axes3D对象。
    - data: 三维numpy数组，其值将用来根据颜色映射确定颜色。(如果data中的值为nan,则不会绘制该体素。)
    - cmap: 颜色映射，默认为CMAP。
    - norm_mode: 归一化模式，默认为'linear'。
    - vmin: 颜色映射的最小值，默认为None。
    - vmax: 颜色映射的最大值，默认为None。
    - norm_kwargs: 传递给get_norm的额外关键字参数。
    - edgecolors: 体素的边框颜色，默认为BLACK。
    - cbar: 是否添加颜色条，默认为True。
    - cbar_position: 颜色条的位置，默认为None。
    - cbar_label: 颜色条的标签，默认为None。
    - mask: 三维numpy数组，其True值将用mask_color绘制。(如果mask为None,则不会绘制mask。)
    - mask_color: mask的颜色，默认为MASK_COLOR。
    - voxel_kwargs: 传递给plt_voxel_3d的额外关键字参数。
    - cbar_kwargs: 传递给add_side_colorbar的额外关键字参数。

    注意:
    - 如果想要呈现无边框的效果，可以将edgecolors设置为None。
    '''
    # 设置默认参数
    cbar_position = update_dict(CBAR_POSITION_3D, cbar_position)
    voxel_kwargs = update_dict({}, voxel_kwargs)
    cbar_kwargs = update_dict({}, cbar_kwargs)
    if vmin is None:
        vmin = np.nanmin(data)
    elif vmin > np.nanmin(data):
        cbar_kwargs['add_leq'] = True
    if vmax is None:
        vmax = np.nanmax(data)
    elif vmax < np.nanmax(data):
        cbar_kwargs['add_geq'] = True
    
    # 获取norm
    norm = get_norm(norm_mode, vmin=vmin, vmax=vmax, norm_kwargs=norm_kwargs)

    # 根据归一化后的data值获取颜色
    facecolors = cmap(norm(data))
    
    # 绘制体素
    show_data = ~np.isnan(data)
    v = plt_voxel_3d(ax, show_data, color=None, facecolors=facecolors, edgecolors=edgecolors, **voxel_kwargs)
    vs = [v]

    # 绘制mask
    if mask is not None:
        use_mask = True
        mask_v = plt_voxel_3d(ax, mask, color=None, facecolors=mask_color, edgecolors=edgecolors, **voxel_kwargs)
        vs.append(mask_v)
    else:
        use_mask = False

    # 添加colorbar
    if cbar:
        cbars = add_side_colorbar(ax, cmap=cmap, vmin=vmin, vmax=vmax, cbar_label=cbar_label, cbar_position=cbar_position, use_mask=use_mask, mask_color=mask_color, **cbar_kwargs)
        return vs, cbars
    else:
        return vs


def plt_vstack(ax, x, y_values, z_sets, cmap=CMAP, alpha=FAINT_ALPHA):
    """
    在三维坐标系中以堆叠方式绘制数据集。
    
    参数：
    - x: 一维数组,代表x轴的数据点
    - y_sets: 一个包含多个y数据集的二维数组(每个数据集对应x轴的数据点)
    - z_values: 每个y数据集的对应z轴值,决定在z轴上的位置(可以是array或者list)
    - colormap: 字符串,指定用于多边形填充颜色的colormap
    - alpha: 透明度值，控制多边形的透明度。

    使用示例:
    x = np.linspace(0, 10, 100)
    y_values = [1, 2, 3]
    z_sets = [np.sin(x), np.sin(2*x), np.sin(3*x)]
    plt_vstack(ax, x, y_values, z_sets)
    """
    
    # 定义辅助函数，生成位于(x, z)曲线之下的多边形的顶点
    def polygon_under_graph(x, z):
        """
        构造顶点列表，定义填充(x, z)曲线之下的多边形区域。
        假设x是升序排列。
        """
        return [(x[0], 0.), *zip(x, z), (x[-1], 0.)]

    # 生成每个数据集的多边形顶点列表
    verts = [polygon_under_graph(x, z) for z in z_sets]

    # 生成对应数量的颜色
    facecolors = cmap(np.linspace(0, 1, len(verts)))

    # 创建PolyCollection对象并添加到3D坐标系中
    poly = mcoll.PolyCollection(verts, facecolors=facecolors, alpha=alpha)
    ax.add_collection3d(poly, zs=y_values, zdir='y')
# endregion


# region 复杂作图函数(matplotlib系列,输入dataframe使用)
def plt_scatter_heatmap(ax, color_data, size_data, cmap=HEATMAP_CMAP, edgecolor=RANA, vnorm_mode='linear', vmin=None, vmax=None, vnorm_kwargs=None, snorm_mode='linear', smin=None, smax=None, snorm_kwargs=None, rel_smap=partial(scale_to_new_range, old_min=0, old_max=1, new_min=0.05, new_max=0.95), smask=None, smask_marker='X', smask_smap_float=1.0, smask_text='mask', cmask=None, cmask_color=MASK_COLOR, cmask_text='mask', add_cbar=True, cbar_position=None, cbar_label=None, size_label=None, align_label_coord=4, xtick_rotation=XTICK_ROTATION, ytick_rotation=YTICK_ROTATION, show_xtick=True, show_ytick=True, show_all_xtick=True, show_all_ytick=True, xtick_fontsize=TICK_SIZE, ytick_fontsize=TICK_SIZE, grid_kwargs=None, scatter_kwargs=None, cbar_kwargs=None, scatter_cbar_kwargs=None, inset_mode='fig'):
    '''
    使用DataFrame绘制圆形热图
    :param ax: matplotlib的轴对象,用于绘制图形
    :param color_data: 颜色数据的DataFrame
    :param size_data: 大小数据的DataFrame
    :param cmap: 颜色映射
    :param edgecolor: 网格的颜色,默认为RANA
    :param vmin: 颜色映射的最小值,默认为None
    :param vmax: 颜色映射的最大值,默认为None
    :param smin: 大小映射的最小值,默认为None
    :param smax: 大小映射的最大值,默认为None
    :param rel_smap: normalize后的值映射到相对大小(相对大小)
    :param add_cbar: 是否添加颜色条,默认为True
    :param cbar_position: 颜色条的位置,默认为None
    :param cbar_label: 颜色条的标签,默认为None
    :param size_label: 大小条的标签,默认为None
    :param grid_kwargs: 传递给add_grid的额外关键字参数
    :param scatter_kwargs: 传递给plt_scatter的额外关键字参数
    :param cbar_kwargs: 传递给add_colorbar的额外关键字参数
    :param scatter_cbar_kwargs: 传递给add_scatter_colorbar的额外关键字参数
    '''
    # 设置默认参数
    cbar_position = update_dict(CBAR_POSITION, cbar_position)
    grid_kwargs = update_dict({}, grid_kwargs)
    scatter_kwargs = update_dict({}, scatter_kwargs)
    cbar_kwargs = update_dict({}, cbar_kwargs)
    scatter_cbar_kwargs = update_dict({}, scatter_cbar_kwargs)
    snorm_kwargs = update_dict({'clip': True}, snorm_kwargs) # 默认clip=True,不然如果产生负的size会报错

    # 将数据转换为dataframe
    if isinstance(color_data, np.ndarray):
        color_data = pd.DataFrame(color_data)
    else:
        color_data = color_data.copy()
    if isinstance(size_data, np.ndarray):
        size_data = pd.DataFrame(size_data)
    else:
        size_data = size_data.copy()
    if isinstance(smask, np.ndarray):
        smask = pd.DataFrame(smask)
    elif smask is not None:
        smask = smask.copy()
    if isinstance(cmask, np.ndarray):
        cmask = pd.DataFrame(cmask)
    elif cmask is not None:
        cmask = cmask.copy()

    # 设定use_mask
    if smask is not None:
        use_smask = True
    else:
        use_smask = False
    if cmask is not None:
        use_cmask = True
    else:
        use_cmask = False

    # 更新vmin,vmax,smin,smax
    if vmin is None:
        vmin = np.nanmin(color_data.values)
    elif vmin > np.nanmin(color_data.values):
        cbar_kwargs['add_leq'] = True
    if vmax is None:
        vmax = np.nanmax(color_data.values)
    elif vmax < np.nanmax(color_data.values):
        cbar_kwargs['add_geq'] = True
    if smin is None:
        smin = np.nanmin(size_data.values)
    elif smin > np.nanmin(size_data.values):
        scatter_cbar_kwargs['add_leq'] = True
    if smax is None:
        smax = np.nanmax(size_data.values)
    elif smax < np.nanmax(size_data.values):
        scatter_cbar_kwargs['add_geq'] = True

    # 画出棋盘格
    add_grid(ax, range(len(color_data.columns)), range(len(color_data.index)), color=edgecolor, zorder=0, **grid_kwargs)
    ax.set_xlim(-1, len(color_data.columns))
    ax.set_ylim(-1, len(color_data.index))

    # 设定tick和ticklabel
    if show_xtick:
        if show_all_xtick:
            ax.set_xticks(np.arange(len(color_data.columns)))
            ax.set_xticklabels(
                color_data.columns, rotation=xtick_rotation, fontsize=xtick_fontsize)
        else:
            ax.set_xticklabels(ax.get_xticklabels(),
                               rotation=xtick_rotation, fontsize=xtick_fontsize)
        ax.xaxis.set_tick_params(labelbottom=show_xtick)
    else:
        ax.set_xticks([])

    if show_ytick:
        if show_all_ytick:
            ax.set_yticks(np.arange(len(color_data.index)))
            ax.set_yticklabels(
                color_data.index, rotation=ytick_rotation, fontsize=ytick_fontsize)
        else:
            ax.set_yticklabels(ax.get_yticklabels(),
                               rotation=ytick_rotation, fontsize=ytick_fontsize)
        ax.yaxis.set_tick_params(labelleft=show_ytick)
    else:
        ax.set_yticks([])

    # 计算datalim意义的1对应的radius的大小
    ax_width, ax_height = get_ax_size(ax)
    radius_width = ax_width / len(color_data.columns) / 2
    radius_height = ax_height / len(color_data.index) / 2
    radius = inch_to_point(min(radius_width, radius_height))

    # 获取smap
    def smap(x):
        return rel_smap(x) * radius**2 * np.pi

    # 获取norm
    cnorm = get_norm(vnorm_mode, vmin=vmin, vmax=vmax, norm_kwargs=vnorm_kwargs)
    snorm = get_norm(snorm_mode, vmin=smin, vmax=smax, norm_kwargs=snorm_kwargs)

    # 将y轴翻转(这样可以让矩阵的0,0在左上角)
    ax.invert_yaxis()

    # 画出每个散点
    for row in color_data.index:
        for column in color_data.columns:
            marker = 'o'
            color = cmap(cnorm(color_data.loc[row, column]))
            size = smap(snorm(size_data.loc[row, column]))
            if smask is not None:
                if smask.loc[row, column]:
                    marker = smask_marker
                    size = smap(smask_smap_float)
            if cmask is not None:
                if cmask.loc[row, column]:
                    color = cmask_color
            plt_scatter(ax, [column], [row], s=size, c=color, marker=marker, **scatter_kwargs)
    
    # 获取side_ax
    side_ax = add_side_ax(ax, position=cbar_position['position'], relative_size=cbar_position['size'], pad=cbar_position['pad'])

    if add_cbar:
        # 假如color_data和size_data一致,则只画一个颜色条;否则画两个颜色条
        if size_data.equals(color_data):
            if cbar_label is None and size_label is not None:
                cbar_label = size_label
            add_scatter_colorbar(side_ax, cmap=cmap, vnorm_mode=vnorm_mode, vmin=vmin, vmax=vmax, vnorm_kwargs=vnorm_kwargs, snorm_mode=snorm_mode, smin=smin, smax=smax, snorm_kwargs=snorm_kwargs, smap=smap, cbar_label=cbar_label, use_mask=use_smask, mask_marker=smask_marker, mask_smap_float=smask_smap_float, mask_text=smask_text, **scatter_cbar_kwargs)
        else:
            if cbar_position['position'] in ['top', 'bottom']:
                split_ncols = 2
                split_nrows = 1
            else:
                split_ncols = 1
                split_nrows = 2
            size_side_ax, cbar_side_ax = split_ax_by_gs(side_ax, nrows=split_nrows, ncols=split_ncols, hspace=0., wspace=0., label='cbar', inset_mode=inset_mode)

            # 添加颜色条
            cbars = add_colorbar(cbar_side_ax, cmap=cmap, vmin=vmin, vmax=vmax, cbar_label=cbar_label, cbar_position=cbar_position, use_mask=use_cmask, mask_color=cmask_color, mask_tick=cmask_text, inset_mode=inset_mode, **cbar_kwargs)

            same_color_cmap = get_cmap([edgecolor, edgecolor]) # 生成一个只有一种颜色的cmap(因为这里只表达大小不表达颜色)
            add_scatter_colorbar(size_side_ax, cmap=same_color_cmap, smin=smin, smax=smax, snorm_mode=snorm_mode, snorm_kwargs=snorm_kwargs, smap=smap, cbar_label=size_label, use_mask=use_smask, mask_marker=smask_marker, mask_smap_float=smask_smap_float, mask_text=smask_text, **scatter_cbar_kwargs)
            align_label_manual([size_side_ax, cbar_side_ax], axis='y', label_coord=align_label_coord)
            return cbars


def plt_group_bar_df(ax, df, bar_width=None, group_by='columns', colors=CMAP, vert=True, **kwargs):
    '''
    基于DataFrame绘制分组的柱状图,根据分组是按照行还是列来自动处理。
    :param ax: matplotlib的轴对象,用于绘制图形
    :param df: 数据存放的DataFrame
    :param bar_width: 单个柱子的宽度,默认为None,自动确定宽度
    :param group_by: 指定分组数据是按行('rows')还是按列('columns'),默认为'columns'
    :param colors: 每个分组的颜色列表;也可以指定cmap,默认为CMAP,然后自动生成颜色序列
    :param vert: 是否是垂直的柱状图,默认为True
    :param kwargs: 其他plt.bar支持的参数
    '''
    if group_by == 'columns':
        x = df.index.tolist()  # x轴标签
        y = df.values.T.tolist()  # 每组的柱状图值
        label_list = df.columns.tolist()  # 每个柱状图的标签
    else:  # 'rows'
        x = df.columns.tolist()  # x轴标签
        y = df.values.tolist()  # 每组的柱状图值
        label_list = df.index.tolist()  # 每个柱状图的标签

    plt_group_bar(ax=ax, x=x, y=y, label_list=label_list,
                       bar_width=bar_width, colors=colors, vert=vert, **kwargs)


def plt_two_side_bar_df():
    pass


def plt_group_box_df(ax, df, group_by='columns', colors=None, **kwargs):
    print('需要加入横纵选项')
    pass


def plt_split_violine_df(ax, df, group_by='columns', colors=None, **kwargs):
    '''
    see https://seaborn.pydata.org/examples/grouped_violinplots.html
    '''
    print('需要加入横纵选项')
    pass


def read_sns():
    print('please see sns gallery and add new function')
# endregion


# region 复杂作图函数(sns系列,输入向量使用)
def sns_band_line(ax, x, y, band_width, label=None, color=BLUE, alpha=FAINT_ALPHA, **kwargs):
    '''
    使用x和y绘制折线图,并添加一个表示误差带的区域
    :param ax: matplotlib的轴对象,用于绘制图形
    :param x: x轴的数据
    :param y: y轴的数据
    :param band_width: 误差带宽度
    :param label: 图例标签,默认为None
    :param color: 折线图的颜色,默认为蓝色
    :param alpha: 误差带的透明度
    :param kwargs: 其他sns.lineplot支持的参数
    '''
    # 绘制基础折线图
    sns_line(ax, x, y, label=label, color=color, **kwargs)

    # 计算误差带的上下界
    y_upper = np.array(y) + np.array(band_width)
    y_lower = np.array(y) - np.array(band_width)

    # 绘制误差带
    ax.fill_between(x, y_lower, y_upper, color=color, alpha=alpha)
# endregion


# region 复杂作图函数(sns系列,输入dataframe使用)
def sns_marginal_heatmap(ax, data, x_side_ax=None, y_side_ax=None, outside=True, x_side_ax_position='top', y_side_ax_position='right', x_side_ax_pad=SIDE_PAD, y_side_ax_pad=SIDE_PAD, x_side_ax_size=0.3, y_side_ax_size=0.3, x_color=BLUE, y_color=BLUE, heatmap_kwargs=None, bar_kwargs=None, rm_tick=True, rm_spine=True, rm_axis=True, inset_mode='fig'):
    '''
    绘制带有边缘分布的热图
    '''
    heatmap_kwargs = update_dict(get_default_param(sns_heatmap), heatmap_kwargs)
    heatmap_kwargs['cbar_position'] = update_dict(CBAR_POSITION, heatmap_kwargs['cbar_position'])
    bar_kwargs = update_dict(get_default_param(plt_bar), bar_kwargs)
    bar_kwargs = update_dict(bar_kwargs, {'width': 1})

    if isinstance(data, np.ndarray):
        local_data = pd.DataFrame(data)
    else:
        local_data = data.copy()
    
    # 获得边缘分布需要的ax
    if x_side_ax is None and y_side_ax is None:
        if outside:
            x_side_ax = add_side_ax(ax, position=x_side_ax_position, relative_size=x_side_ax_size, pad=x_side_ax_pad, sharex=ax, inset_mode=inset_mode, hide_repeat_xaxis=False, hide_repeat_yaxis=False)
            y_side_ax = add_side_ax(ax, position=y_side_ax_position, relative_size=y_side_ax_size, pad=y_side_ax_pad, sharey=ax, inset_mode=inset_mode, hide_repeat_xaxis=False, hide_repeat_yaxis=False)
        else:
            # 当边缘分布在内部时，需要手动分划ax然后获取需要的
            x_side_ax, y_side_ax, ax = split_with_double_marginal_ax(ax=ax, x_side_ax_position=x_side_ax_position, y_side_ax_position=y_side_ax_position, x_side_ax_size=x_side_ax_size, y_side_ax_size=y_side_ax_size, x_side_ax_pad=x_side_ax_pad, y_side_ax_pad=y_side_ax_pad, inset_mode=inset_mode)
    
    # 调整cbar的位置
    if heatmap_kwargs['cbar']:
        ax_width, ax_height = get_ax_size(ax)
        x_side_ax_width, x_side_ax_height = get_ax_size(x_side_ax)
        y_side_ax_width, y_side_ax_height = get_ax_size(y_side_ax)
        if heatmap_kwargs['cbar_position']['position'] == x_side_ax_position:
            heatmap_kwargs['cbar_position']['pad'] += x_side_ax_pad + x_side_ax_height / ax_height
        if heatmap_kwargs['cbar_position']['position'] == y_side_ax_position:
            heatmap_kwargs['cbar_position']['pad'] += y_side_ax_pad + y_side_ax_width / ax_width

    # 绘制热图
    sns_heatmap(ax, local_data, **heatmap_kwargs)

    # 绘制边缘分布
    plt_bar(x_side_ax, x=np.arange(len(local_data.columns)) + 0.5, y=col_sum_df(local_data), **update_dict(bar_kwargs, {'color': x_color}))
    plt_bar(y_side_ax, x=np.arange(len(local_data.index)) + 0.5, y=row_sum_df(local_data), **update_dict(bar_kwargs, {'color': y_color, 'vert': False}))

    # 隐藏边缘分布的刻度和坐标轴
    if rm_tick:
        rm_ax_tick(x_side_ax)
        rm_ax_tick(y_side_ax)
    if rm_spine:
        rm_ax_spine(x_side_ax)
        rm_ax_spine(y_side_ax)
    if rm_axis:
        rm_ax_axis(x_side_ax)
        rm_ax_axis(y_side_ax)

    # 调整边缘分布的方向
    if x_side_ax_position == 'bottom':
        x_side_ax.invert_yaxis()
    if y_side_ax_position == 'left':
        y_side_ax.invert_xaxis()

    return ax, x_side_ax, y_side_ax


def sns_triangle_heatmap(ax, up_data, lower_data, up_mask=None, lower_mask=None, up_mask_color=MASK_COLOR, lower_mask_color=MASK_COLOR, same_cmap=True, cmap=CMAP, up_cmap=CMAP, lower_cmap=CMAP, up_norm_mode='linear', up_vmin=None, up_vmax=None, up_norm_kwargs=None, lower_norm_mode='linear', lower_vmin=None, lower_vmax=None, lower_norm_kwargs=None, cbar=True, cbar_label=None, up_cbar_label='up', lower_cbar_label='lower', two_cbar_pad=0.2, cbar_position=None, heatmap_kwargs=None, inset_mode='fig'):
    '''
    绘制三角形热图
    :param ax: matplotlib的轴对象,用于绘制图形
    :param up_data: 上三角的数据
    :param lower_data: 下三角的数据
    '''
    # 设置默认值
    cbar_position = update_dict(CBAR_POSITION, cbar_position)
    heatmap_kwargs = update_dict({}, heatmap_kwargs)
    heatmap_kwargs['cbar_kwargs'] = update_dict({}, heatmap_kwargs.get('cbar_kwargs'))
    up_cbar_kwargs = update_dict({}, heatmap_kwargs.get('cbar_kwargs'))
    up_cbar_kwargs['inset_mode'] = inset_mode
    lower_cbar_kwargs = update_dict({}, heatmap_kwargs.get('cbar_kwargs'))
    lower_cbar_kwargs['inset_mode'] = inset_mode

    # 得到是否使用mask(注意区分此mask和tri_mask)
    if up_mask is None:
        up_use_mask = False
    else:
        up_use_mask = True
    if lower_mask is None:
        lower_use_mask = False
    else:
        lower_use_mask = True

    # 转换为dataframe
    if isinstance(up_data, np.ndarray):
        up_data = pd.DataFrame(up_data)
    else:
        up_data = up_data.copy()
    if isinstance(lower_data, np.ndarray):
        lower_data = pd.DataFrame(lower_data)
    else:
        lower_data = lower_data.copy()
    if isinstance(up_mask, np.ndarray):
        up_mask = pd.DataFrame(up_mask)
    elif up_mask is not None:
        up_mask = up_mask.copy()
    if isinstance(lower_mask, np.ndarray):
        lower_mask = pd.DataFrame(lower_mask)
    elif lower_mask is not None:
        lower_mask = lower_mask.copy()

    # 得到上下三角的mask
    up_tri_mask = np.triu(np.ones_like(up_data), k=0) != 0
    lower_tri_mask = np.tril(np.ones_like(lower_data), k=0) != 0

    # 利用上下三角的mask分别处理up和lower的mask(防止出现大块的不在三角内的mask)
    if up_use_mask:
        up_mask = up_mask.where(up_tri_mask, False)
    if lower_use_mask:
        lower_mask = lower_mask.where(lower_tri_mask, False)

    # 得到mask后的数据(将不需要的部分设置为nan,这里是包含对角线的上半和下半部分,包含对角线可以让vmin和vmax正确计算)
    up_data = up_data.where(up_tri_mask, np.nan)
    lower_data = lower_data.where(lower_tri_mask, np.nan)

    # 处理vmin和vmax
    if up_vmin is None:
        up_vmin = np.nanmin(up_data.values)
    elif up_vmin > np.nanmin(up_data.values):
        up_cbar_kwargs['add_leq'] = True
    if up_vmax is None:
        up_vmax = np.nanmax(up_data.values)
    elif up_vmax < np.nanmax(up_data.values):
        up_cbar_kwargs['add_geq'] = True
    if lower_vmin is None:
        lower_vmin = np.nanmin(lower_data.values)
    elif lower_vmin > np.nanmin(lower_data.values):
        lower_cbar_kwargs['add_leq'] = True
    if lower_vmax is None:
        lower_vmax = np.nanmax(lower_data.values)
    elif lower_vmax < np.nanmax(lower_data.values):
        lower_cbar_kwargs['add_geq'] = True

    # 得到norm
    up_norm = get_norm(up_norm_mode, vmin=up_vmin, vmax=up_vmax, norm_kwargs=up_norm_kwargs)
    lower_norm = get_norm(lower_norm_mode, vmin=lower_vmin, vmax=lower_vmax, norm_kwargs=lower_norm_kwargs)

    # 对于使用同一个cmap的情况,需要保证两个vmin和vmax一致
    if same_cmap:
        up_vmin = min(up_vmin, lower_vmin)
        up_vmax = max(up_vmax, lower_vmax)
        lower_vmin = up_vmin
        lower_vmax = up_vmax
        up_cmap = cmap
        lower_cmap = cmap
        if up_norm_mode != lower_norm_mode or up_norm_kwargs != lower_norm_kwargs:
            raise ValueError('当same_cmap为True时,up和lower的norm_mode和norm_kwargs必须一致')
        up_norm = get_norm(up_norm_mode, vmin=up_vmin, vmax=up_vmax, norm_kwargs=up_norm_kwargs)
        lower_norm = up_norm
        if up_vmin > np.nanmin(up_data.values) or lower_vmin > np.nanmin(lower_data.values):
            heatmap_kwargs['cbar_kwargs']['add_leq'] = True
        if up_vmax < np.nanmax(up_data.values) or lower_vmax < np.nanmax(lower_data.values):
            heatmap_kwargs['cbar_kwargs']['add_geq'] = True
        if up_use_mask or lower_use_mask:
            up_use_mask = True
            lower_use_mask = True

    # 确保是方阵
    if up_data.shape[0] != up_data.shape[1]:
        raise ValueError('up_data必须是方阵')
    if lower_data.shape[0] != lower_data.shape[1]:
        raise ValueError('lower_data必须是方阵')

    # 绘制上三角(mask掉下三角)
    sns_heatmap(ax, up_data, cmap=up_cmap, norm_mode=up_norm_mode, vmin=up_vmin, vmax=up_vmax, norm_kwargs=up_norm_kwargs, cbar=False, mask=up_mask, mask_color=up_mask_color, **heatmap_kwargs)
    # 绘制下三角(mask掉上三角)
    sns_heatmap(ax, lower_data, cmap=up_cmap, norm_mode=lower_norm_mode, vmin=lower_vmin, vmax=lower_vmax, norm_kwargs=lower_norm_kwargs, cbar=False, mask=lower_mask, mask_color=lower_mask_color, **heatmap_kwargs)
    cbar_list = []
    if cbar:
        if same_cmap:
            # 使用same_cmap的情况,只需要绘制一次cbar
            cbar_list.append(add_side_colorbar(ax, cmap=cmap, vmin=up_vmin, vmax=up_vmax, cbar_position=cbar_position, cbar_label=cbar_label, use_mask=up_use_mask, mask_color=up_mask_color, **heatmap_kwargs['cbar_kwargs']))
        else:
            side_ax = add_side_ax(ax, position=cbar_position['position'], relative_size=cbar_position['size'], pad=cbar_position['pad'])
            if cbar_position['position'] in ['top', 'bottom']:
                split_ncols = 2
                split_nrows = 1
            elif cbar_position['position'] in ['left', 'right']:
                split_ncols = 1
                split_nrows = 2
            up_ax, lower_ax = split_ax_by_gs(side_ax, nrows=split_nrows, ncols=split_ncols, hspace=two_cbar_pad, wspace=two_cbar_pad, label='cbar', inset_mode=inset_mode)
            cbar_list.append(add_colorbar(ax=up_ax, cmap=up_cmap, vmin=up_vmin, vmax=up_vmax, cbar_position=cbar_position, cbar_label=up_cbar_label, use_mask=up_use_mask, mask_color=up_mask_color, **up_cbar_kwargs))
            cbar_list.append(add_colorbar(ax=lower_ax, cmap=lower_cmap, vmin=lower_vmin, vmax=lower_vmax, cbar_position=cbar_position, cbar_label=lower_cbar_label, use_mask=lower_use_mask, mask_color=lower_mask_color, **lower_cbar_kwargs))

    # 给对角线添加线并上色
    for i in range(up_data.shape[0]):
        if up_mask is not None:
            if up_mask.loc[i, i]:
                plt_polygon(ax, [(i, i), (i + 1, i), (i + 1, i + 1)], color=up_mask_color, adjust_lim=False)
            else:
                plt_polygon(ax, [(i, i), (i + 1, i), (i + 1, i + 1)], color=up_cmap(up_norm(up_data.iloc[i, i])), adjust_lim=False)
        else:
            plt_polygon(ax, [(i, i), (i + 1, i), (i + 1, i + 1)], color=up_cmap(up_norm(up_data.iloc[i, i])), adjust_lim=False)
        if lower_mask is not None:
            if lower_mask.loc[i, i]:
                plt_polygon(ax, [(i, i), (i, i + 1), (i + 1, i + 1)], color=lower_mask_color, adjust_lim=False)
            else:
                plt_polygon(ax, [(i, i), (i, i + 1), (i + 1, i + 1)], color=lower_cmap(lower_norm(lower_data.iloc[i, i])), adjust_lim=False)
        else:
            plt_polygon(ax, [(i, i), (i, i + 1), (i + 1, i + 1)], color=lower_cmap(lower_norm(lower_data.iloc[i, i])), adjust_lim=False)
    # 绘制分界线
    plt_line(ax, [0, up_data.shape[0]], [0, up_data.shape[0]], color=WHITE)
    return cbar_list
# endregion


# region 复杂作图函数(添加star)
def add_star_extreme_value(ax, x, y, extreme_type, label=None, marker=STAR, color=None, markersize=STAR_SIZE, linestyle='None', **kwargs):
    '''
    在min或者max位置添加星号
    :param extreme_type: 极值类型,可以是'max'或者'min'
    '''
    # 找到数据中的最大值和最小值位置
    if extreme_type == 'max':
        # 计算最大值
        max_value = np.nanmax(y)
        # 找到所有最大值的位置
        pos = np.where(y == max_value)[0]
    elif extreme_type == 'min':
        # 计算最小值
        min_value = np.nanmin(y)
        # 找到所有最小值的位置
        pos = np.where(y == min_value)[0]

    # 给label自动赋值
    if label is None:
        label = extreme_type

    # 给star_color自动赋值
    if color is None:
        if extreme_type == 'max':
            color = RED
        if extreme_type == 'min':
            color = GREEN

    # 画图
    return add_star(ax, np.array(x)[pos], np.array(y)[pos], label=label, marker=marker, color=color, markersize=markersize, linestyle=linestyle, **kwargs)


def add_star_extreme_value_heatmap(ax, data, extreme_type, label=None, marker=STAR, color=None, markersize=STAR_SIZE, linestyle='None', **kwargs):
    '''
    在热图的极值位置添加星号
    :param extreme_type: 极值类型,可以是'max'或者'min'
    '''
    # 找到数据中的最大值和最小值位置
    if extreme_type == 'max':
        # 找到最大值
        extreme_value = np.nanmax(data)
    elif extreme_type == 'min':
        # 找到最小值
        extreme_value = np.nanmin(data)

    # 找到所有极值的位置
    pos = np.where(data == extreme_value)

    # 给label自动赋值
    if label is None:
        label = extreme_type

    # 给star_color自动赋值
    if color is None:
        if extreme_type == 'max':
            color = RED
        if extreme_type == 'min':
            color = GREEN

    # 画图
    return add_star_heatmap(ax, pos[1], pos[0], label=label, marker=marker, color=color, markersize=markersize, linestyle=linestyle, **kwargs)
# endregion


# region 复杂作图函数(添加辅助线)
def add_vline_extreme_value(ax, x, y, extreme_type, label=None, color=None, linestyle=AUXILIARY_LINE_STYLE, linewidth=LINE_WIDTH, **kwargs):
    '''
    在min或者max位置添加垂直线
    :param extreme_type: 极值类型,可以是'max'或者'min'
    '''
    # 找到数据中的最大值和最小值位置
    if extreme_type == 'max':
        # 计算最大值
        max_value = np.nanmax(y)
        # 找到所有最大值的位置
        pos = np.where(y == max_value)[0]
    elif extreme_type == 'min':
        # 计算最小值
        min_value = np.nanmin(y)
        # 找到所有最小值的位置
        pos = np.where(y == min_value)[0]

    # 给label自动赋值
    if label is None:
        label = extreme_type

    # 给color自动赋值
    if color is None:
        if extreme_type == 'max':
            color = RED
        if extreme_type == 'min':
            color = GREEN

    # 画图
    return add_vline(ax, np.array(x)[pos], label=label, color=color, linestyle=linestyle, linewidth=linewidth, **kwargs)
# endregion


# region 复杂作图函数(添加文字)
def add_text_by_dict(ax, text_dict, show_list=None, round_digit_dict=None, round_format_dict=None, unit_dict=None, unit_mode='()', color_dict=None, fontsize=FONT_SIZE, text_x=TEXT_X, text_y=TEXT_Y, text_process=None, text_kwargs=None):
    '''
    利用dict的key和value添加文字(每个key一行)
    
    参数:
    - ax: matplotlib的轴对象,用于绘制图形
    - text_dict: dict, 文字的dict
    - show_list: list, 需要显示的key的列表
    - round_digit_dict: dict, 每个key对应的小数位数
    - round_format_dict: dict, 每个key对应的小数格式
    - unit_dict: dict, 每个key对应的单位
    - fontsize: 字体大小
    - text_x: 文字的x坐标(默认为TEXT_X)
    - text_y: 文字的y坐标(默认为TEXT_Y)
    - text_process: dict, 文字处理的dict
    - text_kwargs: dict, 文字的其他参数

    注意:
    - show_list为None时,默认显示所有的key,如果想要什么也不显示,可以传入空列表[],(但是这样的话也没有必要运行此函数)
    '''
    if show_list is None:
        show_list = list(text_dict.keys())
    if round_digit_dict is None:
        round_digit_dict = {key: ROUND_DIGITS for key in show_list}
    if round_format_dict is None:
        round_format_dict = {key: 'general' for key in show_list}
    if unit_dict is None:
        unit_dict = {key: None for key in show_list}
    if color_dict is None:
        color_dict = {key: BLACK for key in show_list}
    text_process = update_dict(TEXT_PROCESS, text_process)
    if text_kwargs is None:
        text_kwargs = {}
    
    y_offset = 0.
    for key in show_list:
        if key in text_dict:
            # 格式化key
            local_key = format_text(key, text_process=text_process)
            value = text_dict[key]

            round_value = round_float(value, round_digit_dict[key], round_format_dict[key], latex=True)
            if unit_dict[key] is not None:
                if unit_mode == '()':
                    round_value = f'{round_value} ({unit_dict[key]})'
                elif unit_mode == '[]':
                    round_value = f'{round_value} [{unit_dict[key]}]'
                else:
                    round_value = f'{round_value} {unit_dict[key]}'
            add_text(ax, f'{local_key}: {round_value}', x=text_x, y=text_y - y_offset, fontsize=fontsize, color=color_dict[key], **text_kwargs)
            y_offset += point_to_ax_proportion(ax, fontsize, axis='y')
# endregion


# region 通用函数(颜色)
def rgb_to_rgba(rgb, alpha=1.0):
    """
    将 RGB 颜色转换为 RGBA 颜色

    参数:
    rgb (tuple): RGB 颜色值，范围在 0-1
    alpha (float): 透明度，范围在 0-1

    返回:
    tuple: RGBA 颜色值
    """
    r, g, b = rgb

    # 返回 RGBA 颜色值
    return (r, g, b, alpha)


def get_random_color(n, cmap=CMAP, seed=None):
    '''
    随意获取n个颜色
    '''
    if seed is None:
        seed = get_time_seed()
    rng = get_local_rng(seed)
    colors = cmap(np.linspace(0, 1, n))
    return rng.permutation(colors)


def get_plt_color(n):
    '''
    获取matplotlib默认的颜色
    '''
    prop_cycler = plt.rcParams['axes.prop_cycle']
    cycler_iterator = itertools.cycle(prop_cycler)
    colors = []
    for _ in range(n):
        props = next(cycler_iterator)
        colors.append(props['color'])
    return colors


def scale_color(color, scale):
    '''
    输入(r,g,b)的颜色,和一个缩放比例,返回缩放后的颜色
    '''
    new_color = list(c * scale for c in color)
    for i, c in enumerate(new_color):
        if c > 1.0:
            new_color[i] = 1.0
        elif c < 0.0:
            new_color[i] = 0.0
    return tuple(new_color)
# endregion


# region 通用函数(散点大小)
def get_suitable_s(ax, num):
    '''
    根据ax的大小和num的数量,返回合适的s值,实际使用时,可以先使用这个值,然后再根据实际情况在此基础上进行调整

    注意:
    s指的是面积,并且是以点(points)的平方为单位的
    '''
    ax_width, ax_height = get_ax_size(ax)
    ax_width_point = inch_to_point(ax_width)
    ax_height_point = inch_to_point(ax_height)
    s = np.pi * (np.min([ax_width_point, ax_height_point]) / np.sqrt(num)) ** 2
    s = np.min([s, 70.])
    return s
# endregion


# region 通用函数(transform)
def map_transform(x, y, source_transform, target_transform):
    """
    将坐标从一个 transform 转换到另一个 transform。

    参数：
    - ax: matplotlib 的 Axes 对象。
    - x, y: 要转换的坐标。
    - source_transform: 源坐标系统的 transform。比如ax.transAxes
    - target_transform: 目标坐标系统的 transform。

    返回：
    - 转换后的坐标 (x_new, y_new)。

    注意:
    不要转换data坐标,因为data坐标随着图像继续作画,或者改变坐标轴的范围,会发生变化
    """
    # 将 (x, y) 点转换到 source_transform 的坐标系统中
    source_to_display = source_transform.transform([x, y])
    
    # 获取 target_transform 的逆变换（从 display 转到 target）
    display_to_target = target_transform.inverted()
    
    # 使用逆变换将 display 坐标转换到 target 坐标系统
    x_new, y_new = display_to_target.transform(source_to_display)
    
    return x_new, y_new
# endregion


# region 通用函数(判断是否是xlabel, ylabel, title, ax的外框)
@to_be_improved
def is_xlabel(obj):
    '''
    判断对象是否是x轴标签。
    
    参数：
        obj: Matplotlib对象。
        
    返回：
        is_xlabel: 布尔值，表示对象是否是x轴标签。
    '''
    if isinstance(obj, plt.matplotlib.text.Text):
        if obj.get_text() == obj.axes.get_xlabel() and obj.get_position() == obj.axes.xaxis.label.get_position():
            return True
    return False


@to_be_improved
def is_ylabel(obj):
    '''
    判断对象是否是y轴标签。
    
    参数：
        obj: Matplotlib对象。
        
    返回：
        is_ylabel: 布尔值，表示对象是否是y轴标签。
    '''
    if isinstance(obj, plt.matplotlib.text.Text):
        if obj.get_text() == obj.axes.get_ylabel() and obj.get_position() == obj.axes.yaxis.label.get_position():
            return True
    return False


def is_title(obj):
    '''
    判断对象是否是标题。
    
    参数：
        obj: Matplotlib对象。
        
    返回：
        is_title: 布尔值，表示对象是否是标题。
    '''
    try:
        if obj == obj.axes.title:
            return True
    except:
        return False


@to_be_improved
def is_ax_bounding_box(obj):
    '''
    判断对象是否是坐标轴的边界。
    
    参数：
        obj: Matplotlib对象。
        
    返回：
        is_ax_bounding_box: 布尔值，表示对象是否是坐标轴的边界。
    '''
    if isinstance(obj, plt.matplotlib.patches.Patch):
        if obj.get_xy() == (0, 0) and obj.get_width() == 1 and obj.get_height() == 1 and obj.get_angle() == 0:
            return True
    return False
# endregion


# region 通用函数(zorder)
def set_zorder(obj, new_zorder):
    '''
    设置对象的zorder值。zorder值越高，对象就越靠近顶部。
    
    参数：
        obj: matplotlib图形对象。
        new_zorder: 新的zorder值，类型为整数或浮点数。
    '''
    obj.set_zorder(new_zorder)


def get_zorder(obj):
    '''
    获取给定图形对象的zorder值。
    
    参数：
        obj: Matplotlib图形对象。
        
    返回：
        zorder: 图形对象的zorder值。
    '''
    return obj.get_zorder()


def get_zorder_dict(ax, include_axes=False):
    '''
    获取ax中所有对象及其zorder值，并以字典形式返回。
    
    参数：
        ax: matplotlib的Axes对象。
        include_axes: 布尔值，表示是否包含坐标轴的zorder。
        
    返回：
        zorders: 包含对象及其zorder值的字典。
    '''
    zorder_dict = {}
    for obj in ax.get_children():
        # 排除坐标轴的zorder，如果需要
        if not include_axes:
            if isinstance(obj, (plt.matplotlib.axis.XAxis, plt.matplotlib.axis.YAxis, plt.matplotlib.spines.Spine, plt.matplotlib.axis.Tick, plt.matplotlib.axis.XTick, plt.matplotlib.axis.YTick)):
                continue
            if is_xlabel(obj) or is_ylabel(obj) or is_title(obj):
                continue
            if is_ax_bounding_box(obj):
                continue
        if isinstance(obj, plt.matplotlib.text.Text):
            if len(obj.get_text()) == 0:
                continue
        zorder = obj.get_zorder()
        obj_type = type(obj).__name__  # 获取对象的类型名称
        zorder_dict[(obj, obj_type)] = zorder  # 将对象及其类型名称和zorder值添加到字典中
    return zorder_dict


@to_be_improved
def show_zorder(ax, include_axes=False, font_size=FONT_SIZE, color=RED, alpha=FAINT_ALPHA):
    '''
    在图上显示所有对象的zorder值。
    
    参数：
        ax: matplotlib的Axes对象。
        include_axes: 布尔值，表示是否显示坐标轴的zorder。
        font_size: 文本的字体大小。
        color: 文本的颜色。
        alpha: 文本的透明度。
    '''
    zorder_dict = get_zorder_dict(ax, include_axes)
    max_zorder = np.max(list(zorder_dict.values())) + 1
    
    for (obj, obj_type), zorder in zorder_dict.items():
        x_text, y_text = None, None
        if isinstance(obj, plt.Line2D):
            x, y = obj.get_xdata(), obj.get_ydata()
            x_text, y_text = x[0], y[0]
        elif isinstance(obj, plt.Text):
            x_text, y_text = obj.get_position()
        elif isinstance(obj, mcoll.PathCollection):  # 散点图
            offsets = obj.get_offsets()
            if len(offsets) > 0:
                x_text, y_text = offsets[0]
            else:
                continue
        elif isinstance(obj, mpatches.Patch):  # 包括矩形、圆等图形对象
            try:
                center = obj.get_center()
                x_text, y_text = center
            except AttributeError:
                # 对于没有get_center方法的Patch对象，跳过或使用其他方式获取其位置
                continue
        
        if x_text is not None and y_text is not None:
            add_annotation(ax, f'{obj_type}\nzorder={zorder}', xy=(x_text, y_text), xytext=((x_text+0.05, y_text+0.05)), xycoords='data', fontsize=font_size, color=color, ha='left', va='bottom', zorder=max_zorder, alpha=alpha, arrowprops={'fc': color, 'ec': color, 'head_length': ARROW_HEAD_LENGTH/2, 'head_width': ARROW_HEAD_WIDTH/2, 'linewidth': LINE_WIDTH/2}, bbox={'facecolor': 'white', 'edgecolor': color, 'pad': 0.5, 'alpha': alpha})
# endregion


# region 通用函数(inch,point,ax_proportion单位转换)
def inch_to_point(inch):
    '''
    将plt中使用的inch单位转换为points单位
    '''
    return inch * 72


def point_to_inch(point):
    '''
    将plt中使用的points单位转换为inch单位
    '''
    return point / 72


def ax_proportion_to_inch(ax, proportion, axis='x'):
    '''
    将ax的比例转换为inch单位
    '''
    ax_width, ax_height = get_ax_size(ax)
    if axis == 'x':
        return ax_width * proportion
    elif axis == 'y':
        return ax_height * proportion


def inch_to_ax_proportion(ax, inch, axis='x'):
    '''
    将inch单位转换为ax的比例
    '''
    ax_width, ax_height = get_ax_size(ax)
    if axis == 'x':
        return inch / ax_width
    elif axis == 'y':
        return inch / ax_height


def ax_proportion_to_point(ax, proportion, axis='x'):
    '''
    将ax的比例转换为points单位
    '''
    return inch_to_point(ax_proportion_to_inch(ax, proportion, axis))


def point_to_ax_proportion(ax, point, axis='x'):
    '''
    将points单位转换为ax的比例
    '''
    return inch_to_ax_proportion(ax, point_to_inch(point), axis)
# endregion


# region 通用函数(ax_size)
def get_ax_size(ax):
    '''
    获取给定轴的尺寸(以英寸为单位)
    
    参数:
    - ax: matplotlib轴对象
    
    返回:
    - ax_width: 轴的宽度(英寸)
    - ax_height: 轴的高度(英寸)
    '''
    # 获取图形的尺寸(英寸)
    fig = ax.get_figure()
    fig_width, fig_height = fig.get_size_inches()
    
    # 获取轴的边界框，这是相对于图形大小的
    bbox = ax.get_position()
    
    # 计算轴的实际尺寸（英寸）
    ax_width = fig_width * bbox.width
    ax_height = fig_height * bbox.height
    
    return ax_width, ax_height
# endregion


# region 通用函数(spine)
@iterate_over_axs
def move_spine_to_origin(ax, axis='both', arrow=True):
    '''
    将坐标轴移动到原点

    参数:
    ax: matplotlib.axes.Axes 对象
    axis: str, 要移动的坐标轴, 可选 'x', 'y', 'both', 默认 'both'
    arrow: bool, 是否在坐标轴末端添加箭头, 默认 True
    '''
    if axis == 'x' or axis == 'both':
        ax.spines["bottom"].set_position(("data", 0))
        if arrow:
            ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
    if axis == 'y' or axis == 'both':
        ax.spines["left"].set_position(("data", 0))
        if arrow:
            ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

@iterate_over_axs
def rm_ax_spine(ax, spines_to_remove=None):
    '''
    移除轴的边框。

    参数:
    - ax: matplotlib的Axes对象
    - spines_to_remove: 要移除的边框(边框指的仅仅是轴的线,不包括tick,ticklabel和label,如果想要移除这些,请使用rm_ax_tick,rm_ax_ticklabel,rm_ax_specific_axis)
    '''
    if spines_to_remove is None:
        spines_to_remove = ['top', 'right', 'left', 'bottom']
    if isinstance(spines_to_remove, str):
        spines_to_remove = [spines_to_remove]

    for spine in spines_to_remove:
        ax.spines[spine].set_visible(False)


def rm_cbar_spine(cbar):
    '''
    移除cbar的边框。(这里就不是通过spine的设置来移除了)

    参数:
    - cbar: matplotlib的Colorbar对象
    '''
    cbar.outline.set_visible(False)

@iterate_over_axs
def set_ax_spine(ax, spine_params=None):
    '''
    应用边框参数到轴。

    参数:
    - axs: matplotlib的Axes对象或者可迭代对象
    - spine_params: 边框参数, 例如{'linewidth': 2, 'color': 'red', 'linestyle': '--'}
    '''
    if spine_params:
        for spine in ax.spines.values():
            spine.set(**spine_params)
# endregion


# region 通用函数(tick)
def add_nested_tick(ax, axis, ticks, labels, location=0, length=TICK_MAJOR_SIZE*4, width=0):
    """
    在给定的轴上添加嵌套的刻度线和标签。为了美观起见, 使用了这个函数后再添加xlabel最好在sec上添加, 语法几乎等同于ax.set_xlabel等

    参数:
    ax (matplotlib.axes.Axes): 要添加嵌套刻度线的轴
    axis (str): 要添加嵌套刻度线的轴,可以是'x'或'y'
    ticks (list): 要添加的刻度线的位置
    labels (list): 要添加的标签
    location (float, optional): 次要刻度线的位置。默认为0
    length (float, optional): 刻度线的长度。默认为 None。
    """
    if axis == 'x':
        sec = ax.secondary_xaxis(location=location)
        sec.set_xticks(ticks, labels)
        sec.tick_params(axis='x', length=length, width=width)
        sec.spines['bottom'].set_visible(False)
    elif axis == 'y':
        sec = ax.secondary_yaxis(location=location)
        sec.set_yticks(ticks, labels)
        sec.tick_params(axis='y', length=length, width=width)
        sec.spines['left'].set_visible(False)
    return sec


def add_sep_tick(ax, axis, ticks, length=TICK_MAJOR_SIZE*4, width=TICK_MAJOR_WIDTH*2):
    """
    在给定的轴上添加tick作为某种分界线。为了美观起见, 使用了这个函数后再添加xlabel最好在sec上添加, 语法几乎等同于ax.set_xlabel等

    参数:
    ax (matplotlib.axes.Axes): 要添加嵌套刻度线的轴
    axis (str): 要添加嵌套刻度线的轴,可以是'x'或'y'
    ticks (list): 要添加的刻度线的位置
    length (float, optional): 刻度线的长度。默认为 None。
    width (float, optional): 刻度线的宽度。默认为LINE_WIDTH。
    """
    if axis == 'x':
        sec = ax.secondary_xaxis(location=0)
        sec.set_xticks(ticks, labels=[])
        sec.tick_params('x', length=length, width=width)
        sec.spines['bottom'].set_visible(False)
    elif axis == 'y':
        sec = ax.secondary_yaxis(location=0)
        sec.set_yticks(ticks, labels=[])
        sec.tick_params('y', length=length, width=width)
        sec.spines['left'].set_visible(False)

@iterate_over_axs
def rm_ax_tick(ax, axis=None, which='both'):
    '''
    移除轴的刻度。

    参数:
    - ax: matplotlib的Axes对象
    - axis: 要移除的轴
    - which: 要移除的刻度, 可选 'both', 'major', 'minor'

    注意:
    - ax.set_xticks([])和ax.set_yticks([])可以移除刻度, 但是这样会在sharex和sharey的情况下移除其他轴的刻度
    '''
    if axis is None or axis == 'both':
        axis = ['x', 'y']

    for tick in axis:
        if tick == 'x':
            ax.xaxis.set_tick_params(width=0, which=which)
        elif tick == 'y':
            ax.yaxis.set_tick_params(width=0, which=which)

@iterate_over_axs
def rm_ax_ticklabel(ax, axis=None, which='both'):
    '''
    移除轴的刻度。

    参数:
    - ax: matplotlib的Axes对象
    - axis: 要移除的轴
    - which: 要移除的刻度, 可选 'both', 'major', 'minor'

    注意:
    - ax.set_xticklabels([])和ax.set_yticklabels([])会对所有sharex和sharey的轴生效, 所以不推荐使用
    '''
    if axis is None or axis == 'both':
        axis = ['x', 'y']

    for tick in axis:
        if tick == 'x':
            ax.tick_params(axis='x', which=which, labelbottom=False)
        elif tick == 'y':
            ax.tick_params(axis='y', which=which, labelleft=False)

@iterate_over_axs
def set_ax_tick(ax, ticks, labels, axis, which='major'):
    '''
    设置轴的刻度。
    
    参数:
    - ax: matplotlib的Axes对象
    - ticks: 刻度的位置
    - labels: 刻度的标签
    - axis: 要设置刻度的轴, 可选 'x', 'y', 'z'

    注意:
    - 一般来说,major的情形就够用了,需要设置minor的情形可能是log刻度下想要减少刻度的数量
    - 也可以直接使用set_ax函数来设置刻度,它更加自动化,还可以调整ticklabel的fontsize等,但这个函数更加轻量级
    '''
    if which == 'major':
        minor = False
    elif which == 'minor':
        minor = True
    if axis == 'x':
        ax.set_xticks(ticks, labels=labels, minor=minor)
    elif axis == 'y':
        ax.set_yticks(ticks, labels=labels, minor=minor)
    elif axis == 'z':
        ax.set_zticks(ticks, labels=labels, minor=minor)


def set_cbar_tick(cbar, norm_mode='linear', ticks=None):
    pass
# endregion


# region 通用函数(broken axis, share axis, move axis, rm_ax_axis, set_ax_aspect)
def broken_axis(ax1, ax2, orientation, link_location='auto', share=True, offset=0.05, slope=1., color=BLACK, linewidth=LINE_WIDTH, **line_kwargs):
    '''
    连接两个ax, 并绘制出中断的轴的效果, 可搭配split_ax使用, 注意设置好xlim和ylim等

    参数:
    ax1, ax2: 两个需要连接的ax(按照坐标,从小到大,纵向的时候底部的是ax1,顶部的是ax2,横向的时候左边的是ax1,右边的是ax2)
    orientation: str, 连接的方向, 可选 'vertical', 'horizontal'
    link_location: str, 连接线的位置, 可选 'auto', 'top', 'bottom', 'left', 'right', 'both'
    '''
    # 获取轴的位置信息
    ax1_pos = ax1.get_position()
    ax2_pos = ax2.get_position()
    
    # 根据方向进行处理
    if orientation == 'vertical':
        # 纵向连接
        if ax1_pos.y0 > ax2_pos.y0:
            raise ValueError('ax1应该在ax2的底部,请确保ax的输入顺序正确')
        
        if share:
            share_axis(axs=[ax1, ax2], sharex=True, sharey=False)

        # 隐藏上方轴的底部和下方轴的顶部
        ax2.spines.bottom.set_visible(False)
        ax1.spines.top.set_visible(False)

        # 将上方轴的x轴取消
        ax2.xaxis.set_visible(False)
        
        # 根据spine来得知link_location auto的位置
        link_location_list = []
        if link_location == 'auto':
            if ax1.spines.left.get_visible():
                link_location_list.append('left')
            if ax1.spines.right.get_visible():
                link_location_list.append('right')
        if link_location in ['left', 'right']:
            link_location_list.append(link_location)
        if link_location == 'both':
            link_location_list = ['left', 'right']

        # 在上下轴的边界处绘制连接线
        for lk_loc in link_location_list:
            if lk_loc == 'left':
                ax2.plot([-offset, offset], [-offset*slope, offset*slope], transform=ax2.transAxes, clip_on=False, color=color, linewidth=linewidth, **line_kwargs)
                ax1.plot([-offset, offset], [1-offset*slope, 1+offset*slope], transform=ax1.transAxes, clip_on=False, color=color, linewidth=linewidth, **line_kwargs)
            elif lk_loc == 'right':
                ax2.plot([1-offset, 1+offset], [-offset*slope, offset*slope], transform=ax2.transAxes, clip_on=False, color=color, linewidth=linewidth, **line_kwargs)
                ax1.plot([1-offset, 1+offset], [1-offset*slope, 1+offset*slope], transform=ax1.transAxes, clip_on=False, color=color, linewidth=linewidth, **line_kwargs)
        
    elif orientation == 'horizontal':
        # 横向连接
        if ax1_pos.x0 > ax2_pos.x0:
            raise ValueError('ax1应该在ax2的左边,请确保ax的输入顺序正确')
        
        if share:
            share_axis(axs=[ax1, ax2], sharex=False, sharey=True)

        # 隐藏右侧轴的左边和左侧轴的右边
        ax2.spines.left.set_visible(False)
        ax1.spines.right.set_visible(False)
        
        # 将右侧轴的y轴取消
        ax2.yaxis.set_visible(False)
        
        # 根据spine来得知link_location auto的位置
        link_location_list = []
        if link_location == 'auto':
            if ax1.spines.top.get_visible():
                link_location_list.append('top')
            if ax1.spines.bottom.get_visible():
                link_location_list.append('bottom')
        if link_location in ['top', 'bottom']:
            link_location_list.append(link_location)
        if link_location == 'both':
            link_location_list = ['top', 'bottom']

        # 在左右轴的边界处绘制连接线
        for lk_loc in link_location_list:
            if lk_loc == 'top':
                ax2.plot([-offset, offset], [1-offset*slope, 1+offset*slope], transform=ax2.transAxes, clip_on=False, color=color, linewidth=linewidth, **line_kwargs)
                ax1.plot([1-offset, 1+offset], [1-offset*slope, 1+offset*slope], transform=ax1.transAxes, clip_on=False, color=color, linewidth=linewidth, **line_kwargs)
            elif lk_loc == 'bottom':
                ax2.plot([-offset, offset], [-offset*slope, offset*slope], transform=ax2.transAxes, clip_on=False, color=color, linewidth=linewidth, **line_kwargs)
                ax1.plot([1-offset, 1+offset], [-offset*slope, offset*slope], transform=ax1.transAxes, clip_on=False, color=color, linewidth=linewidth, **line_kwargs)


def broken_axis_multi(axs, orientation, share=True, **kwargs):
    '''
    可以连接多个ax, 并绘制出中断的轴的效果, 可搭配split_ax使用, 注意设置好xlim和ylim等

    参数:
    - axe: 多个ax的列表或者array,必须按照ax的位置排列好后输入(这对于array的ax来说其实比较反人类,因为对于纵向的情形,函数的要求和一般的排列方式是相反的)
    '''
    iterable_axs = get_iterable_ax(axs)
    if share:
        if orientation == 'vertical':
            share_axis(iterable_axs[::-1], sharex=True, sharey=False)
        elif orientation == 'horizontal':
            share_axis(iterable_axs, sharex=False, sharey=True)
    for i in range(len(iterable_axs)):
        if i == 0:
            continue
        broken_axis(iterable_axs[i-1], iterable_axs[i], orientation, share=False, **kwargs)


def split_broken_axis(ax, orientation, num, share=True, split_ax_kwargs=None, broken_ax_kwargs=None):
    '''
    一步完成split和broke_axis,最大限度避免share和重复share可能产生的问题;以及broken_axis需要的顺序容易错误的问题
    '''
    sharex = False # 不管怎么样,share的过程都在broken_axis中完成,所以split_ax中不需要share
    sharey = False
    split_ax_kwargs = update_dict({'sharex': sharex, 'sharey': sharey}, split_ax_kwargs)

    broken_ax_kwargs = update_dict({}, broken_ax_kwargs)

    if orientation == 'vertical':
        nrows, ncols = num, 1
    elif orientation == 'horizontal':
        nrows, ncols = 1, num

    sub_ax = split_ax_by_gs(ax, nrows, ncols, **split_ax_kwargs)

    if orientation == 'vertical':
        broken_axis_multi(sub_ax[::-1], orientation, share=share, **broken_ax_kwargs)
    elif orientation == 'horizontal':
        broken_axis_multi(sub_ax, orientation, share=share, **broken_ax_kwargs)
    return sub_ax


def share_axis(axs, sharex=True, sharey=True):
    '''
    share_axis新版,不一定需要接收输出,可以直接在原地修改

    注意:
    如果ax1.sharex(ax2)和ax1.sharex(ax3),如果直接这么写则会报错,但是ax2.sharex(ax1)和ax3.sharex(ax1)是可以的;所以本函数有可能报错,如果遇到已经share的情况,这时候需要用户在创建时不要share,而是在后续调用本函数时share,或者调整本函数的使用顺序(如果比较难,可以使用share_axis_to_target)
    '''
    if not isinstance(sharex, str):
        sharex = "all" if sharex else "none"
    if not isinstance(sharey, str):
        sharey = "all" if sharey else "none"

    # 2d numpy array的情形,可以按行或者按列共享
    if isinstance(axs, np.ndarray):
        if axs.ndim == 2:
            nrows, ncols = axs.shape
            for i in range(nrows):
                for j in range(ncols):
                    shared_with = {"none": None, "all": axs[0, 0],
                                    "row": axs[i, 0], "col": axs[0, j]}
                    if shared_with[sharex] is not None:
                        # add_subplot是可以输入None的,但是sharex和sharey不行
                        axs[i, j].sharex(shared_with[sharex])
                    if shared_with[sharey] is not None:
                        axs[i, j].sharey(shared_with[sharey])
            return axs

    # 非2d numpy array的情形,只能按照all或者none共享
    if sharex in ['row', 'col'] or sharey in ['row', 'col']:
        print("Warning: sharex and sharey can only be 'all' or 'none' when axs is not a 2D numpy array.")
    iterable_axs = get_iterable_ax(axs)
    for i, ax in enumerate(iterable_axs):
        if i == 0:
            pass
        else:
            if sharex == 'all':
                ax.sharex(iterable_axs[0])
            if sharey == 'all':
                ax.sharey(iterable_axs[0])
    # 其实不需要返回,因为是原地修改;但是为了兼容,还是返回一下;如果是单个的ax输入,get_iterable_ax会把他变成一个list,所以这里需要返回第一个
    if not isinstance(axs, plt.Axes):
        return rebuild_ax(iterable_axs, axs)
    else:
        return iterable_axs[0]


def share_axis_to_target(axs, target_ax, sharex=True, sharey=True):
    '''
    share_axis_to_target新版,不一定需要接收输出,可以直接在原地修改

    注意:
    如果ax1.sharex(ax2)和ax1.sharex(ax3),如果直接这么写则会报错,但是ax2.sharex(ax1)和ax3.sharex(ax1)是可以的;所以本函数有可能报错,如果遇到已经share的情况,这时候需要用户在创建时不要share,而是在后续调用本函数时share,或者调整本函数的使用顺序
    '''
    if sharex == 'none':
        sharex = False
    if sharey == 'none':
        sharey = False
    iterable_axs = get_iterable_ax(axs)
    for ax in iterable_axs:
        if sharex:
            ax.sharex(target_ax)
        if sharey:
            ax.sharey(target_ax)
    # 其实不需要返回,因为是原地修改;但是为了兼容,还是返回一下;如果是单个的ax输入,get_iterable_ax会把他变成一个list,所以这里需要返回第一个
    if not isinstance(axs, plt.Axes):
        return rebuild_ax(iterable_axs, axs)
    else:
        return iterable_axs[0]

@iterate_over_axs
def move_axis(ax, axis, position):
    '''
    将ax的某个轴移动到指定位置。(与move ax区分,move ax是移动整个ax的位置)
    '''
    if axis == 'y':
        if position == 'right':
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position('right')
        elif position == 'left':
            ax.yaxis.tick_left()
            ax.yaxis.set_label_position('left')
    elif axis == 'x':
        if position == 'top':
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position('top')
        elif position == 'bottom':
            ax.xaxis.tick_bottom()
            ax.xaxis.set_label_position('bottom')

@iterate_over_axs
def rm_ax_axis(ax):
    '''
    移除轴的坐标轴。(除了title,其他的都会被移除,比如外框,刻度,刻度标签等)

    参数:
    - ax: matplotlib的Axes对象
    '''
    ax.axis('off')

@iterate_over_axs
def rm_ax_specific_axis(ax, axis=None):
    '''
    移除轴的特定坐标轴。

    参数:
    - ax: matplotlib的Axes对象
    - axis: 要移除的轴, 可选 'x', 'y', 'both'
    '''
    if axis is None or axis == 'both':
        axis = ['x', 'y']

    for ax_type in axis:
        if ax_type == 'x':
            ax.spines['bottom'].set_visible(False)
            ax.spines['top'].set_visible(False)
        elif ax_type == 'y':
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
    
    rm_ax_tick(ax, axis)
    rm_ax_ticklabel(ax, axis)

@iterate_over_axs
def set_ax_aspect(ax, aspect=1, adjustable='datalim', **kwargs):
    '''
    设置轴的纵横比。

    参数:
    - ax: matplotlib的Axes对象
    - aspect: 纵横比
    - adjustable: 调整方式,默认为'datalim',还可以选值为'box'
    - kwargs: 传递给set_box_aspect的额外关键字参数
    '''
    ax.set_aspect(aspect, adjustable=adjustable, **kwargs)

@iterate_over_axs
def set_ax_aspect_3d(ax, aspect=(1, 1, 1), adjustable='datalim', **kwargs):
    '''
    设置3D轴的x、y和z方向的比例

    参数:
    - ax: matplotlib的Axes对象
    - aspect: 比例
    - adjustable: 调整方式,默认为'datalim',还可以选值为'box'
    - kwargs: 传递给set_box_aspect的额外关键字参数
    '''
    if adjustable == 'datalim':
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        zlim = ax.get_zlim()

        xmean = np.mean(xlim)
        ymean = np.mean(ylim)
        zmean = np.mean(zlim)

        plot_radius = max([abs(lim - mean) for lims, mean in ((xlim, xmean), (ylim, ymean), (zlim, zmean)) for lim in lims])

        local_aspect = np.array(aspect) / np.max(aspect)

        ax.set_xlim3d([xmean - plot_radius/local_aspect[0], xmean + plot_radius/local_aspect[0]])
        ax.set_ylim3d([ymean - plot_radius/local_aspect[1], ymean + plot_radius/local_aspect[1]])
        ax.set_zlim3d([zmean - plot_radius/local_aspect[2], zmean + plot_radius/local_aspect[2]])
    elif adjustable == 'box':
        ax.set_box_aspect(aspect, **kwargs)
# endregion


# region 通用函数(坐标轴scale)
@iterate_over_axs
def set_symlog_scale(ax, axis, linthresh, linscale=1, **kwargs):
    '''
    设置对数坐标轴的对称对数刻度。

    参数:
    - ax (matplotlib.axes.Axes): matplotlib的Axes对象
    - axis (str): 要设置的坐标轴,可以是'x', 'y', 'z'
    - linthresh (float): 线性区域的阈值(如果没有这个,在零处就有无穷个刻度,所以有设置的意义)
    - linscale (float, optional): 线性区域的缩放因子(默认为1,意为线性区域从-linthresh到linthresh的视觉上等于对数区域的一格)
    - **kwargs: 传递给set_xscale/set_yscale/set_zscale的其他参数
    '''
    if axis == 'x':
        ax.set_xscale('symlog', linthresh=linthresh, linscale=linscale, **kwargs)
    elif axis == 'y':
        ax.set_yscale('symlog', linthresh=linthresh, linscale=linscale, **kwargs)
    elif axis == 'z':
        ax.set_zscale('symlog', linthresh=linthresh, linscale=linscale, **kwargs)
# endregion


# region 通用函数(title, label, tick调整, 对齐)
def align_label(axs, axis, fig=None):
    '''
    将多个轴的标签对齐。

    参数:
    - axs: matplotlib的Axes对象或对象列表,array([ax1, ax2, ...])
    - axis: 对齐的轴,可以是'x', 'y'
    - fig: matplotlib的Figure对象

    注意:
    - 假如某个ax的轴在左侧,而其他的在右侧,那么无法对齐。
    - 如果出现无法对齐的情况,可以尝试align_label_manual。
    '''
    local_axs = get_iterable_ax(axs)
    if fig is None:
        fig = local_axs[0].get_figure()
    if axis == 'x':
        fig.align_xlabels(local_axs)
    elif axis == 'y':
        fig.align_ylabels(local_axs)


def align_label_manual(axs, axis, label_coord):
    '''
    将多个轴的标签对齐。

    参数:
    - axs: matplotlib的Axes对象或对象列表,array([ax1, ax2, ...])
    - axis: 对齐的轴,可以是'x', 'y'
    - label_coord: 标签的坐标,如果是0.则紧贴轴,如果希望ylabel放在轴靠左的位置,可以设置为-0.1,其余同理
    '''
    local_axs = get_iterable_ax(axs)
    if axis == 'x':
        for ax in local_axs:
            ax.xaxis.set_label_coords(0.5, label_coord)
    elif axis == 'y':
        for ax in local_axs:
            ax.yaxis.set_label_coords(label_coord, 0.5)


def get_label_obj(ax, axis):
    if axis == 'x':
        return ax.xaxis.get_label()
    elif axis == 'y':
        return ax.yaxis.get_label()
    elif axis == 'z':
        return ax.zaxis.get_label()


def get_label_pad(ax, axis):
    if axis == 'x':
        return ax.xaxis.labelpad
    elif axis == 'y':
        return ax.yaxis.labelpad
    elif axis == 'z':
        return ax.zaxis.labelpad


def get_label_text(ax, axis):
    return get_label_obj(ax, axis).get_text()


def adjust_label(ax, axis, **kwargs):
    if axis == 'x':
        ax.set_xlabel(get_label_text(ax, axis), **kwargs)
    elif axis == 'y':
        ax.set_ylabel(get_label_text(ax, axis), **kwargs)
    elif axis == 'z':
        ax.set_zlabel(get_label_text(ax, axis), **kwargs)

@iterate_over_axs
def rm_ax_xlabel(ax):
    ax.set_xlabel('')

@iterate_over_axs
def rm_ax_ylabel(ax):
    ax.set_ylabel('')


def get_title_obj(ax):
    return ax.title


def get_title_pad(ax):
    print("gpt don't know how to get title pad")
    return None


def get_title_text(ax):
    return ax.title.get_text()


def adjust_title(ax, **kwargs):
    ax.set_title(get_title_text(ax), **kwargs)


def label_title_process(x_label, y_label, z_label, title, text_process=None):
    '''
    处理标签和标题
    '''
    text_process = update_dict(TEXT_PROCESS, text_process)
    return format_text(x_label, text_process), format_text(y_label, text_process), format_text(z_label, text_process), format_text(title, text_process)


def set_fig_title(fig, title, text_process=None, title_size=SUP_TITLE_SIZE, **kwargs):
    '''
    设置图形的标题

    参数:
    - fig: matplotlib的Figure对象
    - title: 图形的标题
    - text_process: 文本处理参数
    - title_size: 标题字体大小
    - kwargs: 传递给fig.suptitle的其他参数,比如y值,可以控制标题的位置
    '''
    text_process = update_dict(TEXT_PROCESS, text_process)

    # 处理标题
    local_title = format_text(title, text_process)

    # 设置标题
    fig.suptitle(local_title, fontsize=title_size, **kwargs)


def add_fig_title(*args, **kwargs):
    return set_fig_title(*args, **kwargs)


def add_axes_title(axes, title, text_process=None, title_size=TITLE_SIZE, pad=50., **kwargs):
    '''
    在多个ax的中间添加标题
    '''
    text_process = update_dict(TEXT_PROCESS, text_process)
    title = format_text(title, text_process)

    merged_ax = merge_ax(axes, rm_mode=False)
    merged_ax.set_title(title, pad=pad, fontsize=title_size, **kwargs)

    rm_ax_axis(merged_ax)


def get_tick_size(ax):
    x_tick_size = ax.xaxis.get_ticklabels()[0].get_fontsize()
    y_tick_size = ax.yaxis.get_ticklabels()[0].get_fontsize()
    return x_tick_size, y_tick_size


def suitable_tick_size(num_ticks, plt_size, tick_size=TICK_SIZE, proportion=TICK_PROPORTION):
    '''
    Adjusts the font size of the tick labels based on the number of ticks and the size of the axis.
    '''
    if num_ticks == 0:
        return tick_size
    suitable_tick_size = inch_to_point(plt_size) / num_ticks * proportion
    return min(suitable_tick_size, tick_size)

@iterate_over_axs
def adjust_ax_tick(ax, xtick_rotation=XTICK_ROTATION, ytick_rotation=YTICK_ROTATION, proportion=TICK_PROPORTION):
    '''
    x轴和y轴的刻度标签字体大小根据刻度数量和轴的大小进行调整。(要写在set_ax之后,否则会被覆盖)
    自动旋转x轴刻度标签。
    '''
    ax_width, ax_height = get_ax_size(ax)
    # 获取x轴和y轴的刻度数量
    x_num_ticks = len(ax.get_xticks())
    y_num_ticks = len(ax.get_yticks())
    
    x_tick_size, y_tick_size = get_tick_size(ax)
    if x_num_ticks > 0:
        ax.tick_params(axis='x', labelsize=suitable_tick_size(
            x_num_ticks, ax_width, x_tick_size, proportion))
    if y_num_ticks > 0:
        ax.tick_params(axis='y', labelsize=suitable_tick_size(
            y_num_ticks, ax_height, y_tick_size, proportion))

    # 旋转x轴刻度标签
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=xtick_rotation)

    # 旋转y轴刻度标签
    ax.set_yticks(ax.get_yticks())
    ax.set_yticklabels(ax.get_yticklabels(), rotation=ytick_rotation)


def get_tick_rotation(ax, axis):
    '''
    获取轴的刻度标签的旋转角度
    '''
    try:
        # 尝试获取第一个刻度标签的旋转角度
        if axis == 'x':
            return ax.xaxis.get_ticklabels()[0].get_rotation()
        elif axis == 'y':
            return ax.yaxis.get_ticklabels()[0].get_rotation()
    except IndexError:
        # 如果没有刻度标签,返回默认角度
        if axis == 'x':
            return XTICK_ROTATION
        elif axis == 'y':
            return YTICK_ROTATION
# endregion


# region 通用函数(legend)
@iterate_over_axs
def rm_ax_legend(ax):
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()

@iterate_over_axs
def set_ax_legend(ax, loc=LEGEND_LOC, fontsize=LEGEND_SIZE, bbox_to_anchor=None, text_process=None, rm_exist_legend=True, ncols=1, facecolor='inherit', edgecolor='0.8', labelcolor=None, **kwargs):
    '''
    facecolor: 图例框的颜色,默认为'inherit';如果想要不设置,可以设置为'none'或者'None'(注意这个是字符串)
    edgecolor: 图例框的边框颜色,默认为'0.8';如果想要不设置,可以设置为'none'或者'None'(注意这个是字符串)
    labelcolor: label中文字的颜色,如果需要和line,scatter一致,可以设置为'linecolor'
    '''
    if rm_exist_legend:
        # 去掉已有的图例
        rm_ax_legend(ax)
    # 检查是否有图例标签
    handles, labels = ax.get_legend_handles_labels()
    # 去掉重复
    handles, labels = dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys()
    if labels:
        labels = [format_text(label, text_process) for label in labels]
        ax.legend(handles, labels, loc=loc,
                fontsize=fontsize, bbox_to_anchor=bbox_to_anchor, ncols=ncols, facecolor=facecolor, edgecolor=edgecolor, labelcolor=labelcolor, **kwargs)
# endregion


# region 通用函数(一键调整ax)
@iterate_over_axs
def set_ax(ax, xlabel=None, ylabel=None, zlabel=None, xlabel_pad=LABEL_PAD, ylabel_pad=LABEL_PAD, zlabel_pad=LABEL_PAD, title=None, title_pad=TITLE_PAD, text_process=None, title_size=TITLE_SIZE, label_size=LABEL_SIZE, tick_size=TICK_SIZE, xtick=None, ytick=None, ztick=None, xtick_label=None, ytick_label=None, ztick_label=None, xtick_size=None, ytick_size=None, ztick_size=None, xtick_rotation=None, ytick_rotation=None, adjust_tick_size=True, tick_proportion=TICK_PROPORTION, legend=True, legend_size=LEGEND_SIZE, xlim=None, ylim=None, zlim=None, xlog=False, ylog=False, zlog=False, elev=None, azim=None, legend_loc=LEGEND_LOC, bbox_to_anchor=None, rm_exist_legend=True, legend_kwargs=None, tight_layout=False, reset_scale=False):
    '''
    设置图表的轴、标题、范围和图例

    参数:
    ax - 绘图的Axes对象
    xlabel, ylabel, zlabel - x轴、y轴和z轴的标签
    xlabel_pad, ylabel_pad, zlabel_pad - x轴、y轴和z轴标签的间距
    title - 图标题
    title_pad - 标题的间距
    text_process - 文本处理参数
    title_size - 标题字体大小
    label_size - 标签字体大小
    tick_size - 刻度标签字体大小
    xtick_size, ytick_size - x轴和y轴刻度标签字体大小
    legend_size - 图例字体大小
    xlim, ylim - 坐标轴范围
    legend_loc - 图例位置
    bbox_to_anchor - 图例的位置参数(示例:(1, 1), 配合legend_loc='upper left'表示图例框的upper left角放在坐标(1, 1)处)
    tight_layout - 是否启用紧凑布局,默认为False,如果输入dict,则作为tight_layout的参数
    reset_scale - 是否在不设定xlog等的时候重新set为linear(如果True则会进行一步linear,但是这有可能破坏一些xticklabel之类的设置)
    '''
    text_process = update_dict(TEXT_PROCESS, text_process)
    legend_kwargs = update_dict({}, legend_kwargs)
    if xtick_rotation is None:
        xtick_rotation = get_tick_rotation(ax, 'x')
    if ytick_rotation is None:
        ytick_rotation = get_tick_rotation(ax, 'y')
    is_3d = isinstance(ax, Axes3D)

    # 尝试获取x_label和y_label
    xlabel = ax.get_xlabel() if xlabel is None else xlabel
    ylabel = ax.get_ylabel() if ylabel is None else ylabel
    if is_3d:
        zlabel = ax.get_zlabel() if zlabel is None else zlabel

    # 处理标签和标题
    local_xlabel, local_ylabel, local_zlabel, local_title = label_title_process(
        xlabel, ylabel, zlabel, title, text_process)

    # 设置标签和标题
    ax.set_xlabel(local_xlabel, fontsize=label_size, labelpad=xlabel_pad)
    ax.set_ylabel(local_ylabel, fontsize=label_size, labelpad=ylabel_pad)
    if is_3d:
        ax.set_zlabel(local_zlabel, fontsize=label_size, labelpad=zlabel_pad)
    ax.set_title(local_title, fontsize=title_size, pad=title_pad)

    # 设置tick
    if xtick is not None:
        set_ax_tick(ax, ticks=xtick, labels=xtick_label, axis='x')
    if ytick is not None:
        set_ax_tick(ax, ticks=ytick, labels=ytick_label, axis='y')
    if is_3d and ztick is not None:
        set_ax_tick(ax, ticks=ztick, labels=ztick_label, axis='z')

    # 设置字体
    if xtick_size is None:
        xtick_size = tick_size
    if ytick_size is None:
        ytick_size = tick_size
    
    if (not is_3d) and adjust_tick_size:
        width, height = get_ax_size(ax)
        xtick_size = suitable_tick_size(len(ax.get_xticks()), width, tick_size, tick_proportion)
        ytick_size = suitable_tick_size(len(ax.get_yticks()), height, tick_size, tick_proportion)

    ax.tick_params(axis='x', labelsize=xtick_size, rotation=xtick_rotation)
    ax.tick_params(axis='x', which='minor', rotation=xtick_rotation)
    ax.tick_params(axis='y', labelsize=ytick_size, rotation=ytick_rotation)
    ax.tick_params(axis='y', which='minor', rotation=ytick_rotation)

    if is_3d:
        if ztick_size is None:
            ztick_size = tick_size
        ax.tick_params(axis='z', labelsize=ztick_size)

    # 设置对数坐标轴
    if xlog:
        ax.set_xscale('log')
    elif reset_scale:
        ax.set_xscale('linear')
    if ylog:
        ax.set_yscale('log')
    elif reset_scale:
        ax.set_yscale('linear')
    if is_3d:
        if zlog:
            ax.set_zscale('log')
        elif reset_scale:
            ax.set_zscale('linear')

    # 设置坐标轴范围
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if is_3d and zlim is not None:
        ax.set_zlim(zlim)

    if legend:
        set_ax_legend(ax, loc=legend_loc, fontsize=legend_size, bbox_to_anchor=bbox_to_anchor, text_process=text_process, rm_exist_legend=rm_exist_legend, **legend_kwargs)

    if tight_layout:
        plt.tight_layout(**tight_layout)

    if is_3d:
        set_ax_view_3d(ax, elev=elev, azim=azim)
# endregion


# region 通用函数(ax的label管理,此处的label不是xlabel, ylabel, zlabel, 而是ax的标签)
def set_ax_label(ax, label=None):
    '''
    添加轴的标签(方便后续直接从fig获取时,可以知道这个ax是干什么的)

    如果label是None,则设置默认标签;否则,label需要和ax的数据类型相同,且形状相同
    '''
    if isinstance(ax, dict):
        # 如果 ax 是字典
        if label is None:
            # 如果 label 为 None, 则设置默认标签
            for k, v in ax.items():
                v.set_label(k)
        elif isinstance(label, dict):
            # 如果 label 是字典
            for k in ax.keys():
                if k in label:  # 确保键在 label 中
                    ax[k].set_label(label[k])

    elif isinstance(ax, np.ndarray):
        # 如果 ax 是二维数组
        if ax.ndim == 2:
            rows, cols = ax.shape
            if label is None:
                # 如果 label 为 None, 则设置默认标签
                for i in range(rows):
                    for j in range(cols):
                        ax[i, j].set_label(f"row_{i}_col_{j}")
            elif isinstance(label, np.ndarray) and label.shape == ax.shape:
                # 如果 label 是二维数组且形状与 ax 相同
                for i in range(rows):
                    for j in range(cols):
                        ax[i, j].set_label(label[i, j])

        # 如果 ax 是一维数组
        elif ax.ndim == 1:
            length = ax.shape[0]
            if label is None:
                # 如果 label 为 None, 则设置默认标签
                for i in range(length):
                    ax[i].set_label(f"index_{i}")
            elif isinstance(label, list) and len(label) == length:
                # 如果 label 是列表且长度与 ax 相同
                for i in range(length):
                    ax[i].set_label(label[i])
    
    elif isinstance(ax, plt.Axes):
        # 如果 ax 是单个 Axes 对象
        ax.set_label(label)


def find_ax_by_label(fig, label):
    '''
    利用label找到对应的ax
    '''
    for ax in fig.axes:
        if ax.get_label() == label:
            return ax
    return None
# endregion


# region 通用函数(获取合适的margin和adjust_params)
def get_margin_basic(margin=None, left_space=False, bottom_space=False, right_space=False, top_space=False, left_ratio=1., bottom_ratio=1., right_ratio=1., top_ratio=1.):
    '''
    获取合适的margin

    参数:
    - left_space: 是否加大留给侧边的间距
    - bottom_space: 是否加大留给底部的间距
    - right_space: 是否加大留给右侧的间距
    - top_space: 是否加大留给顶部的间距
    '''
    margin = update_dict(MARGIN, margin)
    if left_space:
        margin['left'] += 0.05 * left_ratio
    if bottom_space:
        margin['bottom'] += 0.05 * bottom_ratio
    if right_space:
        margin['right'] -= 0.05 * right_ratio
    if top_space:
        margin['top'] -= 0.05 * top_ratio
    return margin


def get_margin_custom(margin=None, left_space=False, bottom_space=False, right_space=False, top_space=False, left_ratio=1., bottom_ratio=1., right_ratio=1., top_ratio=1.):
    margin = update_dict(MARGIN, margin)
    return get_margin_basic(margin, left_space, bottom_space, right_space, top_space, left_ratio, bottom_ratio, right_ratio, top_ratio)


def get_margin_custom_3d(margin=None, left_space=False, bottom_space=False, right_space=False, top_space=False, left_ratio=1., bottom_ratio=1., right_ratio=1., top_ratio=1.):
    margin = update_dict(MARGIN_3D, margin)
    return get_margin_basic(margin, left_space, bottom_space, right_space, top_space, left_ratio, bottom_ratio, right_ratio, top_ratio)


def get_suitable_adjust_params_basic(adjust_params_custom, left_space=False, bottom_space=False, right_space=False, top_space=False, w_space=False, h_space=False, left_ratio=1., bottom_ratio=1., right_ratio=1., top_ratio=1., w_ratio=1., h_ratio=1.):
    '''
    获取合适的adjust_params_custom

    参数:
    - left_space: 是否加大留给侧边的间距
    - bottom_space: 是否加大留给底部的间距
    - right_space: 是否加大留给右侧的间距
    - top_space: 是否加大留给顶部的间距
    - w_space: 是否加大水平间距
    - h_space: 是否加大垂直间距
    '''
    adjust_params_custom = update_dict(ADJUST_PARAMS_CUSTOM, adjust_params_custom)
    if left_space:
        adjust_params_custom['left'] += 0.05 * left_ratio
    if bottom_space:
        adjust_params_custom['bottom'] += 0.05 * bottom_ratio
    if right_space:
        adjust_params_custom['right'] -= 0.05 * right_ratio
    if top_space:
        adjust_params_custom['top'] -= 0.05 * top_ratio
    if w_space:
        adjust_params_custom['wspace'] += 0.1 * w_ratio
    if h_space:
        adjust_params_custom['hspace'] += 0.1 * h_ratio
    return adjust_params_custom


def get_suitable_adjust_params_custom(adjust_params_custom=None, left_space=False, bottom_space=False, right_space=False, top_space=False, w_space=False, h_space=False, left_ratio=1., bottom_ratio=1., right_ratio=1., top_ratio=1., w_ratio=1., h_ratio=1.):
    adjust_params_custom = update_dict(ADJUST_PARAMS_CUSTOM, adjust_params_custom)
    return get_suitable_adjust_params_basic(adjust_params_custom, left_space, bottom_space, right_space, top_space, w_space, h_space, left_ratio, bottom_ratio, right_ratio, top_ratio, w_ratio, h_ratio)


def get_suitable_adjust_params_custom_3d(adjust_params_custom=None, left_space=False, bottom_space=False, right_space=False, top_space=False, w_space=False, h_space=False, left_ratio=1., bottom_ratio=1., right_ratio=1., top_ratio=1., w_ratio=1., h_ratio=1.):
    adjust_params_custom = update_dict(ADJUST_PARAMS_CUSTOM_3D, adjust_params_custom)
    return get_suitable_adjust_params_basic(adjust_params_custom, left_space, bottom_space, right_space, top_space, w_space, h_space, left_ratio, bottom_ratio, right_ratio, top_ratio, w_ratio, h_ratio)
# endregion


# region 通用函数(获取,调整figsize)
def get_adjust_params_from_custom(nrows=1, ncols=1, adjust_params_custom=None):
    '''
        adjust_params_custom 含义: 此处left, right, top, bottom的值相对于ax, 而不是fig
    '''
    adjust_params = adjust_params_custom.copy()
    adjust_params['left'] = adjust_params_custom['left'] / (ncols + adjust_params_custom['wspace'] * (ncols - 1) + 1 - adjust_params_custom['right'] + adjust_params_custom['left'])
    adjust_params['right'] = 1 - (1 - adjust_params_custom['right']) / (ncols + adjust_params_custom['wspace'] * (ncols - 1) + 1 - adjust_params_custom['right'] + adjust_params_custom['left'])
    adjust_params['bottom'] = adjust_params_custom['bottom'] / (nrows + adjust_params_custom['hspace'] * (nrows - 1) + 1 - adjust_params_custom['top'] + adjust_params_custom['bottom'])
    adjust_params['top'] = 1 - (1 - adjust_params_custom['top']) / (nrows + adjust_params_custom['hspace'] * (nrows - 1) + 1 - adjust_params_custom['top'] + adjust_params_custom['bottom'])
    return adjust_params


def get_suitable_fig_size(nrows=1, ncols=1, ax_width=AX_WIDTH, ax_height=AX_HEIGHT, margin=None, adjust_params=None, adjust_params_custom=None, which='auto'):
    '''
        adjust_params_custom 优先级高于 adjust_params 优先级高于 margin, 优先级高的参数会覆盖优先级低的参数(当which为auto时)

        adjust_params_custom 含义: 此处left, right, top, bottom的值相对于ax, 而不是fig
    '''
    margin = update_dict(MARGIN, margin)
    adjust_params_custom = update_dict(ADJUST_PARAMS_CUSTOM, adjust_params_custom)
    if adjust_params_custom and which in ['adjust_params_custom', 'auto']:
        fig_width = ax_width * (ncols + adjust_params_custom['wspace'] * (ncols - 1)) + ax_width * (1 - adjust_params_custom['right'] + adjust_params_custom['left'])
        fig_height = ax_height * (nrows + adjust_params_custom['hspace'] * (nrows - 1)) + ax_height * (1 - adjust_params_custom['top'] + adjust_params_custom['bottom'])
    elif adjust_params and which in ['adjust_params', 'auto']:
        fig_width = (ax_width * (ncols + adjust_params['wspace'] * (ncols - 1))) / (adjust_params['right'] - adjust_params['left'])
        fig_height = (ax_height * (nrows + adjust_params['hspace'] * (nrows - 1))) / (adjust_params['top'] - adjust_params['bottom'])
    elif margin and which in ['margin', 'auto']:
        fig_width = ax_width / (margin['right'] - margin['left']) * ncols
        fig_height = ax_height / (margin['top'] - margin['bottom']) * nrows
    return fig_width, fig_height


def set_fig_size(fig, width=None, height=None):
    fig.set_size_inches(width, height)
# endregion


# region 通用函数(判断是fig还是subfig)
def get_fig_type(obj):
    """
    判断给定的 matplotlib 对象是 Figure 还是 SubFigure.

    参数:
        obj: 要判断的对象，可以是 Figure, SubFigure 或其他 matplotlib 对象.

    返回:
        str: 返回 'fig' 如果对象是 Figure;
             返回 'subfig' 如果对象是 SubFigure;
             返回 'unknown' 如果对象不是这两者之一.
    """
    if isinstance(obj, Figure):
        return 'fig'
    elif isinstance(obj, SubFigure):
        return 'subfig'
    else:
        return 'unknown'
# endregion


# region 通用函数(创建fig, ax, gs)
def get_fig(width=AX_WIDTH, height=AX_HEIGHT, dpi=FIG_DPI, **kwargs):
    '''
        创建一个空白图形。
    '''
    return plt.figure(figsize=(width, height), dpi=dpi, **kwargs)

@iterate_over_subfigs
def set_subfig_size_inches(subfig, ncols, nrows):
    '''
    为 subfig 设置一个 get_size_inches 方法, 使得可以通过 subfig.get_size_inches() 获取子图的大小
    '''
    def get_size_inches(self):
        return (
            self.figure.get_size_inches()[0] / ncols,
            self.figure.get_size_inches()[1] / nrows
        )
    add_method_to_instance(subfig, get_size_inches, 'get_size_inches')


def get_subfig(fig=None, nrows=1, ncols=1, squeeze=True, wspace=None, hspace=None, width_ratios=None, height_ratios=None, **kwargs):
    if fig is None:
        fig = plt.gcf()
    subfig = fig.subfigures(nrows=nrows, ncols=ncols, squeeze=squeeze, wspace=wspace, hspace=hspace, width_ratios=width_ratios, height_ratios=height_ratios, **kwargs)

    for s in get_iterable_ax(subfig):
        set_subfig_size_inches(s, ncols, nrows)
    return subfig


def get_fig_subfig(nrows=1, ncols=1, subfig_width=AX_WIDTH, subfig_height=AX_HEIGHT, fig_width=None, fig_height=None, dpi=FIG_DPI, get_fig_kwargs=None, subfig_kwargs=None):
    '''
        创建一个图形和子图对象。

        利用subfig有几个好处: 
        可以创建不均匀的ax,比如说左侧三个,右侧两个ax(先创建两个subfig然后利用get_ax分别创建ax,记得设置adjust_params或者adjust_params_custom而不是margin使得边框对齐)
        可以更好的share_axis(比如说画两列图,左侧的图sharex,右侧的图sharey,这样就可以分别设置sharex和sharey)
        可以更好的set_fig_title

        利用subfig的缺点:
        无法很好的控制ax_width和ax_height,因为fig_width和fig_height由subfig_width和subfig_height决定(需要在get_ax之后,根据subfig的数量和每个subfig的大小(利用get_suitable_fig_size)来手动重制fig的大小(set_fig_size),或者在创建时计算好fig的大小,利用get_suitable_fig_size)
    '''
    get_fig_kwargs = update_dict({}, get_fig_kwargs)
    subfig_kwargs = update_dict({}, subfig_kwargs)
    adjust_params = {'left': 0., 'right': 1., 'top': 1., 'bottom': 0., 'wspace': 0., 'hspace': 0.}
    adjust_params = update_dict(adjust_params, subfig_kwargs)
    local_fig_width, local_fig_height = get_suitable_fig_size(nrows, ncols, subfig_width, subfig_height, adjust_params=adjust_params, which='adjust_params')
    if fig_width is None:
        fig_width = local_fig_width
    if fig_height is None:
        fig_height = local_fig_height

    fig = get_fig(fig_width, fig_height, dpi=dpi, **get_fig_kwargs)
    subfig = get_subfig(nrows=nrows, ncols=ncols, fig=fig, **subfig_kwargs)
    return fig, subfig


def get_ax(fig=None, nrows=1, ncols=1, sharex=False, sharey=False, rm_repeat_tick_label_when_share=RM_REPEAT_TICK_LABEL_WHEN_SHARE, margin=None, squeeze=True, label='ax', subplots_params=None):
    '''
        在一个fig上创建多个ax

        fig: 可以是fig也可以是subfig,如果是None,则使用plt.gcf()获取当前的fig
        nrows, ncols: 行数和列数
        sharex, sharey: 是否共享x轴和y轴(可以是'all', 'none', 'row', 'col', True, False)
        rm_repeat_tick_label_when_share: 是否移除多余的tick label(当sharex或者sharey时)
        margin: 边框空白大小(默认为None,即使用MARGIN)
        squeeze: 是否压缩(nrows, ncols)中的1维
        adjust_params: 字典,包含用于调用subplots_adjust的参数(不推荐使用,因为随着子图的数量的增加,这个参数会变得很难调整,因为这里的left,right,top,bottom是相对于整个fig的位置,而随着ax数量增加,fig的大小会变化)(如果要使用这个参数,推荐使用get_fig_gs,然后从gs获取ax,或者创建好ax之后,使用adjust_ax)
        adjust_params_custom: 字典,含义: 此处left, right, top, bottom的值相对于ax, 而不是fig(推荐使用,不受子图数量的影响,但无法保证不同子图数量情形保证边框对齐)(如果要使用这个参数,推荐使用get_fig_gs,然后从gs获取ax,或者创建好ax之后,使用adjust_ax_custom)

        注意: 在某个fig或者subfig上创建ax,使用ax.figure或者fig.get_figure()都可以得到这个fig或者subfig(而不是最一开始最大的fig)
    '''
    if fig is None:
        fig = plt.gcf()
    margin = update_dict(MARGIN, margin)
    subplots_params = update_dict({}, subplots_params)

    if not isinstance(sharex, str):
        sharex = "all" if sharex else "none"
    if not isinstance(sharey, str):
        sharey = "all" if sharey else "none"

    ax = np.empty((nrows, ncols), dtype=object)
    for i in range(nrows):
        for j in range(ncols):
            shared_with = {"none": None, "all": ax[0, 0],
                            "row": ax[i, 0], "col": ax[0, j]}
            ax[i, j] = fig.add_subplot(nrows, ncols, i * ncols + j + 1, sharex=shared_with[sharex], sharey=shared_with[sharey], **subplots_params)

    if rm_repeat_tick_label_when_share:
        # 移除多余的label
        if sharex in ["col", "all"]:
            for a in ax.flat:
                a._label_outer_xaxis(skip_non_rectangular_axes=True)
        if sharey in ["row", "all"]:
            for a in ax.flat:
                a._label_outer_yaxis(skip_non_rectangular_axes=True)

    # 设置ax的位置
    set_relative_ax_position(ax, nrows, ncols, margin=margin)
    
    if squeeze:
        ax = squeeze_ax(ax)

    # 添加label
    set_ax_label(ax)

    # 略微修改label
    for a in get_iterable_ax(ax):
        set_ax_label(a, cat(label, a.get_label()))
    return ax


def adjust_ax(ax, adjust_params=None):
    '''
    调整ax的位置

    注意:
    对于后续生成的ax_inside_ax,无法使用这个函数调整位置,并且使用这个函数还会影响原先的ax的位置;但是对于subfig内部的ax,可以使用这个函数调整位置(并且会相对于subfig调整位置)
    '''
    adjust_params = update_dict(ADJUST_PARAMS_CUSTOM, adjust_params)
    for a in get_iterable_ax(ax):
        a.figure.subplots_adjust(**adjust_params)
        break
    return ax # 不接收也可以


def adjust_ax_custom(ax, ncols=None, nrows=None, adjust_params_custom=None):
    '''
    调整ax的位置,此处left, right, top, bottom的值相对于ax, 而不是fig

    注意:
    对于后续生成的ax_inside_ax,无法使用这个函数调整位置,并且使用这个函数还会影响原先的ax的位置;但是对于subfig内部的ax,可以使用这个函数调整位置(并且会相对于subfig调整位置)
    '''
    # 从ax中获取ncols和nrows
    if isinstance(ax, plt.Axes):
        ncols, nrows = 1, 1
    elif isinstance(ax, np.ndarray):
        if ax.ndim == 1:
            raise ValueError("The number of rows and columns cannot be inferred from a 1D array of axes, must provide 'ncols' and 'nrows'.")
        nrows, ncols = ax.shape

    # 调整
    adjust_params_custom = update_dict(ADJUST_PARAMS_CUSTOM, adjust_params_custom)
    adjust_params = get_adjust_params_from_custom(ncols=ncols, nrows=nrows, adjust_params_custom=adjust_params_custom)
    adjust_ax(ax, adjust_params)
    return ax # 不接收也可以


def get_fig_ax(nrows=1, ncols=1, ax_width=AX_WIDTH, ax_height=AX_HEIGHT, fig_width=None, fig_height=None, sharex=False, sharey=False, rm_repeat_tick_label_when_share=RM_REPEAT_TICK_LABEL_WHEN_SHARE, subplots_params=None, squeeze=True, margin=None, label='ax'):
    '''
    创建一个图形和轴对象，并根据提供的参数调整布局和轴的方框边缘。
    推荐的方式是设定ax_width和ax_height，而不是fig_width和fig_height。当设定ax_width和ax_height时，fig_width和fig_height会自动计算, 此时设定margin或adjust_params不会破坏ax框的比例
    如果想要先把fig分成等分,然后在每个等分里设置框的位置,使用margin
    如果想要最外层的图有自己单独的距离图像边框的范围,使用adjust_params和adjust_params_custom,并且使用get_fig_gs,再从gs获取ax;示例:adjust_params={'left': 0.2, 'right': 0.8, 'top': 0.8, 'bottom': 0.2, 'wspace': 0.5, 'hspace': 0.5},这里的left,right,top,bottom是相对于整个fig的位置(可以理解为最外围的pad),wspace和hspace是子图之间的间距(相对于average width和average height)

    注意:
    本函数只支持等大的ax,如果需要不等大的,可以使用merge_ax,split_ax来获得;或者使用subfig来创建不等大的ax

    Parameters:
    - figsize: 元组，指定图形的宽度和高度。
    - nrows, ncols: 整数，指定子图的行数和列数。
    - ax_width, ax_height: 浮点数，指定子图的宽度和高度。
    - fig_width, fig_height: 浮点数，指定图形的宽度和高度。(优先级高于ax_width和ax_height,如果设置了则会覆盖ax_width和ax_height的设置)
    - sharex, sharey: 布尔值或字符串，指定是否共享x轴和y轴。(可以是'all', 'none', 'row', 'col', True, False)
    - rm_repeat_tick_label_when_share: 布尔值，指定是否移除多余的刻度标签(当sharex或者sharey时)

    Returns:
    - fig, ax: 创建的图形和轴对象。
    '''
    subplots_params = update_dict({}, subplots_params)
    margin = update_dict(MARGIN, margin)

    # 计算fig的宽度和高度
    local_fig_width, local_fig_height = get_suitable_fig_size(nrows=nrows, ncols=ncols, ax_width=ax_width, ax_height=ax_height, margin=margin, which='margin')
    if fig_width is None:
        fig_width = local_fig_width
    if fig_height is None:
        fig_height = local_fig_height

    # 创建图形和轴对象
    fig = get_fig(width=fig_width, height=fig_height)
    ax = get_ax(fig=fig, nrows=nrows, ncols=ncols, sharex=sharex, sharey=sharey, rm_repeat_tick_label_when_share=rm_repeat_tick_label_when_share, margin=margin, squeeze=squeeze, label=label, subplots_params=subplots_params)
    return fig, ax


def gfa(nrows=1, ncols=1, ax_width=AX_WIDTH, ax_height=AX_HEIGHT, fig_width=None, fig_height=None, sharex=False, sharey=False, rm_repeat_tick_label_when_share=RM_REPEAT_TICK_LABEL_WHEN_SHARE, subplots_params=None, squeeze=True, margin=None, label='ax'):
    '''
    get_fig_ax的缩写
    '''
    return get_fig_ax(nrows=nrows, ncols=ncols, ax_width=ax_width, ax_height=ax_height, fig_width=fig_width, fig_height=fig_height, sharex=sharex, sharey=sharey, rm_repeat_tick_label_when_share=rm_repeat_tick_label_when_share, subplots_params=subplots_params, squeeze=squeeze, margin=margin, label=label)


def get_fig_ax_3d(**kwargs):
    '''
    创建一个3D图形和轴对象,提供get_fig_ax的参数作为**kwargs
    '''
    # Ensure 'subplots_params' exists in 'kwargs'
    if 'subplots_params' not in kwargs:
        kwargs['subplots_params'] = {}

    # Ensure 'projection' is set to '3d'
    if 'projection' not in kwargs['subplots_params']:
        kwargs['subplots_params']['projection'] = '3d'

    # Update the margin
    kwargs['margin'] = update_dict(MARGIN_3D, kwargs.get('margin'))
    return get_fig_ax(**kwargs)


def get_gs_original(fig=None, nrows=1, ncols=1, left=None, bottom=None, right=None, top=None, wspace=None, hspace=None, width_ratios=None, height_ratios=None):
    '''
    获取一个GridSpec对象。(不推荐使用,因为随着子图的数量的增加,这个参数会变得很难调整,因为这里的left,right,top,bottom是相对于整个fig的位置,而随着ax数量增加,fig的大小会变化,对于相同的值边框空白大小会不同,无法对齐)

    注意:
    对于gs,其始终是2维的,即使是1行或者1列
    '''
    # 如果没有提供figure,则使用当前的图形
    if fig is None:
        fig = plt.gcf()
    return GridSpec(nrows, ncols, figure=fig, left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace, width_ratios=width_ratios, height_ratios=height_ratios)


def get_gs_custom(fig=None, nrows=1, ncols=1, left=None, bottom=None, right=None, top=None, wspace=None, hspace=None, width_ratios=None, height_ratios=None):
    '''
    获取一个GridSpec对象。但是left等参数是相对于ax的位置,而不是fig的位置

    注意:
    对于gs,其始终是2维的,即使是1行或者1列
    '''
    adjust_params_custom = dict(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
    adjust_params_custom = update_dict_ignore(ADJUST_PARAMS_CUSTOM, adjust_params_custom)
    adjust_params = get_adjust_params_from_custom(nrows=nrows, ncols=ncols, adjust_params_custom=adjust_params_custom)
    return get_gs_original(nrows=nrows, ncols=ncols, fig=fig, width_ratios=width_ratios, height_ratios=height_ratios, **adjust_params)


def get_fig_gs_original(nrows=1, ncols=1, ax_width=AX_WIDTH, ax_height=AX_HEIGHT, fig_width=None, fig_height=None, dpi=FIG_DPI, get_fig_kwargs=None, adjust_params=None):
    '''
    获取fig和gs

    注意:
    此函数只支持等大的ax,如果需要不等大的,可以使用merge_ax,split_ax来获得;或者使用subfig来创建不等大的ax
    根据ax_width和ax_height来计算fig的大小,如果fig_width和fig_height不为None,则会覆盖ax_width和ax_height的设置
    '''
    get_fig_kwargs = update_dict({}, get_fig_kwargs)
    adjust_params = update_dict(ADJUST_PARAMS_CUSTOM, adjust_params)

    # 计算fig的宽度和高度
    local_fig_width, local_fig_height = get_suitable_fig_size(nrows=nrows, ncols=ncols, ax_width=ax_width, ax_height=ax_height, adjust_params=adjust_params, which='adjust_params')
    if fig_width is None:
        fig_width = local_fig_width
    if fig_height is None:
        fig_height = local_fig_height

    # 创建图形和GridSpec对象
    fig = get_fig(width=fig_width, height=fig_height, dpi=dpi, **get_fig_kwargs)
    gs = get_gs_original(nrows=nrows, ncols=ncols, fig=fig, **adjust_params)
    return fig, gs


def get_fig_gs_custom(nrows=1, ncols=1, ax_width=AX_WIDTH, ax_height=AX_HEIGHT, fig_width=None, fig_height=None, dpi=FIG_DPI, get_fig_kwargs=None, adjust_params_custom=None):
    '''
    获取fig和gs

    注意:
    此函数只支持等大的ax,如果需要不等大的,可以使用merge_ax,split_ax来获得;或者使用subfig来创建不等大的ax
    根据ax_width和ax_height来计算fig的大小,如果fig_width和fig_height不为None,则会覆盖ax_width和ax_height的设置
    '''
    get_fig_kwargs = update_dict({}, get_fig_kwargs)
    adjust_params_custom = update_dict(ADJUST_PARAMS_CUSTOM, adjust_params_custom)

    # 计算fig的宽度和高度
    local_fig_width, local_fig_height = get_suitable_fig_size(nrows=nrows, ncols=ncols, ax_width=ax_width, ax_height=ax_height, adjust_params_custom=adjust_params_custom, which='adjust_params_custom')
    if fig_width is None:
        fig_width = local_fig_width
    if fig_height is None:
        fig_height = local_fig_height

    # 创建图形和GridSpec对象
    fig = get_fig(width=fig_width, height=fig_height, dpi=dpi, **get_fig_kwargs)
    gs = get_gs_custom(nrows=nrows, ncols=ncols, fig=fig, **adjust_params_custom)
    return fig, gs


def get_gs_inside_ax(ax, nrows=1, ncols=1, wspace=None, hspace=None, width_ratios=None, height_ratios=None):
    '''
    以ax的位置为基础获取一个GridSpec对象。
    '''
    left, right, bottom, top = get_ax_position_custom(ax)
    fig = ax.figure
    return get_gs_original(fig=fig, nrows=nrows, ncols=ncols, left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace, width_ratios=width_ratios, height_ratios=height_ratios)


def get_all_ax_from_gs(gs, sharex=False, sharey=False, rm_repeat_tick_label_when_share=RM_REPEAT_TICK_LABEL_WHEN_SHARE, squeeze=True, label='gs', **kwargs):
    '''
    从GridSpec对象中获取所有的轴对象。
    '''
    if not isinstance(sharex, str):
        sharex = "all" if sharex else "none"
    if not isinstance(sharey, str):
        sharey = "all" if sharey else "none"

    ax = np.empty((gs.nrows, gs.ncols), dtype=object)
    for i in range(gs.nrows):
        for j in range(gs.ncols):
            shared_with = {"none": None, "all": ax[0, 0],
                            "row": ax[i, 0], "col": ax[0, j]}
            ax[i, j] = get_ax_from_gs(gs, index=(i, j), sharex=shared_with[sharex], sharey=shared_with[sharey], **kwargs)

    if rm_repeat_tick_label_when_share:
        # 移除多余的label
        if sharex in ["col", "all"]:
            for a in ax.flat:
                a._label_outer_xaxis(skip_non_rectangular_axes=True)
        if sharey in ["row", "all"]:
            for a in ax.flat:
                a._label_outer_yaxis(skip_non_rectangular_axes=True)
    
    if squeeze:
        ax = squeeze_ax(ax)
    
    # 添加label
    set_ax_label(ax)

    # 略微修改label
    for a in get_iterable_ax(ax):
        set_ax_label(a, cat(label, a.get_label()))
    return ax


def get_ax_from_gs(gs, index=None, label='gs', **kwargs):
    '''
    从GridSpec对象中获取轴对象。

    参数:
    gs: 可以是完整的gs,也可以是索引之后的gs(如果是索引之后的gs,则index必须为None)
    index: 默认为None,表示获取整个gs的ax,如果是tuple,则表示获取gs[index]的ax;注意,完整的gs无法直接用于获取ax,需要使用[:]索引才可以
    '''
    fig = gs.figure
    if index is None:
        try:
            ax = fig.add_subplot(gs[:], **kwargs) # 使用gs,如果是原始的gs,必须要索引,否则会报错
        except:
            ax = fig.add_subplot(gs, **kwargs) # 为了防止用户传入索引之后的gs,这里使用gs再次尝试
    else:
        if not isinstance(index, tuple):
            raise ValueError('Index must be a tuple.')
        ax = fig.add_subplot(gs[index], **kwargs)
    
    # 添加label
    set_ax_label(ax, index)

    # 略微修改label
    for a in get_iterable_ax(ax):
        set_ax_label(a, cat(label, a.get_label()))
    return ax


def get_all_subfig_from_gs(gs, squeeze=True):
    subfig = np.empty((gs.nrows, gs.ncols), dtype=object)
    for i in range(gs.nrows):
        for j in range(gs.ncols):
            subfig[i, j] = get_subfig_from_gs(gs, index=(i, j))
    if squeeze:
        subfig = squeeze_ax(subfig)
    return subfig


def get_subfig_from_gs(gs, index=None, **kwargs):
    '''
    从GridSpec对象中获取子图对象。

    参数:
    gs: 可以是完整的gs,也可以是索引之后的gs(如果是索引之后的gs,则index必须为None)
    index: 默认为None,表示获取整个gs的subfig,如果是tuple,则表示获取gs[index]的subfig;注意,完整的gs无法直接用于获取subfig,需要使用[:]索引才可以
    '''
    # 利用添加ax来获取需要的width和height
    ax = get_ax_from_gs(gs, index=index)
    width, height = get_ax_size(ax)
    rm_ax(ax)
    
    fig = gs.figure
    if index is None:
        try:
            subfig = fig.add_subfigure(gs[:], **kwargs) # 使用gs,如果是原始的gs,必须要索引,否则会报错
        except:
            subfig = fig.add_subfigure(gs, **kwargs) # 为了防止用户传入索引之后的gs,这里使用gs再次尝试
    else:
        if not isinstance(index, tuple):
            raise ValueError('Index must be a tuple.')
        subfig = fig.add_subfigure(gs[index], **kwargs)

    # 动态给 subfigure 添加一个 get_size_inches 方法(因为set_ax需要使用figsize)
    def get_size_inches(self):
        return width, height

    for s in get_iterable_ax(subfig):
        s.get_size_inches = get_size_inches.__get__(s)
    return subfig


def get_subfig_inside_ax(ax, nrows=1, ncols=1, squeeze=True, wspace=None, hspace=None, width_ratios=None, height_ratios=None):
    '''
    以ax的位置为基础获取一个subfig对象
    '''
    gs = get_gs_inside_ax(ax, nrows=nrows, ncols=ncols, wspace=wspace, hspace=hspace, width_ratios=width_ratios, height_ratios=height_ratios)
    return get_all_subfig_from_gs(gs, squeeze=squeeze)


def get_ax_inside_ax(ax, nrows=1, ncols=1, wspace=None, hspace=None, width_ratios=None, height_ratios=None, sharex=False, sharey=False, squeeze=True, keep_original=False, **kwargs):
    '''
    只是copy了split_ax_by_gs的代码,所有功能一致
    '''
    return split_ax_by_gs(ax, nrows=nrows, ncols=ncols, wspace=wspace, hspace=hspace, width_ratios=width_ratios, height_ratios=height_ratios, sharex=sharex, sharey=sharey, squeeze=squeeze, keep_original=keep_original, **kwargs)


def get_gs_inside_gs(gs, index=None, nrows=1, ncols=1, wspace=None, hspace=None, width_ratios=None, height_ratios=None):
    '''
    从GridSpec对象中获取一个新的GridSpec对象。实现方式是先创建ax,使用get_gs_inside_ax获取gs,再删除ax
    '''
    ax = get_ax_from_gs(gs, index=index)
    sub_gs = get_gs_inside_ax(ax, nrows=nrows, ncols=ncols, wspace=wspace, hspace=hspace, width_ratios=width_ratios, height_ratios=height_ratios)
    rm_ax(ax)
    return sub_gs
# endregion


# region 通用函数(保存图像)
def save_fig(fig, filename, formats=None, dpi=SAVEFIG_DPI, close=True, bbox_inches=BBOX_INCHES, pad_inches=PAD_INCHES, filename_process=None, pkl=SAVEFIG_PKL, ax=None, **kwargs):
    '''
    保存图形到指定的文件格式(搭配concat_str使用,concat_str可以用于生成文件名)

    参数:
    filename - 保存文件的基础名（不建议包含扩展名）
    formats - 要保存的文件格式列表,默认为 [SAVEFIG_FORMAT]
    dpi - 图像的分辨率,默认为 SAVEFIG_DPI
    close - 是否在保存后关闭图形,默认为 True
    bbox_inches - 设置边界框(bounding box)尺寸,默认为 BBOX_INCHES
    '''
    if formats is None:
        formats = [SAVEFIG_FORMAT]
    filename_process = update_dict(FILENAME_PROCESS, filename_process)

    # 对filename进行处理
    filename = format_filename(filename, filename_process)

    # 从filename中获取文件夹名并创建文件夹
    mkdir(os.path.dirname(filename))

    if filename.endswith(('.png', '.pdf', '.eps')):
        # 假如filename以'.png','.pdf'或'.eps'结尾,将后缀名添加到保存的格式列表中
        formats.append(filename.split('.')[-1])
        filename = filename[:-4]
        
    # 按照formats列表中的格式保存
    for fmt in formats:
        fig.savefig(f'{filename}.{fmt}', dpi=dpi,
                    bbox_inches=bbox_inches, pad_inches=pad_inches, **kwargs)

    # 保存fig到pkl
    if pkl:
        save_pkl((fig, ax), filename)

    if close:
        plt.close(fig)
        fig = None


def save_fig_lite(fig, filename, formats=None, dpi=SAVEFIG_DPI/3, close=True, bbox_inches='tight', pad_inches=PAD_INCHES, filename_process=None, pkl=False, ax=None, **kwargs):
    '''
    轻量级的保存图形函数,只保存位图,降低dpi,不保存pkl
    并且bbox_inches的默认值改为'tight',这样有助于防止标题等被裁剪
    '''
    formats = [SAVEFIG_RASTER_FORMAT]
    save_fig(fig, filename, formats=formats, dpi=dpi, close=close, bbox_inches=bbox_inches, pad_inches=pad_inches, filename_process=filename_process, pkl=pkl, ax=ax, **kwargs)


def save_fig_3d(fig, filename, elev_list=None, azim_list=np.arange(0, 360, 30), formats=None, dpi=SAVEFIG_DPI, close=True, bbox_inches=BBOX_INCHES, pkl=True, ax=None, generate_video=False, frame_rate=FRAME_RATE, delete_figs=False, video_formats=None, savefig_kwargs=None):
    '''
    保存3D图形的多个视角，对于每个视角，都会生成图片（并在文件名上加入角度），然后将图片合成视频。
    对于filename,如果以'.png','.pdf'或'.eps'结尾,则按照后缀名保存,否则按照formats列表中的格式保存。
    '''
    if elev_list is None:
        elev_list = [ELEV]
    if formats is None:
        if generate_video:
            # 如果要生成视频,且formats没有指定，只保存位图
            formats = [SAVEFIG_RASTER_FORMAT]
        else:
            # 如果不生成视频,且formats没有指定，只保存矢量图
            formats = [SAVEFIG_VECTOR_FORMAT]
    if video_formats is None:
        video_formats = ['mp4', 'gif']
    if savefig_kwargs is None:
        savefig_kwargs = {}

    if filename.endswith(('.png', '.pdf', '.eps')):
        # 假如filename以'.png','.pdf'或'.eps'结尾,将后缀名添加到保存的格式列表中
        formats.append(filename.split('.')[-1])
        filename = filename[:-4]

    ax = fig.get_axes()

    fig_paths_dict = {}
    fig_paths_list = []
    for elev in elev_list:
        for azim in azim_list:
            set_ax_view_3d(ax, elev=elev, azim=azim)
            local_filename = f'{filename}_elev_{str(int(elev))}_azim_{str(int(azim))}'
            # 注意这里不能close,并且不重复存储pkl
            save_fig(fig=fig, filename=local_filename, formats=formats,
                        dpi=dpi, close=False, bbox_inches=bbox_inches, pkl=False, ax=ax, **savefig_kwargs)
            fig_paths_dict[(elev, azim)] = fig_paths_dict.get((elev, azim), []) + [local_filename]
            fig_paths_list.append(local_filename)

    # 保存fig到pkl
    if pkl:
        save_pkl((fig, ax), filename)

    if close:
        plt.close(fig)

    if generate_video:
        fig_to_video(fig_paths_list, filename, frame_rate=frame_rate, delete_figs=delete_figs, formats=video_formats)
        
    if delete_figs:
        for elev in elev_list:
            for azim in azim_list:
                for fmt in formats:
                    os.remove(f'{filename}_elev_{elev}_azim_{azim}.{fmt}')

    return fig_paths_dict, fig_paths_list


def save_fig_3d_lite(fig, filename, elev_list=None, azim_list=np.arange(0, 360, 60), formats=None, dpi=SAVEFIG_DPI/3, close=True, bbox_inches='tight', pkl=False, ax=None, generate_video=False, frame_rate=FRAME_RATE, delete_figs=False, video_formats=None, savefig_kwargs=None):
    '''
    轻量级的保存3D图形的多个视角,只保存位图,降低dpi,不保存pkl,不生成视频,减少了azim_list的数量
    并且bbox_inches的默认值改为'tight',这样有助于防止标题等被裁剪
    '''
    formats = [SAVEFIG_RASTER_FORMAT]
    save_fig_3d(fig, filename, elev_list=elev_list, azim_list=azim_list, formats=formats, dpi=dpi, close=close, bbox_inches=bbox_inches, pkl=pkl, ax=ax, generate_video=generate_video, frame_rate=frame_rate, delete_figs=delete_figs, video_formats=video_formats, savefig_kwargs=savefig_kwargs)


def save_subfig(subfig, filename, close=False, pkl=False, **kwargs):
    '''
    利用bbox_inches来保存subfig
    '''
    bbox_inches = get_subfig_bbox_inches(subfig)
    fig = subfig.get_figure()
    save_fig(fig, filename, bbox_inches=bbox_inches, close=close, pkl=pkl, **kwargs)


def save_ax(axs, filename, close=False, pkl=False, bbox_inches='tight', **kwargs):
    '''
    利用将其他ax设置为invisiable来保存ax

    对于在subfig状态下创建的ax,无法使用这个函数
    '''
    iterable_ax = get_iterable_ax(axs)
    fig = iterable_ax[0].get_figure()
    if get_fig_type(fig) == 'subfig':
        raise ValueError('Cannot save ax in subfig state using this function.')

    original_visibility = {}

    for a in fig.get_axes():
        original_visibility[a] = a.get_visible()
        if isinstance(iterable_ax, np.ndarray):
            if not np.isin(a, iterable_ax):
                a.set_visible(False)
        else:
            if a not in iterable_ax:
                a.set_visible(False)

    save_fig(fig, filename, close=close, pkl=pkl, bbox_inches=bbox_inches, **kwargs)

    # 恢复其他ax的可见性到原始状态
    for a in fig.get_axes():
        a.set_visible(original_visibility[a])
# endregion


# region 通用函数(复制ax)
def copy_ax_content(source_ax, target_ax):
    """
    将 source_ax 的内容拷贝到 target_ax 中，保持所有图形元素（如线条、散点、图例等）。
    
    参数:
    source_ax (matplotlib.axes.Axes): 源 Axes 对象。
    target_ax (matplotlib.axes.Axes): 目标 Axes 对象。
    """
    # 复制线条
    for line in source_ax.get_lines():
        target_ax.plot(line.get_xdata(), line.get_ydata(), label=line.get_label(),
                       color=line.get_color(), linestyle=line.get_linestyle(),
                       linewidth=line.get_linewidth(), marker=line.get_marker())

    # 复制散点图 (PathCollection)
    for collection in source_ax.collections:
        offsets = collection.get_offsets()  # 获取散点的位置
        colors = collection.get_facecolors()  # 获取散点的颜色
        sizes = collection.get_sizes()  # 获取散点的大小
        target_ax.scatter(offsets[:, 0], offsets[:, 1], c=colors, s=sizes, label=collection.get_label())

    # 复制patches
    for patch in source_ax.patches:
        if isinstance(patch, mpatches.Rectangle):
            new_patch = mpatches.Rectangle(patch.get_xy(), patch.get_width(), patch.get_height(),
                                  angle=patch.angle, color=patch.get_facecolor(),
                                  edgecolor=patch.get_edgecolor(), linewidth=patch.get_linewidth())
        elif isinstance(patch, mpatches.Circle):
            new_patch = mpatches.Circle(patch.center, patch.radius, color=patch.get_facecolor(),
                               edgecolor=patch.get_edgecolor(), linewidth=patch.get_linewidth())
        elif isinstance(patch, mpatches.Polygon):
            new_patch = mpatches.Polygon(patch.get_xy(), closed=patch.get_closed(),
                                color=patch.get_facecolor(), edgecolor=patch.get_edgecolor(),
                                linewidth=patch.get_linewidth())
        else:
            continue  # 如果有不支持的 Patch 类型可以选择跳过
        target_ax.add_patch(new_patch)

    # 复制 imshow 图像
    for img in source_ax.images:
        target_ax.imshow(
            img.get_array(),
            extent=img.get_extent(),
            origin=img.origin,
            cmap=img.get_cmap(),
            norm=img.norm,
            interpolation=img.get_interpolation(),
            alpha=img.get_alpha(),
            aspect='auto',
        )

    # 复制标题和标签
    target_ax.set_title(source_ax.get_title())
    target_ax.set_xlabel(source_ax.get_xlabel())
    target_ax.set_ylabel(source_ax.get_ylabel())

    # 复制图例
    if source_ax.get_legend() is not None:
        handles, labels = source_ax.get_legend_handles_labels()
        target_ax.legend(handles, labels)

    # 复制刻度标签
    target_ax.set_xticks(source_ax.get_xticks())
    target_ax.set_xticklabels(source_ax.get_xticklabels())
    target_ax.set_yticks(source_ax.get_yticks())
    target_ax.set_yticklabels(source_ax.get_yticklabels())

    # 复制坐标轴的限制
    target_ax.set_xlim(source_ax.get_xlim())
    target_ax.set_ylim(source_ax.get_ylim())

    # 复制文本
    for text in source_ax.texts:
        target_ax.text(text.get_position()[0], text.get_position()[1], text.get_text(), fontsize=text.get_fontsize(), color=text.get_color())
# endregion


# region 通用函数(图片格式转换)
def convert_fig(input_file_path, output_format):
    input_format = input_file_path.split('.')[-1].lower()
    output_file_path = input_file_path[:-len(input_format)] + output_format

    if input_format == 'png' and output_format == 'pdf':
        with Image.open(input_file_path) as img:
            img.convert('RGB').save(output_file_path,
                                    'PDF', resolution=SAVEFIG_DPI)

    elif input_format == 'png' and output_format == 'eps':
        print('png to eps is not supported')

    elif input_format == 'eps' and output_format == 'png':
        # # EPS to PNG conversion
        with Image.open(input_file_path) as img:
            img.load(scale=10)  # 不知道这个有什么用,但是有了就会清晰,没有就模糊
            # img.save(output_file_path, 'PNG', dpi=(SAVEFIG_DPI, SAVEFIG_DPI))
            # 创建一个白色背景的画布
            background = Image.new('RGBA', img.size, (255, 255, 255))
            # 将EPS图像合并到背景画布上
            composite = Image.alpha_composite(background, img.convert('RGBA'))
            # 保存为PNG，这时不需要指定DPI，因为我们已经在背景画布上处理了图像
            composite.save(output_file_path, 'PNG',
                           dpi=(SAVEFIG_DPI, SAVEFIG_DPI))

    # PDF to PNG (requires pdf2fig)
    elif input_format == 'pdf' and output_format == 'png':
        # Increased DPI for better quality
        figs = convert_from_path(input_file_path, dpi=SAVEFIG_DPI)
        for i, fig in enumerate(figs):
            fig.save(output_file_path, 'PNG')
            break  # Assuming saving only the first page

    # PDF to EPS (PDF -> PNG -> EPS)
    elif input_format == 'pdf' and output_format == 'eps':
        temp_fig = convert_from_path(input_file_path, dpi=SAVEFIG_DPI)[
            0]  # Increased DPI for better quality
        basedir = os.path.dirname(input_file_path)
        temp_fig_path = os.path.join(basedir, 'temp.png')
        temp_fig.save(temp_fig_path, 'PNG')
        with Image.open(temp_fig_path) as img:
            img.save(output_file_path, 'EPS')
        os.remove(temp_fig_path)  # Clean up the temporary file

    # EPS to PDF (EPS -> PNG -> PDF)
    elif input_format == 'eps' and output_format == 'pdf':
        with Image.open(input_file_path) as img:
            img.load(scale=10)  # 不知道这个有什么用,但是有了就会清晰,没有就模糊
            basedir = os.path.dirname(input_file_path)
            temp_fig_path = os.path.join(basedir, 'temp.png')
            img.save(temp_fig_path, 'PNG', dpi=(SAVEFIG_DPI, SAVEFIG_DPI))
        with Image.open(temp_fig_path) as temp_img:
            temp_img.convert('RGB').save(output_file_path,
                                         'PDF', resolution=SAVEFIG_DPI)
        os.remove(temp_fig_path)  # Clean up the temporary file

    else:
        raise ValueError(
            f"Conversion from {input_format} to {output_format} is not supported.")
# endregion
# endregion


# region 通用函数(创建visualizer)
class SingleVisualizer:
    '''
    可视化单个对象

    save_fig_kwargs的调整可以方便的影响每次的save_fig,方便在测试时使用png和低dpi,而在正式时使用pdf和高dpi
    '''
    def __init__(self, title, save_fig_kwargs=None):
        self.title = title
        self.update_save_fig_kwargs(save_fig_kwargs)

    def get_fig_ax(self, **kwargs):
        self.fig, self.ax = get_fig_ax(**kwargs)
        return self.fig, self.ax

    def update_save_fig_kwargs(self, save_fig_kwargs):
        '''
        更新save_fig_kwargs
        '''
        if not hasattr(self, 'save_fig_kwargs'):
            self.save_fig_kwargs = {}
        self.save_fig_kwargs = update_dict(self.save_fig_kwargs, save_fig_kwargs)

    def save_fig(self, filename, save_fig_kwargs=None, fig=None):
        '''
        保存fig
        '''
        if fig is None:
            fig = self.fig
        save_fig_kwargs = update_dict(self.save_fig_kwargs, save_fig_kwargs)
        save_fig(fig, filename, **save_fig_kwargs)


class MultiVisualizer(SingleVisualizer):
    '''
    可视化多个对象

    MultiVisualizer本身也可以作为一个SingleVisualizer,从而可以嵌套使用
    例如:
    每个脑区拥有一个SingleVisualizer
    全脑拥有多个脑区的SingleVisualizer,成为MultiVisualizer
    全脑的不同参数,可以以全脑的MultiVisualizer作为一个SingleVisualizer
    '''
    def __init__(self, single_visualizer_list, save_fig_kwargs=None, fig_title_list=None, title=None):
        '''
        注意:
        fig_title_list: 用于设置fig的标题,如果不设置,则使用title_list作为标题(所以它可以重复),因为仅仅用于图片内,而不会用于索引single_visualizer_list
        '''
        self.title_list = [single_visualizer.title for single_visualizer in single_visualizer_list]
        self.single_visualizer_list = single_visualizer_list.copy()
        self.set_fig_title_list(fig_title_list)
        super().__init__(title=title, save_fig_kwargs=save_fig_kwargs)
    
    def update_save_fig_kwargs(self, save_fig_kwargs):
        '''
        更新save_fig_kwargs

        注意:
        会同步更新所有的single_visualizer的save_fig_kwargs
        '''
        super().update_save_fig_kwargs(save_fig_kwargs)
        for single_visualizer in self.single_visualizer_list:
            single_visualizer.update_save_fig_kwargs(save_fig_kwargs)

    @property
    def num(self):
        return len(self.single_visualizer_list)

    def set_fig_title_list(self, fig_title_list=None):
        '''
        设置fig的标题(有时我们不希望使用title_list作为标题)
        '''
        if fig_title_list is None:
            self.fig_title_list = self.title_list.copy()
        else:
            self.fig_title_list = fig_title_list.copy()

    def get_single_visualizer(self, title):
        '''
        获取单个visualizer
        '''
        for single_visualizer in self.single_visualizer_list:
            if single_visualizer.title == title:
                return single_visualizer

    @contextlib.contextmanager
    def temporary_disable(self, titles):
        '''
        利用with语句临时移除指定的visualizer

        示例:
        with multi_visualizer.temporary_disable(['title1', 'title2']):
            multi_visualizer.get_fig_subfig_ax(...)
            multi_visualizer.add_tag()
        '''
        try:
            # 临时移除指定 visualizer
            self._original_single_visualizer_list = self.single_visualizer_list.copy()
            self._original_title_list = self.title_list.copy()
            self._original_fig_title_list = self.fig_title_list.copy()

            rm_idx = []
            for title in titles:
                idx = self.title_list.index(title)
                rm_idx.append(idx)
            self.single_visualizer_list = [single_visualizer for i, single_visualizer in enumerate(self.single_visualizer_list) if i not in rm_idx]
            self.title_list = [title for i, title in enumerate(self.title_list) if i not in rm_idx]
            self.fig_title_list = [fig_title for i, fig_title in enumerate(self.fig_title_list) if i not in rm_idx]
            yield
        finally:
            # 恢复原始列表
            self.single_visualizer_list = self._original_single_visualizer_list.copy()
            self.title_list = self._original_title_list.copy()
            self.fig_title_list = self._original_fig_title_list.copy()

    @classmethod
    def from_multi_visualizer_instance(cls, multi_visualizer_instance):
        '''
        从实例化的MultiVisualizer设定参数

        示例:
        new_multi_visualizer = MultiVisualizer.from_multi_visualizer_instance(multi_visualizer)
        '''
        return cls(copy.deepcopy(multi_visualizer_instance.single_visualizer_list), copy.deepcopy(multi_visualizer_instance.save_fig_kwargs), copy.deepcopy(multi_visualizer_instance.fig_title_list))

    def get_fig_subfig_ax(self, subfig_nrows='auto', subfig_ncols='auto', separate=False, get_suitable_fig_size_kwargs=None, get_fig_subfig_kwargs=None, get_ax_kwargs=None):
        '''
        获取fig和ax

        参数:
        subfig_nrows, subfig_ncols: 行数和列数(对于subfig而言,如果separate为True,则nrows和ncols无效)
        separate: True,则每个对象一个fig;False,每个对象一个subfig,但是所有对象一个fig
        '''
        if subfig_nrows == 'auto' and subfig_ncols == 'auto':
            raise ValueError('subfig_nrows and subfig_ncols cannot be both auto.')

        if subfig_nrows == 'auto':
            subfig_nrows = round(self.num / subfig_ncols)
        if subfig_ncols == 'auto':
            subfig_ncols = round(self.num / subfig_nrows)

        self.subfig_nrows = subfig_nrows
        self.subfig_ncols = subfig_ncols
        self.separate = separate
        self.subfig_dict = {}
        self.ax_dict = {}
        get_suitable_fig_size_kwargs = update_dict({}, get_suitable_fig_size_kwargs)
        get_fig_subfig_kwargs = update_dict({}, get_fig_subfig_kwargs)
        get_ax_kwargs = update_dict({}, get_ax_kwargs)
        self.ax_nrows = get_ax_kwargs.get('nrows', 1)
        self.ax_ncols = get_ax_kwargs.get('ncols', 1)

        subfig_width, subfig_height = get_suitable_fig_size(**get_suitable_fig_size_kwargs)

        if separate:
            for title in self.title_list:
                self.subfig_dict[title] = get_fig(subfig_width, subfig_height)
                self.ax_dict[title] = get_ax(fig=self.subfig_dict[title], **get_ax_kwargs)
        else:
            self.fig, subfig = get_fig_subfig(nrows=subfig_nrows, ncols=subfig_ncols, subfig_width=subfig_width, subfig_height=subfig_height, **get_fig_subfig_kwargs)
            if self.num == 1:
                subfig = np.array([[subfig]])
            subfig = subfig.flatten()
            for i, title in enumerate(self.title_list):
                self.subfig_dict[title] = subfig[i]
                self.ax_dict[title] = get_ax(fig=self.subfig_dict[title], **get_ax_kwargs)

        return self.subfig_dict, self.ax_dict

    def set_external_fig_subfig_ax(self, subfig_dict, ax_dict):
        '''
        支持直接导入fig和subfig
        '''
        self.subfig_dict = subfig_dict
        self.ax_dict = ax_dict

    def add_tag(self, **kwargs):
        '''
        添加tag

        注意:
        如果希望以自定义的顺序添加tag,可以自行使用add_axes_list_tag_by_order
        '''
        if self.separate:
            for ax in self.ax_dict.values():
                add_axes_list_tag_by_order(np.array(ax).flatten(), **kwargs)
        else:
            ax_in_order = []
            for title in self.title_list:
                ax_in_order.append(self.ax_dict[title])
            ax_in_order = flatten_list(pure_list(ax_in_order))
            add_axes_list_tag_by_order(ax_in_order, **kwargs)

    def add_subfig_title(self, **kwargs):
        '''
        为每个subfig添加标题
        '''
        for i in range(self.num):
            subfig = self.subfig_dict[self.title_list[i]]
            fig_title = self.fig_title_list[i]
            set_fig_title(subfig, fig_title, **kwargs)

    def rm_redundant_xylabel(self):
        pass

    def add_rounded_rectangle():
        pass

    def multi_ax_plot(self, func_list, kwargs_list):
        '''
        在多个ax上作图
        '''
        for func, kwargs in zip(func_list, kwargs_list):
            func(**kwargs)

    def multi_ax_plot_by_single_visualizer(self, func_name, kwargs_list):
        '''
        通过SingleVisualizer的方法在多个ax上作图
        '''
        func_list = [getattr(single_visualizer, func_name) for single_visualizer in self.single_visualizer_list]
        self.multi_ax_plot(func_list, kwargs_list)

    def save_fig(self, filename, save_fig_kwargs=None):
        '''
        保存fig
        '''
        save_fig_kwargs = update_dict(self.save_fig_kwargs, save_fig_kwargs)
        if self.separate:
            for title in self.title_list:
                save_fig(self.subfig_dict[title], filename+'_'+title, **save_fig_kwargs)
        else:
            save_fig(self.fig, filename, **save_fig_kwargs)


class MultiVisualizerByRow(MultiVisualizer):
    '''
    快捷的可视化多个对象,按行排列

    注意:
    仅仅支持每个subfig内部的ax只有一行
    '''
    def __init__(self, single_visualizer_list, save_fig_kwargs=None, fig_title_list=None):
        super().__init__(single_visualizer_list, save_fig_kwargs, fig_title_list)
    
    def rm_redundant_xylabel(self):
        '''
        去掉多余的xy标签
        '''
        if self.separate:
            pass
        else:
            for row, ax in enumerate(self.ax_dict.values()):
                if row != self.ax_nrows-1:
                    rm_ax_xlabel(ax)

    def get_fig_subfig_ax(self, subfig_ncols=1, separate=False, get_suitable_fig_size_kwargs=None, get_fig_subfig_kwargs=None, get_ax_kwargs=None):
        '''
        使用父类,但是自动设定nrows
        '''
        return super().get_fig_subfig_ax(self.num, subfig_ncols, separate, get_suitable_fig_size_kwargs, get_fig_subfig_kwargs, get_ax_kwargs)


class MultiVisualizerByCol(MultiVisualizer):
    '''
    快捷的可视化多个对象,按列排列

    注意:
    仅仅支持每个subfig内部的ax只有一列
    '''
    def __init__(self, single_visualizer_list, save_fig_kwargs=None, fig_title_list=None):
        super().__init__(single_visualizer_list, save_fig_kwargs, fig_title_list)
    
    def rm_redundant_xylabel(self):
        '''
        去掉多余的xy标签
        '''
        if self.separate:
            pass
        else:
            for col, ax in enumerate(self.ax_dict.values()):
                if col != 0:
                    rm_ax_ylabel(ax)

    def add_tag(self, row_first=True, **kwargs):
        '''
        添加tag

        注意:
        row_first为True,则按行添加tag;False,则按列添加tag
        row_first要求subfig内部的ax只有一列
        '''
        if row_first:
            if self.separate:
                for ax in self.ax_dict.values():
                    add_axes_list_tag_by_order(ax, **kwargs)
            else:
                ax_in_order = []
                for title in self.title_list:
                    ax_in_order.append(self.ax_dict[title])
                ax_in_order = np.array(ax_in_order).T.flatten()
                add_axes_list_tag_by_order(ax_in_order, **kwargs)
        else:
            super().add_tag(**kwargs)
    
    def get_fig_subfig_ax(self, subfig_nrows=1, separate=False, get_suitable_fig_size_kwargs=None, get_fig_subfig_kwargs=None, get_ax_kwargs=None):
        '''
        使用父类,但是自动设定ncols
        '''
        return super().get_fig_subfig_ax(subfig_nrows, self.num, separate, get_suitable_fig_size_kwargs, get_fig_subfig_kwargs, get_ax_kwargs)


class PaperFigVisualizer(SingleVisualizer):
    '''
    为Paper使用,通过统一的figs_dir来保存fig

    注意:
    如果没有输入version,figs_dir后面会被加上时间戳,所以每次运行都会生成新的文件夹,这种方式可以避免覆盖之前的fig
    '''
    def __init__(self, figs_dir, version=None, title=None, save_fig_kwargs=None):
        if version is None:
            version = get_time()
        self.figs_dir = pj(figs_dir, version)
        super().__init__(title=title, save_fig_kwargs=save_fig_kwargs)

    def auto_save_fig(self, filename=None, save_fig_kwargs=None, fig=None):
        '''
        自动按照调用者的函数名保存fig
        '''
        if filename is None:
            filename = pj(self.figs_dir, sys._getframe(1).f_code.co_name)
        self.save_fig(filename, save_fig_kwargs=save_fig_kwargs, fig=fig)


def _init_paper_fig_and_run_method(cls, init_kwargs, method_list):
    '''用于多进程初始化PaperFig并运行方法'''
    paper_fig = cls(**init_kwargs)
    for method in method_list:
        getattr(paper_fig, method)()


def multi_process_fig(cls, init_kwargs, method_list, process_num=10):
    '''多进程运行PaperFig的多个方法'''
    kwargs_list = []
    for method in method_list:
        kwargs_list.append({'cls': cls, 'init_kwargs': init_kwargs, 'method_list': [method]})
    multi_process(process_num=process_num, func=_init_paper_fig_and_run_method, kwargs_list=kwargs_list)
# endregion


# region 拼图相关函数
def concat_fig(fig_paths_grid, filename, formats=None, background='transparent', delete_temp_files=True):
    '''
    根据二维图片路径列表自动拼接成网格布局的图片。

    参数:
        fig_paths_grid (list of list): 二维图片路径列表，每个内部列表代表一行。(如果输入一维,自动转换为二维,并横向排列,如果想要纵向排列,输入时使用[[path1],[path2]])
        filename (str): 拼接后图片的保存路径。
    '''
    if formats is None:
        formats = [SAVEFIG_RASTER_FORMAT]
    
    # 定义临时文件list
    temp_files = []

    # 创建文件夹
    mkdir(os.path.dirname(filename))

    # 如果输入一维,自动转换为二维
    if not isinstance(fig_paths_grid[0], list):
        fig_paths_grid = [fig_paths_grid]

    rows = len(fig_paths_grid)
    cols = max(len(row) for row in fig_paths_grid)

    figs_grid = [[None for _ in range(cols)] for _ in range(rows)]
    max_widths = [0] * cols
    max_heights = [0] * rows

    # 加载所有图片并计算每列最大宽度和每行最大高度
    for row_idx, row in enumerate(fig_paths_grid):
        for col_idx, fig_path in enumerate(row):
            valid_fig_path, fig_exist = find_fig(fig_path, order=['.png', '.pdf', '.eps'])
            if fig_exist:
                if not valid_fig_path.endswith('.png'):
                    convert_fig(valid_fig_path, 'png')
                    valid_fig_path = valid_fig_path[:-3] + 'png'
                    temp_files.append(valid_fig_path)
            else:
                print(f'Please check the path {fig_path} and try again.')
                return
            fig = Image.open(valid_fig_path)
            figs_grid[row_idx][col_idx] = fig
            max_widths[col_idx] = max(max_widths[col_idx], fig.width)
            max_heights[row_idx] = max(max_heights[row_idx], fig.height)

    total_width = sum(max_widths)
    total_height = sum(max_heights)

    # 根据背景选择模式和颜色
    if background == 'transparent':
        new_fig = Image.new('RGBA', (total_width, total_height), (0, 0, 0, 0))
    elif background == 'white':
        new_fig = Image.new('RGB', (total_width, total_height), (255, 255, 255))
    else:
        raise ValueError(f'Background mode {background} is not supported.')

    y_offset = 0
    for row_idx, row in enumerate(figs_grid):
        x_offset = 0
        for col_idx, fig in enumerate(row):
            if fig:  # 如果该位置有图片
                new_fig.paste(fig, (x_offset, y_offset))
            x_offset += max_widths[col_idx]
        y_offset += max_heights[row_idx]

    if filename.endswith(('.png', '.pdf', '.eps')):
        # 假如filename以'.png','.pdf'或'.eps'结尾,将后缀名添加到保存的格式列表中
        formats.append(filename.split('.')[-1])
        filename = filename[:-4]
        
    for fmt in formats:
        if fmt == 'png':
            new_fig.save(f'{filename}.{fmt}')
        else:
            new_fig.save(f'{filename}.png')
            convert_fig(f'{filename}.png', fmt)
    
    # 删除临时文件(比如说转换格式时生成的png文件)
    if delete_temp_files:
        for temp_file in temp_files:
            os.remove(temp_file)


def concat_fig_with_tag(figs_grid, filename, tags=None, formats=None, background='transparent', close=True, tag_size=TAG_SIZE, tag_color=BLACK, tag_position=FIG_TAG_POS, tag_kwargs=None, auto_tag_params=None):
    '''
    根据图片和标签自动拼接成网格布局的图片。(和concat_fig的用法基本相同,但是这里是输入fig对象并添加标签)

    参数:
        figs_grid (list): matplotlib的fig对象列表
        labels (list): 标签列表
        filename (str): 拼接后图片的保存路径
    '''
    if formats is None:
        formats = [SAVEFIG_RASTER_FORMAT]
    if tag_kwargs is None:
        tag_kwargs = {}
    if auto_tag_params is None:
        auto_tag_params = {}

    if filename.endswith(('.png', '.pdf', '.eps')):
        # 假如filename以'.png','.pdf'或'.eps'结尾,将后缀名添加到保存的格式列表中
        formats.append(filename.split('.')[-1])
        filename = filename[:-4]

    fig_paths_grid = []
    flatten_figs = flatten_list(figs_grid)
    if tags is None:
        flatten_tags = get_tag(len(flatten_figs), **auto_tag_params)
    else:
        flatten_tags = flatten_list(tags)
    for fig, tag in zip(flatten_figs, flatten_tags):
        add_fig_tag(fig, tag, x=tag_position[0], y=tag_position[1], fontsize=tag_size, color=tag_color, **tag_kwargs)
        save_fig(fig, concat_str([filename, tag]), formats=formats, close=close)
        fig_paths_grid.append(concat_str([filename, tag]))
    fig_paths_grid = rebuild_list(fig_paths_grid, figs_grid)
    concat_fig(fig_paths_grid, filename, formats=formats, background=background)
# endregion


# region 合成视频相关函数
def fig_to_video(fig_paths, filename, frame_rate=FRAME_RATE, delete_figs=False, formats=None):
    '''
    Combine PNG figs into MP4 videos and/or GIF figs, checking for the existence of each fig.
    假如fig_paths中的图片没有后缀，则会自动寻找有无对应的.png,.pdf或.eps文件,如果有则转换为.png文件，并生成视频。
    Parameters:
    - fig_paths: List of paths to the input figs.
    - filename: Base path for the output video or GIF fig. Extension will be added based on format.
    - frame_rate: Frame rate of the output video (ignored for GIF).
    - delete_figs: Flag to indicate whether to delete the input figs after creating the output.
    - formats: List of formats for the output, including 'mp4' and/or 'gif'.
    '''
    if formats is None:
        formats = ['mp4', 'gif']

    # 创建文件夹
    mkdir(os.path.dirname(filename))

    valid_fig_paths = []

    # Check for the existence of figs and accumulate valid paths
    for fig_path in fig_paths:
        valid_fig_path, fig_exist = find_fig(fig_path, order=['.png', '.pdf', '.eps'])
        if fig_exist:
            if valid_fig_path.endswith('.png'):
                valid_fig_paths.append(valid_fig_path)
            else:
                convert_fig(valid_fig_path, 'png')
                valid_fig_path = valid_fig_path[:-3] + 'png'
                valid_fig_paths.append(valid_fig_path)
        else:
            print(f'Please check the path {fig_path} and try again.')
            return

    # Ensure there are valid figs to process
    if not valid_fig_paths:
        print("No valid figs to process.")
        return

    if 'mp4' in formats:
        # MP4 Video output
        video_filename = f"{filename}.mp4"
        frame = cv2.imread(valid_fig_paths[0])
        height, width, layers = frame.shape
        video_size = (width, height)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_filename, fourcc,
                              frame_rate, video_size)

        for fig_path in valid_fig_paths:
            fig = cv2.imread(fig_path)
            if fig.shape[1] != video_size[0] or fig.shape[0] != video_size[1]:
                fig = cv2.resize(
                    fig, video_size, interpolation=cv2.INTER_LANCZOS4)
            out.write(fig)
        out.release()

    if 'gif' in formats:
        # GIF output
        gif_filename = f"{filename}.gif"
        figs = [Image.open(fig_path) for fig_path in valid_fig_paths]
        resized_figs = [fig.resize(
            (figs[0].width, figs[0].height), Image.LANCZOS) for fig in figs]
        resized_figs[0].save(gif_filename, save_all=True,
                               append_images=resized_figs[1:], duration=1000/frame_rate, loop=0)

    if delete_figs:
        # Delete figs after processing all formats
        for fig_path in valid_fig_paths:
            os.remove(fig_path)
        print("All valid figs deleted.")

    cv2.destroyAllWindows()
# endregion