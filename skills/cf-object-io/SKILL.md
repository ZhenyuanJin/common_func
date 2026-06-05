---
name: cf-object-io
description: 在 common_func 项目中需要保存或读取 Python 对象、dict、numpy array、scipy sparse matrix、pandas DataFrame、txt/yaml，或需要按目录分开保存大型字典时触发。用于指导优先使用 common_functions.py 中的 save_pkl/load_pkl、save_dict/load_dict、save_dict_separate/load_dict_separate、save_array/load_array、save_sps_array/load_sps_array、save_df/load_df 等封装。
---

# CF 对象读写约定

这个 skill 用于在 `common_func` 项目中保存和读取对象时，优先使用 `common_functions.py` 里的 IO 封装。除非任务明确要求原生接口，否则不要直接手写 `pickle.dump`、`joblib.dump`、`np.save`、`pd.to_pickle` 等。

## 通用对象

普通 Python 对象优先用 `save_pkl` 和 `load_pkl`：

```python
import common_functions as cf

cf.save_pkl(obj, filename)
obj = cf.load_pkl(filename)
```

`save_pkl` 默认用 joblib 保存；`load_pkl` 会兼容 `.joblib`、`.pickle` 和 `.pkl`。通常不要手动写扩展名，让 cf 自动补后缀和创建目录。

## 常见数据类型

字典优先用 `save_dict` 和 `load_dict`。默认会保存可读的 txt 预览和 pkl/joblib 对象，适合参数、配置、结果摘要：

```python
cf.save_dict(result_dict, filename)
result_dict = cf.load_dict(filename)
```

numpy array 用 `save_array` / `load_array`，scipy sparse matrix 用 `save_sps_array` / `load_sps_array`，DataFrame 用 `save_df` / `load_df`。

简单可人工编辑的配置可以用 `save_dict_yaml` / `load_dict_yaml`；普通文本或列表文本用 `save_txt`、`save_list_txt`、`load_txt` 等文本函数。

## 大字典和分开保存

当字典很大、每个 key 对应的数据类型不同，或只想按 key 选择性读取时，用 `save_dict_separate` 和 `load_dict_separate`。它会给每个 value 自动选择保存函数，并保存 `metadata` 做键映射：

```python
cf.save_dict_separate(data, save_dir, process_num=cf.PROCESS_NUM)
subset = cf.load_dict_separate(save_dir, key_to_load=['a', 'b'], process_num=cf.PROCESS_NUM)
```

需要向已有 separate dict 追加或覆盖部分 key 时，用 `save_dict_separate_merge_to_saved`；需要把读取结果合并到已有 dict 时，用 `load_dict_separate_merge_to_exist`。

保存 nested dict 时才考虑 `max_depth`。普通字典不要过早使用 separate 模式，先用 `save_dict`。

## 使用原则

路径通常传不带扩展名的 base filename，让 cf 自动处理扩展名和目录创建。

保存前根据数据类型选最具体的函数：对象用 pkl，数组用 array，稀疏矩阵用 sps array，DataFrame 用 df，字典用 dict。不要为了方便把所有东西都塞进一个 pickle，除非它本来就是一个整体对象。

读取外部文件时，如果已经知道确切格式，可以用对应的 `load_*`；如果是 cf 保存的 dict，优先用 `load_dict`，让它自动判断普通 pkl 还是 separate 目录。

涉及大文件、批量 key 或 separate dict 时，可以传 `process_num`，但先用 `process_num=1` 验证路径、key 和格式正确。
