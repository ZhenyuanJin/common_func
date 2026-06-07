---
name: cf-data-keeper
description: 在 common_func 项目中需要选择或配置 DataKeeper 的 data_type、save_load_method、dict/OrderedDataContainer、separate/lmdb 存储后端、key 设计、保存读取策略，或需要在 AbstractTool._config_data_keeper 中设置 self.data_keeper_kwargs 时触发。若问题主要是 Experiment/ComposedExperiment 的 tool 组织、目录复用或整体调试，改用 cf-experiment。
---

# CF DataKeeper 约定

这个 skill 用于选择和使用 `common_functions.py` 中的 `DataKeeper`、`DataContainer` 和 `OrderedDataContainer`。它关注结果容器、key 设计、保存后端和读取方式；不负责解释 `Experiment` / `ComposedExperiment` 的整体数据流。

## 选择规则

默认选择是 `dict + separate`：

```python
self.data_keeper_kwargs = {'data_type': 'dict', 'save_load_method': 'separate'}
```

普通少量结果、summary、数组、DataFrame、模型指标等，优先用默认 `dict + separate`。写入时显式给 `key`：

```python
self.data_keeper.set_value(result, key='result')
result = self.data_keeper.get_value(key='result')
```

大量参数化结果需要用多个参数定位时，用 `OrderedDataContainer`。必须提供 `param_order`；读取和写入用同一组参数：

```python
self.data_keeper_kwargs = {
    'data_type': 'OrderedDataContainer',
    'save_load_method': 'separate',
    'param_order': ['dataset', 'seed', 'alpha'],
}

self.data_keeper.set_value(value, dataset='demo', seed=0, alpha=0.1)
value = self.data_keeper.get_value(dataset='demo', seed=0, alpha=0.1)
```

需要单文件数据库式存储、key 很多且不希望产生大量子文件时，再考虑 `lmdb`。`separate` 更容易人工查看、删除和局部排查；`lmdb` 更像一个数据库文件，适合稳定的大量 key 存储。

## 在 Tool 中配置

在 experiment tool 里只在 `_config_data_keeper` 中设置 `self.data_keeper_kwargs`，不要在 `run_detail` 临时替换 `data_keeper`：

```python
def _config_data_keeper(self):
    super()._config_data_keeper()
    self.data_keeper_kwargs = {'data_type': 'dict', 'save_load_method': 'separate'}
```

如果只使用默认 `dict + separate`，可以不重写 `_config_data_keeper`。`Experiment` 初始化 tool 时会根据 `data_keeper_kwargs` 创建 `DataKeeper(name=self.data_keeper_name, basedir=self.dir_manager.outcomes_dir, ...)`，保存目录通常是 `outcomes/{tool_name}`。

## Key 设计

`dict` 模式下，key 应该是稳定、可读、与结果语义直接对应的字符串，例如 `result`、`summary`、`weights`、`score_by_area`。不要把随机顺序、worker id、临时文件名或运行日志混进 key。

`OrderedDataContainer` 模式下，`param_order` 应该只包含定位结果所需的语义参数，并保持长期稳定。不要把只影响运行方式的参数放进 key，例如 `process_num`、`task_list`、`threshold_gb`。

`included_name_list` 不决定哪些参数参与 key；传给 `set_value(...)` / `get_value(...)` 的参数仍都会参与 key。它只决定哪些参数在生成 key 时同时写入参数名和值；未列入的参数只写入参数值。默认 `included_name_list=[]`，即按 `param_order` 拼接参数值。只有当不同参数可能取到相同值、容易生成歧义 key 时，才把这些参数名加入 `included_name_list`。参数名和值应尽量短且稳定，方便后续 `get_subcontainer(...)` 和局部读取。

## 保存和读取

在 `Experiment` 里通常不用手动 `save()`；tool 的 `after_run` 会调用 `self.data_keeper.save()`，并在 `enable_skip=True` 时写入 `{tool_name}_all_saved` 标记。

独立使用 `DataKeeper` 时，需要自己保存：

```python
keeper = cf.DataKeeper(name='results', basedir=outcomes_dir, data_type='dict', save_load_method='separate')
keeper.set_value(summary, key='summary')
keeper.save()
summary = keeper.get_value(key='summary')
```

`get_value(...)` 会先查内存；内存没有时，会尝试从磁盘按 key 局部加载。大对象用完后可以调用：

```python
self.data_keeper.release_memory(keys_to_keep=['summary'])
```

下游 tool 接收到的前置 `{previous_tool_name}_data_keeper` 是 read-only。下游只读取前置结果，不要往前置 keeper 写入；新结果写入当前 tool 的 `self.data_keeper`。

## separate 与 lmdb 注意点

`separate` 使用 `save_dict_separate_merge_to_saved` / `load_dict_separate_merge_to_exist`，适合可检查、可局部加载和可分 key 删除的结果目录。

`lmdb` 使用 `save_dict_lmdb` / `load_dict_lmdb_merge_to_exist`。如果结果特别大，可能需要关注 `save_dict_lmdb` 的 `size`、`unit`、`max_size` 等参数；`Experiment` 默认 `after_run` 不传额外保存参数，复杂场景应在项目内写薄 subclass 或 wrapper。

常见错误应直接修正而不是静默绕过：

- `OrderedDataContainer` 没有 `param_order` 会报错。
- `dict` 模式下 `get_value()` 不给 `key` 会报错。
- 对 read-only 的前置 `DataKeeper` 调用 `set_value` 或 `save` 会报错。
- key 不稳定会导致复用、局部加载和后续分析难以追溯。

## 验证重点

先用小数据写入、保存、释放内存、重新读取同一个 key。对 `OrderedDataContainer`，至少检查一个完整参数 key。若在 experiment 中使用，还要确认 `outcomes/{tool_name}` 下的实际文件或 lmdb 目录符合预期，并且下游 tool 能通过 `{previous_tool_name}_data_keeper.get_value(...)` 读取。
