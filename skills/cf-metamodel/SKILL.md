---
name: cf-metamodel
description: 在 common_func 项目中需要继承老式 MetaModel、实现 run_detail、用 set_up 配置 params/basedir/code_file_list、保存或读取 simulation_results 和 analysis_results、通过 set_timedir 复用已有结果、做参数搜索复用、运行 analyze/analyze_detail，或用 MetaModelContainer 批量加载旧模型目录时触发。若任务使用新版 AbstractTool/Experiment/ComposedExperiment tool 流程，改用 cf-experiment，不要混用两套框架。
---

# CF MetaModel 约定

这个 skill 用于维护仍基于 `common_functions.py` 的 `MetaModel` 老式模型框架的项目。它强调单个模型对象的参数管理、结果保存、目录复用、analysis 追加和批量加载。新项目优先考虑 `cf-experiment`；只有已有代码已经继承 `cf.MetaModel`，或者需要读取旧结果目录时，才使用本 skill。

## 适用边界

`cf-metamodel` 负责：

- `MetaModel` 子类结构、`set_up`、`run` / `simulate` 和 `run_detail`。
- `simulation_results`、`analysis_results` 的写入、保存、加载和按 key 读取。
- `set_timedir` 读取已有模型目录，并在已有 simulation 上单独追加 analysis。
- `value_dir_key_before`、`both_dir_key_before`、`value_dir_key_after`、`both_dir_key_after`、`ignore_key_list` 对目录和参数搜索的影响。
- `force_run`、参数搜索复用和 incomplete model 检查。
- `MetaModelContainer` 批量加载一个目录下的多个旧模型。

不要在同一个工作流里混用 `MetaModel` 和新版 `AbstractTool` / `Experiment` / `ComposedExperiment`。如果代码已经是 tool pipeline，使用 `cf-experiment`；如果只是 DataKeeper 后端、key 设计或 lmdb/separate 保存策略问题，使用 `cf-data-keeper`。

不要修改 `/data/zyjin/common_func/common_functions.py` 来适配局部项目。需要兼容时，在当前项目内写 wrapper 或子类。

## 核心结构

`MetaModel` 是单对象模型框架。典型生命周期是：

1. 实例化子类，默认初始化 `simulation_results` 和 `analysis_results`。
2. 调用 `set_up(params, basedir, code_file_list, ...)` 设置参数、目录、代码备份列表和搜索配置。
3. 调用 `run()` 或 `simulate()`。
4. 框架先 `search_params()`，若已有完整 simulation 结果则复用；否则调用 `initialize_model()`、子类 `run_detail()`，再保存 simulation。
5. 需要分析时调用 `set_analysis_params(...)` 后 `analyze()`，由子类 `analyze_detail()` 写入 `analysis_results` 并保存。

`run_detail` 只负责核心 simulation 的包装：从 `self.params` 或参数属性取输入，调用普通计算函数，把结果写入 `self.simulation_results`。不要把大量数学逻辑直接堆进 `run_detail`；核心计算应可脱离 `MetaModel` 单独测试。

## 子类模板

```python
import sys
sys.path.append('/data/zyjin/common_func')

import numpy as np
import common_functions as cf

def get_simulation_results(seed, n, scale):
    """Return simulation samples with shape (n,) and dtype float."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=n) * scale

def get_analysis_results(samples):
    """Return summary statistics for a 1D numeric array."""
    return {
        'mean': float(np.mean(samples)),
        'std': float(np.std(samples)),
    }

class MyModel(cf.MetaModel):
    def set_model_name(self, model_name=None):
        self.model_name = 'my_model' if model_name is None else model_name

    def set_optional_params_default(self):
        super().set_optional_params_default()
        self.value_dir_key_before = ['dataset']
        self.both_dir_key_after = ['seed']
        self.ignore_key_list = ['note']

    def run_detail(self):
        samples = get_simulation_results(self.seed, self.n, self.scale)
        self.simulation_results['samples'] = samples

    def analyze_detail(self):
        if self.whether_run_this_analysis_task('summary'):
            samples = self.get_simulation_value('samples')
            self.analysis_results['summary'] = get_analysis_results(samples)
```

运行：

```python
params = {
    'dataset': 'demo',
    'seed': 0,
    'n': 1000,
    'scale': 1.0,
    'note': 'debug label ignored in search',
}

model = MyModel()
model.set_up(
    params=params,
    basedir='../../results/my_project',
    code_file_list=[cf.current_file()],
)
model.run()
```

`set_params` 会把 `params` 中的 key 同步成属性，所以 `run_detail` 可以使用 `self.seed`、`self.n`、`self.scale`。如果参数名可能覆盖已有属性，应先换名，避免隐藏框架属性。

## 参数与目录

`set_up` 是常规入口：

```python
model.set_up(
    params=params,
    basedir=basedir,
    code_file_list=[cf.current_file()],
    value_dir_key_before=['dataset'],
    both_dir_key_after=['seed'],
    ignore_key_list=['note'],
    force_run=False,
)
```

目录结构通常是：

```text
basedir / value_or_both_dir_before... / current_time / value_or_both_dir_after... /
    code/
    logs/
    params/
    outcomes/
    figs/
    models/
```

`value_dir_key_before` 和 `both_dir_key_before` 影响搜索根目录之前的分层；`value_dir_key_after` 和 `both_dir_key_after` 会追加在时间目录之后。`ignore_key_list` 只用于参数搜索忽略不影响结果的键，不代表这些键一定不会保存。`force_run=True` 会跳过复用并强制开新结果。

如果希望复用固定时间字符串，可以先设置：

```python
model = MyModel()
model.set_current_time('2025_11_11_12_00_00')
model.set_up(params, basedir, [cf.current_file()])
model.run()
```

## 运行与复用

`run()` 会自动：

- 搜索相同参数的旧目录；
- 检查 `outcomes/simulation_results/simulation_results_saved` 标记；
- 初始化 logger；
- 备份 `code_file_list` 和 common functions 中的 `.py` 文件；
- 必要时执行 `run_detail()`；
- 保存 `simulation_results` 并写完成标记。

读取已有结果目录时，不要重新 `set_up` 后猜路径，直接使用 `set_timedir`：

```python
model = MyModel()
model.set_timedir('../../results/my_project/demo/2025_11_11_12_00_00/seed_0')
samples = model.get_simulation_value('samples')
```

`set_timedir` 会加载 `params`、`info_container`，并从保存结果恢复 simulation / analysis 的配置。它适合已有 simulation 上追加 analysis 或做只读检查。

## 结果保存

默认 `simulation_results` 和 `analysis_results` 是 `dict`，默认保存方式是 `separate`。少量结果直接写 key：

```python
self.simulation_results['samples'] = samples
self.analysis_results['summary'] = summary
```

需要参数化大量结果时，在子类中改成 `OrderedDataContainer`：

```python
def set_optional_params_default(self):
    super().set_optional_params_default()
    self.set_simulation_results(results_type='OrderedDataContainer', param_order=['seed', 'trial'])
    self.set_analysis_results(results_type='OrderedDataContainer', param_order=['metric', 'seed'])
```

读取接口：

```python
samples = model.get_simulation_value('samples')
summary = model.get_analysis_value('summary')
subset = model.get_analysis_subcontainer(metric='loss')
```

`get_*_subcontainer` 只支持 `OrderedDataContainer`。如果只加载部分 key，可向 `load_simulation_results(...)` 或 `load_analysis_results(...)` 传递底层 load 函数支持的参数，例如 `key_to_load`。

## Analysis 流程

analysis 参数不要放进 simulation 的 `params`，否则会污染参数搜索。用 `set_analysis_params`：

```python
model = MyModel()
model.set_timedir(timedir)
model.set_analysis_task(['summary'])
model.set_analysis_params({'window': 50})
model.analyze()
```

子类如果重写 `set_analysis_params`，必须确保调用或等效执行 `_analysis_params_setted()`，否则 `analyze()` 会报错。`analyze_detail` 中用 `whether_run_this_analysis_task(task)` 控制任务开关；默认任务列表是 `['all']`。

analysis 通常应该读取已有 simulation，写入 `analysis_results`，不应重新运行 simulation。非致命绘图或分析问题应尽量记录并保留可用结果；核心 shape、dtype、非法输入或不可恢复数值问题应直接 `raise`。

## 批量管理

`MetaModelContainer(model_class, dir_before_timedir, dir_after_timedir=None)` 用于从同一个 `dir_before_timedir` 批量加载多个模型。容器初始化会调用 `find_incomplete_model`，跳过没有 `simulation_results_saved` 标记的不完整 simulation。`dir_before_timedir` 应指向包含多个时间目录的上级目录；如果结果还有 time 后缀目录，用 `dir_after_timedir` 指定。

```python
container = cf.MetaModelContainer(
    model_class=MyModel,
    dir_before_timedir='../../results/my_project/demo',
    dir_after_timedir='seed_0',
)

count = container.count_params_by_key('scale')
close_items = container.get_close_items_info_by_func(
    func=lambda model: model.get_analysis_value('summary')['mean'],
    target_value=0.0,
    num=5,
)
```

`count_params_by_key(key)` 会统计 `model.params[key]` 的取值次数。`get_close_items_info_by_func(func, target_value, num=5)` 会找最接近目标值的模型，并在结果字典中附加对应 `timedir`。`func` 返回值可以是标量、`list` / `tuple` 或 `np.ndarray`，距离分别按绝对值或向量范数计算。

`MetaModelContainer` 继承 `InstanceContainer`，因此可直接使用以下通用能力：

- `len(container)`、iteration、indexing：读取容器大小、遍历和按下标取 model。
- `append(item)`、`extend(items)`：追加单个或多个实例。
- `get_filtered_by_func(func)`、`get_filtered_by_attr(**attributes)`：返回新的基础 `InstanceContainer`，不修改原容器。
- `inplace_filter_by_func(func)`、`inplace_filter_by_attr(**attributes)`、`inplace_sort_by_func(key_func, reverse=False)`、`inplace_sort_by_attr(attr, reverse=False)`：原地修改并返回自身，支持链式调用。
- `get_info_by_attr(attribute)`、`get_info_by_func(func)`：从每个实例提取信息列表。
- `get_grouped_container_list_by_func(func)`：按字符串或数字 key 分组，返回排序后的 key 列表和对应 `InstanceContainer` 列表。
- `get_close_items_by_func(func, target_value, num=5)`、`get_close_items_info_by_func(func, target_value, num=5)`：找最接近目标值的实例及其信息。
- `get_one_param_and_property(...)`、`get_two_property(...)`、`get_two_param_and_property(...)`：做参数-指标或指标-指标统计，可传 `save_dir` 保存结果。
- `visualize_one_param_and_property(ax, ...)`、`visualize_two_property_scatter(ax, ...)`、`visualize_two_param_and_property_heatmap(ax, ...)`、`visualize_two_param_and_property_fix_one(ax, ...)`：绘图函数均传入 `ax`，内部使用 common plotting 风格函数。

## 调试与验证

最小检查优先：

- 子类能实例化：`model = MyModel()`。
- `params` 中所有 `run_detail` 使用的 key 都能变成属性。
- 小规模 `params` 可以完成一次 `model.run()`。
- 重复同一 `params` 时能复用已有目录，而不是无意义重跑。
- `set_timedir(timedir)` 后能读取 `params` 和关键 `simulation_results`。
- analysis 单独运行时先 `set_timedir`，再 `set_analysis_params` 和 `analyze()`。

排查路径问题时打印或检查 `model.timedir`、`model.params_dir`、`model.outcomes_dir`。不完整结果目录可以用 `find_incomplete_model(dir_before_timedir, dir_after_timedir)` 只读查看；`clean_incomplete_model` 会删除文件，除非 doctor 明确要求，不要主动运行。
