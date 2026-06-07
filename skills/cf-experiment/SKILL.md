---
name: cf-experiment
description: 在 common_func 项目中需要组织 AbstractTool/Experiment/ComposedExperiment 实验流程、定义 tool 顺序、配置 tool_params_dict/experiment_params_dict、传递前置 tool 结果、复用已有 timedir、skip/search 已完成结果、串联依赖 experiment，或调试完整 experiment 数据流时触发。若问题主要是 DataKeeper 的 data_type/save_load_method/key 设计/separate/lmdb 选择，改用 cf-data-keeper。
---

# CF Experiment 约定

这个 skill 用于在项目中使用 `common_functions.py` 的 `AbstractTool`、`Experiment` 和 `ComposedExperiment` 组织可复现实验。重点是 tool 编排、数据传递、目录管理、结果复用、整体运行和调试。

## 适用边界

`cf-experiment` 负责：

- `AbstractTool` / `Experiment` / `ComposedExperiment` 的职责划分和子类写法。
- `tool_params_dict`、`tool_config_dict`、`experiment_params_dict`、`experiment_config_dict` 的位置。
- tool 之间通过 `{previous_tool_name}_data_keeper`、`{previous_tool_name}_params` 传递数据。
- 共享 `dir_manager`、`timedir`、`params/`、`outcomes/`、`logs/`、`code/`、`figs/` 等目录约定。
- `enable_search`、`enable_skip`、`load`、`force_run`、`re_run_tool_in_experiment` 的使用边界。

`data_keeper` 的详细选择不在本 skill 展开。涉及 `data_type`、`save_load_method`、`OrderedDataContainer`、`lmdb/separate`、key 设计、局部读取策略时，使用 `cf-data-keeper`。在 experiment 内只需要知道默认是 `dict + separate`，并且需要改时在 tool 的 `_config_data_keeper` 中配置 `self.data_keeper_kwargs`。

## 核心结构

一个 `Experiment` 包含多个 tool。每个 tool 继承 `cf.AbstractTool` 或其子类，并定义 `run_detail`；`Experiment` 子类只设置实验名和 tool 顺序。

第一个 tool 决定当前 experiment 的核心结果身份：它负责参数搜索、`timedir` 复用和主要目录层级。后续 analyzer / visualizer 的参数会被保存，但不应决定整个 experiment 的核心 `timedir`。如果某个后续步骤的参数会改变下游核心结果，应把它拆成独立 experiment 的第一个 tool。

tool 名称必须唯一。运行或加载后，framework 会按 `tool.name` 把 tool 注入为 experiment 属性，例如 `experiment.simulator`。不要在 `_minimal_init_tools` 里手动写 `self.simulator = SimTool()`；只需要把 tool 实例放进 `self.tools`。

## run_detail 边界

核心计算逻辑必须封装成普通函数，让它可以脱离 experiment 单独运行、复用和测试。`run_detail` 只做 wrapper：

- 从 `self.params` 属性、前置 `data_keeper` 或前置 params 取输入。
- 调用普通核心函数，例如 `get_simulation_results(...)`、`get_analysis_results(...)`。
- 把结果写入当前 `self.data_keeper`。
- 做必要的数据传递或轻量日志，不在里面堆大段核心数学逻辑。

最小模式：

```python
import sys
sys.path.append('<common_func_root>')
import numpy as np
import common_functions as cf

def get_simulation_results(seed, n, scale):
    rng = np.random.default_rng(seed)
    return rng.normal(size=n) * scale

def get_analysis_results(result):
    return {
        'mean': float(np.mean(result)),
        'std': float(np.std(result)),
    }

class SimTool(cf.Simulator):
    def _set_name(self):
        self.name = 'simulator'

    def _set_required_key_list(self):
        super()._set_required_key_list()
        self.required_key_list.extend(['seed', 'n'])

    def _set_optional_key_value_dict(self):
        super()._set_optional_key_value_dict()
        self.optional_key_value_dict.update({'scale': 1.0})

    def run_detail(self):
        result = get_simulation_results(self.seed, self.n, self.scale)
        self.data_keeper.set_value(result, key='result')

class AnalyzeTool(cf.Analyzer):
    def _set_name(self):
        self.name = 'analyzer'

    def _set_required_key_list(self):
        super()._set_required_key_list()

    def _set_optional_key_value_dict(self):
        super()._set_optional_key_value_dict()

    def run_detail(self):
        result = self.simulator_data_keeper.get_value(key='result')
        analysis = get_analysis_results(result)
        self.data_keeper.set_value(analysis, key='summary')

class MyExperiment(cf.Experiment):
    def _set_name(self):
        self.name = 'my_experiment'

    def _minimal_init_tools(self):
        self.tools = [SimTool(), AnalyzeTool()]
```

## 参数和配置

结果相关参数写在 `tool_params_dict[tool_name]` 中。参数通常会自动变成 tool 属性，例如 `self.seed`、`self.n`。

结果无关的运行配置写在 `tool_config_dict[tool_name]` 中，例如 `process_num`、`task_list`、`already_done`、`enable_delete_after_pipeline`、`threshold_gb`。不要把这些配置混入结果参数，否则参数搜索和复用会失真。

目录规则一般放在 tool 子类的 `_config_dir_manager` 中；`tool_config_dict` 只作为运行配置入口。常用基类：

- `cf.Simulator` / `cf.Trainer`: 默认 `enable_search=True`、`enable_skip=True`，适合可缓存的核心计算。
- `cf.Analyzer` / `cf.Visualizer`: 继承 `FlexibleTool`，默认 `enable_try=True`、`task_list=['all']`，适合可选择执行的分析和绘图任务。
- `cf.NoSkipTool`: 用于传递无法保存或每次都必须重跑的对象。

tool 间传递数据时，后续 tool 使用 `{previous_tool_name}_data_keeper`、`{previous_tool_name}_params` 访问前面 tool 的结果和参数。前置 `data_keeper` 在传播后是 read-only，下游只读前置结果，新结果写入当前 `self.data_keeper`。

## data_keeper 最小规则

tool 结果通过 `self.data_keeper.set_value(value, key=...)` 写入，默认保存到 `outcomes/{tool_name}`。默认配置是：

```python
self.data_keeper_kwargs = {'data_type': 'dict', 'save_load_method': 'separate'}
```

如果只保存普通少量结果，用默认 `dict`。如果保存大量参数化结果，用 `OrderedDataContainer`。如果确实需要单文件数据库式存储，再考虑 `lmdb`。具体 key 设计、后端选择和读取策略查 `cf-data-keeper`。

在 tool 内需要改配置时：

```python
def _config_data_keeper(self):
    super()._config_data_keeper()
    self.data_keeper_kwargs = {
        'data_type': 'OrderedDataContainer',
        'save_load_method': 'separate',
        'param_order': ['dataset', 'seed'],
    }
```

## 目录与复用

每个 experiment 会自动管理结果目录。目录结构通常是：

```text
basedir / value_dir_key_before... / current_time / value_dir_key_after... /
    code/
    logs/
    params/
    outcomes/
    figs/
    models/
```

如果希望参数影响目录层级，在第一个 tool 的 `_config_dir_manager` 中配置：

```python
def _config_dir_manager(self):
    super()._config_dir_manager()
    self.dir_manager_kwargs['value_dir_key_before'].extend(['dataset'])
    self.dir_manager_kwargs['both_dir_key_after'].extend(['seed'])
    self.dir_manager_kwargs['ignore_key_list'].extend(['process_num'])
```

`value_dir_key_*` 只把参数值放入路径，`both_dir_key_*` 把参数名和值都放入路径。`ignore_key_list` 用于参数搜索时忽略不影响结果的键。

需要保存图时优先使用 `Visualizer.auto_save_fig`，它会保存到 `figs/` 下并按调用函数名组织路径。推荐显式传入 `fig`；如果不传 `filename`，同时传 `add_to_filename_dict={}` 或具体参数字典。

## 运行 Experiment

典型运行方式：

```python
basedir = '../../results/my_project'

tool_params_dict = {
    'simulator': {'seed': 0, 'n': 1000, 'scale': 1.0},
    'analyzer': {},
}

tool_config_dict = {
    'analyzer': {'task_list': ['all']},
}

experiment = MyExperiment()
experiment.set_tool_params_dict(tool_params_dict)
experiment.set_tool_config_dict(tool_config_dict)
experiment.set_basedir(basedir)
experiment.set_code_file_list([cf.current_file()])
experiment.run()
```

如果第一个 tool 开启 `enable_search`，重复运行相同参数时会优先复用已完成结果。若已有结果不完整，会创建新的 `timedir`。`set_code_file_list([cf.current_file(), ...])` 应包含当前脚本和项目内关键依赖文件，便于之后追溯结果对应的代码。

指定固定时间目录：

```python
experiment.set_current_time('2025_11_11_12_00_00')
```

这会禁用所有 tool 的参数搜索，并把结果写入指定 `current_time` 对应目录。只加载已有 experiment 时使用：

```python
experiment = MyExperiment()
experiment.load('../../results/my_project/2025_11_11_12_00_00')
result = experiment.simulator.data_keeper.get_value(key='result')
```

`load` 用于读取已有实验目录。加载后可以直接通过各 tool 的 `params` 和 `data_keeper.get_value(...)` 使用已保存结果，通常不会重新计算。

## 使用 ComposedExperiment

当多个 experiment 存在明确前后依赖时，用 `ComposedExperiment`；如果彼此独立，应创建多个普通 `Experiment` 实例，而不是强行 compose。

```python
class MyComposedExperiment(cf.ComposedExperiment):
    def _minimal_init_experiments(self):
        self.experiments = [PreprocessExperiment(), FitExperiment(), PlotExperiment()]
```

运行方式：

```python
experiment_params_dict = {
    'preprocess': {
        'loader': {'dataset': 'demo'},
    },
    'fit': {
        'trainer': {'lr': 0.001},
    },
    'plot': {
        'visualizer': {},
    },
}

experiment_config_dict = {
    'plot': {
        'visualizer': {'task_list': ['plot_summary']},
    },
}

composed_experiment = MyComposedExperiment()
composed_experiment.set_experiment_params_dict(experiment_params_dict)
composed_experiment.set_experiment_config_dict(experiment_config_dict)
composed_experiment.set_basedir('../../results/my_project')
composed_experiment.set_code_file_list([cf.current_file()])
composed_experiment.run()
```

`ComposedExperiment` 会把每个 experiment 放到 `basedir/{experiment.name}` 下，并按顺序运行。前一个 experiment 结束后，后续 experiment 可以通过 `{previous_experiment_name}_data_keeper_dict` 和 `{previous_experiment_name}_params_dict` 访问其所有 tool 的结果和参数。

`ComposedExperiment` 会让后续 experiment 的复用判断包含前置 experiment 的核心参数。通常不需要手动把前置 experiment 的参数塞进下游 `experiment_params_dict`；只需要在下游 tool 里通过 `{previous_experiment_name}_data_keeper_dict` 读取前置结果。

如果外部还要复用原始 `experiment_params_dict`，建议在传给 `set_experiment_params_dict` 前先复制一份。

重要假设：每个 experiment 中只有第一个 tool 的参数决定该 experiment 的核心输出，第二个及后续 tool 主要用于分析和作图。若这个假设不成立，应调整 experiment 划分，把会影响下游结果的步骤放到独立 experiment 的第一个 tool。

加载已有 composed experiment 时，传入最后一个 experiment 的 `timedir`：

```python
composed_experiment = MyComposedExperiment()
composed_experiment.load('../../results/my_project/plot/2025_11_11_12_00_00')
value = composed_experiment.fit.trainer.data_keeper.get_value(key='weight')
```

加载时传入最后一个 experiment 的 `timedir` 即可，前置 experiment 会自动按记录的目录恢复。

## 调试与验证

调试的最小单位应是完整 `Experiment` 或完整 `ComposedExperiment`，而不是直接实例化某个下游 tool 单跑。原因是 tool 之间依赖 `{previous_tool_name}_data_keeper`、`{previous_tool_name}_params`、共享 `dir_manager`、同一个 `timedir`、已保存 params 和 code/log/outcomes 目录。

推荐流程：

1. 用很小参数跑完整 smoke test，例如最小样本数、`process_num=1`、少量 task。
2. 检查 `params/`、`outcomes/`、`logs/`、`code/`、`figs/` 是否生成在预期位置。
3. 检查下游 tool 是否能通过前置 `data_keeper.get_value(...)` 读取结果。
4. 重新运行相同参数，确认第一个 tool 的 search/skip 复用符合预期。
5. 用 `Experiment.load(timedir)` 或 `ComposedExperiment.load(last_timedir)` 恢复已有结果，并再次读取关键输出。

`re_run_tool_in_experiment`、`re_run_tool_in_composed_experiment` 和 tool 的 `force_run` 只作为整体跑通后定位、修复 analyzer / visualizer 的手段，不作为首选调试方式：

```python
experiment.load(timedir)
cf.re_run_tool_in_experiment(
    experiment,
    tool_name='analyzer',
    task_list=['all'],
    tool_params={},
)
```

composed experiment 对应：

```python
composed_experiment.load(last_timedir)
cf.re_run_tool_in_composed_experiment(
    composed_experiment,
    experiment_name='plot',
    tool_name='visualizer',
    task_list=['plot_summary'],
    tool_params={},
)
```

## 测试边界

`cf-experiment` 的测试目标不是验证核心计算数学正确性。核心函数如 `get_simulation_results(...)`、`get_analysis_results(...)` 的单元测试应放在对应计算模块或项目测试中。

本 skill 下更应该测试：

- tool 间数据传递和前置 `data_keeper` 读取。
- 参数保存和加载后 `params` 恢复。
- `params/`、`outcomes/`、`logs/`、`code/`、`figs/` 目录生成。
- 相同参数下的 search/skip 复用。
- `load` 后不重算也能读取结果。
- `ComposedExperiment` 前后依赖和前置 `timedir` 恢复。

## 使用原则

把核心计算、分析和绘图拆成不同 tool；tool 的 `run_detail` 只做本 tool 的 framework wrapper，核心逻辑放到普通函数中。

让第一个 tool 的参数代表当前 experiment 的可复用身份。影响结果的参数进入 `tool_params_dict`，只影响运行方式的配置进入 `tool_config_dict`。

如果发现框架接口不适合当前项目，在项目内写薄 wrapper 或子类适配；不要直接修改 common_func 根目录下的 `common_functions.py`，除非当前任务明确要求维护 common functions 本身。
