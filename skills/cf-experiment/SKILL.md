---
name: cf-experiment
description: 在 common_func 项目中需要组织可复现实验流程、编写 AbstractTool/Experiment/ComposedExperiment 子类、配置 tool_params_dict/experiment_params_dict、复用已有 timedir、跳过已完成结果、串联多个依赖实验，或使用 common_functions.py 中 Experiment 与 ComposedExperiment 框架时触发。
---

# CF Experiment 约定

这个 skill 用于在项目中使用 `/data/zyjin/common_func/common_functions.py` 的 `AbstractTool`、`Experiment` 和 `ComposedExperiment` 组织实验。它适合把计算、分析、作图、保存和复用结果纳入统一目录结构，而不是在脚本里手写一串无状态函数。

## 核心结构

一个 `Experiment` 包含多个 tool。每个 tool 继承 `cf.AbstractTool` 或其子类，并定义 `run_detail`；`Experiment` 子类只负责设置实验名和 tool 顺序。

```python
import sys
sys.path.append('/data/zyjin/common_func')
import numpy as np
import common_functions as cf

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
        rng = np.random.default_rng(self.seed)
        result = rng.normal(size=self.n) * self.scale
        self.data_keeper.set_value(result, key='result')

class MyExperiment(cf.Experiment):
    def _set_name(self):
        self.name = 'my_experiment'

    def _minimal_init_tools(self):
        self.tools = [SimTool()]
```

`tool.name` 必须唯一，并应与后续访问属性一致。运行或加载后可以通过 `experiment.simulator` 访问该 tool。不要在 `_minimal_init_tools` 里手动写 `self.simulator = SimTool()`，只需要把 tool 实例放进 `self.tools`。

## Tool 编写规则

结果相关参数写在 `tool_params_dict[tool_name]` 中。参数通常会自动变成 tool 属性，例如 `self.seed`。

结果无关的运行配置写在 `tool_config_dict[tool_name]` 中，例如 `process_num`、`task_list`、`already_done`、`enable_delete_after_pipeline`、`threshold_gb`。不要把这些配置混入结果参数，否则参数搜索和复用会失真。

目录规则不要通过 `tool_config_dict` 临时修改；普通目录规则应在 tool 子类的 `_config_dir_manager` 中设置。`tool_config_dict` 只用于运行配置，例如 `process_num`、`task_list`、`already_done`、`enable_delete_after_pipeline`。

常用基类：

- `cf.Simulator` / `cf.Trainer`: 默认 `enable_search=True`、`enable_skip=True`，适合可缓存的核心计算。
- `cf.Analyzer` / `cf.Visualizer`: 继承 `FlexibleTool`，默认 `enable_try=True`、`task_list=['all']`，适合可选择执行的分析和绘图任务。
- `cf.NoSkipTool`: 用于传递无法保存或每次都必须重跑的对象。

tool 间传递数据时，后续 tool 中使用 `{previous_tool_name}_data_keeper`、`{previous_tool_name}_params` 访问前面 tool 的结果和参数：

```python
class AnalyzeTool(cf.Analyzer):
    def _set_name(self):
        self.name = 'analyzer'

    def _set_required_key_list(self):
        super()._set_required_key_list()

    def _set_optional_key_value_dict(self):
        super()._set_optional_key_value_dict()

    def run_detail(self):
        result = self.simulator_data_keeper.get_value('result')
        self.data_keeper.set_value(np.mean(result), key='mean')
```

## 目录与保存

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

把第一个 tool 作为当前 experiment 的核心结果身份：它负责决定参数复用和 `timedir`。如果希望参数影响目录层级，在第一个 tool 的 `_config_dir_manager` 中配置：

```python
def _config_dir_manager(self):
    super()._config_dir_manager()
    self.dir_manager_kwargs['value_dir_key_before'].extend(['dataset'])
    self.dir_manager_kwargs['both_dir_key_after'].extend(['seed'])
    self.dir_manager_kwargs['ignore_key_list'].extend(['process_num'])
```

`value_dir_key_*` 只把参数值放入路径，`both_dir_key_*` 把参数名和值都放入路径。`ignore_key_list` 用于参数搜索时忽略不影响结果的键。不要让后续 analyzer 或 visualizer 的参数决定整个 experiment 的 `timedir`；如果某个分析会改变核心结果，应把它做成独立 experiment 的第一个 tool。

后续 tool 的参数会被保存，但不应用来区分整个 experiment 的核心结果。若后续 analyzer 或 visualizer 的参数变化需要重跑，优先显式调用 `re_run_tool_in_experiment`、`force_run`，或把该步骤拆成后续 experiment 的第一个 tool。

tool 结果通过 `self.data_keeper.set_value(value, key=...)` 写入，默认保存到 `outcomes/{tool_name}`。需要保存图时优先使用 `Visualizer.auto_save_fig`，它会保存到 `figs/` 下并按调用函数名组织路径。推荐显式传入 `fig`；如果不传 `filename`，同时传 `add_to_filename_dict={}` 或具体参数字典。

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

如果第一个 tool 开启 `enable_search`，重复运行相同参数时会优先复用已完成结果。若已有结果不完整，会创建新的 `timedir`。

指定固定时间目录：

```python
experiment.set_current_time('2025_11_11_12_00_00')
```

这会禁用所有 tool 的参数搜索，并把结果写入指定 `current_time` 对应目录。只加载已有 experiment 时使用：

```python
experiment = MyExperiment()
experiment.load('../../results/my_project/2025_11_11_12_00_00')
result = experiment.simulator.data_keeper.get_value('result')
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

## ComposedExperiment 的参数复用

`ComposedExperiment` 会让后续 experiment 的复用判断包含前置 experiment 的核心参数。通常不需要手动把前置 experiment 的参数塞进下游 `experiment_params_dict`；只需要在下游 tool 里通过 `{previous_experiment_name}_data_keeper_dict` 读取前置结果。

如果外部还要复用原始 `experiment_params_dict`，建议在传给 `set_experiment_params_dict` 前先复制一份。

重要假设：每个 experiment 中只有第一个 tool 的参数决定该 experiment 的核心输出，第二个及后续 tool 主要用于分析和作图。若这个假设不成立，应调整 experiment 划分，把会影响下游结果的步骤放到独立 experiment 的第一个 tool。

加载已有 composed experiment 时，传入最后一个 experiment 的 `timedir`：

```python
composed_experiment = MyComposedExperiment()
composed_experiment.load('../../results/my_project/plot/2025_11_11_12_00_00')
value = composed_experiment.fit.trainer.data_keeper.get_value('weight')
```

加载时传入最后一个 experiment 的 `timedir` 即可，前置 experiment 会自动按记录的目录恢复。

## 调试与验证

先用小参数和 `process_num=1` 跑通单个 tool，再跑完整 experiment；确认目录、参数文件、`outcomes/`、`logs/`、`code/` 都符合预期后，再提高并行度或批量运行。

调试单个下游 tool 时，可以先 `experiment.load(timedir)`，再用 `cf.re_run_tool_in_experiment(experiment, tool_name, task_list, tool_params)` 或直接调用 tool 的 `force_run`。composed experiment 对应使用 `cf.re_run_tool_in_composed_experiment`。

不要多个 worker 同时写同一个 `DataKeeper` 路径。需要并行时，让每个 task 生成独立 key，或者在 tool 内用 common functions 的多进程封装先收集结果，再由主流程统一写入 `data_keeper`。

## 使用原则

把核心计算、分析和绘图拆成不同 tool；tool 的 `run_detail` 只做本 tool 的工作，保存通过 `data_keeper` 或绘图保存接口完成。

让第一个 tool 的参数代表当前 experiment 的可复用身份。影响结果的参数进入 `tool_params_dict`，只影响运行方式的配置进入 `tool_config_dict`。

`set_code_file_list([cf.current_file(), ...])` 应包含当前脚本和项目内关键依赖文件，便于之后追溯结果对应的代码。

如果发现框架接口不适合当前项目，在项目内写薄 wrapper 或子类适配；不要直接修改 `/data/zyjin/common_func/common_functions.py`，除非 doctor 明确要求维护 common functions 本身。
