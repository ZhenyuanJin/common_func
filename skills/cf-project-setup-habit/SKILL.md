---
name: cf-project-setup-habit
description: 当需要按常见科研计算项目习惯搭建、整理或评审项目结构时触发。适用于从零创建 project skeleton、迁移旧脚本、检查 data/code/results 分层、规划 simulation/analysis/plot/test/utils 目录、补 README、记录路径与结果。
---

# CF Project Setup Habit

这个 skill 简要描述常用的科研计算项目搭建习惯。目标是让新项目容易运行、容易追溯，并且让数据、代码、结果分开管理。

## 推荐根目录

典型项目根目录使用以下结构：

```text
project_root/
    README.md
    data/
    code/
    results/
```

- `README.md`: 简短说明项目目的、主要入口脚本、关键依赖版本、如何复现实验结果。
- `data/`: 原始数据、外部数据、预处理后可复用的数据。按数据来源或任务分子目录，例如 `data/example_dataset/`。
- `code/`: 所有项目代码。不要把核心脚本散放在项目根目录。
- `results/`: 仿真、分析、绘图和中间结果输出。结果应按 workflow 和任务继续分层，例如 `results/simulation/example_task/`。

长期协作项目建议额外维护：

```text
project_root/
    memory.md
    storage.md
```

- `memory.md`: 记录已完成事项、关键参数、验证命令、发现的问题和后续事项。
- `storage.md`: 记录稳定数据路径、结果路径、图路径、日志路径和重要子目录大小。

## code 目录习惯

`code/` 下按职责拆分：

```text
code/
    simulation/
    analysis/
    visualization/
    utils_function/
    test/
```

- `simulation/`: 可直接运行的仿真入口脚本，例如 `run_simulation.py`、`run_parameter_sweep.py`。
- `analysis/`: 后处理和统计分析入口脚本。若项目较小，可以先由 simulation 脚本串联分析，再逐步拆出。
- `visualization/`: 独立绘图入口脚本。绘图函数本身仍应与计算逻辑分离。
- `utils_function/`: 当前项目专用 wrapper、领域模型函数、I/O helper、第三方库适配层。
- `test/`: notebooks、smoke tests 或最小复现脚本。正式可重复检查优先使用 `.py` 脚本。

## 工具函数拆分

项目内工具函数通常按层次拆开：

- `project_helpers.py`: 项目局部常用的小工具、路径、保存读取、绘图基础 helper。
- `utils_function.py`: 项目领域逻辑，包括模型或流程构建、仿真 wrapper、分析函数、可视化调度函数。
- `{library_or_framework}_functions.py` 或类似文件: 第三方框架适配层，例如运行环境设置、模型组件封装、外部 API 适配。

入口脚本应保持为 workflow 编排层：

1. 定义参数。
2. 加载数据。
3. 调用构建、运行、分析、绘图函数。
4. 把结果写入 `results/` 下的任务目录。

不要把大量核心数学逻辑直接堆在入口脚本中。

## 数据和结果路径

路径习惯：

- 输入数据放 `data/{dataset_or_task}/`。
- 结果放 `results/{workflow}/{task}/`。
- 每次重要运行使用时间戳目录或参数目录，避免覆盖旧结果。
- 仿真结果、参数、图文件建议继续拆成 `outcomes/`、`params/`、`figs/`。
- 需要可追溯时，把运行脚本和关键依赖代码保存到结果目录的 `code/` 子目录，或在 `memory.md` 记录代码版本和命令。

## README 最小内容

README 至少写清：

- 项目一句话目标。
- 主要入口脚本及其对应结果，例如运行 `run_simulation.py` 得到 simulation 结果。
- 关键 helper 文件职责。
- 关键依赖版本，尤其是容易破坏兼容性的计算、统计、绘图或数据处理依赖。
- 数据位置和必要外部文件说明。

README 保持短而可执行，不需要写成长文档。

## 验证和运行习惯

正式长任务前先做小规模 smoke test：

- 缩短仿真时间、减少数据规模、减少 seed 或 task 数。
- 检查 import、路径、shape、dtype、保存路径和最小输出图。
- smoke test 通过后再运行完整参数。

耗时任务应估计时间、CPU、内存和存储需求。多进程任务要避免多个 worker 同时写同一个文件，结果汇总应按 task id 排序，保证 single process 和 multiprocessing 的结果一致。

## 绘图习惯

绘图输出放在 `results/` 下的 `figs/` 或任务图目录中。绘图函数必须与计算分析分离：

- 绘图函数名以 `plot_` 开头。
- 如果传入 axes，axes 作为第一个参数。
- 绘图函数内部不做昂贵计算。
- 复用项目内已有的 figure、axes、style 和保存 helper，避免在不同脚本中重复实现绘图基础逻辑。
- 图例和文字标注位置要手动检查，不能只依赖默认 `best`。

如果数据有非致命问题，应尽量先出图，并在最终回复、日志或 `memory.md` 说明可疑点。

## 搭建新项目时的检查清单

- 是否存在 `README.md`，并说明入口脚本、依赖版本和结果复现方式。
- 是否有清晰的 `data/`、`code/`、`results/` 分层。
- 是否把项目专用工具函数放到 `code/utils_function/` 或等价目录。
- 是否把仿真、分析、绘图函数拆开。
- 是否设置稳定输出目录，避免覆盖重要结果。
- 是否准备最小 smoke test 或最小运行参数。
- 长期项目是否需要创建或更新 `memory.md` 和 `storage.md`。
