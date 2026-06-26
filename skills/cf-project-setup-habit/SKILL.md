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
    .gitignore
    memory.md
    storage.md
    data/
        raw_data/
        processed_data/
    code/
        pre_process/
        simulation/
        analysis/
        visualization/
        utils_function/
        test/
    results/
```

- `README.md`: 简短说明项目目的、主要入口脚本、关键依赖版本、如何复现实验结果。
- `.gitignore`: 至少忽略 `results/` 和不应进入版本控制的大型中间文件、缓存文件、memory、storage。
- `memory.md`: 项目过程记录文件；具体记录规则遵守全局指令。
- `storage.md`: 记录稳定数据路径、结果路径、图路径、日志路径和重要子目录大小，用于维护项目当前最新存储状态。
- `data/raw_data/`: 原始输入、外部数据或不可变输入。
- `data/processed_data/`: 处理后的可复用数据。可继续按数据来源或任务分子目录，例如 `data/processed_data/example_dataset/`。
- `code/`: 所有项目代码。不要把核心脚本散放在项目根目录。
- `results/`: 仿真、分析、绘图和中间结果输出。结果应按 workflow 和任务继续分层，例如 `results/simulation/example_task/`。

允许某些子目录暂时为空，但目录职责应清楚。所有 `.py` 脚本必须放在对应子文件夹内，不要直接放在 `code/` 下；如有需要可以新增职责明确的子文件夹。

## Git 和版本控制习惯

- `results/` 必须加入 `.gitignore`，避免昂贵结果、中间结果、图文件和日志默认进入版本控制。
- 大型原始数据、外部数据、缓存、模型权重和临时文件不应直接提交；如果必须引用，应在 `README.md` 或 `storage.md` 中记录来源、路径和获取方式。
- 配置模板可以进入版本控制，机器特定路径、密钥、token 和本地环境文件不应提交。
- 不要依赖未记录的本地文件状态。新建项目时应尽早补齐 `.gitignore`、`README.md`、`memory.md` 和 `storage.md`。

## code 目录习惯

`code/` 下按职责拆分：

```text
code/
    pre_process/
    simulation/
    analysis/
    visualization/
    utils_function/
    test/
```

- `pre_process/`: 随机对象预生成、任务表生成、数据预处理、可复用中间对象生成等代码。
- `simulation/`: 可直接运行的仿真入口脚本，例如 `run_simulation.py`、`run_parameter_sweep.py`。
- `analysis/`: 后处理和统计分析入口脚本。若项目较小，可以先由 simulation 脚本串联分析，再逐步拆出。
- `visualization/`: 独立绘图入口脚本。绘图函数本身仍应与计算逻辑分离。
- `utils_function/`: 当前项目专用 wrapper、领域模型函数、I/O helper、第三方库适配层。
- `test/`: notebooks、smoke tests 或最小复现脚本。正式可重复检查优先使用 `.py` 脚本。

入口脚本推荐命名：

```text
run_preprocess_[object_name].py
run_simulate_[task_name].py
run_analyze_[analysis_name].py
run_plot_[figure_name].py
```

`simulation/`、`analysis/`、`visualization/` 中的脚本应尽量薄，只负责读取参数、调用函数、保存结果和记录日志。可复用函数应放到 `utils_function/` 或等价工具目录中。每个脚本应能从项目根目录运行，避免依赖当前工作目录的隐式假设。

## 工具函数拆分

项目内工具函数通常按层次拆开：

- `{...}_function.py`: 可以按照数据预处理、计算核心、分析结果、绘图等职责分文件；`code/utils_function/` 下这类项目函数文件总数通常不超过 10 个，超过时检查是否拆分过细。
- `{...}_parameter.py`: 记录参数，供入口脚本调用具体函数时读取。参数可以分为计算相关和实验相关：计算相关包括 CPU/GPU、GPU id、进程数量等；实验相关包括随机种子和具体实验参数等。
- `{library_or_framework}_functions.py` 或类似文件: 第三方框架适配层，例如运行环境设置、模型组件封装、外部 API 适配。

`code/utils_function/` 只放项目内可复用函数、项目局部 wrapper、配置读取函数等。不要在这个文件夹里运行主流程。小项目可以先用少量文件承载，避免过早拆成太多模块；大型项目再按职责继续拆分。

入口脚本应保持为 workflow 编排层：

1. 定义参数。
2. 加载数据。
3. 调用构建、运行、分析、绘图函数。
4. 把结果写入 `results/` 下的任务目录。

不要把大量核心数学逻辑直接堆在入口脚本中。

## 项目配置习惯

项目固定路径、外部工具路径、数据根目录、环境名称等信息，应集中写在配置文件或项目说明中，不要散落在脚本内部。

推荐配置位置：

```text
code/utils_function/config.py
```

基础配置项：

```text
PROJECT_ROOT = [项目根目录，优先自动推断]
DATA_ROOT = data/
RAW_DATA_DIR = data/raw_data/
PROCESSED_DATA_DIR = data/processed_data/
RESULTS_ROOT = results/
CONDA_ENV_NAME = [项目指定 conda 环境名称；新项目创建时必须明确。如果 doctor 尚未提供，应先询问；doctor 回答后写入 config.py 或 README/storage.md 中的项目环境记录。]
```

脚本中只读取配置，不要在多个脚本中重复硬编码路径。如果配置缺失，应先报告缺失项；在不影响安全性和可追溯性的情况下，可以给出合理默认方案。

默认使用项目指定环境运行代码和安装依赖。运行脚本、安装依赖、测试代码前，必须确认当前环境是否为项目指定环境。若项目尚未记录指定环境，应先询问 doctor；doctor 明确后，将环境名称记录到 `config.py` 或项目说明文件中。不要未经确认自行创建新环境，也不要未经确认把依赖安装到系统 Python 或非项目环境。

## 数据和结果路径

路径习惯：

- 原始输入数据放 `data/raw_data/{dataset_or_task}/`。
- 处理后可复用数据放 `data/processed_data/{dataset_or_task}/`。
- 结果放 `results/{workflow}/{task}/`。
- 每次重要运行使用时间戳目录或参数目录，避免覆盖旧结果。
- 单次任务目录下可继续拆成 `simulation_results/`、`analysis_results/`、`figs/`、`logs/`、`params/`。
- 每个重要结果文件应包含运行参数 `params` 或旁边保存参数文件，以便追溯。
- 如果某个结果是中间缓存，应在文件名、目录名或 `storage.md` 中说明其用途和可否删除。

## 搭建新项目时的检查清单

- 是否存在 `README.md`，并说明入口脚本、依赖版本和结果复现方式。
- 是否存在 `.gitignore`，并确认 `results/` 和大型缓存不会进入版本控制。
- 是否有清晰的 `data/raw_data/`、`data/processed_data/`、`code/`、`results/` 分层。
- 是否把项目专用工具函数放到 `code/utils_function/` 或等价目录。
- 是否把固定路径、环境名称和外部资源路径集中到配置文件或说明中。
- 是否把仿真、分析、绘图函数拆开。
- 是否设置稳定输出目录，避免覆盖重要结果。
- 是否准备最小 smoke test 或最小运行参数。
- 是否检查新增结果文件、图文件和 task table 的可读取性与路径唯一性。
