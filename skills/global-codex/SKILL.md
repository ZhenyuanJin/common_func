---
name: global-codex
description: Global Codex instructions for doctor projects. Use when the user requests the global Codex instruction skill, global_codex.txt, or wants Codex to follow doctor-specific reply style, workflow, common functions, plotting, memory/storage documentation, Python style, randomness, multiprocessing consistency, imports, code organization, and naming rules.
---

# Global Codex Instructions

## Identity and Reply Style

- Every reply must start with: `doctor`.
- 默认使用中文回答，除非用户明确要求英文。
- 代码、命令、配置项、文件名、API 名称、包名和错误信息保持英文原文。
- 最终回复应简洁说明：
  1. 修改了什么；
  2. 运行或检查了什么；
  3. 是否还有风险、未验证点或需要 doctor 亲自决定的事项。

## Workflow

- 修改代码前，必须先阅读相关文件，不要凭空猜测项目结构、函数接口或已有风格。
- 如果项目中存在 `AGENTS.md`、`memory.md`、`storage.md`、`README.md`、`notes.md` 或类似说明文件，开始工作前应先阅读。
- 如果改动会影响其他代码，必须同步检查并修改相关调用处。
- 修改后应尽量运行最相关的测试、lint、脚本或最小检查命令。
- 运行耗时任务前，应先粗略评估运行时间、CPU、内存和存储需求。
- 如果任务预计较久，应优先用小规模参数做 smoke test，并据此估计大规模任务的运行时间，再决定是否运行正式版本。
- 对于需要等待输出的命令，不要频繁中断。应先按预估运行时间 `sleep`，到达预估时间后再读取输出；如果任务仍未完成，则继续等待一整倍预估时间后再检查。例如预估 10 分钟，则先 `sleep 10m`，未完成再 `sleep 10m`。
- 不要主动执行 Git 操作，包括：
  - `git add`
  - `git commit`
  - `git push`
  - `git reset`
  - `git checkout`
  - `git stash`
- 如确实需要 Git 操作，可以提出建议，经过批准后进行，或者在doctor的主动要求下可以git。

## Running and Validation

- 尽量主动运行代码，而不是只写代码不验证。运行代码的一般要附带图片呈现给doctor。
- 优先运行最小可复现检查，再运行完整任务。
- 如果机器资源充足，可以较大方地使用内存和存储；但初步确认代码可行性时，应先使用小规模参数。
- 对核心计算模块，如果发现严重不一致、非法输入、shape 错误、dtype 错误或不可恢复的数值问题，应严格 `raise`，不要静默失败。
- 对绘图模块，应优先尽可能产出图；同时把可疑问题、异常数据或未完全满足的绘图要求记录下来。
- 非致命绘图问题不应导致整轮绘图完全失败，但必须在最终回复或日志中说明。

## Common Functions and Skills

- 优先使用 doctor 的 common functions：`/data/zyjin/common_func`。
- 写相关功能前，看 skills 和 router：

```text
/data/zyjin/common_func/skills
/data/zyjin/common_func/skills/cf-skill-router/SKILL.md
```

- 不要直接修改 common functions 或 skills；如需适配，在当前项目写 wrapper。
- 若发现 bug、接口不合适或结果可疑，报告问题、复现方式和建议修改。
- 若项目局部说明指定其他 common functions 路径或 wrapper，优先遵守局部说明。

## Plotting

- 每一轮任务中，若涉及分析、仿真、模型验证或结果展示，应尽量产出图。图是最重要的输出之一。
- 绘图应该和计算分析模块完全分离，即绘图函数内部不应该承担计算和分析的功能。
- 绘图必须优先使用 common functions 中已有接口，尤其是创建 figure、布局 axes、设置风格和保存图片的接口。
- 应积极并正确使用 common functions 中的相关函数，例如：

```text
get_fig_ax
plt_*
set_ax
save_fig
```

- 如果 common functions 已经提供某功能，不要绕过它重新调用底层 Matplotlib 或重复造轮子。
- 只有在 common functions 无法满足需求时，才可以使用底层 Matplotlib 接口。
- 禁止在绘图函数中使用依赖全局状态的 Pyplot 调用，例如：
  - `plt.plot()`
  - `plt.scatter()`
  - `plt.imshow()`
- 绘图函数名必须以 `plot_` 开头。
- 如果绘图函数需要传入 axes，则 axes 应作为第一个参数。
- 绘图函数只负责画图，不应在内部进行昂贵的数据计算。
- 绘图文件应保存到项目指定输出目录；如果项目未指定，默认保存到 `results/` 下的子目录。
- 对绘图模块，如果数据有问题，应尽可能先出图，同时记录问题；不要因为非致命问题导致整轮绘图完全失败。

## Plot Legend and Annotation Placement

- 每次绘图时，务必主动检查图例与图中文字标注的位置。
- 不要依赖 Matplotlib 的默认位置或 `best` 参数；应根据当前数据分布、点云密度、曲线位置和图的主要信息区域，手动设置合适的位置。
- 若图例或文字标注会遮挡关键数据、拟合线、误差带、峰值、尾部分布等重要结构，应将其移至更空旷的位置。
- 若图内没有足够的空白区域，应将图例或文字标注放到图外。
- 对于包含多个子图的 figure，需逐一检查每个子图的图例与标注位置，不要假设所有子图可以共用同一个位置。
- 应优先根据该类图的常见信息占位区域选择位置，并在保存前确认没有遮挡。

## Memory, Storage and Documentation

- 继续旧项目或重要运行前，先查 `memory.md` 和 `storage.md`，尤其避免重复运行已有昂贵结果。
- `memory.md` 记录已完成事项、关键参数、验证命令、问题和后续事项；每轮结束追加短记录，新增内容不超过约 100 个中文字符。
- `storage.md` 维护稳定数据路径、结果路径、图路径、日志路径和重要子目录大小；它记录当前最新状态，不按日期写流水账。
- 不把短暂、纯调试信息写入 `memory.md` 或 `storage.md`。
- 若项目缺少这些文件但明显需要长期追踪，建议创建；doctor 明确要求记录时，直接创建或更新。

## Python Code Style

- 不使用类型注解。不要写：

```python
def get_x(x: int) -> np.ndarray:
```

而写：

```python
def get_x(x):
```

- 减少注释，让代码通过命名和结构自明。
- 只有在逻辑不容易直接看懂时才添加简短注释。
- 减少不必要的换行；简单函数签名保持单行。
- 函数应尽量纯净，专注计算并返回结果，避免在函数内部写非必要的 `print` 调试输出。
- 日志、进度条、debug 输出应放在顶层脚本，或者由显式 `verbose` 参数控制。
- 代码要保证灵活性，但不要为了灵活性做过度抽象。
- 不要为了速度牺牲可重复性、可审计性和结果可追溯性。
- 不要写重复的代码。Do not repeat yourself。

## Randomness and Reproducibility

- 不要依赖全局随机状态。
- 明确使用 `numpy.random.default_rng` 在局部生成随机数。
- 每个需要随机性的函数都应接收`seed`，在内部生成rng并由此生成随机数。
- 禁止使用：
  - `np.random.seed()`
  - `np.random.rand()`
  - `np.random.randn()`
  - 其他依赖全局随机状态的接口
- 随机对象应尽量显式生成、显式保存、显式传入核心计算函数。
- 不要让随机数消耗顺序依赖任务执行顺序。
- 如果任务需要可重复性，每个 task 的随机性必须由 task identity 决定，而不是由 worker 顺序、完成顺序或随机数消耗顺序决定。

## Single Process and Multiprocessing Consistency

- 对需要可重复性的项目，代码应保证 single process 和 multiprocessing 对同一个 task 产生完全一致的结果。
- 多进程任务中：
  1. worker 不使用全局 `rng`；
  2. worker 返回结果时必须带上 task id；
  3. 主进程按 task id 汇总、排序、保存结果；
  4. 不允许多个 worker 同时写同一个结果文件；
  5. 每个 worker 可以写独立文件，或者由主进程统一写文件；
  6. 结果不得依赖任务完成顺序。
- 优先使用 common functions 中已有的 multiprocessing 支持；如果已有封装能满足需求，不要重复实现。

## Imports

- 不使用 wildcard imports，例如：

```python
from module import *
```

- 避免混乱的二级导入，例如：

```python
from sklearn import manifold
from scipy.io import loadmat
```

- 优先使用顶层或清晰模块导入，例如：

```python
import numpy as np
import scipy
import sklearn.decomposition
```

- 调用函数时尽量使用完整路径，例如：

```python
scipy.io.loadmat(...)
scipy.spatial.distance.pdist(...)
sklearn.decomposition.PCA(...)
```

## Code Organization and Naming

- 将相关函数按主题或工作流顺序组织。
- 较长文件可以使用清晰的分段注释，例如：

```python
# ============================================================
# Random object generation
# ============================================================
```

- 相似功能使用一致命名。
- 读取文件统一使用 `load_` 开头。
- 获取、生成、计算、分析、分类等核心函数统一使用 `get_` 开头。
- 绘图函数统一使用 `plot_` 开头。
- 模块内私有辅助函数可以使用 `_` 前缀，不强制使用 `get_`。
- 所有核心函数应有 docstring，说明输入、输出、shape、dtype 和随机性假设。
- docstring 不应冗长；优秀函数应主要通过命名和结构自明。
