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
- 如确实需要上述 Git 操作，应先说明目的、影响范围和风险；只有在 doctor 明确批准，或 doctor 主动明确要求执行该具体 Git 操作时，才可以执行。
- 创建或修改 `.gitignore`、查看文件内容、说明 Git 状态不属于上述 Git 操作限制。

## Running and Validation

- 尽量主动运行代码，而不是只写代码不验证。
- 优先运行最小可复现检查，再运行完整任务。
- 对于 import、路径、shape、dtype、文件读写、函数返回类型等纯工程检查，可以不强行产图。
- 只要 validation 涉及数值结果、仿真结果、分析结果、模型行为、数据分布、拟合效果、轨迹、矩阵、解码结果、loss 曲线或其他需要 doctor 判断合理性的输出，就应尽量产出检查图。
- smoke test 如果产生了可视化可检查的最小结果，也应保存对应检查图，而不是只报告代码成功运行。
- 若任务涉及分析、仿真、模型验证或结果展示，应尽量产出图或可检查的输出文件，并在最终回复中说明路径。
- 如果机器资源充足，可以较大方地使用内存和存储；但初步确认代码可行性时，应先使用小规模参数。
- 对核心计算模块，如果发现严重不一致、非法输入、shape 错误、dtype 错误或不可恢复的数值问题，应严格 `raise`，不要静默失败。
- 对绘图模块，应优先尽可能产出图；同时把可疑问题、异常数据或未完全满足的绘图要求记录下来。
- 非致命绘图问题不应导致整轮绘图完全失败，但必须在最终回复或日志中说明。

## Validation and Running

正式长任务前先做小规模 smoke test：

- 缩短仿真时间、减少数据规模、减少 seed 或 task 数。
- 检查 import、路径、shape、dtype、保存路径和最小输出图。
- 检查脚本是否可以从项目根目录启动。
- 如果新增结果文件，检查文件是否实际生成、是否能重新读取、关键数组 shape 是否符合预期。
- 如果新增 task table 或批量任务配置，检查 task id 是否唯一、参数是否完整、输出路径是否不会互相覆盖。
- smoke test 通过后再运行完整参数。

耗时任务应估计时间、CPU、内存和存储需求。多进程任务要避免多个 worker 同时写同一个文件，结果汇总应按 task id 排序，保证 single process 和 multiprocessing 的结果一致。

对于需要多次运行的流程，应优先写成可重复执行的脚本，而不是只写一次性 notebook 逻辑。对于参数扫描、批量仿真、批量分析，应优先建立 task table 或明确任务列表，保证后续可追踪、可恢复、可并行。

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
- 绘图应该和计算分析模块分离；绘图函数内部可以进行轻量的数据整理，但不应承担昂贵计算、主要统计分析或核心模型逻辑。
- 绘图必须优先使用 common functions 中已有接口，尤其是创建 figure、布局 axes、设置风格和保存图片的接口。
例如：

```text
get_fig_ax
plt_*
set_ax
save_fig
```
注意，上面的 `plt_*` 指 common functions 中的绘图 helper；禁止项中的 `plt.plot()`、`plt.scatter()`、`plt.imshow()` 指 `matplotlib.pyplot` 的状态式调用
- 如果 common functions 已经提供某功能，不要绕过它重新调用底层 Matplotlib 或重复造轮子。
- 只有在 common functions 无法满足需求时，才可以使用底层 Matplotlib 接口。
- 禁止在绘图函数中使用依赖全局状态的 Pyplot 调用，例如：
  - `plt.plot()`
  - `plt.scatter()`
  - `plt.imshow()`
- 绘图函数名必须以 `plot_` 开头。
- 如果绘图函数需要传入 axes，则 axes 应作为第一个参数。
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
- `memory.md` 记录已完成事项、关键参数、验证命令、问题和后续事项。对代码修改、重要运行、路径变化、结果产出或发现重要问题的任务，每轮结束追加一条短记录，新增内容不超过约 100 个中文字符；微小或不重要改动可以不更新。
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

## Script Style

- 项目入口脚本默认不使用 `argparse` 或其他命令行参数。
- 入口脚本的参数应写在脚本顶部，或从 `*_parameter.py` 等明确参数文件导入。
- 若已有项目已经使用命令行参数，不要在未检查调用方式、README、脚本依赖和批处理流程前改动现有接口。
- 若 doctor 明确要求提供 CLI，或项目局部指令要求 CLI，可以使用 `argparse`，比如利用 `argparse` 修改运行使用的gpu_id。

## Randomness and Reproducibility

- 不要依赖全局随机状态。
- 明确使用 common functions 中的 `get_local_rng`；如果当前项目无法使用 common functions，则使用 `numpy.random.default_rng` 在局部生成随机数。
- `seed` 参数不允许有默认值；需要随机生成的函数必须显式接收 `seed`。
- 每个负责生成随机数、随机数组、随机参数、随机索引或随机 mask 的函数都应接收 `seed`，在内部生成局部 `rng`，并由此生成随机数。
- 不允许在函数之间传递 `rng` 对象。
- 如果下游核心计算函数需要随机输入，应由上游函数预先生成随机数组、随机参数、随机索引、随机 mask 或其他具体随机对象，再显式传入下游函数。
- 核心计算函数应尽量接收确定性输入并返回确定性结果；随机性应集中在明确的随机生成函数、task 生成函数或入口脚本中。
- 禁止使用：
  - `np.random.seed()`
  - `np.random.rand()`
  - `np.random.randn()`
  - 其他依赖全局随机状态的接口
- 随机数组、随机参数、随机索引、随机 mask 等具体随机输入应尽量显式生成、显式保存、显式传入核心计算函数。
- 不要让随机数消耗顺序依赖任务执行顺序。
- 如果任务需要可重复性，每个 task 的随机性必须由 task identity 决定，而不是由 worker 顺序、完成顺序或随机数消耗顺序决定。
- 对于多阶段随机过程，应为每个阶段定义明确的 seed 生成规则，例如由 `base_seed`、`task_id` 和 `stage_name` 共同决定；不要依赖同一个 `rng` 的连续消耗顺序来区分不同阶段。

## Single Process and Multiprocessing Consistency

- 对需要可重复性的项目，代码应保证 single process 和 multiprocessing 对同一个 task 产生完全一致的结果。
- 多进程任务中：
  1. worker 不使用全局 `rng`；
  2. worker 不接收或传递 `rng` 对象；
  3. worker 的随机性由 task id 或 task-specific seed 决定；
  4. worker 返回结果时必须带上 task id；
  5. 主进程按 task id 汇总、排序、保存结果；
  6. 不允许多个 worker 同时写同一个结果文件；
  7. 每个 worker 可以写独立文件，或者由主进程统一写文件；
  8. 结果不得依赖任务完成顺序。
- 优先使用 common functions 中已有的 multiprocessing 支持；如果已有封装能满足需求，不要重复实现。

## Imports

- 所有import都应该在py文件的顶部
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

## README

README 写清：

- 主要入口脚本及其对应结果，例如运行 `run_simulation.py` 得到 simulation 结果。
- 关键 helper 文件职责。
- 关键依赖版本（根据使用的虚拟环境填写）。
- 数据位置和必要外部文件说明。

README 保持短而可执行，不需要写为长文档。随着项目进行，及时更新README。