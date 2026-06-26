---
name: cf-project-local-instructions
description: 当前 CF 项目后续每次对话都要注重的局部指令。适用于处理本项目任务时检查回复身份、项目目标、主要输入输出、当前优先事项、旧结果复用、可重复脚本、task table、代码组织、结果记录、storage.md、validation focus 和项目特殊说明。初步建立项目结构时使用 cf-project-setup-habit。
---

# Project Codex Instructions

本文件是当前项目的局部指令。若与全局 `Global Codex Instructions` 冲突，以本项目文件为准；若未特别说明，则继续遵守全局规则。

## Reply Identity Check

- 本项目要求：每次回复的第二个词必须是 `Jin`。

## Project Workflow Additions

除全局 workflow 外，本项目额外要求：

1. 开始旧任务前，优先检查 `storage.md` 中是否已有可复用结果，避免重复运行昂贵任务。
2. 如果发现 `memory.md` 或 `storage.md` 与实际文件状态不一致，应说明不一致，并在必要时更新。
3. 对于需要多次运行的流程，应优先写成可重复执行的脚本，而不是只写一次性 notebook 逻辑。
4. 对于参数扫描、批量仿真、批量分析，应优先建立 task table 或明确的任务列表，保证后续可追踪、可恢复、可并行。

## Project Code Organization

推荐脚本命名：

```text
run_preprocess_[object_name].py
run_simulate_[task_name].py
run_analyze_[analysis_name].py
run_plot_[figure_name].py
```

要求：

1. `code/pre_process/` 中的脚本负责生成可复用的中间对象，例如 task table、随机对象、处理后数据等。
2. `code/simulation/` 中的脚本负责核心模拟或主要计算流程。
3. `code/analysis/` 中的脚本负责读取已有结果并进行分析、统计、拟合、降维、可视化前数据整理等。
4. `code/utils_function/` 中只放项目内可复用函数，不放一次性主流程。
5. 顶层运行脚本应尽量薄，只负责读取参数、调用函数、保存结果和记录日志。
6. 如果某个脚本会产出文件，应在脚本顶部或参数区明确输出位置。
7. 如果某个脚本依赖其他脚本的输出，应在 docstring 或顶部说明依赖文件。

## Output Organization

所有昂贵仿真结果、中间结果、分析结果和绘图输出保存到 `results/`。

推荐结果结构：

```text
results/
  task_name_a/
    simulation/
    analysis/
    figures/
    logs/
  task_name_b/
    param_x_1/
      simulation/
      analysis/
      figures/
      logs/
    param_x_2/
      simulation/
      analysis/
      figures/
      logs/
```

要求：

1. 每个重要结果文件应包含运行参数 `params`，以便追溯。
2. 不要把大型结果文件提交到版本控制系统。
3. 不要让多个进程同时写同一个文件。
4. 结果目录应按任务、日期、参数组、figure 编号或其他清晰规则组织，避免所有文件混在同一层。
5. 重要输出必须记录到 `storage.md`。
6. 如果使用cf的metamodel或者experiment等工具，其会自动配置路径。

## Project-Specific Notes

本项目的额外约定、特殊参数、数据说明或临时注意事项写在这里：

```text
[在此填写项目特殊要求。没有则留空。]
```
