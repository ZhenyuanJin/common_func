---
name: cf-project-local-instructions
description: 当前项目的局部指令。适用于处理本项目任务时检查局部回复身份、results 组织方式、向后兼容策略和项目特殊说明。初步建立项目结构时使用 cf-project-setup-habit；通用 workflow、代码组织、README、validation、storage、task table、绘图、随机性、多进程、imports 和命名规则遵守 global-codex。
---

# Project Codex Instructions

本文件是当前项目的局部指令。若与全局 `Global Codex Instructions` 冲突，以本项目文件为准；若未特别说明，则继续遵守全局规则。

## Reply Identity Check

- 本项目要求：每次回复的第二个词必须是 `Jin`。

## Project Results Organization

本项目所有仿真结果、中间结果、分析结果、绘图输出和日志默认保存到 `results/`。
如果本项目使用metamodel、experiment，则路径会被自动管理。否则，`results/` 应按任务组织，而不是按脚本类型或临时运行顺序堆放。

推荐结构：

```text
results/
    [task_name]/
        simulation/
        analysis/
        figures/
        logs/
        params/
```

若同一任务包含多个参数组、数据版本、模型版本、seed 组或实验条件，可在 `[task_name]/` 下继续分层：

```text
results/
    [task_name]/
        [param_or_version_name]/
            simulation/
            analysis/
            figures/
            logs/
            params/
```

规则：

- `[task_name]` 应表示科学任务、仿真任务、分析目标或 figure 目标；一般可对应入口脚本名去掉 `run_` 前缀和 `.py` 后缀后的任务名，避免使用 `test`、`tmp`、`new` 等无信息名称。
- `simulation/` 保存原始仿真输出或核心计算输出。
- `analysis/` 保存后处理、统计、拟合、降维、解码等分析结果。
- `figures/` 保存最终图和检查图。
- `logs/` 保存运行日志、错误信息和耗时记录。
- `params/` 保存运行参数、task table 子集或配置快照。
- 每次重要运行应使用参数目录、时间戳目录或唯一 task id，避免覆盖旧结果。
- 多进程任务不得让多个 worker 同时写同一个结果文件。
- 重要结果应能追溯到生成脚本、输入数据、关键参数和运行环境。
- 重要输出必须记录到 `storage.md`。

本项目实际结果结构：

```text
[在这里填写本项目实际使用的 results/ 结构；如果未填写，按上述推荐默认结构执行。]
```

## Backward Compatibility

本项目是否需要向后兼容旧代码、旧脚本接口、旧参数文件、旧结果目录或旧数据格式，由本节记录。若本节未填写，不要擅自假设必须向后兼容，也不要擅自破坏已有可复用结果。

```text
[在这里填写本项目的向后兼容策略。没有明确要求时留空。]
```

可填写内容包括：

- 是否需要保持旧脚本入口和参数名称不变。
- 是否需要保持旧结果文件名、目录结构或字段格式不变。
- 是否允许迁移旧结果到新结构。
- 是否允许删除、覆盖或重算旧中间结果。
- 如果修改会破坏旧代码或旧结果读取方式，是否需要同时提供 adapter、conversion script 或兼容 wrapper。

## Project-Specific Notes

本项目的额外约定、特殊参数、数据说明或临时注意事项写在这里：

```text
[在此填写项目特殊要求。没有则留空。]
```
