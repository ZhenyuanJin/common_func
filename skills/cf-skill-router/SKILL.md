---
name: cf-skill-router
description: 在 common_func 项目中需要判断用户需求应触发哪个 cf-* skill、需要在多个 cf skill 之间分工、整理 skill 使用边界，或用户直接询问“该用哪个 skill/路由/分流/分类”时触发。优先把常见 common_func 请求路由到 cf-plotting、cf-draw-rules、cf-multiprocess、cf-object-io、cf-neuron-data、cf-ncc-avalanche 等局部或领域技能，并说明主 skill 与辅助 skill 的组合关系。除非用户明确提及或现有代码已经使用 Experiment/ComposedExperiment、MetaModel/ModelContainer、DataKeeper，否则不要主动路由到 cf-experiment、cf-metamodel 或 cf-data-keeper。
---

# CF Skill Router

这个 skill 只负责路由和分工，不替代其他 skill 的具体操作说明。接到 common_func 相关任务时，先判断主任务类型，再加载对应的专业 skill；如果任务跨多个边界，可以同时使用主 skill 和少量辅助 skill。

## 路由原则

默认优先选择局部、轻量的 skill。除非用户明确要求框架化实验管理、旧模型容器、结果容器配置，或当前代码上下文已经在使用这些对象，否则不要主动引入 `cf-experiment`、`cf-metamodel`、`cf-data-keeper`。

不要因为任务包含“实验”“仿真”“分析”“保存结果”等普通词，就推断需要 Experiment、MetaModel、ModelContainer 或 DataKeeper。普通绘图、绘图审美规范、普通并行任务、普通对象读写和常规领域分析，分别路由到 `cf-plotting`、`cf-draw-rules`、`cf-multiprocess`、`cf-object-io` 或领域 skill。

优先选择最贴近用户表面需求的 skill 作为主 skill。只有当用户直接要求重构整体 workflow、复用 timedir、组织 tool pipeline、读取旧 MetaModel 目录，或配置 DataKeeper 后端时，框架类 skill 才能成为主 skill。

如果需求只是某个局部工具函数，例如“保存一个 dict”“画一张 heatmap”“并行跑一个列表”，直接使用对应局部 skill，不必加载 experiment/metamodel。

如果需求同时包含多个步骤，按职责叠加 skill；但不要把 `cf-experiment`、`cf-metamodel`、`cf-data-keeper` 当作默认辅助 skill。只有用户提到这些框架或现有代码已经暴露相关类/API 时，才加载它们。

不要同时把 `cf-experiment` 和 `cf-metamodel` 当作同一工作流的主框架。已有代码继承 `cf.MetaModel` 或读取旧 MetaModel 结果时用 `cf-metamodel`；新版 `AbstractTool` / `Experiment` / `ComposedExperiment` 流程用 `cf-experiment`。

## 主 Skill 选择

使用 `cf-plotting`：

- 写 common_func 风格的绘图代码。
- 创建 `fig/ax`、调用 `cf.plt_*`、`cf.set_ax`、`cf.save_fig`。
- 处理多子图、3D 图、colorbar、paper mode、图内文字标注。
- 给已有分析结果补图或检查图像保存流程。

使用 `cf-draw-rules`：

- 检查或优化科研图、论文图的视觉效果。
- 判断文字、线条、点、tick、spine、legend、留白和对齐是否适合正文展示。
- 统一颜色编码、避免彩虹 colormap、选择符合数据语义的对比型或渐进型 colormap。
- 设计多子图布局、图形类型变化、重点突出方式和图内直接标注。
- 通常与 `cf-plotting` 一起使用：`cf-plotting` 负责怎么调用 common functions 画图，`cf-draw-rules` 负责画出来是否清楚美观。

使用 `cf-multiprocess`：

- 把 `for`、`enumerate`、`dict.items()` 循环改成 common_func 多进程封装。
- 使用 `multi_process`、`multi_process_list_for`、`multi_process_enumerate_for`、`multi_process_items_for`。
- 调整 `process_num`、排查 pickle、seed、嵌套多进程、共享写文件等并行风险。

使用 `cf-object-io`：

- 保存或读取普通 Python 对象、dict、numpy array、scipy sparse matrix、pandas DataFrame、txt、yaml。
- 需要按 key 分开保存大字典，或从 separate dict 中局部读取。
- 任务不涉及 Experiment 的 DataKeeper，只是独立 IO。

使用 `cf-neuron-data`：

- 处理常规神经元 spike times、spike array、firing rate、raster、ISI、ACF/CCF。
- 在事件表和 `(T, N)` spike array 之间转换。
- 用 `neuron_data_functions.py` 做常规神经元时间序列分析和作图。
- 注意：neural avalanche 分析不要用这个 skill。

使用 `cf-ncc-avalanche`：

- 调用 `ncc_avalanche_functions.py` 做 NCC neural avalanche 分析。
- 处理 avalanche size/duration、power-law 拟合、size-duration scaling、crackling relation。
- 保存 avalanche summary 图，检查 duration bin count 拟合、`plparams_kwargs`、`size_duration_kwargs`。

使用 `cf-experiment`：

- 仅在用户明确提到 `Experiment`、`ComposedExperiment`、`AbstractTool`、tool pipeline、`timedir` 复用，或现有代码已经使用这些类时使用。
- 组织新版 `AbstractTool`、`Simulator`、`Analyzer`、`Visualizer`、`Experiment`、`ComposedExperiment`。
- 设计 tool 顺序、tool 间数据传递、`tool_params_dict`、`tool_config_dict`。
- 复用或加载 `timedir`，处理 search/skip、`load`、重跑 analyzer/visualizer。
- 调试完整 experiment 或 composed experiment 数据流。

使用 `cf-metamodel`：

- 仅在用户明确提到 `MetaModel`、`ModelContainer`、`MetaModelContainer`，或现有代码继承 `cf.MetaModel` / 需要读取旧 MetaModel 结果目录时使用。
- 维护老式 `cf.MetaModel` 子类。
- 实现 `set_up`、`run_detail`、`analyze_detail`。
- 保存、读取、复用旧 `simulation_results` / `analysis_results` 目录。
- 用 `MetaModelContainer` 批量加载旧模型目录或做旧参数搜索复用。

使用 `cf-data-keeper`：

- 仅在用户明确提到 `DataKeeper`、`DataContainer`、`OrderedDataContainer`、`data_keeper_kwargs`、`separate/lmdb` 后端，或现有 Experiment tool 代码正在配置结果容器时使用。
- 选择 `DataKeeper` 的 `data_type`、`save_load_method`、`dict` / `OrderedDataContainer`。
- 设计 result key、`param_order`、`included_name_list`。
- 在 tool 的 `_config_data_keeper` 中设置 `self.data_keeper_kwargs`。
- 比较 `separate` 和 `lmdb` 保存后端，处理局部读取、释放内存、read-only 前置 keeper。
