---
name: cf-plotting
description: 在 common_func 项目中使用 common_functions.py 的绘图工具时触发。用于指导如何创建 fig/ax、调用 cf 的 plt_* 函数在 ax 上绘图、用 set_ax 统一设置坐标轴和标题、用 save_fig 保存图片，并尽量沿用 common_functions.py 中已经调好的默认绘图参数。
---

# CF 绘图约定

这个 skill 用于在 `common_func` 项目中写绘图代码时，优先使用 `common_functions.py` 里的绘图封装。除非任务明确要求特殊样式，否则尽量保持默认参数；`common_functions.py` 已经统一设置了字体、线宽、marker、legend、边距、dpi、配色和保存格式。

## 推荐流程

典型流程是：

```python
import common_functions as cf

fig, ax = cf.get_fig_ax()
cf.plt_line(ax, x, y, label='data')
cf.set_ax(ax, xlabel='x', ylabel='y', title='title')
cf.save_fig(fig, filename)
```

多子图时保持同样模式：

```python
fig, ax = cf.get_fig_ax(nrows=2, ncols=2)
cf.plt_scatter(ax[0, 0], x, y)
cf.set_ax(ax[0, 0], xlabel='x', ylabel='y')
cf.save_fig(fig, filename)
```

## 创建图片和坐标轴

优先用 `get_fig_ax` 创建常规二维图。它按 `ax_width`、`ax_height` 和默认 `margin` 自动推导 figure 大小，通常不用手动设置 `figsize`。

写论文或需要按版面宽度控制图片时，优先用 `get_fig_ax_by_fig_width_mm`。`fig_width='single'` 表示 89 mm，`fig_width='double'` 表示 183 mm，也可以直接传入数值，单位是 mm。

使用 `get_fig_ax_by_fig_width_mm`、`get_fig_ax_by_fig_width_mm_3d` 这类按论文版面宽度创建图片的函数前，先调用 `set_font_size_paper_mode()` 和 `set_line_style_paper_mode()`，让全局字体、线宽、tick 和 marker 尺寸切到 paper 模式。当此图像被保存后，调用 `set_font_size()` 和 `set_line_style()` 恢复常规默认设置。

全部子图都是 3D 时，使用 `get_fig_ax_3d` 或 `get_fig_ax_by_fig_width_mm_3d`。如果只有部分子图是 3D，可以先用 `get_fig_ax` 创建整体 2D 布局，再用 `convert_ax_to_3d` 把指定 ax 转成 3D；需要反向转换时用 `convert_ax_to_2d`。不要在外部手动改 projection，优先使用 cf 里的转换函数。

```python
fig, ax = cf.get_fig_ax(nrows=1, ncols=2)
ax[1] = cf.convert_ax_to_3d(ax[1])
cf.plt_line(ax[0], x, y)
cf.plt_scatter_3d(ax[1], x, y, z)
```

需要不均匀布局、嵌套布局或复杂组合时，再考虑 `get_fig_gs_custom`、`get_ax_from_gs`、`get_fig_subfig`、`get_ax_inside_ax` 等布局函数，也可以用 `split_ax`、`split_ax_by_gs` 拆分已有 ax，或用 `merge_ax` 合并多个 ax。普通图不要过早使用复杂布局。

## 在 ax 上作画

写代码时优先调用下面这些函数，而不是直接用 `plt.plot()`、`plt.scatter()` 等依赖全局状态的 pyplot 调用。

常用函数族：

- 基础图：`plt_line`、`plt_scatter`、`plt_bar`、`plt_hist`、`plt_box`、`plt_violin`
- 矩阵和热图：优先用 `sns_heatmap`；按坐标边界显示矩阵时用 `plt_pcolormesh`，直接显示数组时用 `plt_imshow`，二维分布用 `plt_hist_2d`，圆形热图用 `plt_scatter_heatmap`
- 统计增强：`plt_linregress`、`plt_density_scatter`、`plt_errorbar_line`、`plt_band_line`、`plt_kde`、`plt_cdf`
- 分组数据：`plt_group_bar`、`plt_group_box`、`plt_bar_dict`、`plt_group_bar_dict`、`plt_group_bar_df`
- 3D 图：`plt_scatter_3d`、`plt_line_3d`、`plt_bar_3d`、`plt_surface_3d`、`plt_voxel_heatmap`

热图和带颜色映射的图尽量使用 cf 函数自带的 colorbar 逻辑，通过 `cbar=True`、`cbar_label`、`cbar_position`、`cbar_kwargs` 控制；需要单独添加 colorbar 时，用 `add_colorbar` 或 `add_side_colorbar`，散点大小和颜色同时编码时用 `add_scatter_colorbar` 或 `add_side_scatter_colorbar`。不要直接手写新的 colorbar 布局，除非 cf 的默认布局无法满足需求。

默认颜色、colormap、线宽、marker size、直方图 bin 数、colorbar 位置等通常已经合适。cf 里已经全局定义了常用颜色、`CMAP`、`HEATMAP_CMAP`、`DENSITY_CMAP`、`CMAP_DICT` 和 `get_cmap`；只有在数据语义需要区分、论文版式要求或默认效果明显不合适时，才覆盖 `color`、`cmap`、`s`、`linewidth`、`cbar_kwargs` 等参数。

## 设置坐标轴和图标题

画完主要元素后，用 `set_ax` 统一设置坐标轴标签、标题、tick、范围、log scale、legend 和 3D 视角：

```python
cf.set_ax(ax, xlabel='time', ylabel='rate', title='response', legend=True)
```

`set_ax` 会保留已有 label/title，未传入的字段一般不用重复设置。需要图例时，在绘图函数里传 `label`，最后让 `set_ax(..., legend=True)` 统一处理。

Figure 级标题用 `set_fig_title`。子图标签用 `add_ax_tag`、`add_axes_list_tag_by_order` 或 `add_fig_tag`。删除多余元素时用 `rm_ax_xlabel`、`rm_ax_ylabel`、`rm_ax_ticklabel`、`rm_ax_spine`、`rm_ax_legend`、`rm_ax_axis` 等辅助函数。

多子图共享坐标轴时，创建阶段优先传 `sharex`、`sharey`；后处理共享可用 `share_axis`，但要注意已有 share 关系可能影响调用顺序。

## 保存图片

常规保存用 `save_fig(fig, filename)`，通常不要给 `filename` 写扩展名，让函数根据默认格式保存并自动创建目录。

快速预览或临时结果用 `save_fig_lite`，它保存位图、降低 dpi，适合测试。

3D 图用 `save_fig_3d` 或 `save_fig_3d_lite`，可以自动保存多个 `elev`/`azim` 视角，也可以生成视频。

只保存局部区域时可用 `save_ax` 或 `save_subfig`。如果保存后还要继续修改同一张图，给保存函数传 `close=False`；否则保持默认 `close=True`。

## 使用原则

优先沿用 cf 默认值，少改样式参数。默认参数代表项目内已经调过的审美和论文输出约定。

保持 `ax` 显式传递。绘图函数、设置函数、保存函数之间用 `fig` 和 `ax` 串起来，不依赖当前全局 figure。

先完成数据含义，再调整外观。样式定制集中放在 `set_ax`、`set_fig_title`、保存函数和少数必要的绘图参数里，避免在代码中散落大量 matplotlib 原生设置。

需要新绘图函数时，遵循现有模式：第一个参数是 `ax`，默认参数优先使用 `common_functions.py` 里的全局常量，内部操作传入的 `ax` 并返回 matplotlib 对象。

## 图中文字标注

在图中添加指标、拟合参数、统计量、说明文字或面板内文本时，优先使用 common_functions.py 中的文本接口，例如 cf.add_text 和 cf.add_text_by_dict。不要直接调用 Matplotlib 原生 ax.text()，除非 cf 的文本接口无法满足需求。

如果文字属于某个子图的解释或拟合结果，应该在对应子图的绘图函数内部添加，而不是在外层 summary 函数里集中添加，确保单独保存该子图和组合成多子图时文字都能对应正确内容。
