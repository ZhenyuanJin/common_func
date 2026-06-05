---
name: cf-neuron-data
description: 在 common_func 项目中处理神经元 spike times、spike array、firing rate、raster、ISI、ACF/CCF 或常规神经元时间序列分析时触发。用于指导优先使用 neuron_data_functions.py 中的 spike_times_to_array、spike_array_to_times、spike_to_fr、get_fr_each_neuron、raster_plot、fr_plot、EI_fr_plot、get_ISI 等封装，而不是在项目内重复手写 binning、rate 计算和常规神经元数据作图。注意：neural avalanche analysis 不使用本 skill。
---

# CF 神经元数据分析约定

这个 skill 用于在项目中处理神经元 spike times、spike array、firing rate、raster plot、ISI、相关函数等常规神经元时间序列分析时，优先使用 `/data/zyjin/common_func/neuron_data_functions.py`。项目代码只负责把本项目的数据整理成这些函数需要的形状，不重复实现已有分析和绘图逻辑。

重要限制：`neural avalanche analysis` 不使用这个 skill，也不要直接沿用 `neuron_data_functions.py` 里的 avalanche 相关接口。avalanche 的定义、binning、threshold、事件合并和统计流程需要按具体项目单独确认，后续应建立项目内专门流程或单独 skill。

## 导入方式

`neuron_data_functions.py` 依赖同目录下的 `common_functions.py` 和其他平级模块。推荐先把 `/data/zyjin/common_func` 加入 `sys.path`，再平级导入：

```python
import sys
sys.path.append('/data/zyjin/common_func')
import common_functions as cf
import neuron_data_functions as ndf
```

如果项目已经有 wrapper，例如 `code/utils_function/common_utils.py`，优先从 wrapper 中使用 `ndf`，避免每个脚本重复处理路径和 matplotlib cache。

## 数据形状约定

`neuron_data_functions.py` 的核心时间序列格式通常是：

```text
spike: ndarray, shape (T, N)
```

其中 `T` 是时间 bin 数，`N` 是神经元数。每个元素可以是二值 spike，也可以是该 bin 内 spike count。时间轴通常使用：

```python
ts = start_time + np.arange(spike.shape[0]) * dt
```

如果 simulation 输出是事件表格式：

```text
spike_times: ndarray, shape (num_spikes,)
spike_indices: ndarray, shape (num_spikes,)
```

项目内只写薄 adapter，把事件表转换成 list-of-lists 后调用：

```python
spike = ndf.spike_times_to_array(spike_times_list, start_time, end_time, dt, mode='num')
```

不要在项目里重新手写通用 binning、population rate 或 raster index 提取函数，除非 `neuron_data_functions.py` 明确不能满足该数据格式。

## Spike Times 与 Spike Array 转换

事件表先整理成按神经元分组的 list-of-lists：

```python
spike_times_list = [[] for _ in range(N)]
for t, i in zip(spike_times, spike_indices):
    spike_times_list[int(i)].append(float(t))
```

再转为 `(T, N)` array：

```python
spike = ndf.spike_times_to_array(
    spike_times_list,
    start_time=start_time,
    end_time=end_time,
    dt=dt,
    mode='num',
)
```

如果每个 bin 最多只允许一个 spike，用 `mode='binary'`。如果一个 bin 内可能多个 spike，使用 `mode='num'` 保留计数。

从 spike array 反向得到 spike times 时使用：

```python
spike_times_list = ndf.spike_array_to_times(spike, start_time, dt, position='left')
```

## Firing Rate

计算 population firing rate 时用：

```python
fr_E = ndf.spike_to_fr(spike, width=bin_width, dt=dt, neuron_idx=E_indices)
fr_I = ndf.spike_to_fr(spike, width=bin_width, dt=dt, neuron_idx=I_indices)
fr_all = ndf.spike_to_fr(spike, width=bin_width, dt=dt)
```

计算每个神经元随时间变化的 firing rate 时用：

```python
fr_each = ndf.get_fr_each_neuron(spike, width=bin_width, dt=dt, process_num=1)
```

初次验证和小数据优先 `process_num=1`。如果提高进程数，必须确认总内存上限、数据切分和结果拼接都符合项目资源策略。

如果只需要每个神经元在分析窗口内的平均 firing rate，可以在 `fr_each` 上沿时间轴取均值：

```python
mean_fr_each = np.mean(fr_each, axis=0)
```

## Raster 和 Firing Rate 作图

常规 raster plot 直接使用：

```python
fig, ax = cf.get_fig_ax()
ndf.raster_plot(ax, ts, spike, xlim=[start_time, end_time], ylim=[0, spike.shape[1]])
cf.save_fig(fig, filename)
```

E/I population rate 使用：

```python
fig, ax = cf.get_fig_ax()
ndf.EI_fr_plot(ax, ts, fr_E, fr_I)
cf.save_fig(fig, filename)
```

单条 firing rate 使用：

```python
fig, ax = cf.get_fig_ax()
ndf.fr_plot(ax, ts, fr_all, label='all')
cf.save_fig(fig, filename)
```

绘图函数内部已经使用 common_functions 的 `cf.plt_*` 和 `cf.set_ax`。项目内 wrapper 可以负责准备 `fig/ax`、筛选 E/I 神经元、设置保存路径，但不要重复实现底层 raster 或 line plot。

## ISI 和相关函数

ISI 相关分析优先使用：

```python
ISI = ndf.get_ISI(spike, dt, neuron_idx=neuron_idx)
ISI_mean = ndf.get_ISI_mean(spike, dt, neuron_idx=neuron_idx)
ISI_CV = ndf.get_ISI_CV(spike, dt, neuron_idx=neuron_idx)
```

自相关和互相关优先使用：

```python
lag_times, acf = ndf.get_neuron_data_acf(neuron_data, dt, nlags, process_num=1)
lag_times, ccf = ndf.get_neuron_data_ccf(neuron_data_x, neuron_data_y, dt, nlags, process_num=1)
```

这些函数适合常规神经元时间序列分析。再次强调：avalanche analysis 不在这个 skill 的适用范围内。

## 使用原则

项目内只保留薄 adapter：负责事件表到 `(T, N)` spike array 的整理、E/I index 的选择、输出 dict 的命名和保存路径管理。

对于 firing rate、raster、ISI、ACF/CCF 等已有功能，优先调用 `neuron_data_functions.py`。如果发现接口不适合当前项目，先在项目内写 wrapper 适配；不要直接修改 common functions 源文件，除非 doctor 明确要求维护 common functions 本身。

核心计算模块遇到 shape、dtype、时间范围或 neuron index 错误时应直接 `raise`，不要静默修正。绘图模块则尽量先出图，同时在日志或最终回复中说明可疑数据。
