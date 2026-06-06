---
name: cf-ncc-avalanche
description: 在项目中使用 common_func 根目录下的 ncc_avalanche_functions.py 做 NCC neural avalanche 分析、power-law 拟合、size-duration scaling、crackling relation 检查或保存 avalanche summary 图时触发。用于指导 spike_times 输入、分析窗口、bin width、NCC plparams/sizegivdurwls 参数、duration bin count 拟合和 summary 图保存。
---

# CF NCC Avalanche 分析约定

这个 skill 用于调用 `ncc_avalanche_functions.py` 完成 neural avalanche 分析。该模块会切分 avalanche、调用 NCC `plparams` 拟合 size/duration power law、调用 `sizegivdurwls` 拟合 size-duration relation，并输出 crackling relation 指标和三面板 summary 图。

关键约定：duration 的 power-law 拟合使用整数 avalanche duration bin count，不使用时间单位 duration。`get_ncc_avalanche_results` 已经固定使用 `avalanche_duration_bin` 作为 `duration_fit_values`。

## 导入与依赖

先把 common_func 根目录加入 `sys.path`，再平级导入：

```python
import sys
sys.path.append('<common_func_root>')
import ncc_avalanche_functions as naf
```

`ncc_avalanche_functions.py` 会从同一根目录下的 `ncc_toolbox_path.txt` 读取 NCC toolbox 路径。若无法 import `ncc_toolbox`，优先检查这个路径文件，不要在分析脚本里散落多个 NCC 路径。

## 输入格式

`spike_times` 必须是 list-of-arrays/list-of-lists：

```text
spike_times[neuron_id] = 该神经元的 spike time 序列
```

所有 spike time 使用同一时间单位，常见为 ms。`neuron_idx` 可以选择子集；不传则使用全部神经元。

如果原始数据是事件表，先转成 list-of-lists：

```python
spike_times_list = [[] for _ in range(N)]
for t, i in zip(spike_times, spike_indices):
    spike_times_list[int(i)].append(float(t))
```

## 推荐调用

优先显式传入 `start_time`、`end_time` 和 `time_bin_duration`，保证不同实验可比。只有没有明确 bin width 时，才让函数通过 ISI 估计。

```python
results = naf.get_ncc_avalanche_results(
    spike_times_list,
    start_time=warmup,
    end_time=duration,
    time_bin_duration=0.1,
    plparams_kwargs={
        'samples': 100,
        'threshold': 0.2,
        'likelihood': 1e-3,
    },
    seed=0,
)
```

常用参数：

- `start_time` / `end_time`: avalanche 分析窗口；通常使用 post-warmup 区间。
- `time_bin_duration`: avalanche bin width；会传给 `neuron_data_functions.get_avalanche_from_spike_times`。
- `time_bin_estimation`: 未传 `time_bin_duration` 时使用，支持 `mean_population_isi`、`median_population_isi`、`mean_neuron_isi`、`median_neuron_isi`。
- `plparams_kwargs`: 传给 NCC `plparams` 的 name/value 参数，例如 `samples`、`threshold`、`likelihood`。
- `size_duration_kwargs`: 传给 NCC `sizegivdurwls`，可包含 `durmin`、`durmax` 等参数。
- `seed`: 控制 NCC Monte Carlo p-value 的随机性；size fit 用 `seed`，duration fit 用 `seed + 1`。
- `bin_density` / `unique_bins`: 控制 `plplottool` 生成的绘图数据。

## 输出字段

常用结果字段：

```python
results['avalanche_size']
results['avalanche_duration']
results['avalanche_duration_bin']
results['duration_bin_count']
results['duration_fit_values']
results['duration_fit_unit']
results['size_fit']
results['duration_fit']
results['size_duration_fit']
results['size_tau']
results['duration_alpha']
results['gamma']
results['predicted_gamma']
results['difference']
results['ratio']
results['warnings']
```

`avalanche_duration` 使用输入时间单位；`avalanche_duration_bin`、`duration_bin_count` 和 `duration_fit_values` 使用 bin 数。NCC duration 拟合、duration distribution 图和 size-duration WLS 都应使用 bin 数。

`size_fit` 和 `duration_fit` 的关键字段：

```python
fit['tau']
fit['xmin']
fit['xmax']
fit['sigma_tau']
fit['p']
fit['p_crit']
fit['ks']
fit['plot_data']
fit['status']
fit['error']
fit['runtime_warnings']
```

如果 `status` 不是 `'ok'`，先看 `results['warnings']`、`fit['error']` 和 `fit['runtime_warnings']`，不要只根据图判断结果。

## Summary 图

保存标准三面板图：

```python
fig, axes = naf.save_ncc_avalanche_summary(
    results,
    'results/analysis/avalanches/avalanche_summary',
)
```

图包含 size distribution、duration distribution 和 size-duration relation。duration 面板的 x 轴是 `duration (bins)`。

只创建 figure 不保存时：

```python
fig, axes = naf.plot_ncc_avalanche_summary(results)
```

需要单独画某个面板时：

```python
fig, ax = cf.get_fig_ax()
naf.plot_ncc_powerlaw_distribution(ax, results, property_name='duration')
```

## 检查清单

运行后至少检查：

```python
print(results['duration_fit_unit'])
print(results['duration_fit_values'].dtype)
print(results['size_fit']['xmin'], results['size_fit']['xmax'])
print(results['duration_fit']['xmin'], results['duration_fit']['xmax'])
print(results['warnings'])
```

预期：

- `duration_fit_unit` 是 `'bin'`。
- `duration_fit_values` 是整数 bin count。
- `duration_fit` 的 `xmin/xmax` 是 bin 数，不是 ms 等时间单位。
- `warnings` 为空，或每条 warning 都有明确原因。

如果 avalanche 数量、拟合区间或指数和参考结果差异很大，优先检查分析窗口、`time_bin_duration`、神经元子集、spike time 单位、`plparams_kwargs` 和 `seed` 是否一致。
