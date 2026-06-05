---
name: cf-multiprocess
description: 在 common_func 项目中需要并行运行函数、批量处理列表/enumerate/items 循环，或调整 process_num 时触发。用于指导优先使用 common_functions.py 中的 multi_process、multi_process_list_for、multi_process_enumerate_for、multi_process_items_for 等封装，而不是直接手写 multiprocessing 或 ProcessPoolExecutor。
---

# CF Multiprocess 约定

这个 skill 用于在 `common_func` 项目中写多进程代码时，优先使用 `common_functions.py` 里的并行封装。简单任务保持 `process_num=1` 先跑通；确认函数可以独立运行、参数可序列化、结果正确后，再提高 `process_num`。

## 推荐入口

同一个函数、不同参数并行时，用 `multi_process`：

```python
import common_functions as cf

results = cf.multi_process(
    process_num=cf.PROCESS_NUM,
    func=run_one,
    kwargs_list=[{'seed': seed, 'param': param} for seed, param in param_list],
    func_name='run one'
)
```

普通 `for i in for_list` 循环，用 `multi_process_list_for`，让被调用函数接收 `for_idx_name` 对应的关键字参数：

```python
def run_one(i, config):
    return simulate(config, i)

results = cf.multi_process_list_for(
    process_num=cf.PROCESS_NUM,
    func=run_one,
    kwargs={'config': config},
    for_list=range(n),
    for_idx_name='i',
    func_name='simulate'
)
```

需要 `enumerate(for_list)` 时，用 `multi_process_enumerate_for`；需要 `for key, value in for_dict.items()` 时，用 `multi_process_items_for`。只有多个不同函数需要并行时，才用 `multi_process_for_different_func`。

## 参数写法

优先使用 `kwargs_list` 或 `kwargs` 传参，减少位置参数顺序错误。`args_list` 中的单元素元组必须写成 `(x,)`，不要写成 `(x)`。

`multi_process` 中如果 `args_list` 和 `kwargs_list` 都只有 0 或 1 个元素，会把同一个任务重复运行 `process_num` 次；如果提供多个参数项，`process_num` 表示切分任务的进程数。

`multi_process_list_for`、`multi_process_enumerate_for`、`multi_process_items_for` 只适合每个循环项相互独立的任务。如果不同任务会写同一个文件、修改同一个对象或依赖执行顺序，不要直接并行。

## 注意事项

被并行调用的函数应定义在模块顶层，避免使用 lambda、局部函数或不可 pickle 的闭包。

多进程里不要依赖全局随机状态，也不要把同一个 rng 传给多个进程。推荐给每个任务传不同 `seed`，在函数内部创建局部 rng。

在已有 multiprocessing 子进程中继续调用 cf 的多进程封装时，cf 会自动退回单进程，避免嵌套多进程。

`process_num` 不要盲目设太大。默认可用 `cf.PROCESS_NUM`，小任务、内存占用高或 IO 很重的任务可以手动降低。

调试时先用 `process_num=1`，这样报错位置更清楚；确认后再切到多进程。
