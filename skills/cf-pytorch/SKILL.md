---
name: cf-pytorch
description: 在 common_func 项目中使用 PyTorch 写训练、推理、checkpoint、device/dtype 管理、tensor/numpy 转换、局部 torch 随机数、梯度控制、学习率工具或 NaN/Inf 诊断时触发。用于指导优先使用 pytorch_functions.py 中的 get_device、to_device、get_local_rng、to_tensor、save_checkpoint、load_checkpoint、no_grad_eval、concat_outputs、check_tensor_finite 等封装。
---

# CF PyTorch 约定

这个 skill 用于在项目中使用 `pytorch_functions.py` 的 PyTorch 常用封装。目标是减少重复样板代码，让 device、随机性、checkpoint、训练循环和诊断逻辑更一致。

## 导入方式

推荐把 `pytorch_functions.py` 作为一个独立工具模块导入，不要把所有函数 wildcard import 到当前命名空间：

```python
import pytorch_functions as ptf
```

如果当前脚本已经统一用 `sys.path.append('<common_func_root>')` 引入 common functions，则保持项目现有导入风格即可。

## 设备和数据搬运

获取设备优先用 `get_device`。未指定设备时，它会优先选择 CUDA，其次在 `prefer_mps=True` 时选择 MPS，最后回退 CPU：

```python
device = ptf.get_device(device=None)
model = ptf.to_device(model, device)
```

模型、tensor、训练 batch 都统一用 `to_device` 搬到目标设备。它会递归处理 `dict`、`list`、`tuple` 里的 tensor，非 tensor 对象原样保留：

```python
for batch in loader:
    batch = ptf.to_device(batch, device, non_blocking=True)
    output = model(batch["x"])
```

如果 batch 里有自定义对象，需要在项目代码中先转成普通容器，或写薄 wrapper 再调用 `to_device`。

## 局部随机数

需要 PyTorch 随机性时优先用 `get_local_rng(seed, device=...)` 创建局部 `torch.Generator`，不要依赖全局随机状态：

```python
generator = ptf.get_local_rng(seed=0, device="cpu")
x = torch.randn((32, 16), generator=generator)
```

多进程或多任务时，每个 task 应由 task identity 派生自己的 seed，并在 worker 内部创建 generator。不要把同一个 generator 传给多个 worker，也不要用 worker 完成顺序决定随机数消耗。

注意 CUDA generator 的 device 需要和调用的随机算子匹配；不确定时先用 CPU generator 做数据划分、mask 或可重复初始化，再显式搬到目标 device。

## Tensor、NumPy 和标量转换

把输入转成 tensor 用 `to_tensor`，需要 dtype/device 时显式传入：

```python
x = ptf.to_tensor(x_np, dtype=torch.float32, device=device)
```

把 tensor 转回 numpy 用 `to_numpy`。默认会先 `detach()` 再搬到 CPU，适合日志、保存和绘图前处理：

```python
y_np = ptf.to_numpy(y_pred)
```

单元素 tensor 转 Python scalar 用 `to_scalar`。如果 tensor 元素数不是 1，会直接 `raise ValueError`，避免把错误 shape 静默吞掉。

```python
loss_value = ptf.to_scalar(loss)
```

只需要断开计算图并搬到 CPU 时用 `detach_cpu`，保留 tensor 形式，避免过早转 numpy。

## 参数、冻结和梯度

统计参数量用 `count_parameters`。调试模型规模时可以用 `print_trainable_parameters`，它会打印并返回可训练参数量和总参数量：

```python
trainable = ptf.count_parameters(model, trainable_only=True)
trainable, total = ptf.print_trainable_parameters(model)
```

冻结或解冻模块用 `freeze_module`、`unfreeze_module`，底层是对所有参数设置 `requires_grad`：

```python
ptf.freeze_module(backbone)
ptf.unfreeze_module(head)
```

训练中记录梯度范数用 `grad_norm`，裁剪梯度用 `clip_grad_norm`。这两个函数既可接收 `model`，也可接收参数 iterable：

```python
total_norm = ptf.grad_norm(model)
clipped_norm = ptf.clip_grad_norm(model, max_norm=1.0)
```

如果没有任何参数带梯度，`grad_norm` 返回 `torch.tensor(0.0)`。这通常意味着还没有 backward，或参数被冻结，需要结合 `find_none_grad_parameters` 检查。

## 模型模式和推理

临时切换模型模式优先用 context manager。验证和推理最常用 `no_grad_eval`，它会进入 eval 模式并关闭梯度，退出后恢复原来的 training 状态：

```python
with ptf.no_grad_eval(model):
    output = model(x)
```

只需要临时 eval 或 train 时分别用 `eval_mode`、`train_mode`：

```python
with ptf.eval_mode(model_a, model_b):
    score = get_score(model_a, model_b, batch)
```

不要在验证函数里手动 `model.eval()` 后忘记恢复。用这些 context manager 可以避免训练状态被验证流程污染。

## Checkpoint 保存和加载

常规训练 checkpoint 用 `save_checkpoint`，它会保存 `model.state_dict()`，并可选保存 optimizer、scheduler、epoch、step、metrics 和 extra：

```python
ptf.save_checkpoint(
    path=checkpoint_path,
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    epoch=epoch,
    step=global_step,
    metrics={"val_loss": val_loss},
    extra={"seed": seed},
)
```

恢复时用 `load_checkpoint`，可以只读 checkpoint dict，也可以同时恢复 model、optimizer、scheduler：

```python
checkpoint = ptf.load_checkpoint(
    checkpoint_path,
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    map_location=device,
    strict=True,
)
start_epoch = checkpoint.get("epoch", 0) + 1
```

只保存或加载模型权重时，用 `save_model_state` 和 `load_model_state`：

```python
ptf.save_model_state(model_path, model)
ptf.load_model_state(model_path, model, map_location=device, strict=True)
```

路径目录会自动创建。`strict=False` 只应在明确知道缺失或新增哪些参数时使用，并在日志或最终说明中记录原因。

## 训练循环小工具

清梯度用 `zero_grad`，默认 `set_to_none=True`：

```python
ptf.zero_grad(optimizer)
loss.backward()
ptf.clip_grad_norm(model, max_norm=1.0)
optimizer.step()
```

读取学习率用 `get_lr` 或 `get_lrs`，需要把所有 param group 设成同一个学习率时用 `set_lr`：

```python
current_lr = ptf.get_lr(optimizer)
ptf.set_lr(optimizer, lr=1e-4)
```

梯度累积时用 `should_step(global_step, accumulation_steps)` 判断是否执行 `optimizer.step()`。`global_step` 从 1 开始最直观；`accumulation_steps <= 0` 会直接报错。

```python
loss = loss / accumulation_steps
loss.backward()

if ptf.should_step(global_step, accumulation_steps):
    ptf.clip_grad_norm(model, max_norm=1.0)
    optimizer.step()
    ptf.zero_grad(optimizer)
```

## 推理输出整理

多个 batch 的输出合并用 `concat_outputs`。它支持 tensor list、dict 输出，以及 list/tuple 嵌套输出：

```python
outputs = []
with ptf.no_grad_eval(model):
    for batch in loader:
        batch = ptf.to_device(batch, device)
        outputs.append(model(batch["x"]))

outputs = ptf.concat_outputs(outputs, dim=0)
```

如果输出不是 tensor、dict、tuple 或 list，`concat_outputs` 会返回普通 list。对于 dict 输出，所有 batch 的 key 必须一致；否则应先在模型输出处统一结构。

## NaN/Inf 诊断

核心 tensor 检查用 `check_tensor_finite`。核心计算模块发现不可恢复的 NaN/Inf 时，设置 `raise_error=True`，不要静默继续：

```python
ptf.check_tensor_finite(loss, name="loss", raise_error=True)
```

快速判断用 `has_nan`、`has_inf`。排查模型参数和梯度时用：

```python
bad = ptf.find_bad_parameters(model, check_grad=True, check_data=True, max_items=20)
none_grad = ptf.find_none_grad_parameters(model)
```

记录 tensor 摘要用 `summarize_tensor`，它会返回 shape、dtype、device、numel、finite，以及浮点 tensor 的 mean/std/min/max：

```python
summary = ptf.summarize_tensor(activation, name="activation")
```

这些诊断函数适合写进训练日志、异常分支或 smoke test。正式训练中发现严重数值问题时，应优先 fail fast，并保存足够的 batch id、seed、checkpoint path 和参数信息方便复现。

## 最小训练骨架

```python
import torch
import pytorch_functions as ptf

device = ptf.get_device()
model = ptf.to_device(model, device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for epoch in range(num_epochs):
    model.train()
    for step, batch in enumerate(train_loader, start=1):
        batch = ptf.to_device(batch, device, non_blocking=True)
        output = model(batch["x"])
        loss = loss_fn(output, batch["y"])
        ptf.check_tensor_finite(loss, name="loss", raise_error=True)

        ptf.zero_grad(optimizer)
        loss.backward()
        ptf.clip_grad_norm(model, max_norm=1.0)
        optimizer.step()

    val_outputs = []
    with ptf.no_grad_eval(model):
        for batch in val_loader:
            batch = ptf.to_device(batch, device, non_blocking=True)
            val_outputs.append(model(batch["x"]))
    val_outputs = ptf.concat_outputs(val_outputs)
```

项目里如果已经使用 `Experiment` / `DataKeeper`，训练结果和 checkpoint 路径仍应交给对应 tool 的目录管理；`pytorch_functions.py` 只负责 PyTorch 层面的通用操作。
