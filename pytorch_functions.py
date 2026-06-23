from contextlib import contextmanager
from pathlib import Path

import torch


# region 设备与 dtype 管理
def get_device(device=None, prefer_mps=True):
    """
    获取可用的 torch.device.

    device 为 None 时优先 CUDA, 其次按 prefer_mps 选择 MPS, 最后回退 CPU.
    如果需要指定具体 CUDA 卡, 直接传入 "cuda:0", "cuda:1" 等标准设备字符串.
    """
    if device is not None:
        return torch.device(device)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if prefer_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def to_device(obj, device, non_blocking=False):
    """
    递归地把 tensor/module 或容器里的 tensor 搬到 device.

    device 可以是 "cpu", "cuda", "cuda:0", "cuda:1" 或 torch.device.
    支持 torch.Tensor, torch.nn.Module, dict, list, tuple. 其他对象原样返回.
    """
    device = get_device(device)

    if isinstance(obj, torch.Tensor):
        return obj.to(device=device, non_blocking=non_blocking)

    if isinstance(obj, torch.nn.Module):
        return obj.to(device=device)

    if isinstance(obj, dict):
        return {key: to_device(value, device, non_blocking=non_blocking) for key, value in obj.items()}

    if isinstance(obj, list):
        return [to_device(value, device, non_blocking=non_blocking) for value in obj]

    if isinstance(obj, tuple):
        return tuple(to_device(value, device, non_blocking=non_blocking) for value in obj)

    return obj


def batch_to_device(batch, device, non_blocking=False):
    """to_device 的语义别名, 用在 DataLoader batch 上更直观."""
    return to_device(batch, device=device, non_blocking=non_blocking)
# endregion


# region 局部随机生成器
def get_local_rng(seed, device="cpu"):
    """
    创建局部 torch.Generator, 不修改 PyTorch 全局随机状态.
    """
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator
# endregion


# region Tensor / NumPy / 标量转换
def to_tensor(x, dtype=None, device=None):
    """把 x 转为 tensor, 可选 dtype 和 device."""
    if isinstance(x, torch.Tensor):
        tensor = x
        if dtype is not None or device is not None:
            tensor = tensor.to(dtype=dtype, device=device)
        return tensor

    return torch.as_tensor(x, dtype=dtype, device=device)


def detach_cpu(x):
    """对 tensor 执行 detach().cpu(), 非 tensor 原样返回."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    return x


def to_numpy(x, detach=True, cpu=True):
    """把 tensor 安全转为 numpy array, 非 tensor 调用 torch.as_tensor 后再转换."""
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)

    if detach:
        x = x.detach()

    if cpu:
        x = x.cpu()

    return x.numpy()


def to_scalar(x):
    """把单元素 tensor 或 Python 数值转为 Python scalar."""
    if isinstance(x, torch.Tensor):
        if x.numel() != 1:
            raise ValueError("to_scalar only accepts tensors with exactly one element.")
        return x.detach().cpu().item()

    if hasattr(x, "item"):
        return x.item()

    return x
# endregion


# region 模型参数与梯度控制
def count_parameters(model, trainable_only=True):
    """统计模型参数量."""
    parameters = model.parameters()
    if trainable_only:
        parameters = (p for p in parameters if p.requires_grad)
    return sum(p.numel() for p in parameters)


def set_requires_grad(module, requires_grad):
    """设置 module 所有参数的 requires_grad, 并返回 module 方便链式使用."""
    for parameter in module.parameters():
        parameter.requires_grad_(requires_grad)
    return module


def freeze_module(module):
    """冻结 module 参数."""
    return set_requires_grad(module, False)


def unfreeze_module(module):
    """解冻 module 参数."""
    return set_requires_grad(module, True)


def _as_parameter_list(parameters):
    if isinstance(parameters, torch.nn.Module):
        return list(parameters.parameters())
    return list(parameters)


def grad_norm(parameters, norm_type=2.0):
    """计算有梯度参数的梯度范数."""
    parameters = [p for p in _as_parameter_list(parameters) if p.grad is not None]
    if len(parameters) == 0:
        return torch.tensor(0.0)

    device = parameters[0].grad.device
    norms = [torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]
    return torch.norm(torch.stack(norms), norm_type)


def clip_grad_norm(parameters, max_norm, norm_type=2.0):
    """裁剪梯度范数, 返回裁剪前的总范数."""
    return torch.nn.utils.clip_grad_norm_(_as_parameter_list(parameters), max_norm, norm_type=norm_type)


def print_trainable_parameters(model):
    """打印并返回模型的可训练参数量和总参数量."""
    trainable = count_parameters(model, trainable_only=True)
    total = count_parameters(model, trainable_only=False)
    ratio = 0.0 if total == 0 else trainable / total
    print(f"trainable params: {trainable:,} || all params: {total:,} || trainable%: {ratio:.2%}")
    return trainable, total
# endregion


# region 模型模式与上下文管理
@contextmanager
def eval_mode(*modules):
    """临时切换到 eval 模式, 退出后恢复原 training 状态."""
    old_states = [module.training for module in modules]
    try:
        for module in modules:
            module.eval()
        yield
    finally:
        for module, training in zip(modules, old_states):
            module.train(training)


@contextmanager
def train_mode(*modules):
    """临时切换到 train 模式, 退出后恢复原 training 状态."""
    old_states = [module.training for module in modules]
    try:
        for module in modules:
            module.train()
        yield
    finally:
        for module, training in zip(modules, old_states):
            module.train(training)


@contextmanager
def no_grad_eval(*modules):
    """临时 eval 并关闭梯度, 适合验证和推理."""
    with eval_mode(*modules), torch.no_grad():
        yield
# endregion


# region checkpoint 保存与加载
def save_checkpoint(path, model, optimizer=None, scheduler=None, epoch=None, step=None, metrics=None, extra=None):
    """保存常见训练 checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {"model": model.state_dict()}

    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()
    if epoch is not None:
        checkpoint["epoch"] = epoch
    if step is not None:
        checkpoint["step"] = step
    if metrics is not None:
        checkpoint["metrics"] = metrics
    if extra is not None:
        checkpoint["extra"] = extra

    torch.save(checkpoint, path)
    return checkpoint


def load_checkpoint(path, model=None, optimizer=None, scheduler=None, map_location=None, strict=True):
    """
    加载 checkpoint, 并可选恢复 model/optimizer/scheduler.

    返回原始 checkpoint dict.
    """
    if map_location is None:
        map_location = get_device()

    checkpoint = torch.load(path, map_location=map_location)

    if model is not None:
        state_dict = checkpoint.get("model", checkpoint)
        model.load_state_dict(state_dict, strict=strict)

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    return checkpoint


def save_model_state(path, model):
    """只保存 model.state_dict()."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model_state(path, model, map_location=None, strict=True):
    """只加载 model.state_dict()."""
    if map_location is None:
        map_location = get_device()
    state_dict = torch.load(path, map_location=map_location)
    return model.load_state_dict(state_dict, strict=strict)
# endregion


# region 训练循环小工具
def zero_grad(optimizer, set_to_none=True):
    """清空 optimizer 梯度."""
    optimizer.zero_grad(set_to_none=set_to_none)


def get_lr(optimizer):
    """返回 optimizer 第一个 param group 的学习率."""
    return optimizer.param_groups[0]["lr"]


def get_lrs(optimizer):
    """返回 optimizer 所有 param groups 的学习率."""
    return [group["lr"] for group in optimizer.param_groups]


def set_lr(optimizer, lr):
    """把 optimizer 所有 param groups 的学习率设为同一个值."""
    for group in optimizer.param_groups:
        group["lr"] = lr


def should_step(global_step, accumulation_steps):
    """判断当前 step 是否应该执行 optimizer.step. global_step 从 1 开始最直观."""
    if accumulation_steps <= 0:
        raise ValueError("accumulation_steps must be positive.")
    return global_step % accumulation_steps == 0
# endregion


# region 推理与输出整理
def concat_outputs(outputs, dim=0):
    """
    拼接模型输出.

    支持 tensor list, dict 输出, list/tuple 嵌套输出. 其他类型用 list 返回.
    """
    outputs = list(outputs)
    if len(outputs) == 0:
        return outputs

    first = outputs[0]
    if isinstance(first, torch.Tensor):
        return torch.cat(outputs, dim=dim)

    if isinstance(first, dict):
        return {key: concat_outputs([output[key] for output in outputs], dim=dim) for key in first}

    if isinstance(first, tuple):
        return tuple(concat_outputs(values, dim=dim) for values in zip(*outputs))

    if isinstance(first, list):
        return [concat_outputs(values, dim=dim) for values in zip(*outputs)]

    return outputs
# endregion


# region 调试与诊断
def has_nan(x):
    """判断 tensor 中是否存在 NaN."""
    return bool(torch.isnan(x).any().item())


def has_inf(x):
    """判断 tensor 中是否存在 Inf."""
    return bool(torch.isinf(x).any().item())


def check_tensor_finite(x, name=None, raise_error=False):
    """
    检查 tensor 是否全为有限值.

    返回 True/False; raise_error=True 时遇到 NaN/Inf 直接抛错.
    """
    is_finite = bool(torch.isfinite(x).all().item())
    if not is_finite and raise_error:
        prefix = "tensor" if name is None else name
        raise ValueError(f"{prefix} contains NaN or Inf.")
    return is_finite


def find_bad_parameters(model, check_grad=True, check_data=True, max_items=None):
    """返回包含 NaN/Inf 的参数或梯度名称."""
    bad = []

    for name, parameter in model.named_parameters():
        if check_data and not check_tensor_finite(parameter.data):
            bad.append((name, "data"))

        if check_grad and parameter.grad is not None and not check_tensor_finite(parameter.grad):
            bad.append((name, "grad"))

        if max_items is not None and len(bad) >= max_items:
            break

    return bad


def find_none_grad_parameters(model):
    """返回 requires_grad=True 但 grad is None 的参数名称."""
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad and parameter.grad is None]


def summarize_tensor(x, name=None):
    """返回 tensor 的基础统计信息 dict, 便于调试打印或记录日志."""
    x_detached = x.detach()
    summary = {
        "shape": tuple(x_detached.shape),
        "dtype": x_detached.dtype,
        "device": x_detached.device,
        "numel": x_detached.numel(),
        "finite": check_tensor_finite(x_detached),
    }

    if x_detached.numel() > 0 and torch.is_floating_point(x_detached):
        x_float = x_detached.float()
        summary.update(
            {
                "mean": to_scalar(x_float.mean()),
                "std": to_scalar(x_float.std(unbiased=False)),
                "min": to_scalar(x_float.min()),
                "max": to_scalar(x_float.max()),
            }
        )

    if name is not None:
        summary = {"name": name, **summary}

    return summary
# endregion
