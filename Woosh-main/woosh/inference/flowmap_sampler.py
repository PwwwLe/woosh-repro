"""Woosh-DFlow/DVFlow 的 FlowMap Euler sampler。

distilled 模型一次接收当前时间 ``t`` 和目标时间 ``r``，预测从 ``t`` 到 ``r``
的 velocity。该 sampler 使用少步线性 schedule，可选 renoise，并把 ``cfg`` 写入
条件字典供 ``FlowMapPreprocessing`` 使用。
"""

import torch
from woosh.model.ldm import LatentDiffusionModelFlowMapPipeline


@torch.inference_mode()
def sample_euler(
    model: LatentDiffusionModelFlowMapPipeline,
    noise,
    cond,
    num_steps=4,
    renoise=0.0,
    cfg=4.0,
    t=None,
):
    """使用 Euler steps 采样 distilled FlowMap 模型。

    Args:
        model: FlowMap pipeline，需实现 ``_denoise_dict_no_param(x_t, t, r, cond)``。
        noise: 初始 Gaussian latent，形状 ``[B, C, frames]``。
        cond: 条件字典；函数会原地加入 ``cond["cfg"]``。
        num_steps: Euler 步数，官方 demo 通常使用 4。
        renoise: 每步额外噪声比例，float 或长度为 ``num_steps`` 的列表。
        cfg: classifier-free guidance scale。

    Returns:
        torch.Tensor: 生成 latent，形状与 ``noise`` 相同。
    """
    device = noise.device
    batch_size = noise.size(0)
    cond["cfg"] = cfg * torch.ones((batch_size,), device=device)

    # Define renoise schedule
    if isinstance(renoise, (float, int)):
        renoise_schedule = [renoise] * num_steps
    elif isinstance(renoise, (list, tuple)) and len(renoise) == num_steps:
        renoise_schedule = renoise
    else:
        raise TypeError("renoise must be a float or a list with num_steps values.")

    # Define linear step schedule and reshape as (batch_size, num_steps + 1)
    if t is None:
        t_vals = torch.linspace(1, 0, num_steps + 1)
    else:
        t_vals = t
    t_vals = t_vals.unsqueeze(0).repeat(batch_size, 1).to(device)

    # Denoising steps using Euler
    for i in range(num_steps):
        t, r = t_vals[:, i], t_vals[:, i + 1]
        renoise_i = renoise_schedule[i]

        # Increase noise temporarily.
        if renoise_i > 0:
            gamma = renoise_i * (t - r)
            t_hat = torch.clamp(t + gamma, max=1.0)

            scale_ = (1 - t_hat) / (1 - t + 1e-12)
            std_ = (t_hat**2 - (t * scale_) ** 2).sqrt()[:, None, None]
            new_noise = scale_[:, None, None] * noise + std_ * torch.randn_like(noise)

            # Only renoise for t_hat > t (otherwise we lose the original noise when t=t_hat=1)
            mask_ = t_hat > t
            noise = torch.where(mask_[:, None, None], new_noise, noise)
            t = torch.where(mask_, t_hat, t)

        u = model._denoise_dict_no_param(x_t=noise, t=t, r=r, cond=cond)["x_hat"]
        noise = noise - (t - r)[:, None, None] * u

    return noise
