"""Woosh-Flow 的 flow-matching ODE sampler。

采样从 latent noise ``[B, C, T]`` 出发，在 ``t=0 -> 1`` 上调用
``LatentDiffusionModelPipeline._denoise_dict_no_param`` 预测 velocity，并用
classifier-free guidance 合成条件/无条件结果。
"""

# Naive CFG
import torch
from torchdiffeq import odeint

from woosh.model.ldm import LatentDiffusionModel


def flowmatching_integrate(
    ldm: LatentDiffusionModel,
    noise: torch.Tensor,
    cond: dict,
    cond_neg: dict = None,
    negative_text_only: bool = True,
    cfg: float = 4.5,
    device: str = "cuda",
    method="dopri5",
    rtol=1e-3,
    atol=1e-3,
    return_steps=False,
    **fm_kwargs,
):
    """使用 ODE solver 积分 flow-matching 轨迹。

    输入和输出都在 AE latent space 中，形状通常为 ``[B, 128, 501]`` 或
    ``[B, 128, 801]``。函数不会调用 autoencoder 解码，调用方需要自行执行
    ``ldm.autoencoder.inverse``。

    Returns data BEFORE post processing (i.e., in latent space).

    Args:
        ldm (LatentDiffusionModel): The latent diffusion model used to predict the denoised output
            at each ODE step.
        noise (torch.Tensor): Initial noise tensor of shape ``(batch_size, *latent_shape)``
            from which integration begins.
        cond (dict): 正向条件字典，常含 ``cross_attn_cond`` 和
            ``cross_attn_cond_mask``；VFlow 还包含 ``video_features``。
        cond_neg (dict, optional): Negative conditioning dictionary used for CFG. If ``None``,
            an unconditional (dropout) forward pass is used as the negative condition.
            Defaults to ``None``.
        negative_text_only (bool, optional): If ``True`` and ``cond_neg`` is provided, only the
            cross-attention text conditioning is replaced with the negative counterpart, leaving
            other conditioning signals from the unconditional pass intact. Defaults to ``False``.
        cfg (float, optional): Classifier-free guidance scale. The guidance is applied as
            ``pred = pred_cond + cfg * (pred_cond - pred_uncond)``. Defaults to ``4.5``.
        device (str, optional): Device on which to run conditioning computation. Defaults to
            ``"cuda"``.
        method (str, optional): ODE solver method. Defaults to ``"dopri5"``.
        rtol (float, optional): Relative tolerance for ODE solver. Defaults to ``1e-3``.
        atol (float, optional): Absolute tolerance for ODE solver. Defaults to ``1e-3``.
        **fm_kwargs: Additional keyword arguments forwarded to ``torchdiffeq.odeint``.

    Returns:
        torch.Tensor | tuple[torch.Tensor, int]: 生成 latent；当
        ``return_steps=True`` 时同时返回 ODE function evaluation 次数。
    """

    batch_size = noise.size(0)

    no_cond = ldm.get_cond(
        {"audio": noise, **cond},
        no_dropout=True,
        device=device,  # we must provide device if we don't provide x
        no_cond=True,
    )
    if cond_neg is not None:
        if negative_text_only:
            # only replace text conditioning
            # print("Using negative TEXT-ONLY conditioning")
            no_cond["cross_attn_cond"] = cond_neg["cross_attn_cond"]
            no_cond["cross_attn_cond_mask"] = cond_neg["cross_attn_cond_mask"]
        else:
            # print("Using negative conditioning")
            no_cond = cond_neg
    step = 0

    def f(t, y):
        c = cond
        nc = no_cond
        nonlocal step
        step += 1

        res = ldm._denoise_dict_no_param(y, 1 - t, c)["x_hat"]
        res_nc = ldm._denoise_dict_no_param(y, 1 - t, nc)["x_hat"]

        res = res + cfg * (res - res_nc)
        return res

    # t = [0, 1]
    t = torch.linspace(0, 1, steps=2, device=noise.device)

    fakes = odeint(f, noise, t, atol=atol, rtol=rtol, method=method, options=fm_kwargs)[-1]

    # print(f"Integrating finished in {step + 1} steps")
    if return_steps:
        return fakes, step + 1
    return fakes
