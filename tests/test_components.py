from __future__ import annotations

import os

import pytest
import torch

from woosh_repro import (
    FlowCondition,
    WooshAE,
    WooshAEConfig,
    WooshCLAP,
    WooshDFlow,
    WooshDVFlow,
    WooshFlow,
    WooshFlowConfig,
    WooshVFlow,
)
from woosh_repro.utils import assert_finite, synthetic_audio, synthetic_video


@pytest.fixture(scope="session")
def device() -> torch.device:
    requested = os.environ.get("WOOSH_TEST_DEVICE")
    if requested:
        return torch.device(requested)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _has_finite_grad(module: torch.nn.Module) -> bool:
    return any(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())


def test_woosh_ae_shapes_loss_backward_and_reconstruction(device: torch.device) -> None:
    torch.manual_seed(0)
    model = WooshAE(WooshAEConfig()).to(device)
    audio = synthetic_audio(batch_size=2, length=512, device=device)
    latents, reconstructed = model.reconstruct(audio)
    assert latents.shape[0] == audio.shape[0]
    assert reconstructed.shape == audio.shape
    assert_finite(reconstructed, "reconstructed")
    assert torch.mean((reconstructed - audio) ** 2).item() < 1e-8
    losses = model.training_loss(audio)
    assert_finite(losses["loss"], "ae_loss")
    losses["loss"].backward()
    assert _has_finite_grad(model)


def test_woosh_clap_contrastive_loss_and_scores(device: torch.device) -> None:
    torch.manual_seed(1)
    model = WooshCLAP().to(device)
    audio = synthetic_audio(batch_size=3, length=512, device=device)
    texts = ["glass breaking click", "muddy footsteps", "short engine burst"]
    out = model.contrastive_loss(audio, texts)
    assert out["logits"].shape == (3, 3)
    assert out["text_tokens"].shape[:2] == (3, model.config.max_tokens)
    assert_finite(out["loss"], "clap_loss")
    assert torch.allclose(out["audio_embedding"].norm(dim=-1), torch.ones(3, device=device), atol=1e-5)
    out["loss"].backward()
    assert _has_finite_grad(model)


def test_woosh_flow_loss_backward_and_conditioned_generation(device: torch.device) -> None:
    torch.manual_seed(2)
    ae = WooshAE(WooshAEConfig()).to(device)
    clap = WooshCLAP().to(device)
    audio = synthetic_audio(batch_size=2, length=512, device=device)
    text_tokens = clap.encode_text(["sharp glass snap", "distant soft footstep"])[1].detach()
    latents = ae.encode(audio).detach()
    model = WooshFlow(WooshFlowConfig(latent_dim=latents.shape[1])).to(device)
    cond = FlowCondition(text_tokens=text_tokens)
    losses = model.training_loss(latents, cond)
    assert_finite(losses["loss"], "flow_loss")
    losses["loss"].backward()
    assert _has_finite_grad(model)
    noise = torch.randn_like(latents)
    gen_a = model.sample(tuple(latents.shape), cond, steps=3, cfg_scale=1.5, device=device, noise=noise)
    other_cond = FlowCondition(text_tokens=torch.flip(text_tokens, dims=[0]))
    gen_b = model.sample(tuple(latents.shape), other_cond, steps=3, cfg_scale=1.5, device=device, noise=noise)
    assert gen_a.shape == latents.shape
    assert_finite(gen_a, "flow_sample")
    assert not torch.allclose(gen_a, gen_b)
    decoded = ae.inverse(gen_a, length=audio.shape[-1])
    assert decoded.shape == audio.shape
    assert_finite(decoded, "flow_decoded")


def test_woosh_vflow_video_conditioning(device: torch.device) -> None:
    torch.manual_seed(3)
    ae = WooshAE(WooshAEConfig()).to(device)
    clap = WooshCLAP().to(device)
    audio = synthetic_audio(batch_size=2, length=512, device=device)
    text_tokens = clap.encode_text(["door latch", "metal slide"])[1].detach()
    video = synthetic_video(batch_size=2, frames=4, device=device)
    latents = ae.encode(audio).detach()
    model = WooshVFlow(WooshFlowConfig(latent_dim=latents.shape[1])).to(device)
    cond = model.condition(text_tokens, video)
    assert cond.video_tokens is not None and cond.video_tokens.shape[:2] == (2, 4)
    losses = model.training_loss(latents, cond)
    assert_finite(losses["loss"], "vflow_loss")
    losses["loss"].backward()
    assert _has_finite_grad(model)
    noise = torch.randn_like(latents)
    gen_a = model.sample(tuple(latents.shape), cond, steps=3, cfg_scale=1.2, device=device, noise=noise)
    gen_b = model.sample(
        tuple(latents.shape),
        model.condition(text_tokens, torch.flip(video, dims=[1])),
        steps=3,
        cfg_scale=1.2,
        device=device,
        noise=noise,
    )
    assert gen_a.shape == latents.shape
    assert_finite(gen_a, "vflow_sample")
    assert not torch.allclose(gen_a, gen_b)


def test_minimal_dflow_and_dvflow_support(device: torch.device) -> None:
    torch.manual_seed(4)
    ae = WooshAE(WooshAEConfig()).to(device)
    clap = WooshCLAP().to(device)
    audio = synthetic_audio(batch_size=2, length=512, device=device)
    latents = ae.encode(audio).detach()
    text_tokens = clap.encode_text(["small impact", "cloth movement"])[1].detach()
    video = synthetic_video(batch_size=2, frames=4, device=device)
    cfg = WooshFlowConfig(latent_dim=latents.shape[1])
    teacher = WooshFlow(cfg).to(device)
    dflow = WooshDFlow(cfg).to(device)
    cond = FlowCondition(text_tokens=text_tokens)
    loss = dflow.training_loss(latents, cond, teacher=teacher)
    assert_finite(loss["loss"], "dflow_loss")
    loss["loss"].backward()
    assert _has_finite_grad(dflow)
    sample = dflow.sample(tuple(latents.shape), cond, steps=2, device=device)
    assert sample.shape == latents.shape
    assert_finite(sample, "dflow_sample")

    vteacher = WooshVFlow(cfg).to(device)
    dvflow = WooshDVFlow(cfg).to(device)
    vcond = dvflow.condition(text_tokens, video)
    vloss = dvflow.training_loss(latents, vcond, teacher=vteacher)
    assert_finite(vloss["loss"], "dvflow_loss")
    vloss["loss"].backward()
    assert _has_finite_grad(dvflow)
    vsample = dvflow.sample(tuple(latents.shape), vcond, steps=2, device=device)
    assert vsample.shape == latents.shape
    assert_finite(vsample, "dvflow_sample")
