from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
from woosh_repro.utils import count_parameters, pick_device, synthetic_audio, synthetic_video


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()
    device = pick_device() if args.device == "auto" else torch.device(args.device)
    torch.manual_seed(7)

    audio = synthetic_audio(batch_size=2, length=512, device=device)
    texts = ["glass click with bright transient", "soft muddy footsteps"]
    video = synthetic_video(batch_size=2, frames=4, device=device)

    ae = WooshAE(WooshAEConfig()).to(device)
    clap = WooshCLAP().to(device)
    latent_shape = ae.encode(audio).shape
    flow_cfg = WooshFlowConfig(latent_dim=latent_shape[1])
    flow = WooshFlow(flow_cfg).to(device)
    vflow = WooshVFlow(flow_cfg).to(device)
    dflow = WooshDFlow(flow_cfg).to(device)
    dvflow = WooshDVFlow(flow_cfg).to(device)

    ae_loss = ae.training_loss(audio)
    clap_out = clap.contrastive_loss(audio, texts)
    text_tokens = clap_out["text_tokens"].detach()
    latents = ae.encode(audio).detach()

    cond = FlowCondition(text_tokens=text_tokens)
    flow_loss = flow.training_loss(latents, cond)
    video_cond = vflow.condition(text_tokens, video)
    vflow_loss = vflow.training_loss(latents, video_cond)
    dflow_loss = dflow.training_loss(latents, cond, teacher=flow)
    dvflow_loss = dvflow.training_loss(latents, dvflow.condition(text_tokens, video), teacher=vflow)

    generated_latents = flow.sample(tuple(latents.shape), cond, steps=4, cfg_scale=1.5, device=device)
    generated_audio = ae.inverse(generated_latents, length=audio.shape[-1])

    print(f"device={device}")
    print(f"ae_params={count_parameters(ae)} clap_params={count_parameters(clap)} flow_params={count_parameters(flow)}")
    print(f"latent_shape={tuple(latents.shape)} generated_audio_shape={tuple(generated_audio.shape)}")
    print(f"ae_loss={ae_loss['loss'].item():.6f} clap_loss={clap_out['loss'].item():.6f}")
    print(f"flow_loss={flow_loss['loss'].item():.6f} vflow_loss={vflow_loss['loss'].item():.6f}")
    print(f"dflow_loss={dflow_loss['loss'].item():.6f} dvflow_loss={dvflow_loss['loss'].item():.6f}")


if __name__ == "__main__":
    main()
