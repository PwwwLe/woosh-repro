# Woosh Minimal Reproduction

This repository is a small, runnable reproduction of the main public Woosh components described in the paper and official SonyResearch/Woosh release. It is intended for smoke tests and code orientation, not full-scale training or numerical reproduction.

## Setup

```bash
conda env create -f environment.yml
conda activate woosh
pytest
python examples/end_to_end.py --device auto
```

On this host the detected GPUs are two NVIDIA A40 cards with driver 570.133.20
and CUDA 12.8 support. The environment uses `torch==2.5.1+cu121`, which is
compatible with that driver and was verified on `cuda:0`.

The code also runs on CPU:

```bash
WOOSH_TEST_DEVICE=cpu pytest
python examples/end_to_end.py --device cpu
```

## Project Structure

```text
woosh_repro/
  ae.py        Tiny STFT/iSTFT Woosh-AE analogue
  clap.py      Tiny CLAP-style text/audio alignment model
  flow.py      Tiny Woosh-Flow, Woosh-VFlow, DFlow, and DVFlow modules
  utils.py     Synthetic audio/video and device helpers
examples/
  end_to_end.py
tests/
  test_components.py
environment.yml
```

## What Is Implemented

- `WooshAE`: monaural STFT-domain latent encoder/decoder with spectral reconstruction loss.
- `WooshCLAP`: text and audio encoders trained with symmetric contrastive loss and text token latents for conditioning.
- `WooshFlow`: latent flow-matching text-to-audio model with classifier-free guidance sampling.
- `WooshVFlow`: video-to-audio extension with projected video tokens and joint text/video/audio attention.
- `WooshDFlow` / `WooshDVFlow`: minimal distilled mean-flow style loss and few-step sampling support.

## Main Differences From Full Woosh

- No pretrained Sony weights, datasets, large downloads, Gradio app, API server, or media I/O.
- Tiny dimensions and synthetic audio/text/video inputs only.
- The AE uses an identity-initialized STFT coefficient path instead of a large VOCOS ConvNeXt GAN vocoder.
- CLAP uses compact internal text/audio encoders instead of RoBERTa-Large and PaSST.
- Flow models use small Transformer encoders instead of the full multimodal LDM architecture.
- Distillation support is smoke-test level: it exercises teacher targets and few-step sampling, but not adversarial distillation training.
