from pathlib import Path
import importlib
import sys


print("python", sys.version.replace("\n", " "))
for name in ["torch", "torchaudio", "omegaconf", "hydra", "gradio"]:
    try:
        module = importlib.import_module(name)
        print(f"import {name}: ok ({getattr(module, '__version__', 'unknown')})")
    except Exception as exc:
        print(f"import {name}: FAIL ({type(exc).__name__}: {exc})")

try:
    import torch

    print("cuda_available", torch.cuda.is_available())
    print("mps_available", torch.backends.mps.is_available())
except Exception:
    pass

standard_weight_dirs = [
    "checkpoints/Woosh-AE",
    "checkpoints/TextConditionerA",
    "checkpoints/TextConditionerV",
    "checkpoints/Woosh-Flow",
    "checkpoints/Woosh-DFlow",
    "checkpoints/Woosh-VFlow-8s",
    "checkpoints/Woosh-DVFlow-8s",
]
for rel in standard_weight_dirs:
    path = Path(rel)
    has_config = (path / "config.yaml").is_file()
    has_weights = (path / "weights.safetensors").is_file() or (
        path / "weights.pt"
    ).is_file()
    print(f"{rel}: config={has_config} weights={has_weights}")

clap_path = Path("checkpoints/Woosh-CLAP")
clap_has_config = (clap_path / "config.yaml").is_file()
clap_has_weights = (clap_path / "weights_audio.safetensors").is_file() and (
    clap_path / "weights_text.safetensors"
).is_file()
print(f"checkpoints/Woosh-CLAP: config={clap_has_config} split_weights={clap_has_weights}")

for rel in ["samples/810333__mokasza__glass-breaking.mp3", "samples/video_sample.mp4"]:
    print(f"{rel}: exists={Path(rel).is_file()}")
