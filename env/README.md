# Environment Notes

This project keeps the official Woosh code under `Woosh-main/` and expects a conda environment named `woosh` for reproduction work.

Create the base conda environment:

```bash
conda env create -f env/conda-woosh.example.yml
conda activate woosh
```

Install the official package from inside `Woosh-main/`:

```bash
python -m pip install --upgrade pip uv
cd Woosh-main
uv pip install --python "$CONDA_PREFIX/bin/python" -e ".[cpu]" safetensors soundfile fastapi uvicorn
```

The official README also supports `uv sync --extra cpu` and `uv sync --extra cuda`. Use those commands when you want uv to manage the project environment directly.

Minimal verification:

```bash
conda run -n woosh python --version
conda run -n woosh python -c "import torch, torchaudio; print(torch.__version__, torchaudio.__version__); print(torch.backends.mps.is_available())"
```
