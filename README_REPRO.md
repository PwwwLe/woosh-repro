# Woosh 复现项目初始化说明

本仓库用于在官方 SonyResearch/Woosh 实现之上做最小复现初始化。当前代码主体位于 `Woosh-main/`，保留官方模型包、配置、测试脚本、Gradio demo 和 API 结构；本次只新增复现说明、环境记录、路径模板、运行脚本和数据/权重/日志目录说明，不重写 Woosh-AE、Woosh-CLAP、Woosh-Flow、Woosh-DFlow、Woosh-VFlow 或 Woosh-DVFlow。

## 当前项目结构

```text
.
├── README_REPRO.md                 # 本中文复现说明
├── configs/
│   └── repro_paths.example.env      # 本地路径和默认模型变量模板
├── env/
│   ├── README.md                    # 环境说明
│   └── conda-woosh.example.yml      # conda 环境示例
├── experiments/
│   ├── README.md                    # 未来实验记录约定
│   ├── logs/                        # 运行日志目录，真实日志不入库
│   └── runs/                        # 实验输出目录，真实产物不入库
├── scripts/
│   ├── check_official_env.sh        # 轻量环境和路径检查
│   └── run_official_dflow.sh        # 直接调用官方 DFlow 推理脚本
└── Woosh-main/                      # 官方 Woosh 代码主体
    ├── checkpoints/                 # 官方 checkpoint 放置位置
    ├── samples/                     # 官方 sample media 放置位置
    ├── outputs/                     # 官方脚本默认输出位置
    ├── test_Woosh-*.py              # 官方推理/检查入口
    ├── gradio_Woosh-*.py            # 官方 Gradio demo
    ├── api/                         # 官方 API server
    └── woosh/                       # 官方 Python package
```

## 官方代码如何作为 baseline 使用

官方 README 明确说明该公开仓库提供推理代码和公开权重，当前主要入口是 `Woosh-main/test_Woosh-*.py`、`Woosh-main/gradio_Woosh-*.py` 和 `Woosh-main/api/`。本复现项目不改变这些入口，运行脚本只负责进入 `Woosh-main/` 后调用官方脚本。

官方相对路径约定很重要：所有测试脚本应从 `Woosh-main/` 目录运行，因为它们使用 `checkpoints/...`、`samples/...` 和 `outputs/...` 这样的相对路径。

## Conda 环境创建

用户指定使用名为 `woosh` 的 conda 环境。建议从根目录执行：

```bash
conda env create -f env/conda-woosh.example.yml
conda activate woosh
python -m pip install --upgrade pip uv
cd Woosh-main
uv pip install --python "$CONDA_PREFIX/bin/python" -e ".[cpu]" safetensors soundfile fastapi uvicorn
```

官方 README 的原始安装命令是：

```bash
cd Woosh-main
uv sync --extra cpu
# 或 CUDA 机器：
uv sync --extra cuda
```

在 Apple Silicon Mac 上，如果希望优先使用 PyTorch 的 MPS 后端，建议在 `woosh` conda 环境中先确认 PyTorch 是否来自支持 macOS/MPS 的 wheel：

```bash
conda run -n woosh python -c "import torch; print(torch.__version__); print(torch.backends.mps.is_available())"
```

## 权重和样例数据路径

官方权重应从 SonyResearch/woosh-sfx release 下载并解压到 `Woosh-main/checkpoints/MODEL_NAME/`。每个模型目录至少需要：

```text
Woosh-main/checkpoints/MODEL_NAME/config.yaml
Woosh-main/checkpoints/MODEL_NAME/weights.safetensors
# 或 weights.pt
```

当前仓库只保留了官方 `config.yaml`，没有真实 `weights.*`。因此模型实例化和推理会在缺少权重时停止，这是当前真实状态。

需要的主要 checkpoint 目录：

- `Woosh-main/checkpoints/Woosh-AE`
- `Woosh-main/checkpoints/Woosh-CLAP`
- `Woosh-main/checkpoints/TextConditionerA`
- `Woosh-main/checkpoints/TextConditionerV`
- `Woosh-main/checkpoints/Woosh-Flow`
- `Woosh-main/checkpoints/Woosh-DFlow`
- `Woosh-main/checkpoints/Woosh-VFlow-8s`
- `Woosh-main/checkpoints/Woosh-DVFlow-8s`

官方 sample media 应放在：

```text
Woosh-main/samples/
```

已知官方测试脚本使用：

- `Woosh-main/samples/810333__mokasza__glass-breaking.mp3`
- `Woosh-main/samples/video_sample.mp4`

## 最小运行命令

环境检查：

```bash
bash scripts/check_official_env.sh
```

直接调用官方 distilled text-to-audio baseline：

```bash
bash scripts/run_official_dflow.sh
```

等价的官方命令是：

```bash
cd Woosh-main
conda run -n woosh python test_Woosh-DFlow.py
```
如果权重、样例数据或依赖缺失，上述命令应直接失败并打印具体缺失项；不要把这种失败记录成复现成功。

## 后续复现实验扩展方式

后续实验应继续保持官方 baseline 不变：

1. 先补齐 `Woosh-main/checkpoints/` 和 `Woosh-main/samples/`。
2. 先运行 `Woosh-DFlow`，因为它是 distilled T2A 入口，推理步数少，适合本机 smoke test。
3. 将每次运行命令、环境、checkpoint 版本、prompt、seed、输出路径和日志记录到 `experiments/logs/` 或 `experiments/runs/`。
4. 如需做小规模对比实验，优先新增 wrapper、配置模板或实验记录，不直接修改 `Woosh-main/woosh/` 模型实现。
5. 如确实需要修改官方代码，应先创建单独分支，并在实验日志中解释修改目的和差异。

## 当前尚未完成的部分

- 尚未下载官方 checkpoint 权重。
- 尚未下载官方 sample media。
- 尚未完成任何真实音频生成。
- 尚未进行定量指标复现或大规模训练。
- 本项目当前只完成复现工程初始化、路径约定、轻量脚本和真实环境阻断记录。
