# Woosh 项目结构与代码流说明

本文档覆盖当前工作区中的官方仓库 `Woosh-main/`、最小复现目录
`woosh_repro/`，以及根目录示例、测试和环境配置。目标是说明文件职责、
模型加载流程，以及 Woosh-AE、Woosh-CLAP、Woosh-Flow、Woosh-VFlow、
Woosh-DFlow、Woosh-DVFlow 的调用和数据流。

## 顶层目录树

```text
.
├── README.md                         # 最小复现说明
├── environment.yml                   # Conda 环境 woosh 的导出文件
├── pyproject.toml                    # 最小复现包和 pytest 配置
├── PROJECT_STRUCTURE_ZH.md           # 本中文项目结构文档
├── examples/
│   └── end_to_end.py                 # 最小复现端到端示例
├── tests/
│   └── test_components.py            # 最小复现 smoke tests
├── woosh_repro/
│   ├── __init__.py                   # 最小复现公开 API
│   ├── ae.py                         # 小型 Woosh-AE
│   ├── clap.py                       # 小型 Woosh-CLAP
│   ├── flow.py                       # 小型 Flow/VFlow/DFlow/DVFlow
│   └── utils.py                      # synthetic data 与设备工具
└── Woosh-main/
    ├── README.md                     # 官方仓库使用说明
    ├── pyproject.toml                # 官方依赖与 uv extras
    ├── LICENSE.*                     # MIT/Apache/Freesound 许可证
    ├── .gitattributes/.gitignore/.python-version
    ├── test_Woosh-*.py               # 官方 checkpoint 推理脚本
    ├── gradio_Woosh-*.py             # 官方 Gradio demos
    ├── api/                          # FastAPI 服务与推理 agent
    ├── checkpoints/                  # 各模型 config.yaml；权重应放在同目录
    ├── reaper_script/                # REAPER 集成示例
    └── woosh/
        ├── components/               # 可加载组件与 conditioner 包装
        ├── inference/                # Flow/FlowMap 采样器
        ├── model/                    # LDM、DiT、VFlow、DFlow/DVFlow wrapper
        ├── module/                   # 底层 AE、CLAP、PaSST、Vocos 网络
        └── utils/                    # 加载、视频、Synchformer/ViT 工具
```

未展开列出 `.git/`、`.agents/`、`.codex/`、`__pycache__/`、
`.pytest_cache/` 等元数据和缓存目录；这些不是项目运行逻辑的一部分。

## 官方模型总体数据流

### 模型加载流程

官方组件统一使用 `LoadConfig(path=...)` 加载：

1. 入口脚本、Gradio、API 创建 `LatentDiffusionModel`、`VideoKontext` 或
   `FlowMapFromPretrained`。
2. `BaseComponent.init_from_config()` 调用 `resolve_config()`。若配置含
   `path`，则 `_config_and_weightspath_from_path()` 查找
   `config.yaml` 与 `weights.safetensors`/`weights.pt`。
3. 组件构造子模块，例如 `LatentDiffusionModel` 创建 `SFXFlow`、`AudioAutoEncoder`
   和 `SFXCLAPTextConditioner`。
4. `register_subcomponent()` 记录哪些子组件从父 checkpoint 排除，典型情况是
   LDM checkpoint 不重复保存 Woosh-AE 和 TextConditioner 权重。
5. `load_from_config()` 加载当前组件权重，并递归加载被排除的子组件。

当前工作区 `checkpoints/*/config.yaml` 存在，但权重文件未在清单中出现；因此
官方大模型推理脚本需要补齐 release 权重和 samples 后才能实际运行。

### Woosh-AE

`AudioAutoEncoder` 包装底层 `VocosAutoEncoder`。输入 waveform 通常为
`[B, 1, T]`，`forward()` 输出 latent `[B, z_dim, frames]`，官方配置
`z_dim=128`、`sample_rate=48000`、`hop_length=480`。生成模型输出 latent 后，
通过 `AudioAutoEncoder.inverse()` 解码回 waveform `[B, 1, T']`。

### Woosh-CLAP 与 TextConditioner

完整 CLAP 检索模型在 `AudioRetrievalModel` 中组合 PaSST audio frontend 与
RoBERTa text frontend，输出归一化 shared-space embedding。生成模型只使用
文本分支：`SFXCLAPTextConditioner.forward()` 从 batch 的 `description` 列表
生成 `text_cond` `[B, max_sentence_tokens, 1024]` 和 `text_mask`
`[B, max_sentence_tokens]`，随后 `LatentDiffusionModel.get_cond()` 把它们映射为
`cross_attn_cond` 与 `cross_attn_cond_mask`。

### Woosh-Flow

`LatentDiffusionModel` 组合：

- `SFXFlow`：DiT velocity 网络。
- `AudioAutoEncoder`：latent/audio 编解码。
- `SFXCLAPTextConditioner`：文本 cross-attention 条件。

推理时入口构造 noise `[B, 128, 501]`，调用 `get_cond()` 获得文本条件，
`flowmatching_integrate()` 在 latent space 上用 ODE solver 调
`_denoise_dict_no_param()`。DiT 的数据路径为：

`InputProcessing` 将 latent `[B, C, T]` patchify 为 `[B, tokens, dim]`，
拼接文本 token；`MMMBlock` 与 `MultimodalitySingleStreamBlock` 融合模态；
`PostProcessing` 投影回 velocity latent `[B, C, T]`。

### Woosh-VFlow

`VideoKontext` 先加载一个 text-to-audio LDM，然后添加
`VideoEncoderConditioner`。视频帧由 `extract_video_frames()` 抽取，
`SynchformerProcessor` 输出 `synch_out`，形状通常为 `[B, video_tokens, 768]`。
`VideoEncoderConditioner` 投影为 `video_features` `[B, video_tokens, dim]`，
`NewPreprocessing` 把该 key 写入 DiT 计算字典，`VideoKontext` 扩展每个多模态
block，使 audio、text、video 三种 modality 共同注意力。

### Woosh-DFlow 与 Woosh-DVFlow

`FlowMapFromPretrained` 复用预训练 Woosh-Flow 或 Woosh-VFlow 的 DiT layers、
autoencoder 与 conditioners，但替换：

- `FlowMapPreprocessing`：使用当前时间 `t`、目标时间 `r`、CFG scale `cfg`
  生成新的 time embedding 和 logvar。
- `FlipSignPostprocessing`：翻转输出符号以匹配 distilled FlowMap 目标。

`sample_euler()` 使用少步 schedule。输入 noise 和输出 latent 同形状
`[B, C, frames]`；每步调用 `_denoise_dict_no_param(x_t, t, r, cond)`，可选
renoise，最后仍由 Woosh-AE 解码。

## 最小复现数据流

`woosh_repro` 不加载官方权重，所有输入均为 synthetic data：

1. `synthetic_audio()` 生成 `[B, 1, T]` transient waveform。
2. `WooshAE.encode()` 产生 STFT latent `[B, latent_dim, frames]`。
3. `WooshCLAP.contrastive_loss()` 同时给出 text/audio embedding 和
   `text_tokens` `[B, max_tokens, width]`。
4. `WooshFlow.training_loss()` 在 `x0` noise 与 `x1` latent 之间做 linear
   interpolation，预测 velocity。
5. `WooshVFlow.condition()` 额外把 synthetic video `[B, F, C, H, W]` 编码为
   `video_tokens`。
6. `WooshDFlow`/`WooshDVFlow` 接收 teacher velocity 或退化目标，支持少步
   distilled sampling。
7. `WooshAE.inverse()` 把生成 latent 解码回 `[B, 1, T]`。

## 根目录文件

- `README.md`：最小复现说明，包含 Conda setup、测试命令、组件范围和与官方
  实现的差异。
- `environment.yml`：Conda 环境 `woosh`。当前导出包含 Python 3.12、numpy、
  pytest、einops、`torch==2.5.1+cu121`。
- `pyproject.toml`：最小复现包元数据与 pytest 配置，`testpaths=["tests"]`。
- `examples/end_to_end.py`：端到端示例。主函数创建 AE、CLAP、Flow、VFlow、
  DFlow、DVFlow，计算各自 loss，运行 conditioned generation，并打印参数量、
  latent shape 与 loss。主要张量：audio `[2,1,512]`，video `[2,4,3,16,16]`，
  latent `[2, latent_dim, frames]`。
- `tests/test_components.py`：pytest smoke tests。覆盖 AE 重建和 backward、
  CLAP logits/归一化 embedding、Flow/VFlow conditioned sampling、DFlow/DVFlow
  distilled loss 和 sampling。使用 `WOOSH_TEST_DEVICE` 可强制 CPU/GPU。

## `woosh_repro/` 文件

- `woosh_repro/__init__.py`：导出最小复现公开类：
  `WooshAE`、`WooshCLAP`、`WooshFlow`、`WooshVFlow`、`WooshDFlow`、
  `WooshDVFlow`、`FlowCondition` 及配置类。
- `woosh_repro/ae.py`：
  - `WooshAEConfig`：STFT 配置，含 `n_fft`、`hop_length`、`latent_dim`。
  - `WooshAE`：小型 STFT-domain AE。`encode(audio)` 接收 `[B,1,T]`，
    输出 `[B, latent_dim, frames]`；`decode(latents, length)` 输出
    `[B,1,length]`；`training_loss()` 返回 waveform L1 与多尺度 spectral L1。
  - 调用关系：Flow 模型消费 AE latent，采样结果再由 AE 解码。
- `woosh_repro/clap.py`：
  - `WooshCLAPConfig`：hash vocab、token 长度、audio STFT 与 hidden size 配置。
  - `_stable_hash_token()`：稳定 tokenizer。
  - `WooshCLAP`：文本/音频双塔。`encode_text()` 返回
    `[B, embed_dim]` 和 `[B, max_tokens, width]`；`encode_audio()` 返回
    `[B, embed_dim]`；`contrastive_loss()` 返回 `[B,B]` logits 和对称 CE loss。
  - 调用关系：`text_tokens` 作为 `FlowCondition.text_tokens`。
- `woosh_repro/flow.py`：
  - `WooshFlowConfig`、`FlowCondition`：定义 latent/condition 维度和条件容器。
  - `_time_features()`：`[B] -> [B,width]` sinusoidal time features。
  - `TinyJointTransformer`：拼接 text/video/audio token，输出 velocity
    `[B, latent_dim, frames]`。
  - `VideoConditioner`：`[B,F,C,H,W] -> [B,F,cond_dim]`。
  - `WooshFlow`：基础 flow matching loss 与 Euler sampling。
  - `WooshVFlow`：增加 video condition。
  - `WooshDFlow`：增加 teacher target、第二时间 `r`、renoise sampling。
  - `WooshDVFlow`：DFlow + video condition。
- `woosh_repro/utils.py`：`pick_device()`、`synthetic_audio()`、
  `synthetic_video()`、`count_parameters()`、`assert_finite()`。

## `Woosh-main/` 官方文件

- `.gitattributes`：Git 属性配置。
- `.gitignore`：忽略缓存、输出、构建物等。
- `.python-version`：官方开发环境的 Python 版本提示。
- `LICENSE.Apachev2`、`LICENSE.Freesound`、`LICENSE.MIT`：分别覆盖改编代码、
  样例数据许可和主要代码许可。
- `README.md`：官方安装、权重下载、测试脚本、Gradio/API 使用和论文引用说明。
- `pyproject.toml`：官方包 `woosh` 的依赖。核心依赖含 torch/torchaudio、
  hydra、pydantic、omegaconf、transformers、hear21passt、torchdiffeq、av、
  gradio；extras 包含 CPU/CUDA、API、REAPER、audio I/O、demo、dev。

### 官方入口脚本

- `test_Woosh-AE.py`：加载 `AudioAutoEncoder(LoadConfig("checkpoints/Woosh-AE"))`，
  读取 sample mp3，执行 encode/decode 并保存 wav。输入 `[1,1,T]`，latent
  `[1,128,frames]`。
- `test_Woosh-CLAP.py`：加载 `AudioRetrievalModel` 和 text/audio safetensors
  权重，计算文本与音频 embedding 的点积得分。
- `test_Woosh-Flow.py`：加载 `LatentDiffusionModel`，构造 `[1,128,501]` noise，
  文本条件经 `get_cond()` 后交给 `flowmatching_integrate()`，再 AE 解码。
- `test_Woosh-DFlow.py`：加载 `FlowMapFromPretrained`，使用 `sample_euler()`
  4-step distilled sampling，输出 wav。
- `test_Woosh-VFlow.py`：加载 `VideoKontext` 与 `SynchformerProcessor`，抽取
  `samples/video_sample.mp4` 特征，运行 VFlow 并 remux mp4。
- `test_Woosh-DVFlow.py`：VFlow 数据准备 + FlowMap distilled sampler。
- `gradio_Woosh-Flow.py`：文本到音频 Gradio demo。核心函数：
  `load_model()`、`generate()`、`build_ui()`、`main()`。
- `gradio_Woosh-DFlow.py`：distilled 文本到音频 Gradio demo，调用
  `FlowMapFromPretrained` 与 `sample_euler()`。
- `gradio_Woosh-VFlow.py`：视频+文本到音频 Gradio demo。调用
  `extract_video_frames()`、`SynchformerProcessor`、`VideoKontext`、
  `flowmatching_integrate()`、`remux_video()`。

### `Woosh-main/api/`

- `api/README.md`：API server 启动和客户端测试命令。
- `api/__init__.py`：API 包说明。
- `api/api_server.py`：
  - `GenerateRequest`、`QueueStatusResponse`、`ActiveRequest`：HTTP schema 与
    内存请求状态。
  - `lifespan()`：启动队列 worker 并加载 compute agent。
  - `run_generate()`：调用 `compute_agent.generate()`。
  - `compress_audio()`：`[C,T]` waveform -> FLAC buffer。
  - `process_queue()`、`cleanup_old_requests()`：异步队列处理。
  - FastAPI routes：`/ping`、`/queue/status`、`/generate/queue`、
    `/result/{id}`、`/result/await/{id}`、`/generate`、`/generate/priority`。
- `api/compute_agent.py`：
  - `GenerateArgs`：API 生成参数。
  - `GenerateAgentInterface`：agent 协议。
  - `GenerateBasicAgent`：加载普通 `LatentDiffusionModel`。
  - `FlowMapGenerateAgent`：加载 DFlow/DVFlow wrapper，构造 `[1,128,501]`
    noise，prompt -> text conditioner -> `sample_euler()` -> AE inverse。
  - `MultimodelGenerateAgent`：管理多个 agent 并按 `args.model` 路由。
- `api/test_api.py`：手动 HTTP 客户端测试，向 `/generate` 发送 Woosh-DFlow
  请求并保存 FLAC。
- `api/utils.py`：`short_prompt()` 和 `CLAPCaptionPostprocessTransform`。

### `Woosh-main/reaper_script/`

- `reaper_script/README.md`：REAPER/ReaScript/reapy/Tkinter 安装与使用说明。
- `reaper_script/reapy_script.py`：
  - `get_tcl_lib_dirs_from_brew()`：macOS 上查找 Tk/Tcl 路径。
  - `generate(prompt)`：调用 API 生成临时 FLAC。
  - `insert_file_at_cursor()`：把音频插入 REAPER timeline。
  - `ui_main()`：Tkinter prompt 输入框。
  - `cli()`、`entry()`：Click CLI 入口。

### `Woosh-main/checkpoints/`

此目录按模型名分组，每个子目录至少应含 `config.yaml`，权重文件通常为
`weights.safetensors`、`weights.pt` 或分支权重。当前文件清单中仅看到配置。

- `Woosh-AE/config.yaml`：VocosAutoEncoder 配置，`z_dim=128`，
  `sample_rate=48000`，`n_fft=960`，`hop_length=480`。
- `Woosh-CLAP/config.yaml`：完整 audio/text retrieval model 配置，audio 为
  PaSST，text 为 RoBERTa-large。
- `TextConditionerA/config.yaml`、`TextConditionerV/config.yaml`：生成模型使用的
  CLAP text conditioner 配置。
- `Woosh-Flow/config.yaml`：text-to-audio LDM，引用 `TextConditionerA` 和
  `Woosh-AE`。
- `Woosh-DFlow/config.yaml`：FlowMap distilled text-to-audio wrapper，内部引用
  text-to-audio LDM 配置。
- `Woosh-VFlow-8s/config.yaml`：VideoKontext LDM，引用 `TextConditionerV`、
  `Woosh-AE`，并配置 video fps/audio fps 与 Synchformer key。
- `Woosh-DVFlow-8s/config.yaml`：FlowMap distilled video-to-audio wrapper。

## 官方 `woosh/` 包

### `woosh/components/`

- `woosh/__init__.py`：官方包说明。
- `woosh/components/__init__.py`：导出 `AudioAutoEncoder` 与
  `SFXCLAPTextConditioner`。
- `woosh/components/base.py`：
  - `ComponentConfig`、`LoadConfig`：Pydantic 配置基类。
  - `_is_load_config()`：Pydantic discriminator。
  - `recursive_update_config()`：磁盘配置和覆盖参数合并。
  - `find_common_tensors_from_storage()`：识别父/子组件共享参数。
  - `BaseComponent`：`init_from_config()`、`resolve_config()`、
    `register_subcomponent()`、`save()`、`load_from_config()`、
    `from_pretrained()`、`freeze_non_trainable_components()`。
- `woosh/components/autoencoders.py`：
  - `AudioAutoEncoderConfig`：开放配置。
  - `AudioAutoEncoder`：从 Hydra `_target_` 实例化底层 AE；`forward(x)`
    输入 `[B,1,T]` 输出 normalized latent `[B,z_dim,frames]`；
    `inverse(x)` 反归一化并解码。
- `woosh/components/conditioners.py`：
  - `ConditionConfig`：声明 conditioner 输出形状和类型。
  - `DiffusionConditioner`：`output` 属性和 `forward()` 抽象接口。
- `woosh/components/clap_conditioners.py`：
  - `SFXCLAPTextConditionerConfig`：RoBERTa/text branch 配置。
  - `freeze_model()`：冻结参数。
  - `SFXCLAPTextConditioner`：`tokenize_text()` 和 `forward()` 从
    `description` 生成 `text_cond` `[B,77,1024]` 与 `text_mask` `[B,77]`。
  - `from_audioretrieval_module()`：从完整 CLAP 模块导出 text conditioner。

### `woosh/inference/`

- `flowmatching_sampler.py`：
  - `flowmatching_integrate()`：ODE sampler。输入 noise `[B,C,T]`、条件字典；
    内部构造 no-cond 分支做 CFG，调用 `ldm._denoise_dict_no_param()`。
- `flowmap_sampler.py`：
  - `sample_euler()`：DFlow/DVFlow 的少步 Euler sampler。写入 `cond["cfg"]`，
    每步调用 `model._denoise_dict_no_param(x_t, t, r, cond)`，输出 latent
    `[B,C,T]`。

### `woosh/model/`

- `dit_types.py`：
  - `DictTensor`：`dict[str, Tensor]` 类型别名。
  - `MMDiTArgs`/`DiTArgs`：DiT 结构配置，包括 `max_seq_len`、`dim`、
    `n_layers`、`n_heads`、`io_channels`、`cond_token_dim`、RoPE 参数等。
- `dit_blocks.py`：
  - `precompute_freqs_cis()`、`apply_rotary_emb()`：RoPE 频率和应用。
  - `RMSNorm`、`FourierFeaturesTime`、`FixedFourierFeaturesTime`、`MLP`。
  - `ModalityAttention`、`ModalityBlock`：单 modality `[B,tokens,dim]`。
  - `MMMAttention`、`MMMBlock`：多 modality q/k/v 拼接融合。
  - `MultimodalitySingleStreamBlock`：audio/text/video 拼成单流 joint block。
  - `SelfAttention`：Video encoder UMBlock 使用的 DictTensor attention。
- `dit_pipeline.py`：
  - `mask_out()`、`unmask_out()`、`mask_out_freqs()`：按 mask 裁剪/恢复 token。
  - `DiTPipeline`：preprocessing -> layers -> postprocessing。
  - `DiTFlowMapPipeline`：同上，但 forward 额外接收 `r`。
- `dit_flows.py`：
  - `InputProcessing`：latent `[B,C,T]` -> audio tokens `[B,tokens,dim]`；
    文本条件 `[B,text_tokens,cond_token_dim]` -> `[B,text_tokens,dim]`。
  - `PostProcessing`：hidden tokens -> latent velocity `[B,C,T]`。
  - `SFXFlow`：组装 Woosh-Flow 的多模态 DiT。
- `ldm.py`：
  - `LatentDiffusionModelArgs`/`LatentDiffusionModelConfig`。
  - `LatentDiffusionModelPipeline`：`get_cond()`、`no_cond()`、
    `_batch_cond_nocond()`、`denoise()`、`_denoise_dict()`、
    `_denoise_dict_no_param()`。
  - `LatentDiffusionModel`：实例化 `SFXFlow`、`AudioAutoEncoder`、
    `SFXCLAPTextConditioner`。
  - `LatentDiffusionModelFlowMapPipeline`：DFlow/DVFlow 使用的 `t,r` denoise。
- `video_kontext.py`：
  - `VideoKontextArgs`/`VideoKontextConfig`。
  - `UMBlock`：video feature encoder block。
  - `VideoEncoderConditioner`：`[B,video_tokens,embed_dim]` ->
    `video_features` `[B,video_tokens,dim]`。
  - `NewPreprocessing`：把 video condition 写入 DiT 计算字典。
  - `VideoKontext`：扩展 LDM，使 video modality 加入 DiT blocks。
- `flowmap_from_pretrained.py`：
  - `FlowMapPretrainedArgs`/`FlowMapPretrainedConfig`。
  - `FlipSignPostprocessing`、`FlowMapPreprocessing`。
  - `FlowMapFromPretrained`：复用预训练 LDM/VFlow，替换 preprocessing 和
    postprocessing，形成 DFlow/DVFlow。

### `woosh/module/`

- `woosh/module/audioretrieval_module.py`：
  - `get_audio_frontend_model()`：构建 PaSST audio frontend。
  - `get_sentence_frontend_model()`：加载 RoBERTa/AutoModel 和 tokenizer。
  - `get_audio_head_model()`、`get_sentence_head_model()`：projection heads。
  - `AudioRetrievalModel`：audio/text CLAP 双塔。音频路径或 waveform ->
    audio embedding `[B,shared_dim]`；文本 -> text embedding `[B,shared_dim]`。
- `woosh/module/model/__init__.py`：导出 `VocosAutoEncoder`。
- `woosh/module/model/autoencoder.py`：
  - `AutoEncoder`：普通 encoder/decoder 容器，`forward()` 返回 `(decoded,z)`。
  - `VariationalAutoEncoder`：返回 `(decoded, posterior)`。
- `woosh/module/model/blocks.py`：
  - `DiagonalGaussianDistribution`：VAE posterior。
  - `Upsample1d`、`Downsample1d`、`ResnetBlock`、`LinearAttention`、
    `AttnBlock`、`FourierFeatures` 等底层 block。
- `woosh/module/model/vocos_blocks.py`：
  - `safe_log()`、`symlog()`、`symexp()`。
  - `EMANormalization`、`ContinuousAdaLayerNorm`、`IdentityAdaLayerNorm`。
  - `STFTEmbedding`：waveform `[B,1,T]` -> STFT/mel-like features
    `[B,features,frames]`。
- `woosh/module/model/vocos.py`：
  - `ISTFT`、`IMDCT`：逆谱变换。
  - `FourierHead` 及多种 `ISTFT*Head`/`IMDCT*Head`：hidden -> waveform。
  - `MelSpectrogramFeatures`、`ConvNeXtBlock`、`AdaLayerNorm`、`ResBlock1`。
  - `VocosBackbone`、`VocosResNetBackbone`。
  - `ZeroDropoutTransform`、`ParamDropoutTransform`。
  - `VocosEncoder`、`VocosDecoder`、`VocosAutoEncoder`、
    `DACVocosAutoEncoder`、`VocosVariationalAutoEncoder`。
- `woosh/module/model/retrieval/passt.py`：PaSST/hear21passt 适配代码。
  `create_passt_model()` 构建 audio frontend；`MaskedAttention`、`MaskedBlock`、
  `MaskedPaSST` 和 patch 函数为 PaSST 增加 padding mask 与 compile-compatible
  mel STFT。该文件依赖外部 PaSST 实现，源码未改动，仅在本文档说明。

### `woosh/utils/`

- `loading.py`：
  - `catchtime`：记录加载耗时。
  - `lazy_loading`：临时设置 `lazy_loading_enabled`，用于构建模型结构但跳过
    实际权重加载。
- `video.py`：
  - `get_synchformer()`：从 Hugging Face Hub 获取 Synchformer 权重。
  - `downsample()`、`process_synchformer_transform()`。
  - `SynchformerProcessor.forward(images, fps)`：输入 raw frames `[T,H,W,C]`，
    输出 `synch_out`、`synch_pts_seconds`、`sync_hop_size_ms`。
- `videoio.py`：
  - `extract_video_frames()`：视频路径 -> frames、fps、pts。
  - `remux_video()`：将生成音频与原视频帧封装为 mp4。
- `synchformer.py`：改编自 MMAudio/MotionFormer 的 Synchformer 视觉特征抽取器。
  主要类为 `MotionFormer`、`Synchformer`，函数 `encode_video_with_sync()` 把
  `[B,T,C,224,224]` 切成 segment 并输出 video feature。该文件有明确上游来源，
  本次未修改源码。
- `vit.py`、`vit_helper.py`：MotionFormer/ViT 相关改编实现，包含
  `VisionTransformer`、`DividedAttention`、`DividedSpaceTimeBlock`、
  `PatchEmbed`、`PatchEmbed3D` 等。作为第三方/改编代码记录，未修改源码。

## 非源码资产与生成物分类

- checkpoint：`Woosh-main/checkpoints/*/config.yaml` 是模型结构和子组件路径配置。
  完整推理还需要对应权重文件，当前未逐项列出。
- datasets/samples：官方 README 提到 release 中的 `samples.zip`，当前结构中未看到
  `samples/` 内容。官方 `test_Woosh-*.py` 依赖这些样例媒体。
- outputs：官方脚本和 API 测试会写 `outputs/` 或临时 FLAC/MP4；这些是运行生成物。
- caches/build artifacts：`__pycache__/`、`.pytest_cache/`、构建目录、包缓存等不属于
  代码结构，只需按类别清理或忽略。
- repository metadata：`.git/`、`.agents/`、`.codex/` 为版本控制和工具状态，不参与
  Woosh runtime。

## 本次未直接修改的第三方/改编代码

为遵守“不修改第三方代码”的要求，明确带有上游来源或强依赖外部实现的文件只在
本文档中说明，未添加源码注释：

- `Woosh-main/woosh/utils/synchformer.py`
- `Woosh-main/woosh/utils/vit.py`
- `Woosh-main/woosh/utils/vit_helper.py`
- `Woosh-main/woosh/module/model/retrieval/passt.py`

