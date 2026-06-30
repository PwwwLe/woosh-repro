"""Woosh-VFlow 的 video conditioning wrapper。

``VideoKontext`` 在已有 text-to-audio LDM 上追加 ``VideoEncoderConditioner``：
video features（通常为 Synchformer 输出 ``[B, frames, 768]``）被投影到 DiT
hidden size，并作为 ``video_features`` modality 加入多模态 attention。
"""

import copy
import logging
from typing import Annotated, Dict, Mapping, Optional, Union, Literal
from einops import rearrange
from pydantic import Discriminator, Tag
import torch
from torch import nn

from woosh.model.dit_blocks import (
    MLP,
    MMMBlock,
    ModalityBlock,
    MultimodalitySingleStreamBlock,
    precompute_freqs_cis,
    SelfAttention,
)
from woosh.model.dit_pipeline import DiTPipeline, DictTensor
from woosh.model.dit_types import DiTArgs, MMDiTArgs

from woosh.model.ldm import (
    LatentDiffusionModel,
    LatentDiffusionModelConfig,
    LatentDiffusionModelPipeline,
)
from woosh.components.base import (
    BaseComponent,
    ComponentConfig,
    LoadConfig,
    _is_load_config,
)
from woosh.components.conditioners import ConditionConfig, DiffusionConditioner


# get logger
log = logging.getLogger(__name__)


class VideoKontextArgs(ComponentConfig):
    """VideoKontext 配置，描述底层 LDM 与 video feature 采样率/键名。"""

    model_type: Literal["VideoKontextLDM"] = "VideoKontextLDM"

    ldm: LatentDiffusionModelConfig
    audio_fps: int = 100

    # video
    video_fps: int = 24
    # embed encoder dim
    embed_dim: int = 768
    # the key in the batch that contains the image embeddings
    embed_key: str = "image_embeds"
    # the key in the batch that contains the pts seconds
    pts_seconds_key: str = "pts_seconds"

    non_checkpoint_layers: int = 0
    n_layers_encoder: int = 0

    trainable_no_cond: bool = False
    use_batch_mask: bool = True


VideoKontextConfig = Annotated[
    Union[
        Annotated[LoadConfig, Tag("load_config")],
        Annotated[VideoKontextArgs, Tag("component_args")],
    ],
    Discriminator(discriminator=_is_load_config),
]


class UMBlock(nn.Module):
    """
    UMBlock with SelfAttention and MLP, used in the VideoEncoderConditioner.
    """

    def __init__(
        self,
        layer_id: int,
        args: DiTArgs,
        qkv_key: str = "x",
        mod_key: Optional[str] = "t",
        freqs_cis_key: Optional[str] = "freqs_cis",
    ):
        """
        Initializes the Transformer block.

        Default behaviour is
        Modulated SelfAttention with rope
        followed by
        Modulated FFN

        with main key 'x' and modulation key 't'

        Args:
            layer_id (int): Layer index in the transformer.
            args (ModelArgs): Model arguments containing block parameters.
            qkv_key: main key, used for self attention and ffn
            mod_key (optional str): modulation key used to modulate layer norms.
            no modulation if None
            freqs_cis_key: key for the rotary embeddings, if None,


        """
        super().__init__()
        self.attn = SelfAttention(
            args, qkv_key=qkv_key, mod_key=mod_key, freqs_cis_key=freqs_cis_key
        )
        self.ffn = MLP(
            args,
            main_key=qkv_key,
            mod_key=mod_key,
        )

        self.layer_id = layer_id

        self.qkv_key = qkv_key
        self.mod_key = mod_key
        self.freqs_cis_key = freqs_cis_key

    def forward(
        self,
        d: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the Transformer block.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position in the sequence.
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
            mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions from attention.

        Returns:
            torch.Tensor: Output tensor after block computation.
        """
        d = d.copy()  # for checkpointing
        d = self.attn.forward(d)
        d = self.ffn.forward(d)
        return d


class VideoEncoderConditioner(nn.Module, DiffusionConditioner):
    """把外部 video embedding 编码成 LDM 条件。"""

    def __init__(self, args: VideoKontextArgs):
        super().__init__()
        # We reuse parameters from the ldm's dit
        # we need to load the config explicitely as it can be a
        # pretrained model initiated using LoadConfig
        dit_args: DiTArgs = LatentDiffusionModel.resolve_config(args.ldm).dit  # type: ignore
        self.config = args

        # precompute_freqs_cis
        # to_audio_fps_multiplier is not needed here since the we only use UMBlocks
        freqs_cis = precompute_freqs_cis(
            dit_args,
            # to_audio_fps_multiplier=args.audio_fps / args.video_fps,
        )
        self.register_buffer(
            "freqs_cis",
            freqs_cis,
            persistent=False,
        )

        self.linear_encoder = nn.Linear(args.embed_dim, dit_args.dim, bias=False)
        self.layers_encoder = torch.nn.Sequential()
        for layer_id in range(args.n_layers_encoder):
            # Block operate over x_downscaled
            self.layers_encoder.append(
                UMBlock(
                    layer_id,
                    dit_args,
                    qkv_key="video_features",
                    mod_key=None,
                    freqs_cis_key="video_freqs_cis",
                )
            )

        self.output_size = dit_args.dim

        if args.trainable_no_cond:
            # If trainable_no_cond is True, we add a trainable parameter
            self.no_cond_token = nn.Parameter(
                torch.zeros(1, 1, self.output_size, dtype=torch.float32)
            )
        else:
            # If trainable_no_cond is False, we use a fixed no_cond_token
            self.register_buffer(
                "no_cond_token",
                torch.zeros(1, dtype=torch.float32),
                persistent=False,
            )

    @property
    def output(self) -> Mapping[str, ConditionConfig]:
        """声明输出的 ``video_features`` 条件类型。"""
        return {
            "video_features": ConditionConfig(
                id="video_features", shape=[self.output_size], type="video_features"
            )
        }

    def forward(
        self, batch, condition_dropout=0.0, no_cond=False, device=None, **kwargs
    ):
        """从 batch 中读取 video feature 并返回 ``{"video_features": Tensor}``。

        输入 key 由 ``config.embed_key`` 控制，官方 VFlow 使用 ``synch_out``，
        形状通常为 ``[B, video_tokens, embed_dim]``。
        """
        if device is None:
            device = (
                batch["audio"].device
                if "audio" in batch
                else next(self.parameters()).device
            )
        if "audio" in batch and batch["audio"] is not None:
            batch_size = batch["audio"].shape[0]
        elif "description" in batch and batch["description"] is not None:
            batch_size = len(batch["description"])
        else:
            batch_size = 1

        # minimium sequence length for no cond
        min_seq_len = 1
        embed_key = self.config.embed_key

        # If no_cond is True, we return the no_cond_token
        if no_cond:
            # print(f"VIDEOKONTEXT: No Cond={no_cond}, batch={batch} ")
            if "video_features" in batch:
                min_seq_len = batch["video_features"].shape[1]

            d = dict(
                video_features=self.no_cond_token.expand(
                    batch_size, min_seq_len, self.output_size
                ),
                video_freqs_cis=self.freqs_cis,
            )
        else:
            # Create compute dictionary
            d = dict(
                video_features=self.linear_encoder(batch[embed_key]),
                video_freqs_cis=self.freqs_cis,
            )
            # batch[synch_out]: torch.Size([10, 72, 768])

            # Pass through multiple transformer blocks
            for layer in self.layers_encoder:
                d = layer(d)

        # video_features: torch.Size([10, 72, 1024])
        video_features: torch.Tensor = d["video_features"]
        # apply batch mask
        mask = torch.rand_like(video_features[..., 0]) >= condition_dropout

        if self.config.use_batch_mask:
            batch_mask = batch.get(f"{embed_key}_mask", None)
            if batch_mask is not None:
                # 2d mask need to .unsqueeze(1)
                mask = mask * batch_mask.unsqueeze(1)
                if not self.training and f"{embed_key}_mask" in batch:
                    log.warning(
                        "Using batch mask %s_mask in inference: %s",
                        embed_key,
                        batch_mask,
                    )
            else:
                if not no_cond:
                    log.warning(
                        "Using no mask for %s no '%s_mask' in batch, batch_keys=%s",
                        embed_key,
                        embed_key,
                        list(batch.keys()),
                    )

        video_features[mask == 0] = self.no_cond_token.to(video_features.dtype).expand(
            1, 1, self.output_size
        )[0, 0]

        return {
            "video_features": video_features,
        }


class NewPreprocessing(nn.Module):
    """复用旧 LDM preprocessing，并把 video modality 写入计算字典。"""

    def __init__(self, args: VideoKontextArgs, old_preprocessing):
        super().__init__()
        self.old_preprocessing = old_preprocessing
        dit_args: DiTArgs = LatentDiffusionModel.resolve_config(args.ldm).dit  # type: ignore

        # to_audio_fps_multiplier is needed here to allow audio/video correct rope
        freqs_cis = precompute_freqs_cis(
            dit_args,
            to_audio_fps_multiplier=args.audio_fps / args.video_fps,
        )
        self.register_buffer(
            "freqs_cis",
            freqs_cis,
            persistent=False,
        )
        self.n_memory_tokens_rope: int = dit_args.n_memory_tokens_rope

    def forward(self, x, t, cond, mask):
        """添加 ``video_features`` 与按 audio/video fps 对齐的 RoPE 频率。"""
        d = self.old_preprocessing(x, t, cond, mask)
        # precompute_freqs_cis

        d.update(
            dict(
                video_features=cond["video_features"],
                video_freqs_cis=self.freqs_cis,
            )
        )
        return d


class VideoKontext(nn.Module, BaseComponent, LatentDiffusionModelPipeline):
    """完整 video-to-audio LDM 组件。

    构造时先加载底层 ``LatentDiffusionModel``，再把 video conditioner 插入
    conditioner 列表，并扩展每个多模态 DiT block 的 modality 列表。
    """

    config_class = VideoKontextArgs

    def __init__(self, config: VideoKontextConfig):
        # Step 1: init of nn.Module
        super().__init__()

        # Step 2: init of BaseComponent
        self.init_from_config(config)
        # now we use self.config and we know it has been validated
        self.config: VideoKontextArgs

        # ========= part to fill starts here ==========
        # Step 3: init of LatentDiffusionModelPipeline

        # init of dit pipeline
        ldm = LatentDiffusionModel(self.config.ldm)
        dit_config: DiTArgs = ldm.config.dit
        assert ldm.config.dit.model_type in ("mmmflux", "mmmssflux"), (
            f"AudioToAudio only supports mmmflux, got {ldm.config.dit.model_type}"
        )

        # Create new conditioners
        video_condtioner = VideoEncoderConditioner(self.config)
        # note that we DON'T copy ldm.conditioners
        # note that we have to put them in ModuleDict
        conditioners = nn.ModuleDict(
            {
                **ldm.conditioners,
                "video_features": video_condtioner,
            }
        )

        # Create new ModalityBlocks
        new_layers = []
        for layer_id, layer in enumerate(ldm.dit.layers):
            # # =================
            # to debug
            # if layer_id == 2:
            #     break
            # # =================
            if isinstance(layer, MMMBlock):
                new_modality_block = ModalityBlock(
                    dit_config,
                    x_key="video_features",
                    freqs_cis_key="video_freqs_cis",
                )
                old_modality_block_dict = layer.get_modality_block_dict()
                new_layer = MMMBlock(
                    layer_id,
                    modality_block_dict={
                        **old_modality_block_dict,
                        "video_features": new_modality_block,
                    },
                )
                new_layers.append(new_layer)
            elif isinstance(layer, MultimodalitySingleStreamBlock):
                # Add lora weights if necessary
                new_layer = layer

                new_layer.x_keys.append("video_features")  # type: ignore
                # append dae_features to processed keys
                new_layer.freqs_cis_keys.append("video_freqs_cis")
                new_layers.append(layer)
            else:
                # If the layer is not a MMMBlock, we keep it as is
                new_layers.append(layer)
        new_layers = nn.ModuleList(new_layers)

        # Create new preprocessing
        # (copies dea_features from cond to compute dict)
        new_preprocessing = NewPreprocessing(
            self.config, old_preprocessing=ldm.dit.preprocessing
        )

        # Creates new dit pipeline
        dit = DiTPipeline(
            preprocessing=new_preprocessing,
            postprocessing=ldm.dit.postprocessing,
            layers=new_layers,
            non_checkpoint_layers=self.config.non_checkpoint_layers,
            mask_out_before=dit_config.mask_out_before,
        )

        # Creates new LDM pipeline
        self.init_pipeline(
            dit=dit,
            autoencoder=ldm.autoencoder,
            conditioners=conditioners,
            sigma_data=ldm.sigma_data,
        )

        # Step 4 : Register subcomponents
        self.register_subcomponent(
            "backbone_ldm",
            subcomponent=ldm,
        )

        # ========= part to fill ends here ==========
        # After registering all subcomponents, we can finally
        # load the state dict from its internal _weights_path
        self.load_from_config()
