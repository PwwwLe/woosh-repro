"""DiT/MMDiT 配置类型定义。"""

from typing import Annotated, Dict, Literal, Union
from pydantic import BaseModel, ConfigDict, Field
import torch

type DictTensor = Dict[str, torch.Tensor]


class MMDiTArgs(BaseModel):
    """Woosh multimodal DiT 的结构超参数。

    关键形状：输入 latent channel 为 ``io_channels``，序列长度上限为
    ``max_seq_len``；文本条件 token 维度为 ``cond_token_dim``，进入 DiT 后会
    投影到 ``dim``。``qk_rope_head_dim`` 与 ``qk_nope_head_dim`` 共同决定每个
    attention head 的维度。
    """

    model_config = ConfigDict(extra="forbid")

    model_type: Literal["mmmssflux",] = "mmmssflux"
    max_description_length: int = 77
    max_seq_len: int = 501
    rope_len_multiplier: Union[int, None] = (
        None  # if not None, multiply rope seq len by this factor, useful for finetuning without scaling frequencies
    )

    dim: int = 2048
    inter_dim: int = 10944
    fixed_timestep_features: bool = False
    timestep_features_dim: int = 256
    n_layers: int = 27
    n_heads: int = 16
    n_multimodal_layers: int = 27
    # mla
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    qkv_head_dim: int = 128

    # memory tokens
    n_memory_tokens_rope: int = 0
    n_memory_tokens_description: int = 0

    # yarn
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1

    # IO
    io_channels: int = 128
    cond_token_dim: int = 1024
    adaln_last_layer: bool = False
    adaln_last_layer_nomod: bool = False  # if adaln_last_layer, do not modulate

    # Optim
    non_checkpoint_layers: int = 0  # checkpoint all layers
    mask_out_before: int = -1  # mask out before layer #n: -1 for no masking

    #
    estimate_logvar: bool = False
    no_description_mask: bool = False
    symmetric_attention_init: bool = False

    patch_size: int = 1

    num_sinks: int = 0
    mlp_act: str = "gelu"  # gelu # swiglu


# DiTConfig can be any the config of any other model
DiTArgs = Annotated[Union[MMDiTArgs], Field(discriminator="model_type")]
