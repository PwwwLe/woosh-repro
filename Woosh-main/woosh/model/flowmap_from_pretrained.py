"""从预训练 Woosh-Flow/VFlow 构造 distilled FlowMap student。

包装器复用 teacher 的 DiT layers、autoencoder 和 conditioners，只替换
preprocessing/postprocessing：preprocessing 增加 ``t``、``r``、``cfg`` 的时间
嵌入，postprocessing 将输出符号翻转以匹配 FlowMap student 目标。
"""

from pydantic import Discriminator, Tag
from typing import Annotated, Union, Literal

import torch
from torch import nn

from woosh.model.dit_blocks import FourierFeaturesTime, FixedFourierFeaturesTime
from woosh.model.dit_pipeline import DiTFlowMapPipeline
from woosh.model.dit_types import DiTArgs, DictTensor
from woosh.model.ldm import (
    LatentDiffusionModel,
    LatentDiffusionModelConfig,
    LatentDiffusionModelFlowMapPipeline,
)
from woosh.components.base import (
    BaseComponent,
    ComponentConfig,
    LoadConfig,
    _is_load_config,
)
from woosh.model.video_kontext import VideoKontext, VideoKontextArgs


class FlowMapPretrainedArgs(ComponentConfig):
    """FlowMap wrapper 配置。

    ``ldm`` 可以是普通 ``LatentDiffusionModel`` 配置，也可以是
    ``VideoKontext`` 配置；``pretrained_model_type`` 决定实例化路径。
    """

    ldm: LatentDiffusionModelConfig | VideoKontextArgs
    pretrained_model_type: Literal["ldm", "videokontext"] = "ldm"


FlowMapPretrainedConfig = Annotated[
    Union[
        Annotated[LoadConfig, Tag("load_config")],
        Annotated[FlowMapPretrainedArgs, Tag("component_args")],
    ],
    Discriminator(discriminator=_is_load_config),
]


class FlipSignPostprocessing(nn.Module):
    """翻转 teacher postprocessing 输出符号以匹配 FlowMap student。"""

    def __init__(self, args: FlowMapPretrainedArgs, old_postprocessing):
        super().__init__()
        self.old_postprocessing = old_postprocessing

    def forward(self, d: DictTensor) -> DictTensor:
        d = self.old_postprocessing(d)
        d["x"] = -d["x"]
        return d


class FlowMapPreprocessing(nn.Module):
    """为旧 DiT preprocessing 增加 ``t``、``r`` 和 ``cfg`` 条件。"""

    def __init__(
        self,
        args: FlowMapPretrainedArgs,
        dit_args: DiTArgs,
        old_preprocessing: nn.Module,
    ):
        """
        Args:
            args: configuration arguments for FlowMap wrapper.
            dit_args: configuration arguments for the pretrained DiT model.
            old_preprocessing: original preprocessing module from pretrained model.

        Note that dit_args can be obtained directly from args, but because its
        location depends on the model type, we pass it here as a new arg.

        """
        super().__init__()
        self.old_preprocessing = old_preprocessing

        # Define new embedding modules for timestep (t, r) and cfg
        self.timestep_features_t = FixedFourierFeaturesTime(
            1, dit_args.timestep_features_dim, time_factor=1.0
        )
        self.timestep_features_r = FixedFourierFeaturesTime(
            1, dit_args.timestep_features_dim, time_factor=1.0
        )

        self.cfg_features = FixedFourierFeaturesTime(
            1, dit_args.timestep_features_dim, time_factor=1.0
        )
        cfg_features_dim = dit_args.timestep_features_dim

        self.to_timestep_embed = nn.Sequential(
            nn.Linear(
                dit_args.timestep_features_dim * 2 + cfg_features_dim,
                dit_args.inter_dim,
                bias=True,
            ),
            nn.SiLU(),
            nn.Linear(dit_args.inter_dim, dit_args.dim, bias=True),
            nn.SiLU(),
        )

        # Define new timestep embedding modules for logvar
        self.timestep_logvar = FourierFeaturesTime(1, dit_args.timestep_features_dim)
        self.to_logvar = nn.Sequential(
            nn.Linear(dit_args.timestep_features_dim * 2, 128, bias=True),
            nn.SiLU(),
            nn.Linear(128, 1, bias=True),
        )

    def forward(self, x, t, r, cond, mask) -> DictTensor:
        """运行旧 preprocessing 后替换 FlowMap 专用时间 embedding。"""
        # Run old preprocessing without r
        d = self.old_preprocessing(x, t, cond, mask)

        # Replace t and logvar with new preprocessing
        d["t"] = self.to_timestep_embed(
            torch.cat(
                [
                    self.timestep_features_t(t[:, None]),
                    self.timestep_features_r(r[:, None]),
                    self.cfg_features(cond["cfg"][:, None]),
                ],
                dim=-1,
            )
        )

        d["logvar"] = self.to_logvar(  # (b,)
            torch.cat(
                [
                    self.timestep_logvar(t[:, None]),
                    self.timestep_logvar(r[:, None]),
                ],
                dim=-1,
            )
        )[:, 0]

        return d


class FlowMapFromPretrained(
    nn.Module, BaseComponent, LatentDiffusionModelFlowMapPipeline
):
    """distilled Woosh-DFlow/DVFlow 的可加载 pipeline 组件。"""

    config_class = FlowMapPretrainedArgs

    def init_pretrained_model(self):
        """
        Initializes the pretrained model pipeline based on the configuration.
        Returns:
            Tuple:
                - ldm: The instantiated pretrained model (LatentDiffusionModel).
                - dit_config: The DiT configuration associated with the model.
        Raises:
            ValueError: If the model type is unsupported or unknown.
        """
        if self.config.pretrained_model_type == "ldm":
            ldm = LatentDiffusionModel(self.config.ldm)
            dit_config: DiTArgs = ldm.config.dit
            supported_models = ["mmmflux", "mmmssflux"]
            if ldm.config.dit.model_type not in supported_models:
                raise ValueError(
                    f"FlowMapPretrained only supports {supported_models}, got {ldm.config.dit.model_type}"
                )
        elif self.config.pretrained_model_type == "videokontext":
            ldm = VideoKontext(self.config.ldm)
            dit_config: DiTArgs = LatentDiffusionModel.resolve_config(
                ldm.config.ldm
            ).dit
        else:
            raise ValueError(
                f"Unknown pretrained_model_type {self.config.pretrained_model_type}"
            )
        return ldm, dit_config

    def __init__(self, config: FlowMapPretrainedConfig):
        # ==== Step 1: init of nn.Module
        super().__init__()

        # ==== Step 2: init of BaseComponent
        self.init_from_config(config)
        self.config: FlowMapPretrainedArgs  # now we use self.config knowing it has been validated

        # ==== Step 3: init student's LDM pipeline with adapted pre and postprocessing
        ldm, dit_config = self.init_pretrained_model()

        # Define student preprocessing (adapted for 2nd time step r)
        new_preprocessing = FlowMapPreprocessing(
            args=self.config,
            dit_args=dit_config,
            old_preprocessing=ldm.dit.preprocessing,
        )

        # Define student postprocessing (flip sign of teacher output)
        new_postprocessing = FlipSignPostprocessing(self.config, ldm.dit.postprocessing)

        # Cast v in float32 for JVP compatibility
        ldm.dit.set_cast_v(cast_v=True)
        new_layers = ldm.dit.layers

        # Creates new DiT and LDM pipelines for student
        dit = DiTFlowMapPipeline(
            preprocessing=new_preprocessing,
            postprocessing=new_postprocessing,
            layers=new_layers,
            non_checkpoint_layers=len(new_layers) + 1,  # we can't use checkpoints
            mask_out_before=dit_config.mask_out_before,
        )
        self.init_pipeline(
            dit=dit,
            autoencoder=ldm.autoencoder,
            conditioners=ldm.conditioners,
            sigma_data=ldm.sigma_data,
            pred_type=ldm.pred_type,
        )

        # Load state dict from its internal _weights_path
        self.load_from_config()
