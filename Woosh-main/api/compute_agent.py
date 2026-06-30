"""API 侧模型加载与推理 agent。

agent 将 HTTP 请求参数转换为 Woosh 模型调用：加载 checkpoint、构造 latent
noise、准备文本条件、运行 FlowMap sampler，并通过 Woosh-AE 解码为 waveform。
"""

from abc import ABC, abstractmethod
import logging
import os
from typing import Mapping, Optional

from pydantic import BaseModel, ConfigDict
import torch
from enum import Enum
from .utils import CLAPCaptionPostprocessTransform
from woosh.inference.flowmap_sampler import sample_euler
from woosh.model.ldm import LatentDiffusionModel
from woosh.model.flowmap_from_pretrained import FlowMapFromPretrained
from woosh.components.base import LoadConfig
from woosh.utils.loading import catchtime

log = logging.getLogger(__name__)


# enum allows api verifications
class NoiseSchedulerEnum(str, Enum):
    karras = "karras"
    linear = "linear"
    sigmoid = "sigmoid"
    cosine = "cosine"


class SamplersEnum(str, Enum):
    heun = "heun"
    cfgpp = "cfgpp"


class GenerateArgs(BaseModel):
    """API 暴露的生成参数。

    当前实现主要使用 ``prompt``、``cfg``、``num_steps``、``seed`` 与
    ``model``；其余字段保留用于兼容其他 sampler 配置。
    """

    # special field to force not having extra arguments
    model_config = ConfigDict(extra="forbid")

    # request arguments:
    prompt: str = ""

    # general args
    cfg: float = 1
    sampler: SamplersEnum = SamplersEnum.heun
    num_steps: int = 100

    sigma_min: float = 1e-5
    sigma_max: float = 80
    rho: float = 7
    S_churn: float = 1
    S_min: float = 0
    S_max: float = float("inf")
    S_noise: float = 1
    guidance_scale: float = 7.5
    noise_scheduler: NoiseSchedulerEnum = NoiseSchedulerEnum.karras

    seed: Optional[int] = None

    model: str = "Woosh-DFlow"


class GenerateAgentInterface(ABC):
    @abstractmethod
    def load_model(self):
        """Loads the model into memory."""
        pass

    @abstractmethod
    def generate(self, *args, **kwargs) -> dict:
        """Generates audio from the model.

        Returns:
            dict: {"audio": waveform, "sample_rate": sample_rate}
        """
        pass

    @abstractmethod
    def cpu(self, *args, **kwargs):
        """moves the model to cpu, freeing gpu memory"""
        pass

    @abstractmethod
    def gpu(self, *args, **kwargs):
        """moves the model to gpu"""
        pass

    @abstractmethod
    def ready(self) -> bool:
        """Whether the model is loaded, on device, and ready for inference.

        Returns:
            bool: the agent is ready for inference, by calling generate.
        """
        pass


normalize_transform = CLAPCaptionPostprocessTransform(remove_punctuation=False)


def normalize_text(text):
    """复用 CLAP caption 后处理逻辑，规范化用户 prompt。"""
    # Normalize the text by removing special characters and replacing spaces with underscores
    res = normalize_transform({"captions": [text]})
    text = res["captions"][0]
    print("normalized text:", text)
    return text


class GenerateBasicAgent(GenerateAgentInterface):
    """基础 LDM agent，负责加载非蒸馏 LatentDiffusionModel。"""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        components_path="checkpoints/",
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.components_path = components_path

    def load_model(self):
        """从 ``components_path/model_name`` 加载 LatentDiffusionModel。"""
        model_name = self.model_name.strip()
        log.info(f"Loading `{model_name}` to  {self.device}")

        model_path = os.path.join(self.components_path, model_name)

        ldm = LatentDiffusionModel(config=LoadConfig(path=model_path))
        ldm._component_summary()
        ldm.eval()
        ldm = ldm.to(self.device)
        self.ldm = ldm

    def cpu(self):
        self.ldm = self.ldm.cpu()

    def gpu(self):
        self.ldm = self.ldm.to(self.device)

    def ready(self) -> bool:
        ldm_device = next(self.ldm.parameters()).device
        print(
            f"model {self.model_name} device: {ldm_device}, self.device: {torch.device(self.device)}"
        )
        return ldm_device != torch.device("cpu")

    def generate(self, args: GenerateArgs):
        raise NotImplementedError


class FlowMapGenerateAgent(GenerateBasicAgent):
    """面向 Woosh-DFlow/DVFlow 的 FlowMap 推理 agent。"""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        components_path="checkpoints/",
    ) -> None:
        super().__init__(model_name, device, components_path)

    def load_model(self):
        """加载 ``FlowMapFromPretrained`` student wrapper 及其子组件。"""
        model_name = self.model_name.strip()
        log.info(f"Loading `{model_name}` to  {self.device}")

        # TODO use path
        model_path = os.path.join(self.components_path, model_name)
        ldm = FlowMapFromPretrained(LoadConfig(path=model_path))
        ldm._component_summary()
        ldm.eval()
        ldm = ldm.to(self.device)
        self.ldm = ldm

    def generate(self, args: GenerateArgs):
        """执行一次 text-to-audio distilled 生成。

        数据流为 prompt -> text conditioner -> FlowMap Euler sampler ->
        AE inverse；latent shape 固定为 ``[1, 128, 501]``，对应约 5 秒音频。
        """
        rng_gen = torch.Generator()
        if args.seed is not None:
            rng_gen.manual_seed(args.seed)
        # @TODO do arbitrary length
        batch_size = 1
        length = 501
        dim = 128
        noise = torch.normal(
            0, 1, size=(batch_size, dim, length), generator=rng_gen
        ).to(self.device)

        description = args.prompt
        description = normalize_text(description)
        batch = {
            "description": [description] * batch_size,
            "audio": noise,
            # "x_original": reference,
            # "x2": reference,
            # "mask": mask,
        }

        cond = self.ldm.get_cond(
            batch,
            no_dropout=True,
            device=self.device,  # we must provide device if we don't provide x
        )
        cond["cfg"] = torch.ones((batch_size,), device=self.device) * args.cfg
        log.info(f"Generating with condition: {cond.keys()}")

        with torch.inference_mode():
            # with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            z = sample_euler(
                self.ldm, noise, cond=cond, num_steps=args.num_steps, renoise=0.5
            )
            x = self.ldm.autoencoder.inverse(z)
            audio_max = torch.max(torch.abs(x)).item()
            if audio_max > 1.0:
                x = x / audio_max
            x = x.cpu()
        # TODO add optional sample_rate arg
        sample_rate = 48000
        # if os.environ.get("WOOSH_RETURN_FP16") == "1":
        #     pass
        # else:
        #     x = x.half()
        log.info(f"Generated audio shape: {x.shape}, sample rate: {sample_rate}")
        x = x[0]
        audio = {"audio": x.cpu(), "sample_rate": sample_rate}
        return audio


class MultimodelGenerateAgent(GenerateAgentInterface):
    """管理多个模型 agent，并在推理前把目标模型移动到推理设备。"""

    def __init__(
        self,
        models: Mapping[str, str],
        device: str = "cuda",
        components_path="checkpoints/",
    ) -> None:
        """Multimodal generation agent for handling multiple models.

        Args:
            models (Mapping[str, str]): model_name to type (e.g. {"dfix-diffv3-ft2-ds3-b96": "ldm})
            device (str, optional): inference device. Defaults to "cuda".
            components_path (str, optional): components path. Defaults to "cache_dir/components/".
        """
        super().__init__()
        self.agents = {}
        for model_name, model_type in models.items():
            model_type = model_type.lower().strip()
            if model_type == "ldm":
                agent = GenerateBasicAgent(
                    model_name=model_name,
                    device=device,
                    components_path=components_path,
                )
            elif model_type == "flowmap":
                agent = FlowMapGenerateAgent(
                    model_name=model_name,
                    device=device,
                    components_path=components_path,
                )
            else:
                raise ValueError(
                    f"Unknown model type: {model_type} for model {model_name}"
                )
            self.agents[model_name] = agent

    def load_model(self):
        for name, agent in self.agents.items():
            log.info(f"Loading model: {name}")
            with catchtime(f"Loaded model {name}"):
                agent.load_model()
                agent.cpu()

    def cpu(self):
        for agent in self.agents.values():
            agent.cpu()

    def gpu(self):
        for agent in self.agents.values():
            agent.gpu()

    def ready(self) -> bool:
        # does not need to be on gpu for inference
        return True

    def generate(self, args: GenerateArgs):
        """按 ``args.model`` 选择子 agent 并返回生成音频字典。"""
        model_name = args.model
        if model_name not in self.agents:
            raise ValueError(f"Unknown model: {model_name}")
        agent = self.agents[model_name]
        if not agent.ready():
            log.info(f"Moving model `{model_name}` to {agent.device} for inference")
            self.cpu()
            agent.gpu()

        audio = agent.generate(args)
        return audio
