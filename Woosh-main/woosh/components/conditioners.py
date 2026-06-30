"""Diffusion conditioner 的抽象接口和输出规格。"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Mapping
import torch.nn as nn


# get logger
log = logging.getLogger(__name__)


@dataclass
class ConditionConfig:
    """描述 conditioner 输出如何进入 diffusion model。

    Attributes:
        id: conditioner 内部返回字典的 key。
        shape: 单样本条件张量除 batch 外的逻辑形状。
        type: LDM 条件字典中的标准类型名，如 ``cross_attn_cond``。
    """

    id: str
    shape: List[int]
    type: str


class DiffusionConditioner(ABC):
    r"""Diffusion/LDM conditioner 的最小协议。"""

    def __init__(self) -> None:
        assert isinstance(self, nn.Module)

    @property
    @abstractmethod
    def output(self) -> Mapping[str, ConditionConfig]:
        r"""返回 conditioner 输出 key 到条件类型的映射。"""
        pass

    # @property
    # @abstractmethod
    # def trainable(self) -> bool:
    #     r"""
    #     deteremin if the conditioner is trainable and should be saved in the training checkpoint.
    #     """
    #     pass

    @abstractmethod
    def forward(
        self, batch, condition_dropout=0.0, no_cond=False, device=None, **kwargs
    ) -> Mapping:
        """根据 batch 构造条件字典，供 LDM ``get_cond`` 汇总。"""
        pass
