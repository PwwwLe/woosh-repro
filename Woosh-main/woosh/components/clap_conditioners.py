"""Woosh-CLAP 文本 conditioner。

该文件只使用 CLAP 的文本分支，把 prompt 编码成 LDM cross-attention token。
官方配置通常使用 RoBERTa-large，输出 ``text_cond`` 为
``[B, max_sentence_tokens, 1024]``，``text_mask`` 为 ``[B, max_sentence_tokens]``。
"""

import contextlib
import logging
from typing import Any, Mapping, Optional

import torch
from omegaconf import OmegaConf

from woosh.module.audioretrieval_module import (
    get_sentence_frontend_model,
    get_sentence_head_model,
    get_text_preprocessing_func,
)
from woosh.components.base import BaseComponent, ComponentConfig, LoadConfig
from woosh.utils.loading import lazy_loading

from .conditioners import ConditionConfig, DiffusionConditioner

# get logger
log = logging.getLogger(__name__)


class SFXCLAPTextConditionerConfig(ComponentConfig):
    """文本 conditioner 配置，通常由 Woosh-CLAP/TextConditioner checkpoint 提供。"""

    # sentence_config: OmegaConf
    sentence_config: Any
    last_hidden_state: bool = True
    use_shared_space: bool = False
    normalize_shared_space: bool = True
    freeze_clap: bool = True
    lhs_index: int = -2
    remove_special_tokens: bool = False
    eval_mode: bool = True
    trainable: bool = False

    # populated from pl.lightening module
    text_preprocessing: Optional[str] = None
    shared_representation_size: int = 512


def freeze_model(model):
    """冻结模块的所有参数并返回该模块。"""
    for param in model.parameters():
        param.requires_grad = False
    return model


class SFXCLAPTextConditioner(torch.nn.Module, BaseComponent, DiffusionConditioner):
    r"""将 batch 中的 ``description`` 文本编码为 diffusion 条件序列。"""

    config_class = SFXCLAPTextConditionerConfig

    def __init__(self, config: SFXCLAPTextConditionerConfig | LoadConfig, **kwargs):
        super().__init__()
        self.init_from_config(config, **kwargs)
        self.config: SFXCLAPTextConditionerConfig
        # self.config = SFXCLAPTextConditionerConfig(**self.config)  # type: ignore
        # init text preprocessing
        self.text_preprocessing = get_text_preprocessing_func(
            self.config.text_preprocessing
        )
        with lazy_loading():
            self.sentence_frontend, self.tokenizer, text_output_size = (
                get_sentence_frontend_model(self.config.sentence_config)
            )

        self.sentence_head = get_sentence_head_model(
            self.config.sentence_config,
            self.config.shared_representation_size,
            text_output_size,
        )
        if not self.config.trainable:
            freeze_model(self.sentence_frontend)
            freeze_model(self.sentence_head)

    @property
    def output(self) -> Mapping[str, ConditionConfig]:
        """声明 forward 返回的条件 key 与 LDM 标准条件类型。"""
        if self.config.last_hidden_state:
            return {
                "text_cond": ConditionConfig(
                    id="text_cond",
                    shape=[self.config.sentence_config["max_sentence_tokens"]],
                    type="cross_attn_cond",
                ),
                "text_mask": ConditionConfig(
                    id="text_mask",
                    shape=[self.config.sentence_config["max_sentence_tokens"]],
                    type="cross_attn_cond_mask",
                ),
            }
        return {
            "text_global": ConditionConfig(
                id="text_cond",
                shape=[self.config.sentence_config["max_sentence_tokens"]],
                type="global_cond",
            )
        }

    @property
    def trainable(self) -> bool:
        return self.config.trainable

    def freeze_grad_context(self):
        """根据配置返回 no_grad/nullcontext，避免冻结模型产生梯度。"""
        if self.config.trainable:
            cm = torch.no_grad()
        else:
            cm = contextlib.nullcontext()
        return cm

    def tokenize_text(self, text_list):
        """预处理并 tokenizer 文本，返回 Hugging Face token batch 与 captions。"""
        captions = self.text_preprocessing(text_list)  # type: ignore

        tokenized = self.tokenizer(
            captions,
            add_special_tokens=True,
            # padding=True,
            padding="max_length",
            truncation=True,  # truncate to longest in batch, otherwise to max_length
            return_tensors="pt",
            max_length=self.config.sentence_config.max_sentence_tokens,  # type: ignore
        )
        return tokenized, captions

    def forward(
        self, batch, condition_dropout=0.0, no_cond=False, device=None, **kwargs
    ) -> Mapping:
        """从 batch 构造文本条件。

        Args:
            batch: 至少包含 ``audio`` 或 ``id``；可选 ``description``。
            condition_dropout: 训练时随机替换为空 prompt 的概率。
            no_cond: 为 CFG 构造无条件分支时设为 True。

        Returns:
            Mapping: ``text_cond``/``text_mask`` 或 ``text_global``。
        """
        if self.config.eval_mode:
            self.sentence_frontend.eval()
            self.sentence_head.eval()

        device = device if device is not None else batch["audio"].device

        if "description" in batch:
            descriptions = [
                desc if desc is not None else "" for desc in batch["description"]
            ]
        else:
            descriptions = [
                ""
                for _ in range(
                    len(batch["id"]) if "id" in batch else batch.get("audio").shape[0]
                )
            ]

        # @TODO use clap normaizing text transform

        # batch_size
        B = len(descriptions)
        # if no_dropout=False, condition dropout is enabled with p=self.condition_dropout
        dropout_description = kwargs.get("dropout_description", "")

        if condition_dropout > 0.0:
            descriptions = [
                desc if u.item() > condition_dropout else dropout_description
                for desc, u in zip(descriptions, torch.rand((B,)))
            ]
        else:
            descriptions = list(descriptions)

        if no_cond:
            descriptions = [dropout_description for _ in descriptions]

        # tokenize the descriptions
        tokenized, captions = self.tokenize_text(descriptions)
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)

        with self.freeze_grad_context():
            sentence_out = self.sentence_frontend(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            if self.config.last_hidden_state:
                lhs_embedings = sentence_out["last_hidden_state"]
                if self.config.lhs_index is not None:
                    lhs_embedings = sentence_out["hidden_states"][self.config.lhs_index]
                if self.config.remove_special_tokens:
                    # mask out special tokens
                    # POS=0, EOS=2, empty=1
                    # checked for roberta, if it is not roberta, check double the input ids of special tokens
                    attention_mask[input_ids <= 2] = 0
                cond = {
                    "text_cond": lhs_embedings,
                    "text_mask": attention_mask,
                }
                return cond

            token_embeddings = sentence_out["last_hidden_state"]
            if self.config.sentence_config.get("pool_type", "eos") == "eos":
                sentence_features = token_embeddings[:, 0, :]
            else:
                raise NotImplementedError(
                    f"{self.config.sentence_config['pool_type']} not implemented"
                )
            if self.config.use_shared_space:
                shared_sentence_features = self.sentence_head(sentence_features)
                if self.config.normalize_shared_space:
                    shared_sentence_features = torch.nn.functional.normalize(
                        shared_sentence_features, p=2, dim=1
                    )
                sentence_features = shared_sentence_features

            cond = {
                "text_global": sentence_features,
                "description": captions,
            }
        return cond

    @classmethod
    def from_audioretrieval_module(
        cls,
        module,
        **kwargs,
    ) -> "SFXCLAPTextConditioner":
        """从完整 CLAP 检索模块导出仅文本分支的 conditioner。"""
        plconfig = OmegaConf.create(module._hydra_external_config)
        config = SFXCLAPTextConditionerConfig(
            sentence_config=plconfig.sentence,
            shared_representation_size=plconfig.shared_representation_size,
            text_preprocessing=plconfig.get("text_preprocessing"),
        )

        model = SFXCLAPTextConditioner(config)
        model.sentence_frontend.load_state_dict(module.sentence_frontend.state_dict())
        model.sentence_head.load_state_dict(module.sentence_head.state_dict())

        # test model
        test_clap_conditioner(model, module)
        return model


def test_clap_conditioner(clapconditioner: SFXCLAPTextConditioner, module):
    """对比导出的 conditioner 与原 CLAP 模块隐藏层输出是否一致。"""
    module.cuda()
    clapconditioner.cuda()
    module.eval()
    # test model
    desc = ["hello test", "world, test, other"]
    ar_batch = {
        "id": [str(i) for i in range(len(desc))],
        "audio": torch.zeros((len(desc), 1, 16000)).cuda(),
        "captions": desc,
    }

    cbatch = {
        "id": [str(i) for i in range(len(desc))],
        "audio": torch.zeros((len(desc), 1, 16000)).cuda(),
        "description": desc,
    }
    with torch.no_grad():
        ar_batch = module.forward_sentence_model(
            ar_batch,
            return_last_hidden_state=True,
            output_hidden_states=True,
        )

    clapcond = clapconditioner(cbatch)

    x1 = clapcond["text_cond"]
    x2 = ar_batch["hidden_states"][clapconditioner.config.lhs_index]
    assert torch.allclose(x1, x2, atol=1e-3)
    module.cpu()
    clapconditioner.cpu()
