"""Woosh 组件的配置、保存和加载基础设施。

所有可持久化模型组件都继承 ``BaseComponent``：构造时解析 Pydantic 配置，
按 ``LoadConfig(path=...)`` 定位 ``config.yaml`` 与 ``weights.*``，并通过
子组件注册机制控制哪些参数写入父 checkpoint。
"""

import logging
import os

from typing import Any, Dict, List, Mapping, Optional, Set, Union
import copy
import torch

from omegaconf import OmegaConf
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn
from pydantic import BaseModel, ConfigDict


rank = 0
# get logger
log = logging.getLogger(__name__)


def _is_load_config(v) -> str:
    """区分 ``LoadConfig`` 和普通组件参数。

    该函数被 Pydantic ``Discriminator`` 调用：只要配置中存在非空 ``path``
    就表示该组件应从磁盘目录加载已有配置和权重。

    Returns:
        str: ``"load_config"`` 或 ``"component_args"``。
    """
    if "path" in v and v["path"] is not None:
        return "load_config"
    if hasattr(v, "path") and v.path is not None:
        return "load_config"
    return "component_args"


class ComponentConfig(BaseModel):
    """所有 Woosh component 共享的最小配置字段。"""

    # === special field to force not having extra arguments
    model_config = ConfigDict(extra="forbid")
    # all components must define exclude_from_checkpoint
    exclude_from_checkpoint: bool = False
    trainable: bool = True


class LoadConfig(ComponentConfig):
    """指向已有 checkpoint 目录的配置。

    除 ``path`` 外允许额外字段，这些字段会覆盖磁盘上的 ``config.yaml``。
    """

    # allow extra args
    # these will be used to overwrite the config
    path: str
    model_config = ConfigDict(
        extra="allow",
    )


def human_format(num):
    """把参数量等数字格式化为 K/M/B/T 后缀。"""
    num = float("{:.3g}".format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return "{}{}".format(
        "{:f}".format(num).rstrip("0").rstrip("."), ["", "K", "M", "B", "T"][magnitude]
    )


def recursive_update_config(config, update):
    r"""
    Recursively update a config dictionary with another dictionary.
    Return a new instance.
    """
    for k, v in update.items():
        if isinstance(v, Mapping):
            config[k] = recursive_update_config(config.get(k, {}), v)
        else:
            config[k] = v
    return config


def find_common_tensors_from_storage(state_dict_a, state_dict_b):
    """按底层 storage 地址找出两个 state_dict 共享的 Tensor。

    用于父组件注册子组件时识别同一参数在两个 state_dict 中的不同 key, 从而
    只保存需要由父 checkpoint 管理的参数。
    """

    def mem(p):
        # return p.data.storage().data_ptr()
        return p.data.untyped_storage().data_ptr()

    mem_a = {mem(p) for p in state_dict_a.values()}
    mem_b = {mem(p) for p in state_dict_b.values()}
    intersection = mem_a & mem_b

    filtered_state_dict_a = {
        k: v for k, v in state_dict_a.items() if mem(v) not in intersection
    }
    filtered_state_dict_b = {
        k: v for k, v in state_dict_b.items() if mem(v) not in intersection
    }
    shared_state_dict_a = {
        k: v for k, v in state_dict_a.items() if mem(v) in intersection
    }

    inverse_mapping_b = {
        mem(v): k for k, v in state_dict_b.items() if mem(v) in intersection
    }

    a_to_b_mapping = {
        k: inverse_mapping_b[mem(v)] for k, v in shared_state_dict_a.items()
    }

    return (
        filtered_state_dict_a,
        filtered_state_dict_b,
        shared_state_dict_a,
        a_to_b_mapping,
    )


class BaseComponent:
    r"""所有可加载/保存 Woosh component 的基类。

    子类通常同时继承 ``torch.nn.Module``，并在 ``__init__`` 中调用
    ``init_from_config``。``config_class`` 指定 Pydantic 配置模型。
    """

    config_class = ComponentConfig
    available_config_formats = ["yaml"]
    available_weight_formats = ["safetensors", "pt"]

    def __init__(self):
        assert isinstance(self, nn.Module)
        super(BaseComponent, self).__init__()

    def init_from_config(self, config):
        """解析配置并初始化 checkpoint 相关状态。

        如果传入 ``LoadConfig``，会先读取磁盘配置并记录权重路径；如果传入普通
        dict/Pydantic 配置，则直接验证为 ``self.config_class``。
        """
        # Components must have an attached Pydantic BaseModel class
        assert self.config_class is not None
        assert issubclass(self.config_class, BaseModel)

        # if config is a ComponentConfig,
        # then the model will being loaded from a pretrained model
        # and _weights_path will be set
        self.config, self._weights_path = self.resolve_config(
            config, return_weights_path=True
        )
        # assert isinstance(self.config, ComponentConfig)

        self._subcomponents: Dict[str, BaseComponent] = {}
        self._subcomponents_configs = []

        # exclude_from_checkpoint is a boolean that indicates if the component
        # should be saved in the checkpoint WHEN part of a larger component
        # unused otherwise
        assert hasattr(self.config, "exclude_from_checkpoint"), (
            f"The configuration for {type(self).__name__} must have an "
            f"'exclude_from_checkpoint' attribute."
            f"Consider overriding it when loading the model from a checkpoint."
        )

        self._exclude_from_checkpoint = self.config.exclude_from_checkpoint
        self._trainable = self.config.trainable

        # exclude_subcomponents_from_checkpoint is a list of subcomponents that should not be saved in the checkpoint
        self._exclude_subcomponents_from_checkpoint = []

        # unset, will be computed after the model has been initialized
        # when registering subcomponents
        self._excluded_parameters_: Optional[Set[str]] = None
        self._included_parameters_: Optional[Set[str]] = None

    @property
    def _included_parameters(self):
        """
        This is to ensure a default behaviour if we don't register any component
        """
        assert isinstance(self, nn.Module)
        if self._included_parameters_ is None:
            log.debug(
                "Initializing included parameters, make sure this is called once all submodules & parameters are registered"
            )
            self._included_parameters_ = set(self.state_dict().keys())
            self._excluded_parameters_ = set()

        return self._included_parameters_

    @_included_parameters.setter
    def _included_parameters(self, value):
        self._included_parameters_ = value

    @property
    def _excluded_parameters(self):
        """
        This is to ensure a default behaviour if we don't register any component
        """
        assert isinstance(self, nn.Module)
        if self._excluded_parameters_ is None:
            log.debug(
                "Initializing excluded parameters, make sure this is called once all submodules & parameters are registered"
            )
            self._excluded_parameters_ = set()
            self._included_parameters_ = set(self.state_dict().keys())

        return self._excluded_parameters_

    @_excluded_parameters.setter
    def _excluded_parameters(self, value):
        self._excluded_parameters_ = value

    @classmethod
    def resolve_config(cls, config, return_weights_path=False):
        """解析单个组件配置，不递归初始化子组件。"""
        overwrite_kwargs = {}
        weights_path = None

        # check if LoadConfig is passed
        # knowing that it can be given as a plain dict with path field
        # get the config with component_args and load the model
        # or a model parameters
        # this sets the _weights_path

        if _is_load_config(config) == "load_config":
            if isinstance(config, Mapping):
                config = LoadConfig(**config)
            # overwrite kwargs are in the config
            # everything else than path
            overwrite_kwargs = config.model_dump(exclude={"path"})
            config, weights_path = cls._config_and_weightspath_from_path(config.path)
            # config is now supposed to be a config_class, not LoadConfig

        # Validate the config using the config_class
        if not isinstance(config, cls.config_class):
            config = cls.config_class(**config)

        # add extra kwargs if present in LoadConfig
        # and validate
        # config = config.model_copy(update=overwrite_kwargs)
        # model_copy(update=overwrite_kwargs) doesn't support nested dicts
        # see https://github.com/pydantic/pydantic/issues/7387
        # see https://github.com/SonyResearch/project_mfm_sfxfm/issues/1302
        config = recursive_update_config(config.model_dump(), overwrite_kwargs)
        config = cls.config_class.model_validate(config)

        if return_weights_path:
            return config, weights_path
        else:
            return config

    def register_subcomponent(
        self,
        name: str,
        subcomponent: "BaseComponent",
        subkey: Optional[str] = None,
    ):
        """注册子组件并更新父组件保存/排除参数集合。

        ``exclude_from_checkpoint=True`` 的子组件只在父配置里保存路径，权重由
        子目录独立加载；否则会只排除子组件自己声明要排除的参数。
        """
        assert isinstance(self, nn.Module) and isinstance(subcomponent, nn.Module)
        self._subcomponents: Dict[str, BaseComponent]
        self._exclude_subcomponents_from_checkpoint: List[str]

        exclude_from_checkpoint = subcomponent._exclude_from_checkpoint

        self._subcomponents_configs.append(
            {
                "config": subcomponent.config,
                "subcomponent_path": name,  # place of the subcomponent relative to its parent component
                "exclude_from_checkpoint": exclude_from_checkpoint,
            }
        )

        # subcomponent = getattr(self, name)
        if subkey is not None:
            # subcomponent = getattr(subcomponent, subkey)
            name = f"{name}.{subkey}"
        self._subcomponents.update({name: subcomponent})

        # Compute parameters to remove from self.state_dict
        subcomponent_state_dict_to_exclude = subcomponent.state_dict()
        if exclude_from_checkpoint:
            self._exclude_subcomponents_from_checkpoint.append(name)
        else:
            # Do not exclude all parameters of the subcomponent
            # if exclude_from_checkpoint is False
            subcomponent_state_dict_to_exclude = {
                k: v
                for k, v in subcomponent_state_dict_to_exclude.items()
                if k not in subcomponent._included_parameters
            }

        A, B, C, A_to_B = find_common_tensors_from_storage(
            state_dict_a=self.state_dict(),
            state_dict_b=subcomponent_state_dict_to_exclude,
        )

        self._included_parameters = self._included_parameters - set(C.keys())
        assert self._excluded_parameters_ is not None
        self._excluded_parameters_ = self._excluded_parameters_ | set(C.keys())

    def register_subcomponent_dict(
        self,
        name: str,
        component_dict: Dict[str, Any] | nn.ModuleDict = {},
    ):
        """注册 ``ModuleDict``/dict 中的所有 ``BaseComponent`` 子项。"""
        if component_dict == {}:
            log.warning(
                f"No subcomponents to register in register_subcomponent_dict for attribute {name} as component_dict is empty"
            )
        for k, component in component_dict.items():
            if isinstance(component, BaseComponent):
                self.register_subcomponent(name, subcomponent=component, subkey=k)

    def save(self, path, config_format="yaml", weights_format="safetensors"):
        """保存 ``config.yaml`` 与过滤后的 ``weights.*``。"""
        assert isinstance(self, nn.Module)
        if rank != 0:
            return

        assert config_format in self.available_config_formats, (
            f"config_format={config_format} should be one of {(*self.available_config_formats,)}"
        )
        assert weights_format in self.available_weight_formats, (
            f"weights_format={weights_format} should be one of {(*self.available_weight_formats,)}"
        )

        try:
            umask = os.umask(0o002)

            # creat the save directory
            os.makedirs(path, exist_ok=True)
            config_dict = self.config.model_dump()  # type: ignore

            # if is_dataclass(config_dict):
            #     config_dict = asdict(config_dict)  # type: ignore
            # config_dict = OmegaConf.create(config_dict)

            # saving config
            save_config_path = os.path.join(path, f"config.{config_format}")
            log.info(f"Saving config of {type(self)} to {save_config_path}")
            if config_format == "yaml":
                with open(save_config_path, "w") as outfile:
                    OmegaConf.save(config_dict, outfile)
                    # yaml.dump(config_dict, outfile, default_flow_style=False)
            else:
                # should never happen
                raise NotImplementedError(
                    f"config_format={config_format} not implemented"
                )
            # saving weights
            save_wights_parh = os.path.join(path, f"weights.{weights_format}")
            log.info(f"Saving weights of {type(self)} to {save_wights_parh}")

            state_dict = self.state_dict()
            self.filter_state_dict_(state_dict)

            if weights_format == "safetensors":
                save_file(state_dict, save_wights_parh)
            elif weights_format == "pt":
                torch.save(state_dict, save_wights_parh)
            else:
                # should never happen
                raise NotImplementedError(
                    f"config_format={config_format} not implemented"
                )
        finally:
            os.umask(umask)  # type: ignore

    def filter_state_dict_(self, state_dict, prefix="") -> None:
        """原地删除不应由当前组件保存的 state_dict key。"""
        if prefix != "":
            assert prefix.endswith(".")
        for k in copy.copy(list(state_dict.keys())):
            if (
                k.startswith(prefix)
                and k.removeprefix(prefix) in self._excluded_parameters
            ):
                del state_dict[k]

    def add_filtered_state_dict_keys_(self, incomplete_state_dict, prefix="") -> None:
        """
        In place
        Adds filtered out keys from the full (unfiltered) state dict to the incomplete_state_dict
        This is the opposite of filter_state_dict

        We only add the keys that were excluded from the incomplete_state_dict
        to ensure that other missing keys are still

        prefix: where to insert the keys in the incomplete_state_dict
        """
        assert isinstance(self, nn.Module)
        if prefix != "":
            assert prefix.endswith(".")
        state_dict = self.state_dict()
        incomplete_state_dict.update(
            {
                prefix + k: v
                for k, v in state_dict.items()
                if k in self._excluded_parameters
            }
        )

    def load_from_config(self):
        """按 ``_weights_path`` 加载当前组件，再递归加载被排除的子组件。"""
        # load non excluded_from_checkpoints parameters of the component
        if self._weights_path is not None:
            self._load_statedict_from_disk()

        # load excluded_from_checkpoints subcomponents
        for component_name, component in self._subcomponents.items():
            if component_name in self._exclude_subcomponents_from_checkpoint:
                component.load_from_config()

    def _load_statedict_from_disk(self, only_return_state_dict=False, strict=True):
        """_summary_

        Args:
            only_return_state_dict (bool, optional): Don't loaded the state_dict into the model but rather return it. Can be useful for lazyloading. Defaults to False.

        Raises:
            ValueError: if the _weights_path is not set, can be due to not using the from_pretrained method.
            NotImplementedError: if the format of the weights file is not implemented.

        Returns:
            dict: loaded state_dict if only_return_state_dict is True
        """
        if self._weights_path is None:
            raise ValueError(
                f"Cannot load weights, _weights_path is not set. Did you load this model using {type(self)}.from_pretrained?"
            )
        weights_path = self._weights_path
        weights_format = weights_path.rsplit(".", 1)[-1]
        log.info(f"Loading weights for {type(self).__name__} from {weights_path}")
        if weights_format == "pt":
            state_dict = torch.load(weights_path)
            if only_return_state_dict:
                return state_dict
            self._load_state_dict(state_dict)
        elif weights_format == "safetensors":
            state_dict = {}
            with safe_open(weights_path, framework="pt") as f:  # type: ignore
                for k in f.keys():
                    state_dict[k] = f.get_tensor(k)
            if only_return_state_dict:
                return state_dict

            self._load_state_dict(state_dict)
        else:
            # should never happen
            raise NotImplementedError(
                f"weights_format={weights_format} not implemented"
            )

    def _load_state_dict(self, state_dict):
        """
        Tries to load_state_dict in strict mode, if it fails, retries in non-strict mode
        Logs the success or failure of the loading

        Components that have subcomponents excluded from the checkpoint
        are expected to fail with strict=True
        """
        assert isinstance(self, nn.Module)
        try:
            self.load_state_dict(state_dict, strict=True)
            log.info(f"Loaded state_dict for {type(self).__name__} in strict mode")
        except RuntimeError as e:
            if os.environ.get("WOOSH_VERBOSE_LOADING_ERROR", "0") == "1":
                log.error(
                    f"Error loading state_dict in strict mode for {type(self).__name__}: {e}"
                )
            log.info(f"Error loading state_dict in strict mode: {type(self).__name__}")
            log.info("Retrying in non-strict mode")
            self.load_state_dict(state_dict, strict=False)

    def _load_from_module_checkpoint(
        self,
        checkpoint,
        prefix="",
    ):
        """
        Select only the relevant keys as specified by prefix
        and adapt their names
        prefix must contain the trailing dot and be the full path to the component
        in the module e.g. "ldm."

        """
        if prefix != "":
            assert prefix.endswith(".")

        state_dict = checkpoint["state_dict"]

        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith(prefix):
                new_state_dict[k[len(prefix) :]] = v

        self._load_state_dict(new_state_dict)

    def config_from_pretrained(
        self,
        path: Optional[Union[str, os.PathLike]],
    ):
        """Loads a config from a path
        and sets the _weights_path to the weights file

        Args:
            path (Optional[Union[str, os.PathLike]]): path to the model

        Raises:
            FileNotFoundError: if the save path does not have config and weights.

        Returns:
            BaseComponent: the loaded model.
        """
        config, weights_path = self._config_and_weightspath_from_path(path)

        self._weights_path = weights_path

        return config

    @classmethod
    def _config_and_weightspath_from_path(cls, path):
        """
        Securely loads a config from a path
        and returns the config and the weights path
        """
        if path is None:
            raise NotImplementedError("path must be provided")
        # finding suitable config
        config_format = "invalid"
        for config_format in cls.available_config_formats:
            if os.path.isfile(os.path.join(path, f"config.{config_format}")):
                break
        else:
            # no config file
            raise FileNotFoundError(
                f"No config file found in {path}, make sure that config.{(*cls.available_config_formats,)} exists."
            )

        # finding suitable weights file
        for weights_format in cls.available_weight_formats:
            if os.path.isfile(os.path.join(path, f"weights.{weights_format}")):
                break
        else:
            # no config file
            raise FileNotFoundError(
                f"No weights file found in {path}, make sure that config.{(*cls.available_weight_formats,)} exists."
            )
        # loading config
        config_path = os.path.join(path, f"config.{config_format}")
        log.info(f"Loading config from {config_path}")
        # log.warning(
        #     f"Weights are not loaded for {config_path}, don't forget to call load_from_config"
        # )
        if config_format == "yaml":
            config = OmegaConf.load(config_path)
            # with open(config_path, "r") as infile:
            #     config = yaml.load(infile, Loader=yaml.FullLoader)
        else:
            # should never happen
            raise NotImplementedError(f"config_format={config_format} not implemented")

        # cast and verify config
        if cls.config_class is not None:
            config = cls.config_class(**config)

        weights_path = os.path.join(path, f"weights.{weights_format}")
        return config, weights_path

    @classmethod
    def from_pretrained(
        cls,
        path: Optional[Union[str, os.PathLike]],
        *model_args,
        **kwargs,
    ):
        """Load a model from a path

        Args:
            path (Optional[Union[str, os.PathLike]]): path to the model
            *model_args: not supported yet.
            **kwargs: config updates to the pretrained model.

        Raises:
            FileNotFoundError: if the save path does not have config and weights.

        Returns:
            BaseComponent: the loaded model.
        """
        if path is None:
            raise NotImplementedError("path must be provided")
        # finding suitable config
        config_format = "invalid"
        for config_format in cls.available_config_formats:
            if os.path.isfile(os.path.join(path, f"config.{config_format}")):
                break
        else:
            # no config file
            raise FileNotFoundError(
                f"No config file found in {path}, make sure that config.{(*cls.available_config_formats,)} exists."
            )

        # finding suitable weights file
        for weights_format in cls.available_weight_formats:
            if os.path.isfile(os.path.join(path, f"weights.{weights_format}")):
                break
        else:
            # no config file
            raise FileNotFoundError(
                f"No weights file found in {path}, make sure that config.{(*cls.available_weight_formats,)} exists."
            )
        # loading config
        config_path = os.path.join(path, f"config.{config_format}")
        log.info(f"Loading config for {cls.__name__} from {config_path}")
        if config_format == "yaml":
            config = OmegaConf.load(config_path)
            # with open(config_path, "r") as infile:
            #     config = yaml.load(infile, Loader=yaml.FullLoader)
        else:
            # should never happen
            raise NotImplementedError(f"config_format={config_format} not implemented")

        # cast and verify config
        if cls.config_class is not None:
            config = cls.config_class(**config)

        # init object
        obj = cls(config, *model_args, **kwargs)

        # loading weights
        weights_path = os.path.join(path, f"weights.{weights_format}")
        obj._weights_path = weights_path

        # TODO: lazy loading?!
        # if not in a lazy loading, do actually load the weights
        if not woosh.utils.loading.lazy_loading_enabled:
            obj._load_statedict_from_disk()

        return obj

    def freeze_non_trainable_components(self):
        """冻结配置中 ``trainable=False`` 的组件树分支。"""
        assert isinstance(self, nn.Module)
        if not self._trainable:
            assert self._exclude_from_checkpoint, (
                "Strange case, where parameters are not trainable but should not be saved in the checkpoint"
            )
            self.requires_grad_(False)
            # self.eval()
        else:
            for k, subcomponent in self._subcomponents.items():
                subcomponent.freeze_non_trainable_components()

    def _component_summary(self, prefix="", depth=0):
        """打印组件树、参数量和 checkpoint 排除关系，供调试加载流程使用。"""
        assert isinstance(self, nn.Module)

        filtered_state_dict = self.state_dict()
        self.filter_state_dict_(filtered_state_dict, prefix="")
        num_params = sum(
            [p.numel() for p in filtered_state_dict.values()]  # type: ignore
        )  # type: ignore

        if prefix != "":
            prefix = prefix + "."

        num_params_all = sum(p.numel() for p in self.state_dict().values())
        print(f"{' | ' * depth + '--- '}{prefix}{type(self).__name__}")
        print(f"{' | ' * (depth + 1) + '     * '}(from_weights={self._weights_path})")
        print(f"{' | ' * (depth + 1) + '     * '}(trainable={self._trainable})")
        print(
            f"{' | ' * (depth + 1) + '     * '}(excluded_from_checkpoint={self._exclude_subcomponents_from_checkpoint})"
        )
        print(
            f"{' | ' * (depth + 1) + '     * '}(Total number of parameters={human_format(num_params_all)})"
        )
        print(
            f"{' | ' * (depth + 1) + '     * '}(Component parameters={human_format(num_params)})"
        )
        print(
            f"{' | ' * (depth + 1) + '     * '}(num_included_tensors={len(self._included_parameters):,})"
        )
        assert self._excluded_parameters_ is not None
        print(
            f"{' | ' * (depth + 1) + '     * '}(num_excluded_tensors={len(self._excluded_parameters_):,})"
        )

        for component_name, component in self._subcomponents.items():
            component._component_summary(
                prefix=prefix + f"{component_name}", depth=depth + 1
            )
